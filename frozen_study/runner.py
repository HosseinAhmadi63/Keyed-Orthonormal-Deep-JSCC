import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse, atexit, contextlib, copy, csv, ctypes, datetime, gc, hashlib, importlib.metadata, json, math, platform, secrets, shutil, sys, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from config import CONFIG
from codec_models import Encoder, Decoder
from crypto_scrambler import CryptoScrambler, nonce_features, run_self_tests, build_test_vectors
from experiment_utils import OnlineClassifier, write_json, save_torch, sha, normalize, cipher_torch, load_data, split_indices, encode, image_metrics, summarize_images, class_metrics, return_archives
from fit_attackers import fit_all_and_lock, evaluate_all, load_candidate, fit_candidate, features
HERE = Path(__file__).resolve().parent
VERSION = '2.0.0-gpu1'

def prefix(stage, tag):
    return int.from_bytes(stage.encode('ascii') + int(tag).to_bytes(2, 'big') + bytes(8), 'big')

def context(device, cfg):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and cfg['amp_bfloat16'] else contextlib.nullcontext()

def train_codec(images, labels, fit, val, seed, weight, key_hex, tag, cfg, folder, device, smoke):
    folder.mkdir(exist_ok=True, parents=True)
    if (folder / 'training_complete.json').exists():
        return
    torch.manual_seed(seed)
    enc = Encoder(256).to(device)
    dec = Decoder(256).to(device)
    clf = OnlineClassifier().to(device)
    optimizer = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=cfg['codec_lr'], weight_decay=0.0001)
    opt_clf = torch.optim.AdamW(clf.parameters(), lr=0.0002, weight_decay=0.0001)
    crypto = CryptoScrambler(bytes.fromhex(key_hex), 256)
    history = []
    best = float('inf')
    first = 1
    epochs = 2 if smoke else cfg['codec_epochs']
    batch = 4 if smoke else cfg['codec_batch']
    last = folder / 'last.pt'
    if last.exists():
        saved = torch.load(last, map_location=device, weights_only=False)
        enc.load_state_dict(saved['enc'])
        dec.load_state_dict(saved['dec'])
        clf.load_state_dict(saved['clf'])
        optimizer.load_state_dict(saved['optimizer'])
        opt_clf.load_state_dict(saved['opt_clf'])
        history = saved['history']
        best = saved['best']
        first = saved['epoch'] + 1
    valnoise = np.random.default_rng(cfg['split_seed'] + 90).normal(0, 10 ** (-cfg['snr_bob_db'] / 20), (len(val), 256)).astype(np.float32)
    for epoch in range(first, epochs + 1):
        start = time.monotonic()
        enc.train()
        dec.train()
        clf.train()
        order = np.random.default_rng(seed + 1000 * epoch).permutation(fit)
        noise = torch.Generator(device=device).manual_seed(seed + 100000 * epoch)
        sums = {'mse': 0.0, 'kl_uniform': 0.0, 'online_cross_entropy': 0.0}
        seen = 0
        for i in range(0, len(order), batch):
            ii = order[i:i + batch]
            x = torch.as_tensor(images[ii].copy(), device=device).float() / 255
            y = torch.as_tensor(labels[ii], device=device)
            ns = [prefix('TR', tag) + (epoch - 1) * len(fit) + i + j for j in range(len(ii))]
            nf = torch.as_tensor(nonce_features(ns), device=device)
            params = crypto.parameters_batch(ns)
            with context(device, cfg):
                z = enc(x)
            u = normalize(z)
            cipher = cipher_torch(u, params)
            eve = cipher + torch.randn(cipher.shape, device=device, generator=noise) * 10 ** (-cfg['snr_eve_db'] / 20)
            for _ in range(cfg['online_classifier_steps']):
                with context(device, cfg):
                    logits = clf(eve.detach(), nf)
                ce = F.cross_entropy(logits.float(), y)
                opt_clf.zero_grad(set_to_none=True)
                ce.backward()
                nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
                opt_clf.step()
            for p in clf.parameters():
                p.requires_grad_(False)
            bob = u + torch.randn(u.shape, device=device, generator=noise) * 10 ** (-cfg['snr_bob_db'] / 20)
            with context(device, cfg):
                pred = dec(bob)
                adv = clf(eve, nf)
            mse = F.mse_loss(pred.float(), x)
            logp = F.log_softmax(adv.float(), dim=1).clamp(-30, 0)
            kl = F.kl_div(logp, torch.full_like(logp, 0.1), reduction='batchmean')
            loss = mse + weight * kl
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite codec loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
            optimizer.step()
            for p in clf.parameters():
                p.requires_grad_(True)
            for k, v in [('mse', mse), ('kl_uniform', kl), ('online_cross_entropy', ce)]:
                sums[k] += float(v.detach()) * len(ii)
            seen += len(ii)
        enc.eval()
        dec.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for i in range(0, len(val), batch):
                ii = val[i:i + batch]
                x = torch.as_tensor(images[ii].copy(), device=device).float() / 255
                u = normalize(enc(x))
                p = dec(u + torch.as_tensor(valnoise[i:i + len(ii)], device=device))
                total += float((p - x).square().mean((1, 2, 3)).sum())
                n += len(ii)
        metric = total / n
        row = {'epoch': epoch, 'validation_mse': metric, 'seconds': time.monotonic() - start, 'fitting': {k: v / seen for k, v in sums.items()}, 'codec_updates': math.ceil(len(fit) / batch), 'new_unique_training_packets': seen, 'nonce_first': hex(prefix('TR', tag) + (epoch - 1) * len(fit)), 'nonce_last': hex(prefix('TR', tag) + epoch * len(fit) - 1)}
        history.append(row)
        if metric < best:
            best = metric
            save_torch(folder / 'best_codec.pt', {'enc': {k: v.detach().cpu() for k, v in enc.state_dict().items()}, 'dec': {k: v.detach().cpu() for k, v in dec.state_dict().items()}, 'seed': seed, 'defense_weight': weight, 'selected_epoch': epoch, 'validation_mse': metric, 'latent_n': 256})
        save_torch(last, {'enc': enc.state_dict(), 'dec': dec.state_dict(), 'clf': clf.state_dict(), 'optimizer': optimizer.state_dict(), 'opt_clf': opt_clf.state_dict(), 'history': history, 'best': best, 'epoch': epoch})
        write_json(folder / 'training_history.json', history)
        remaining = epochs - epoch
        eta = np.mean([r['seconds'] for r in history[-3:]]) * remaining / 60
        print(f"codec seed{seed} lambda{weight:g} epoch{epoch}/{epochs}: valMSE={metric:.7f}, {row['seconds']:.1f}s, this codec ~{eta:.1f}min left", flush=True)
    selected = torch.load(folder / 'best_codec.pt', map_location='cpu', weights_only=True)
    write_json(folder / 'training_complete.json', {'seed': seed, 'defense_weight': weight, 'epochs': len(history), 'selected_epoch': selected['selected_epoch'], 'selected_validation_mse': selected['validation_mse'], 'fitting_n': len(fit), 'validation_n': len(val), 'online_classifier_steps': cfg['online_classifier_steps'], 'test_used_for_selection': False})
    del enc, dec, clf, optimizer, opt_clf
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

def evaluate_cohort(images, labels, test_images, test_y, fit, val, seed, weight, key_hex, repeat_keys, tag, cfg, folder, device, smoke):
    if (folder / 'evaluation_complete.json').exists():
        return
    saved = torch.load(folder / 'best_codec.pt', map_location='cpu', weights_only=True)
    enc = Encoder(256).to(device).eval()
    dec = Decoder(256).to(device).eval()
    enc.load_state_dict(saved['enc'])
    dec.load_state_dict(saved['dec'])
    latent_file = folder / 'latents.npz'
    if latent_file.exists():
        with np.load(latent_file) as a:
            train_u = a['training']
            test_u = a['holdout']
    else:
        train_u = encode(enc, images, device)
        test_u = encode(enc, test_images, device)
        np.savez_compressed(latent_file, training=train_u, holdout=test_u)
    rng = np.random.default_rng(cfg['split_seed'] + 200 + seed)
    ns = [prefix('AC', tag) + int(i) for i in rng.permutation(len(train_u) + len(test_u))]
    train_ns, test_ns = (ns[:len(train_u)], ns[len(train_u):])
    crypto = CryptoScrambler(bytes.fromhex(key_hex), 256)
    train_parameters = crypto.parameters_batch(train_ns)
    train_noise = rng.normal(0, 10 ** (-cfg['snr_eve_db'] / 20), train_u.shape).astype(np.float32)
    train_obs = train_parameters.forward(train_u) + train_noise
    train_nf = nonce_features(train_ns)
    afolder = folder / 'attacks'
    afolder.mkdir(exist_ok=True)
    lock = fit_all_and_lock(train_obs, train_nf, train_u, images, labels, fit, val, cfg, afolder, device, smoke)
    controls = []
    control_specs = [('identity', dict(name='identity_classifier', kind='class', view='raw', nonce=True, aug=False, seed=91)), ('identity', dict(name='identity_cnn', kind='cnn', view='raw_fusion', nonce=False, aug=False, seed=81)), ('permutation_only', dict(name='permutation_classifier', kind='class', view='fusion', nonce=False, aug=False, seed=92))]
    for protocol, spec in control_specs:
        observation = train_u + train_noise if protocol == 'identity' else np.take_along_axis(train_u, train_parameters.perm, axis=1) + train_noise
        result = fit_candidate(spec, observation, train_nf, train_u, images, labels, fit, val, cfg, afolder / spec['name'], device, smoke)
        controls.append({'protocol': protocol, 'fit': result, 'spec': spec})
    write_json(afolder / 'positive_controls_locked.json', controls)
    parameters = crypto.parameters_batch(test_ns)
    test_obs = parameters.forward(test_u) + rng.normal(0, 10 ** (-cfg['snr_eve_db'] / 20), test_u.shape).astype(np.float32)
    test_nf = nonce_features(test_ns)
    allmetrics = evaluate_all(test_obs, test_nf, test_images, test_y, cfg, afolder, device, smoke)
    control_metrics = {}
    for control in controls:
        protocol = control['protocol']
        spec = control['spec']
        observation = test_u if protocol == 'identity' else np.take_along_axis(test_u, parameters.perm, axis=1)
        observation = observation + (test_obs - parameters.forward(test_u))
        predict, _ = load_candidate(afolder / spec['name'] / 'best.pt', device)
        if spec['kind'] == 'class':
            probabilities = predict(observation, test_nf)
            metric = class_metrics(test_y, probabilities)
            np.savez_compressed(afolder / spec['name'] / 'holdout.npz', y_true=test_y, probabilities=probabilities)
        else:
            f = np.concatenate((observation, test_nf), 1)
            rr = image_metrics(lambda a, n: torch.as_tensor(predict(a[:, :256], a[:, 256:]), device=device), test_images, f, device)
            metric = summarize_images(rr)
            np.savez_compressed(afolder / spec['name'] / 'holdout.npz', **rr)
        control_metrics[spec['name']] = metric
        write_json(afolder / spec['name'] / 'holdout_metrics.json', metric)
    prior = torch.as_tensor(images[fit].mean(0, dtype=np.float64) / 255, device=device, dtype=torch.float32)
    prior_rows = image_metrics(lambda _, n: prior.expand(n, -1, -1, -1), test_images, None, device)
    np.savez_compressed(folder / 'prior_holdout.npz', **prior_rows)
    selected_cls, _ = load_candidate(afolder / lock['classifier'] / 'best.pt', device)
    selected_rec, _ = load_candidate(afolder / lock['reconstructor'] / 'best.pt', device)
    keys = [key_hex] + repeat_keys
    keys = keys[:1 if smoke else cfg['repeat_keys']]
    draws = 1 if smoke else cfg['repeat_noise_draws']
    repeats = []
    for ki, key in enumerate(keys):
        p = CryptoScrambler(bytes.fromhex(key), 256).parameters_batch(test_ns)
        cipher = p.forward(test_u)
        for draw in range(draws):
            r = np.random.default_rng(cfg['split_seed'] + 900 + draw)
            standard_bob = r.normal(size=test_u.shape).astype(np.float32)
            standard_eve = r.normal(size=test_u.shape).astype(np.float32)
            bob_z = p.inverse(cipher + standard_bob * 10 ** (-cfg['snr_bob_db'] / 20))
            eve = cipher + standard_eve * 10 ** (-cfg['snr_eve_db'] / 20)
            rows = image_metrics(lambda f, n: dec(torch.as_tensor(f, device=device)), test_images, bob_z, device)
            probs = selected_cls(eve, test_nf)
            joined = np.concatenate((eve, test_nf), 1)
            recrows = image_metrics(lambda f, n: torch.as_tensor(selected_rec(f[:, :256], f[:, 256:]), device=device), test_images, joined, device)
            name = f'key{ki}_noise{draw}'
            np.savez_compressed(folder / (name + '_per_image.npz'), bob_mse=rows['mse'], bob_ssim_legacy=rows['ssim_legacy'], bob_ssim_valid=rows['ssim_valid'], eve_mse=recrows['mse'], eve_ssim_legacy=recrows['ssim_legacy'], eve_ssim_valid=recrows['ssim_valid'], y_true=test_y, probabilities=probs)
            repeats.append({'key_index': ki, 'noise_draw': draw, 'bob': summarize_images(rows), 'selected_reconstruction': summarize_images(recrows), 'selected_classification': class_metrics(test_y, probs)})
    report = {'seed': seed, 'defense_weight': weight, 'codec_selected_epoch': saved['selected_epoch'], 'attacker_selection': lock, 'fitting_mean_prior': summarize_images(prior_rows), 'key_noise_repetitions': repeats, 'all_attacker_metrics': allmetrics, 'positive_controls': control_metrics, 'holdout_indices': list(range(cfg['holdout_start'], cfg['holdout_start'] + len(test_y))) if not smoke else list(range(len(test_y))), 'metadata_repeated_across_keys': True, 'test_used_for_fit_or_selection': False}
    write_json(folder / 'packet_manifest.json', {'key_hex': key_hex, 'additional_evaluation_keys': repeat_keys, 'nonce_hex': [f'{n:024x}' for n in ns], 'assignment_seed': cfg['split_seed'] + 200 + seed})
    write_json(folder / 'evaluation_complete.json', report)
    del enc, dec
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--stage', choices=['all', 'train', 'attacks', 'package'], default='all')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text()) if args.config is not None else copy.deepcopy(CONFIG)
    if args.stage == 'package':
        results = HERE / ('results_smoke' if args.smoke else 'results')
        if not (results / 'COMPLETE.json').exists():
            raise RuntimeError('Experiment incomplete')
        if sha(results / 'EXPERIMENT_KEYS_PUBLIC_FIXTURES.json') != json.loads((results / 'KEY_LOCK.json').read_text())['sha256']:
            raise RuntimeError('Key fixture checksum mismatch')
        return_archives(results, HERE / ('SMOKE_RETURN' if args.smoke else 'RETURN_ASSETS'), cfg['max_return_part_bytes'])
        return
    if args.device == 'cpu' and (not args.smoke):
        raise RuntimeError('Full training requires CUDA')
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA unavailable')
    device = torch.device(args.device)
    torch.set_num_threads(2 if args.smoke else cfg['threads'])
    if os.name == 'nt':
        if ctypes.windll.kernel32.SetThreadExecutionState(2147483649):
            atexit.register(lambda: ctypes.windll.kernel32.SetThreadExecutionState(2147483648))
    if device.type == 'cuda':
        if cfg['amp_bfloat16'] and (not torch.cuda.is_bf16_supported()):
            raise RuntimeError('BF16 support required')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    results = HERE / ('results_smoke' if args.smoke else 'results')
    results.mkdir(exist_ok=True)
    fingerprints = {p.name: sha(p) for p in sorted(HERE.glob('*.py'))}
    identity = {'version': VERSION, 'config': cfg, 'source_sha256': fingerprints, 'smoke': args.smoke}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    lockpath = results / 'RUN_LOCK.json'
    if lockpath.exists() and json.loads(lockpath.read_text())['identity_sha256'] != identity_hash:
        raise RuntimeError('Existing run configuration or code mismatch')
    write_json(lockpath, {'identity_sha256': identity_hash, **identity})
    environment = {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None, 'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory if device.type == 'cuda' else None, 'utc_start': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'smoke_is_synthetic_not_scientific': args.smoke, 'distributions': {name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'cryptography', 'matplotlib']}, 'torch_threads': torch.get_num_threads(), 'cudnn_version': torch.backends.cudnn.version(), 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark, 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'amp_bfloat16': cfg['amp_bfloat16'] and device.type == 'cuda'}
    if not (results / 'environment.json').exists():
        write_json(results / 'environment.json', environment)
    invocations = results / 'invocations'
    invocations.mkdir(exist_ok=True)
    write_json(invocations / (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json'), environment)
    source_snapshot = results / 'source'
    source_snapshot.mkdir(exist_ok=True)
    for p in HERE.iterdir():
        if p.is_file() and p.suffix == '.py':
            shutil.copyfile(p, source_snapshot / p.name)
    write_json(source_snapshot / 'effective_config.json', cfg)
    write_json(results / 'protocol_selftests.json', run_self_tests())
    write_json(results / 'protocol_vectors.json', build_test_vectors())
    images, y, test_images, test_y = load_data(HERE / 'data', cfg, args.smoke)
    fit, val = split_indices(y, cfg, args.smoke)
    np.savez_compressed(results / 'split_manifest.npz', fitting=fit, validation=val, holdout_indices=np.arange(cfg['holdout_start'], cfg['holdout_start'] + len(test_y)) if not args.smoke else np.arange(len(test_y)))
    seeds = cfg['seeds'][:1] if args.smoke else cfg['seeds']
    weights = cfg['defense_weights']
    fixtures_path = results / 'EXPERIMENT_KEYS_PUBLIC_FIXTURES.json'
    if fixtures_path.exists():
        fixtures = json.loads(fixtures_path.read_text())
    else:
        if any(results.glob('seed*/best_codec.pt')):
            raise RuntimeError('Existing run key fixture missing')
        fixtures = {}
        for s in seeds:
            paired = {'training_key': secrets.token_hex(32), 'extra_keys': [secrets.token_hex(32) for _ in range(cfg['repeat_keys'] - 1)]}
            for w in weights:
                fixtures[f'seed{s}_lambda{w:g}'] = dict(paired)
        write_json(fixtures_path, fixtures)
    keylock = results / 'KEY_LOCK.json'
    keyhash = sha(fixtures_path)
    if keylock.exists() and json.loads(keylock.read_text())['sha256'] != keyhash:
        raise RuntimeError('Existing run key fixture checksum mismatch')
    if not keylock.exists() and any(results.glob('seed*/best_codec.pt')):
        raise RuntimeError('Existing run key lock missing')
    write_json(keylock, {'sha256': keyhash, 'fixture': fixtures_path.name})
    cases = [(s, w, f'seed{s}_lambda{w:g}') for s in seeds for w in weights]
    print('Synthetic smoke test' if args.smoke else 'Training and evaluation', flush=True)
    if args.stage in ['all', 'train']:
        for seed, weight, name in cases:
            tag = seeds.index(seed)
            train_codec(images, y, fit, val, seed, weight, fixtures[name]['training_key'], tag, cfg, results / name, device, args.smoke)
    if args.stage in ['all', 'attacks']:
        for seed, weight, name in cases:
            tag = seeds.index(seed)
            if not (results / name / 'training_complete.json').exists():
                raise RuntimeError(f'Codec training missing: {name}')
            evaluate_cohort(images, y, test_images, test_y, fit, val, seed, weight, fixtures[name]['training_key'], fixtures[name]['extra_keys'], tag, cfg, results / name, device, args.smoke)
        reports = [json.loads((results / name / 'evaluation_complete.json').read_text()) for _, _, name in cases]
        write_json(results / 'ALL_RESULTS.json', {'config': cfg, 'version': VERSION, 'cases': reports, 'smoke': args.smoke})
        with (results / 'SUMMARY.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['seed', 'lambda', 'codec_epoch', 'bob_ssim_legacy', 'bob_ssim_valid', 'eve_selected_ssim_legacy', 'classifier_accuracy'])
            for r in reports:
                rr = r['key_noise_repetitions'][0]
                writer.writerow([r['seed'], r['defense_weight'], r['codec_selected_epoch'], rr['bob']['mean_ssim_legacy'], rr['bob']['mean_ssim_valid'], rr['selected_reconstruction']['mean_ssim_legacy'], rr['selected_classification']['accuracy']])
        write_json(results / 'COMPLETE.json', {'complete': True, 'cases': len(cases), 'smoke': args.smoke})
        return_archives(results, HERE / ('SMOKE_RETURN' if args.smoke else 'RETURN_ASSETS'), cfg['max_return_part_bytes'])
        print('Completed', flush=True)
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped', flush=True)
        sys.exit(130)
    except Exception:
        import traceback
        traceback.print_exc()
        print('\nRun failed', flush=True)
        sys.exit(1)
