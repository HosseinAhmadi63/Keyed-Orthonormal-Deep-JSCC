from __future__ import annotations
import argparse, csv, hashlib, json, pickle, sys, tarfile
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def write(p, d):
    Path(p).write_text(json.dumps(d, indent=2, allow_nan=False) + '\n')

def csvwrite(p, rows):
    if not rows:
        return
    with Path(p).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def extract(results):
    dest = BASE / 'data'
    dest.mkdir(parents=True, exist_ok=True)
    history = []
    attacks = []
    candidates = []
    cells = []
    controls = []
    cases = []

    def source_path(name):
        p = results / name
        if p.exists():
            return p
        if name == 'RETURN_INVENTORY.json':
            return results.parent / name
        if name.startswith('source/'):
            return results.parent / name[7:]
        return p
    sources = {p: sha(source_path(p)) for p in ['ALL_RESULTS.json', 'COMPLETE.json', 'hardware_report.json'] if source_path(p).is_file()}
    for seed in [1001, 1002, 1003]:
        for weight in [0.0, 0.2]:
            name = f'seed{seed}_lambda{weight:g}'
            folder = results / name
            d = read(folder / 'evaluation_complete.json')
            lock = d['attacker_selection']
            assert d['seed'] == seed and d['defense_weight'] == weight and (len(d['key_noise_repetitions']) == 9)
            assert d['test_used_for_fit_or_selection'] is False
            prior = d['fitting_mean_prior']
            for row in read(folder / 'training_history.json'):
                history.append(dict(case=name, seed=seed, weight=weight, epoch=row['epoch'], fitting_mse=row['fitting']['mse'], validation_mse=row['validation_mse'], kl_uniform=row['fitting']['kl_uniform'], online_cross_entropy=row['fitting']['online_cross_entropy'], selected_epoch=d['codec_selected_epoch'], updates=row['codec_updates']))
            for report in lock['reports']:
                spec = report['spec']
                candidate = spec['name']
                metric = d['all_attacker_metrics'][candidate]
                c = dict(case=name, seed=seed, weight=weight, candidate=candidate, kind=spec['kind'], augmented=spec['aug'], nonce=spec['nonce'], view=spec['view'], selected=int(candidate in [lock['classifier'], lock['reconstructor']]), selected_epoch=report['selected_epoch'], completed_epochs=report['completed_epochs'], updates=report['updates'], fresh_presentations=report['fresh_simulated_presentations'], accuracy=metric.get('accuracy'), wilson_low=metric.get('wilson95', [None, None])[0], wilson_high=metric.get('wilson95', [None, None])[1], mse=metric.get('mse'), valid_ssim=metric.get('mean_ssim_valid'), legacy_ssim=metric.get('mean_ssim_legacy'), valid_gain=None if 'mse' not in metric else metric['mean_ssim_valid'] - prior['mean_ssim_valid'], legacy_gain=None if 'mse' not in metric else metric['mean_ssim_legacy'] - prior['mean_ssim_legacy'], mse_gain=None if 'mse' not in metric else prior['mse'] - metric['mse'])
                candidates.append(c)
                sources[str(Path(name) / 'attacks' / candidate / 'history.json')] = sha(folder / 'attacks' / candidate / 'history.json')
                for h in read(folder / 'attacks' / candidate / 'history.json'):
                    attacks.append(dict(case=name, candidate=candidate, kind=spec['kind'], augmented=spec['aug'], epoch=h['epoch'], fitting_loss=h['fitting_loss'], validation_loss=h['validation_loss'], selected_epoch=report['selected_epoch'], updates=h['updates'], fresh_presentations=h['fresh_simulated_presentations']))
            for candidate, metric in d['positive_controls'].items():
                controls.append(dict(case=name, candidate=candidate, accuracy=metric.get('accuracy'), valid_ssim=metric.get('mean_ssim_valid'), legacy_ssim=metric.get('mean_ssim_legacy')))
            for row in d['key_noise_repetitions']:
                b, e, c = (row['bob'], row['selected_reconstruction'], row['selected_classification'])
                cells.append(dict(case=name, seed=seed, weight=weight, key=row['key_index'], noise=row['noise_draw'], bob_mse=b['mse'], bob_psnr=b['dataset_psnr'], bob_valid=b['mean_ssim_valid'], bob_legacy=b['mean_ssim_legacy'], eve_mse=e['mse'], eve_psnr=e['dataset_psnr'], eve_valid=e['mean_ssim_valid'], eve_legacy=e['mean_ssim_legacy'], accuracy=c['accuracy'], correct=c['correct'], wilson_low=c['wilson95'][0], wilson_high=c['wilson95'][1], bob_eve_valid=b['mean_ssim_valid'] - e['mean_ssim_valid'], bob_eve_legacy=b['mean_ssim_legacy'] - e['mean_ssim_legacy'], eve_prior_valid=e['mean_ssim_valid'] - prior['mean_ssim_valid'], eve_prior_legacy=e['mean_ssim_legacy'] - prior['mean_ssim_legacy'], eve_prior_mse=prior['mse'] - e['mse']))
            these = [r for r in cells if r['case'] == name]
            aggregate = {k: float(np.mean([r[k] for r in these])) for k in these[0] if k not in ['case', 'seed', 'weight', 'key', 'noise', 'correct', 'wilson_low', 'wilson_high']}
            aggregate['bob_psnr'] = -10 * np.log10(aggregate['bob_mse'])
            aggregate['eve_psnr'] = -10 * np.log10(aggregate['eve_mse'])
            cases.append(dict(case=name, seed=seed, weight=weight, selected_classifier=lock['classifier'], selected_reconstructor=lock['reconstructor'], codec_selected_epoch=d['codec_selected_epoch'], prior=prior, aggregate=aggregate))
            for rel in ['evaluation_complete.json', 'training_history.json', 'attacks/locked_selection.json']:
                sources[str(Path(name) / rel)] = sha(folder / rel)
    csvwrite(dest / 'codec_convergence.csv', history)
    csvwrite(dest / 'attacker_convergence.csv', attacks)
    csvwrite(dest / 'attack_candidates.csv', candidates)
    csvwrite(dest / 'key_noise_cells.csv', cells)
    csvwrite(dest / 'positive_controls.csv', controls)
    write(dest / 'plotdata.json', dict(cases=cases, codec_history=history, attacker_history=attacks, candidates=candidates, key_noise_cells=cells, positive_controls=controls, source_sha256=sources, scope='three paired independent codec seeds; nine conditional key/noise cells per arm; no pooling as54 independent codecs', primary_ssim='GPU-return valid-window unclipped RGB Gaussian11 sigma1.5 population SSIM in float32', selection='codec: minimum validationMSE; classifier: validationCE; reconstructor: validationMSE; suite extreme is descriptive only'))
    print('Extracted six codecs,', len(candidates), 'candidates and', len(cells), 'conditional cells', flush=True)

def qualitative(results, archive):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from keyed_jscc.codec_models import Decoder
    from keyed_jscc.crypto_scrambler import CryptoScrambler, nonce_features
    from keyed_jscc.fit_attackers import load_candidate
    from keyed_jscc.experiment_utils import valid_ssim
    raw = archive.read_bytes()
    assert hashlib.md5(raw).hexdigest() == 'c58f30108f718f92721af3b95e74349a'
    with tarfile.open(archive, 'r:gz') as tar:
        d = pickle.load(tar.extractfile('cifar-10-batches-py/test_batch'), encoding='bytes')
    truth = d[b'data'].reshape(-1, 3, 32, 32)[6400:6404].astype(np.float32) / 255
    output = {'truth': truth, 'holdout_indices': np.arange(6400, 6404)}
    metadata = {}
    for seed in [1001, 1002, 1003]:
        for weight in [0.0, 0.2]:
            name = f'seed{seed}_lambda{weight:g}'
            folder = results / name
            evaluation = read(folder / 'evaluation_complete.json')
            packet = read(folder / 'packet_manifest.json')
            with np.load(folder / 'latents.npz') as a:
                u = a['holdout']
            assert u.shape == (3600, 256)
            ns = [int(x, 16) for x in packet['nonce_hex'][-3600:]]
            params = CryptoScrambler(bytes.fromhex(packet['key_hex']), 256).parameters_batch(ns[:4])
            cipher = params.forward(u[:4])
            rng = np.random.default_rng(20261015 + 900)
            bob_noise = rng.normal(size=u.shape).astype(np.float32)[:4]
            eve_noise = rng.normal(size=u.shape).astype(np.float32)[:4]
            z = params.inverse(cipher + bob_noise * 10 ** (-12 / 20))
            model = Decoder(256).eval()
            saved = torch.load(folder / 'best_codec.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(saved['dec'])
            with torch.inference_mode():
                bob = model(torch.from_numpy(z).float()).numpy()
            selected = evaluation['attacker_selection']['reconstructor']
            predict, _ = load_candidate(folder / 'attacks' / selected / 'best.pt', torch.device('cpu'))
            eve = predict(cipher + eve_noise * 10 ** (-20 / 20), nonce_features(ns[:4]))
            with np.load(folder / 'prior_holdout.npz') as a:
                prior = a['first_examples'][:4]
            output[name + '__bob'] = bob
            output[name + '__eve'] = eve
            output[name + '__prior'] = prior
            mse = lambda x: np.square(x - truth).mean((1, 2, 3))
            with torch.inference_mode():
                bs = valid_ssim(torch.from_numpy(bob), torch.from_numpy(truth)).numpy()
                es = valid_ssim(torch.from_numpy(eve), torch.from_numpy(truth)).numpy()
            with np.load(folder / 'key0_noise0_per_image.npz') as a:
                error = {'bob_mse_max_abs': float(np.max(np.abs(mse(bob) - a['bob_mse'][:4]))), 'eve_mse_max_abs': float(np.max(np.abs(mse(eve) - a['eve_mse'][:4]))), 'bob_valid_ssim_max_abs': float(np.max(np.abs(bs - a['bob_ssim_valid'][:4]))), 'eve_valid_ssim_max_abs': float(np.max(np.abs(es - a['eve_ssim_valid'][:4])))}
                assert error['bob_mse_max_abs'] < 1e-06 and error['eve_mse_max_abs'] < 1e-06, error
                assert error['bob_valid_ssim_max_abs'] < 0.0001 and error['eve_valid_ssim_max_abs'] < 0.0001, error
            metadata[name] = dict(selected_reconstructor=selected, checkpoint_sha256=sha(folder / 'best_codec.pt'), attacker_checkpoint_sha256=sha(folder / 'attacks' / selected / 'best.pt'), reproduction_errors=error)
            print('Qualitative regenerated', name, error, flush=True)
    np.savez_compressed(BASE / 'data/qualitative_examples.npz', **output)
    write(BASE / 'data/qualitative_provenance.json', dict(source_archive_md5=hashlib.md5(raw).hexdigest(), source_archive_sha256=hashlib.sha256(raw).hexdigest(), test_indices=[6400, 6401, 6402, 6403], cases=metadata))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--qualitative', action='store_true')
    p.add_argument('--cifar-archive', type=Path)
    a = p.parse_args()
    extract(a.results)
    if a.qualitative:
        qualitative(a.results, a.cifar_archive)
