from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from evidence_common import BASE, ASSETS, INPUTS, load_decoder, load_mlp, predict, metric_arrays, bootstrap_summary, assemble, write_json, sha256, EveReconNoNonce, EveBobStyleDecoder
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'frozen_codec/evaluation'))
from crypto_scrambler import CryptoScrambler

def cifar(output, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    with np.load(INPUTS / 'cifar_holdout.npz', allow_pickle=False) as a:
        images, obs, nonce, u = [a[k] for k in ('test_images', 'test_obs', 'test_nonce_features', 'test_latents')]
        metadata = json.loads(str(a['metadata_json'].item()))
    n = limit or len(images)
    images, obs, nonce, u = (images[:n], obs[:n], nonce[:n], u[:n])
    key = bytes.fromhex({'key_hex': 'aa47c63ebedee13713bbd1e4ed31f1849fed3be178b9fa04ef79da31fdc4a212', 'wrong_key_hex': 'eb47ecb539e8f5f3e365eafa8213f19aa7979fb1fea47f2f63abccff2f17193e'}['key_hex'])
    order = np.random.default_rng(20261010).permutation(53600)
    nonces = [23002549206610311231137906688 + int(i) for i in order[50000:50000 + n]]
    params = CryptoScrambler(key, 256).parameters_batch(nonces)
    rng = np.random.default_rng(20261012)
    rng.normal(0, 0.1, (50000, 256))
    old_eve_noise = rng.normal(0, 0.1, (3600, 256)).astype(np.float32)
    old_bob_noise = rng.normal(0, 10 ** (-12 / 20), (3600, 256)).astype(np.float32)
    max_observation_error = float(np.max(np.abs(params.forward(u) + old_eve_noise[:n] - obs)))
    assert max_observation_error < 2e-06, max_observation_error
    decoder, decoder_path = load_decoder()
    arrays = {}
    checks = {}
    sources = {'decoder': sha256(decoder_path)}
    historical = np.load(ASSETS / 'evaluation/results/reconstruction_crypto_final/reconstruction_per_image.npz')
    prior = historical['fitting_mean_image_first16_reconstructions'][0]
    cases = [('bob', predict(decoder, params.inverse(params.forward(u) + old_bob_noise[:n]))), ('fitting_mean_image', np.broadcast_to(prior, (n, *prior.shape)))]
    for name, prediction in cases:
        arrays[name] = metric_arrays(prediction, images)
        print('CIFAR metric complete', name, flush=True)
    for protocol in ('crypto', 'identity', 'fixed_crypto'):
        model, path = load_mlp(protocol)
        sources[protocol + '_model'] = sha256(path)
        if protocol == 'crypto':
            po, pn = (obs, nonce)
        elif protocol == 'identity':
            po, pn = (u + old_eve_noise[:n], nonce)
        else:
            po = CryptoScrambler(key, 256).parameters(23002549206610311231137906688).forward(u) + old_eve_noise[:n]
            pn = np.zeros((n, 6), np.float32)
        pred = predict(model, po, pn)
        label = protocol + '_selected_mlp'
        arrays[label] = metric_arrays(pred, images)
        hp = ASSETS / 'evaluation/results' / f'reconstruction_{protocol}_final/reconstruction_per_image.npz'
        family = {'crypto': 'mlp_seed_3', 'identity': 'mlp_seed_3', 'fixed_crypto': 'mlp_seed_1'}[protocol]
        with np.load(hp) as a:
            expected = a[family + '_ssim'][:n]
        checks[label] = {'max_abs_legacy_reproduction_error': float(np.max(np.abs(arrays[label]['legacy_zero_clipped'] - expected)))}
        assert checks[label]['max_abs_legacy_reproduction_error'] < 5e-05, checks[label]
        print('CIFAR metric complete', label, flush=True)
    coefficient = INPUTS / 'ridge_raw_nonce_coefficients.npz'
    if coefficient.exists():
        with np.load(coefficient) as a:
            feat = np.column_stack((obs, nonce)).astype(np.float64)
            pred = np.clip((feat - a['feature_mean']) / a['feature_scale'] @ a['coefficients'] + a['mean_image'], 0, 1).astype(np.float32).reshape(-1, 3, 32, 32)
        arrays['crypto_raw_ridge'] = metric_arrays(pred, images)
        sources['ridge_coefficients'] = sha256(coefficient)
        checks['crypto_raw_ridge'] = {'max_abs_legacy_reproduction_error': float(np.max(np.abs(arrays['crypto_raw_ridge']['legacy_zero_clipped'] - historical['ridge_raw_nonce_ssim'][:n])))}
        assert checks['crypto_raw_ridge']['max_abs_legacy_reproduction_error'] < 5e-05, checks['crypto_raw_ridge']
    report = bootstrap_summary(arrays, prior='fitting_mean_image', draws=10000)
    report.update({'publication_run': n == 3600, 'padding_fraction_32': 1 - (22 / 32) ** 2, 'legacy_reproduction_checks': checks, 'max_reproduced_observation_error': max_observation_error, 'source_hashes': sources, 'definitions': definitions()})
    write_json(output / 'ssim_sensitivity.json', report)
    np.savez(output / 'ssim_per_image.npz', **{name + '__' + key: value for name, row in arrays.items() for key, value in row.items()})
    return report

def definitions():
    return {'legacy_zero_clipped': 'float32;11x11 Gaussian sigma1.5;zero padding5;population variances clamp_min(0);denominator+1e-8;average across RGB/spatial map THEN clamp scalar image score to[0,1]; no pixelwise score clipping', 'zero_unclipped': 'same archived arithmetic and guards; only final per-image scalar clamp removed', 'valid_same_guards_unclipped': 'same float32 score map; crop5 pixels each edge;no final clipping', 'reference_valid_unclipped': 'independent float64 separable Gaussian11/sigma1.5;population variances/covariance;valid centers;no variance clipping,denominator epsilon or score clipping;RGB channel average;no grayscale/downsampling', 'references': ['https://ece.uwaterloo.ca/~z70wang/research/ssim/', 'https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity']}

def kodak(output, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    saved = json.loads((ASSETS / 'evaluation/results/kodak_fresh/kodak_per_image.json').read_text())
    rows = saved['images'][:limit] if limit else saved['images']
    with np.load(INPUTS / 'kodak_packets.npz', allow_pickle=False) as a:
        images, u, obs, nonce, index, nu = [a[k] for k in ('images', 'normalized_latents', 'eve_observations', 'nonce_features', 'image_index', 'nonce_u32')]
    key = bytes.fromhex({'key_hex': 'aa47c63ebedee13713bbd1e4ed31f1849fed3be178b9fa04ef79da31fdc4a212', 'wrong_key_hex': 'eb47ecb539e8f5f3e365eafa8213f19aa7979fb1fea47f2f63abccff2f17193e'}['key_hex'])
    crypto = CryptoScrambler(key, 256)
    rng = np.random.default_rng(20261019)
    decoder, decoder_path = load_decoder()
    fresh, fresh_path = load_mlp('crypto')
    archived_path = ASSETS / 'models/pt/best_response_attackers.pt'
    ck = torch.load(archived_path, map_location='cpu', weights_only=True)
    eves = {'eve_no_nonce_transfer': EveReconNoNonce(256).eval(), 'eve_bob_style_transfer': EveBobStyleDecoder(256).eval()}
    for name, oldname in [('eve_no_nonce_transfer', 'no_nonce'), ('eve_bob_style_transfer', 'bob_style')]:
        eves[name].load_state_dict(ck['eve_models'][oldname])
    prior = np.load(INPUTS / 'all50k_mean_image.npy')
    values = {}
    checks = {}
    input_error = []
    started = time.monotonic()
    for i, row in enumerate(rows):
        take = np.flatnonzero(index == i)
        patches = images[take]
        lat = u[take]
        ob = obs[take]
        nf = nonce[take]
        nonces = [int(t[0]) << 64 | int(t[1]) << 32 | int(t[2]) for t in nu[take]]
        parameters = crypto.parameters_batch(nonces)
        cipher = parameters.forward(lat)
        bn = rng.normal(0, 10 ** (-12 / 20), cipher.shape).astype(np.float32)
        en = rng.normal(0, 0.1, cipher.shape).astype(np.float32)
        error = float(np.max(np.abs(cipher + en - ob)))
        input_error.append(error)
        assert error < 2e-06, error
        predictions = {'bob': predict(decoder, parameters.inverse(cipher + bn), batch=32), 'mean_training_image': np.broadcast_to(prior, patches.shape), 'fresh_mlp_seed_3': predict(fresh, ob, nf, batch=256)}
        predictions.update({name: predict(model, ob, batch=32) for name, model in eves.items()})
        target = assemble(patches, row['height'], row['width']).transpose(2, 0, 1)[None]
        checks[row['name']] = {}
        for name, tiles in predictions.items():
            full = assemble(tiles, row['height'], row['width']).transpose(2, 0, 1)[None]
            metrics = metric_arrays(full, target, batch=1)
            for metric, value in metrics.items():
                values.setdefault(name, {}).setdefault(metric, []).append(float(value[0]))
            expected = row['models'][name]
            delta = abs(float(metrics['legacy_zero_clipped'][0]) - expected['ssim'])
            checks[row['name']][name] = {'legacy_ssim_abs_error': delta, 'mse_abs_error': abs(float(metrics['mse'][0]) - expected['mse'])}
            assert delta < 5e-05, (name, row['name'], delta)
        print(f"Kodak SSIM {i + 1}/{len(rows)} {row['name']} {time.monotonic() - started:.1f}s", flush=True)
        write_json(output / 'progress.json', {'completed_images': i + 1, 'total': len(rows), 'legacy_reproduction': checks})
    arrays = {name: {k: np.asarray(v) for k, v in d.items()} for name, d in values.items()}
    report = bootstrap_summary(arrays, prior='mean_training_image', draws=10000)
    report.update({'publication_run': len(rows) == 24, 'definitions': definitions(), 'legacy_reproduction_checks': checks, 'max_reproduced_observation_error': max(input_error), 'source_hashes': {'decoder_checkpoint': sha256(decoder_path), 'fresh_MLP': sha256(fresh_path), 'archived_attackers': sha256(archived_path), 'all50k_mean_image': sha256(INPUTS / 'all50k_mean_image.npy')}})
    write_json(output / 'ssim_sensitivity.json', report)
    np.savez(output / 'ssim_per_image.npz', source_names=np.array([r['name'] for r in rows]), **{name + '__' + k: v for name, row in arrays.items() for k, v in row.items()})
    return report
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scope', choices=['cifar', 'kodak'], required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    out = a.output or BASE / 'results' / ('ssim_' + a.scope)
    result = (cifar if a.scope == 'cifar' else kodak)(out, a.limit)
    print(json.dumps({'scope': a.scope, 'n': result['n'], 'models': {k: {v: row[v]['mean'] for v in ['legacy_zero_clipped', 'zero_unclipped', 'reference_valid_unclipped']} for k, row in result['models'].items()}}, indent=2))
