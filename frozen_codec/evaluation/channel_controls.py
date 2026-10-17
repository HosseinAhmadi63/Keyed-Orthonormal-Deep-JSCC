from pathlib import Path
import json, time, sys
import numpy as np
import torch
from crypto_scrambler import CryptoScrambler
from prepare_evaluations import ROOT, BASE, DATA, RESULTS, predict, summary, dump
from models.autoencoder import Decoder
torch.set_num_threads(2)

def main():
    with np.load(DATA / 'basic_holdout.npz', allow_pickle=False) as a:
        raw = a['test_latents']
        images = a['test_images']
        triples = a['crypto_nonces_u32']
        indices = a['test_indices']
    u = raw / np.sqrt(np.maximum(np.mean(raw * raw, axis=1, keepdims=True), 1e-06))
    nonces = [int(t[0]) << 64 | int(t[1]) << 32 | int(t[2]) for t in triples]
    keys = {'key_hex': 'aa47c63ebedee13713bbd1e4ed31f1849fed3be178b9fa04ef79da31fdc4a212', 'wrong_key_hex': 'eb47ecb539e8f5f3e365eafa8213f19aa7979fb1fea47f2f63abccff2f17193e'}
    crypto = CryptoScrambler(bytes.fromhex(keys['key_hex']), 256)
    p = crypto.parameters_batch(nonces)
    s = p.forward(u)
    ck = torch.load(BASE / 'pt/best_main.pt', map_location='cpu', weights_only=True)
    dec = Decoder(256).eval()
    dec.load_state_dict(ck['dec'])
    rng = np.random.default_rng(20261021)
    standard_w = rng.normal(size=u.shape).astype(np.float32)
    models = {}
    out = {}
    beg = time.monotonic()
    scenarios = []
    for snr in [0, 6, 12, 18, 24]:
        scenarios.append((f'AWGN_{snr}dB', p.inverse(s + standard_w * 10 ** (-snr / 20)), {'model': 'AWGN', 'snr_db': snr}))
    scenarios.append(('public_inverse_20dB', p.inverse(s + standard_w * 0.1), {'model': 'public-inverse control', 'snr_db': 20, 'key_given_to_attacker': True}))
    for degrees in [5, 15, 45]:
        pairs = s.reshape(-1, 128, 2).copy()
        ang = np.deg2rad(degrees)
        rot = pairs.copy()
        rot[:, :, 0] = np.cos(ang) * pairs[:, :, 0] - np.sin(ang) * pairs[:, :, 1]
        rot[:, :, 1] = np.sin(ang) * pairs[:, :, 0] + np.cos(ang) * pairs[:, :, 1]
        scenarios.append((f'phase_residual_{degrees}deg', p.inverse(rot.reshape(-1, 256) + standard_w * 10 ** (-12 / 20)), {'model': 'constant phase residual in adjacent real I/Q pairs', 'phase_degrees': degrees, 'snr_db': 12, 'phase_equalized': False}))
    fading = rng.rayleigh(1 / np.sqrt(2), (len(u), 1)).astype(np.float32)
    noise = standard_w * 10 ** (-12 / 20)
    received = fading * s + noise
    scenarios.append(('Rayleigh_perfect_CSI', p.inverse(received / np.maximum(fading, 0.1)), {'model': 'real positive scalar block Rayleigh gain', 'gain_second_moment': 1, 'snr_db': 12, 'csi': 'perfect gain with0.1 equalizer floor', 'equalizer_floor': 0.1}))
    estimated = np.maximum(fading + rng.normal(0, 0.1, fading.shape).astype(np.float32), 0.1)
    scenarios.append(('Rayleigh_noisy_CSI', p.inverse(received / estimated), {'model': 'real positive scalar block Rayleigh gain', 'snr_db': 12, 'csi': 'gain plus independent Gaussian estimate error', 'estimate_error_sd': 0.1, 'equalizer_floor': 0.1}))
    scenarios.append(('residual_ISI', p.inverse(s + 0.2 * np.roll(s, 1, axis=1) + noise), {'model': 'residual real intersymbol mixing I+0.2circularshift', 'snr_db': 12, 'equalized': False}))
    for name, z, spec in scenarios:
        m, ss, ex = predict(dec, z, images)
        models[name] = {**summary(m, ss), 'specification': spec}
        out[name + '_mse'] = m
        out[name + '_ssim'] = ss
        out[name + '_first16'] = ex
        dump(RESULTS / 'channel_control_metrics.json', {'holdout_n': len(u), 'frozen_codec': True, 'seed': 20261021, 'models': models})
        np.savez(RESULTS / 'channel_control_per_image.npz', test_indices=indices, **out)
        print(f"Channel control {name}: SSIM {models[name]['ssim']:.6f}; {time.monotonic() - beg:.1f}s total", flush=True)
    with np.load(DATA / 'cache_fixed_crypto.npz', allow_pickle=False) as a:
        train_u = a['train_latents']
        train_v = a['train_obs']
        test_v = a['test_obs']
    split = np.random.default_rng(20261010).permutation(len(train_u))
    val, fit = (split[:5000], split[5000:])
    kpa = {}
    per = {}
    for count in [256, 4096, 45000]:
        ii = fit[:count]
        U = train_u[ii].astype(np.float64)
        V = train_v[ii].astype(np.float64)
        cross = V.T @ U
        left, _, right = np.linalg.svd(cross, full_matrices=False)
        estimated_S = left @ right
        val_error = float(np.mean((train_v[val] @ estimated_S - train_u[val]) ** 2))
        aligned = (test_v @ estimated_S).astype(np.float32)
        m, ss, ex = predict(dec, aligned, images)
        kpa[str(count)] = {**summary(m, ss), 'fitting_pairs': count, 'validation_latent_mse': val_error, 'rank': int(np.linalg.matrix_rank(U)), 'algorithm': 'orthogonal Procrustes, SVD of V^T U; public trained encoder supplies normalized plaintext latents', 'test_selection': False}
        per[f'known_{count}_mse'] = m
        per[f'known_{count}_ssim'] = ss
        print(f"Fixed-transform known-plaintext {count} pairs: SSIM {kpa[str(count)]['ssim']:.6f}", flush=True)
    dump(RESULTS / 'known_plaintext_metrics.json', {'protocol': 'fixed_crypto', 'noise_snr_db': 20, 'holdout_n': 3600, 'models': kpa})
    np.savez(RESULTS / 'known_plaintext_per_image.npz', **per)
    print('Channel and known-plaintext controls completed.', flush=True)
if __name__ == '__main__':
    main()
