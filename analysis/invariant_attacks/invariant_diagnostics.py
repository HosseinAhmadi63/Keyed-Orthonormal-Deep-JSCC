from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
from public_sampling import sample_packets, theoretical_moments
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def ahash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()

def fhash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def eta2(a, y):
    a = np.asarray(a, dtype=np.float64)
    one = a.ndim == 1
    if one:
        a = a[:, None]
    centre = a.mean(0)
    total = ((a - centre) ** 2).sum(0)
    between = sum((np.sum(y == k) * (a[y == k].mean(0) - centre) ** 2 for k in range(10)))
    return (between / np.maximum(total, 1e-100))[0] if one else between / np.maximum(total, 1e-100)

def stats(x):
    x = np.asarray(x, dtype=np.float64)
    sq = x * x
    power = sq.mean(1)
    fourth = (sq * sq).mean(1)
    sixth = (sq * sq * sq).mean(1)
    return {'power': power, 'fourth': fourth, 'sixth': sixth, 'signed_mean': x.mean(1), 'signed_third': (sq * x).mean(1), 'negative_fraction': (x < 0).mean(1), 'max_absolute': np.abs(x).max(1), 'kurtosis': fourth / np.maximum(power ** 2, 1e-100)}

def class_means(a, y, boot, rng):
    out = []
    for k in range(10):
        v = np.asarray(a[y == k], dtype=np.float64)
        means = v[rng.integers(len(v), size=(boot, len(v)))].mean(1)
        out.append({'class': k, 'name': CLASS_NAMES[k], 'n': len(v), 'mean': float(v.mean()), 'standard_deviation': float(v.std(ddof=1)), 'mean_percentile_95': np.quantile(means, [0.025, 0.975]).tolist()})
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--draws', type=int, default=16)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--permutations', type=int, default=2000)
    p.add_argument('--seed', type=int, default=2026101503)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with np.load(a.split, allow_pickle=False) as d:
        idx = d['validation']
    assert len(idx) == 5000
    with np.load(a.cache, allow_pickle=False) as d:
        u = d['train_latents'][idx]
        r = d['train_obs'][idx]
        y = d['y_train'][idx]
    assert np.array_equal(np.bincount(y, minlength=10), np.repeat(500, 10))
    rng = np.random.default_rng(a.seed)
    ls = stats(u)
    rs = stats(r)
    n = u.shape[1]
    ipr = n * ls['power'] ** 2 / ls['fourth']
    pred = theoretical_moments(u, 0.01)
    measured = {'latent_' + k: v for k, v in ls.items()}
    measured.update({'observed_' + k: v for k, v in rs.items()})
    measured['latent_participation_ratio'] = ipr
    names = list(measured)
    matrix = np.column_stack([measured[k] for k in names])
    observed_eta = eta2(matrix, y)
    null = np.stack([eta2(matrix, rng.permutation(y)) for _ in range(a.permutations)])
    summaries = {}
    for j, k in enumerate(names):
        values = measured[k]
        summaries[k] = {'mean': float(values.mean()), 'sd': float(values.std(ddof=1)), 'min': float(values.min()), 'quantiles': np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).tolist(), 'max': float(values.max()), 'class_variance_fraction_eta2': float(observed_eta[j]), 'permuted_label_eta2_95': np.quantile(null[:, j], [0.025, 0.975]).tolist(), 'permutation_p_one_sided': float((1 + np.sum(null[:, j] >= observed_eta[j])) / (a.permutations + 1)), 'class_means': class_means(values, y, a.bootstrap, rng)}
    snrs = [None, 30.0, 20.0, 12.0, 6.0, 0.0]
    sim = {str(s): {'eta2_fourth': [], 'mean_fourth': [], 'class_means_fourth': [], 'moment_prediction_rmse': []} for s in snrs}
    monte_mean = np.zeros(len(u), dtype=np.float64)
    for repeat in range(a.draws):
        clean = sample_packets(u, rng, np.inf, 'full')
        z = rng.standard_normal(u.shape, dtype=np.float32)
        for snr in snrs:
            sigma = 0.0 if snr is None else 10 ** (-snr / 20)
            rr = clean + sigma * z
            sq = rr.astype(np.float64) ** 2
            m4 = (sq * sq).mean(1)
            key = str(snr)
            ex = theoretical_moments(u, sigma * sigma)['fourth']
            sim[key]['eta2_fourth'].append(float(eta2(m4, y)))
            sim[key]['mean_fourth'].append(float(m4.mean()))
            sim[key]['class_means_fourth'].append([float(m4[y == k].mean()) for k in range(10)])
            sim[key]['moment_prediction_rmse'].append(float(np.sqrt(np.mean((m4 - ex) ** 2))))
            if snr == 20.0:
                monte_mean += m4 / a.draws
        print('Invariant simulation draw ' + str(repeat + 1) + '/' + str(a.draws), flush=True)
    toy = np.zeros((2, n), np.float32)
    toy[0, 0] = np.sqrt(n)
    toy[1, :2] = np.sqrt(n / 2)
    toy_actual = []
    for _ in range(16):
        q = sample_packets(toy, rng, np.inf, 'full')
        s = q.astype(np.float64) ** 2
        toy_actual.append((s * s).mean(1).tolist())
    result = {'seed': a.seed, 'n': len(y), 'dimension': n, 'class_names': CLASS_NAMES, 'bootstrap_resamples': a.bootstrap, 'label_permutations': a.permutations, 'ideal_family_draws_per_snr': a.draws, 'source_sha256': {'script': fhash(__file__), 'sampling': fhash(Path(__file__).parent / 'public_sampling.py')}, 'validation_indices_sha256': ahash(idx), 'latent_sha256': ahash(u), 'observation_sha256': ahash(r), 'label_sha256': ahash(y), 'exchangeable_label_expected_eta2': 9 / (len(y) - 1), 'statistics': summaries, 'ideal_sign_moment_identity': {'second': 'v=q2/N+sigma2', 'fourth': '3*v^2-2*q4/N^2', 'sixth': '15*v^3-30*v*q4/N^2+16*q6/N^3', 'q_definition': 'qk=sum_i u_i^k', 'normalization': 'mean_i u_i^2=1, up to floating-point error', 'fourth_latent_class_contrast_scale': -2 / n, 'interpretation': 'At equal power, a contrast in class mean latent fourth moment becomes -2/N times that contrast in the conditional expected observed fourth moment. AWGN adds a class-common offset. Realized packet fluctuations and other statistics are not bounded by this identity.'}, 'conditional_expected_fourth_20dB': {'mean': float(pred['fourth'].mean()), 'minimum': float(pred['fourth'].min()), 'maximum': float(pred['fourth'].max()), 'class_means': class_means(pred['fourth'], y, a.bootstrap, rng)}, 'finite_draw_conditional_mean_fourth_20dB': {'prediction_rmse': float(np.sqrt(np.mean((monte_mean - pred['fourth']) ** 2))), 'per_image_sd': float(monte_mean.std(ddof=1))}, 'simulations': sim, 'toy_counterexample': {'power': 1, 'fourth_noiseless_exact': [1.0, 2.0], 'participation_ratio': [1.0, 2.0], 'sampled_fourth_16draws': toy_actual}, 'elapsed_seconds': time.monotonic() - start}
    write(a.output / 'invariant_diagnostics.json', result)
    np.savez_compressed(a.output / 'invariant_diagnostics_per_image.npz', validation_indices=idx, y=y, latent_fourth=ls['fourth'], latent_sixth=ls['sixth'], latent_participation_ratio=ipr, observed_fourth=rs['fourth'], observed_sixth=rs['sixth'], expected_fourth_20dB=pred['fourth'], mc_conditional_fourth_20dB=monte_mean, eta2_null=null, statistic_names=np.asarray(names))
    print(json.dumps({'latent_IPR_median': float(np.median(ipr)), 'latent_fourth_mean': float(ls['fourth'].mean()), 'observed_fourth_mean': float(rs['fourth'].mean()), 'expected_fourth_mean': float(pred['fourth'].mean()), 'latent_fourth_eta2': float(observed_eta[names.index('latent_fourth')]), 'observed_fourth_eta2': float(observed_eta[names.index('observed_fourth')]), 'elapsed_seconds': time.monotonic() - start}), flush=True)
if __name__ == '__main__':
    main()
