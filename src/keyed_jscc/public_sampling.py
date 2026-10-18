from __future__ import annotations
import numpy as np

def fwht(x):
    x = np.array(x, dtype=np.float32, copy=True, order='C')
    n = x.shape[-1]
    if n < 1 or n & n - 1:
        raise ValueError('Hadamard dimension must be a power of two')
    h = 1
    while h < n:
        blocks = x.reshape(-1, n // (2 * h), 2, h)
        left = blocks[:, :, 0, :].copy()
        right = blocks[:, :, 1, :].copy()
        blocks[:, :, 0, :] = left + right
        blocks[:, :, 1, :] = left - right
        h *= 2
    return x / np.sqrt(np.float32(n))

def sample_packets(u, rng, snr_db=20.0, family='full'):
    u = np.asarray(u, dtype=np.float32)
    b, n = u.shape
    if family == 'full':
        sign1 = (1 - 2 * rng.integers(0, 2, size=u.shape, dtype=np.int8)).astype(np.float32)
        sign2 = (1 - 2 * rng.integers(0, 2, size=u.shape, dtype=np.int8)).astype(np.float32)
        z = fwht(u * sign1) * sign2
    elif family == 'permutation_only':
        z = u
    elif family == 'identity':
        z = u
    else:
        raise ValueError(family)
    if family != 'identity':
        p = rng.permuted(np.broadcast_to(np.arange(n, dtype=np.int64), (b, n)), axis=1)
        z = np.take_along_axis(z, p, axis=1)
    if np.isfinite(snr_db):
        z = z + rng.normal(0, 10.0 ** (-snr_db / 20), u.shape).astype(np.float32)
    return np.asarray(z, dtype=np.float32)

def invariant_features(x, view='fusion'):
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    moments = np.column_stack([x.mean(1), x.std(1), x.min(1), x.max(1), np.mean(x * x, 1), np.mean(x ** 3, 1), np.mean(x ** 4, 1), np.mean(x ** 5, 1), np.mean(x ** 6, 1), np.mean(x < 0, 1), np.quantile(x, 0.1, axis=1), np.quantile(x, 0.25, axis=1), np.quantile(x, 0.5, axis=1), np.quantile(x, 0.75, axis=1), np.quantile(x, 0.9, axis=1)])
    if view == 'fusion':
        columns = [np.sort(x, axis=1), np.sort(a, axis=1), moments]
    elif view == 'magnitude':
        columns = [np.sort(a, axis=1), moments[:, [1, 4, 6, 8]]]
    elif view == 'signed':
        columns = [np.sort(x, axis=1), moments]
    elif view == 'raw_fusion':
        columns = [x, np.sort(x, axis=1), np.sort(a, axis=1), moments]
    elif view == 'raw':
        columns = [x]
    else:
        raise ValueError(view)
    return np.asarray(np.column_stack(columns), dtype=np.float32)

def theoretical_moments(u, sigma2=0.0):
    u = np.asarray(u, dtype=np.float64)
    n = u.shape[1]
    q2 = np.sum(u * u, axis=1)
    q4 = np.sum(u ** 4, axis=1)
    q6 = np.sum(u ** 6, axis=1)
    variance = q2 / n + sigma2
    return {'second': variance, 'fourth': 3 * variance ** 2 - 2 * q4 / n ** 2, 'sixth': 15 * variance ** 3 - 30 * variance * q4 / n ** 2 + 16 * q6 / n ** 3}
