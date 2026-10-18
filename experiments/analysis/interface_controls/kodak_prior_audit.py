from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def audit(path: Path, output: Path, draws: int=10000, seed: int=20261015):
    raw = path.read_bytes()
    saved = json.loads(raw)
    rows = saved['images']
    if len(rows) != 24:
        raise ValueError('Expected all24 source frames')
    names = list(rows[0]['models'])
    if any((set(row['models']) != set(names) for row in rows)):
        raise ValueError('Inconsistent models')
    ss = np.column_stack([[r['models'][m]['ssim'] for r in rows] for m in names]).astype(np.float64)
    mm = np.column_stack([[r['models'][m]['mse'] for r in rows] for m in names]).astype(np.float64)
    prior_name = 'mean_training_image'
    prior = names.index(prior_name)
    bob = names.index('bob')
    if not np.isfinite(ss).all() or not np.isfinite(mm).all():
        raise ValueError('Nonfinite metrics')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(draws, len(rows)))
    bs, bm = (ss[indices].mean(axis=1), mm[indices].mean(axis=1))
    ci = lambda x: np.quantile(x, [0.025, 0.975]).tolist()
    summary = {}
    for j, name in enumerate(names):
        ds = ss[:, j] - ss[:, prior]
        summary[name] = {'mean_ssim': float(ss[:, j].mean()), 'ssim_ci95': ci(bs[:, j]), 'dataset_psnr_db': float(-10 * np.log10(mm[:, j].mean())), 'dataset_psnr_ci95': ci(-10 * np.log10(bm[:, j])), 'ssim_minus_prior': float(ds.mean()), 'ssim_minus_prior_ci95': ci(bs[:, j] - bs[:, prior]), 'positive_ssim_improvement_images': int(np.sum(ds > 0)), 'mse_reduction_from_prior': float((mm[:, prior] - mm[:, j]).mean()), 'mse_reduction_from_prior_ci95': ci(bm[:, prior] - bm[:, j]), 'dataset_psnr_gain_over_prior_db': float(-10 * np.log10(mm[:, j].mean() / mm[:, prior].mean())), 'dataset_psnr_gain_over_prior_ci95': ci(-10 * np.log10(bm[:, j] / bm[:, prior]))}
    nonbob = [m for m in names if m != 'bob']
    strongest = max(nonbob, key=lambda m: summary[m]['mean_ssim'])
    result = {'source': str(path), 'source_sha256': hashlib.sha256(raw).hexdigest(), 'source_images': 24, 'tiles': 9216, 'metrics_before_png_rounding': True, 'bootstrap': {'unit': 'source image', 'draws': draws, 'seed': seed, 'interval': 'pointwise paired percentile95%'}, 'reference': prior_name, 'prior_population': 'all50000 CIFAR-10 official training images', 'strongest_observed_nonbob_mean_ssim_model': strongest, 'models': summary, 'bob_minus_prior': summary['bob'], 'per_image': [{'name': row['name'], 'models': {name: {'ssim': float(ss[i, j]), 'mse': float(mm[i, j]), 'ssim_minus_prior': float(ss[i, j] - ss[i, prior]), 'mse_reduction_from_prior': float(mm[i, prior] - mm[i, j])} for j, name in enumerate(names)}} for i, row in enumerate(rows)]}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'kodak_prior_contrasts.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez(output / 'kodak_prior_per_image.npz', names=np.asarray(names), source_names=np.asarray([row['name'] for row in rows]), ssim=ss, mse=mm, ssim_minus_prior=ss - ss[:, prior, None], mse_reduction_from_prior=mm[:, prior, None] - mm)
    print(json.dumps({'strongest_baseline': strongest, 'bob_minus_prior': summary['bob']['ssim_minus_prior'], 'bob_minus_prior_ci95': summary['bob']['ssim_minus_prior_ci95'], 'attacker_minus_prior': {m: {'mean': summary[m]['ssim_minus_prior'], 'ci95': summary[m]['ssim_minus_prior_ci95']} for m in nonbob if m != prior_name}}, indent=2))
    return result
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2] / 'frozen_codec/evaluation/results/kodak_fresh/kodak_per_image.json')
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'results/kodak_prior')
    a = p.parse_args()
    audit(a.source, a.output)
