from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np

def paired_bootstrap_means(values, *, seed=20261028, draws=10000, chunk=64):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Finite image-by-statistic array required')
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, draws, chunk):
        indices = rng.integers(0, len(values), size=(min(chunk, draws - start), len(values)))
        means.append(values[indices].mean(axis=1))
    return np.concatenate(means, axis=0)

def interval(samples):
    return np.quantile(samples, [0.025, 0.975]).tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    protocols = ('crypto', 'fixed_crypto', 'identity')
    families = ('mlp_seed_1', 'mlp_seed_2', 'mlp_seed_3', 'ridge_raw_nonce', 'ridge_sorted_abs_nonce', 'fitting_mean_image')
    records = {}
    arrays = {}
    indices = None
    splits = None
    for protocol in protocols:
        directory = args.results / f'reconstruction_{protocol}_final'
        report = json.loads((directory / 'reconstruction_metrics.json').read_text())
        if not report.get('total_elapsed_s') or set(report['families']) != set(families):
            raise ValueError(f'Incomplete primary fit/evaluation: {protocol}')
        if report['training']['holdout_used_for_selection']:
            raise ValueError('Holdout-dependent selection invalidates this audit')
        data = np.load(directory / 'reconstruction_per_image.npz', allow_pickle=False)
        split = np.load(directory / 'attacker_split_indices.npz', allow_pickle=False)
        current_indices = data['test_indices']
        if indices is None:
            indices = current_indices
            splits = {k: split[k] for k in split.files}
        else:
            assert np.array_equal(indices, current_indices)
            assert all((np.array_equal(splits[k], split[k]) for k in splits))
        assert np.array_equal(current_indices, np.arange(6400, 10000))
        assert not np.intersect1d(split['fitting'], split['validation']).size
        records[protocol] = report
        arrays[protocol] = {k: data[k].astype(np.float64) for f in families for k in (f + '_mse', f + '_ssim')}
        for family in families:
            mse = arrays[protocol][family + '_mse']
            ssim = arrays[protocol][family + '_ssim']
            assert mse.shape == ssim.shape == (3600,)
            actual = report['families'][family]
            assert np.isclose(mse.mean(), actual['holdout_mse'], rtol=1e-12, atol=1e-12)
            assert np.isclose(ssim.mean(), actual['holdout_ssim'], rtol=1e-12, atol=1e-12)
            if family.startswith('mlp_'):
                epochs = actual['epochs']
                assert len(epochs) == 8 and sum((e['updates'] for e in epochs)) == 1400
                assert all((e['fitting_images_seen'] == 44800 for e in epochs))
                selected = min(epochs, key=lambda e: e['validation_mse'])
                assert selected['epoch'] == actual['selected_validation_epoch']
                assert selected['validation_mse'] == actual['selected_validation_mse']
    names = [(p, f) for p in protocols for f in families]
    mean_ssim = np.column_stack([arrays[p][f + '_ssim'] for p, f in names])
    mean_mse = np.column_stack([arrays[p][f + '_mse'] for p, f in names])
    samples = paired_bootstrap_means(np.column_stack((mean_ssim, mean_mse)))
    boot_ssim, boot_mse = (samples[:, :len(names)], samples[:, len(names):])
    output = {'method': {'n': 3600, 'bootstrap_draws': 10000, 'bootstrap_seed': 20261028, 'interval': 'pointwise 95% paired-image percentile bootstrap', 'psnr': '-10 log10(mean image MSE), transformed separately for each resample', 'no_holdout_selection': True, 'cross_protocol_fitting_validation_indices_identical': True, 'actual_numpy_version': np.__version__}, 'protocols': {}, 'paired_protocol_contrasts': {}}
    for protocol in protocols:
        r = records[protocol]
        selected_mlp = min((f for f in families if f.startswith('mlp_')), key=lambda f: r['families'][f]['selected_validation_mse'])
        prior = families.index('fitting_mean_image') + protocols.index(protocol) * len(families)
        output['protocols'][protocol] = {'training': r['training'], 'runtime_versions': r['runtime_versions'], 'selected_mlp_family_by_validation_mse': selected_mlp, 'families': {}}
        for family in families:
            i = names.index((protocol, family))
            old = r['families'][family]
            metrics = {k: v for k, v in old.items() if k != 'epochs'}
            metrics.update({'ssim_ci95': interval(boot_ssim[:, i]), 'dataset_psnr_ci95': interval(-10 * np.log10(boot_mse[:, i])), 'paired_ssim_minus_fitting_mean': float((mean_ssim[:, i] - mean_ssim[:, prior]).mean()), 'paired_ssim_minus_fitting_mean_ci95': interval(boot_ssim[:, i] - boot_ssim[:, prior]), 'paired_psnr_minus_fitting_mean_db': float(-10 * np.log10(mean_mse[:, i].mean() / mean_mse[:, prior].mean())), 'paired_psnr_minus_fitting_mean_ci95': interval(-10 * np.log10(boot_mse[:, i] / boot_mse[:, prior])), 'ssim_iqr': np.quantile(mean_ssim[:, i], [0.25, 0.75]).tolist()})
            output['protocols'][protocol]['families'][family] = metrics
    for comparator in ('fixed_crypto', 'identity'):
        contrast = {}
        for family in families:
            a = names.index((comparator, family))
            b = names.index(('crypto', family))
            contrast[family] = {'ssim_difference': float((mean_ssim[:, a] - mean_ssim[:, b]).mean()), 'ssim_difference_ci95': interval(boot_ssim[:, a] - boot_ssim[:, b]), 'psnr_difference_db': float(-10 * np.log10(mean_mse[:, a].mean() / mean_mse[:, b].mean())), 'psnr_difference_ci95': interval(-10 * np.log10(boot_mse[:, a] / boot_mse[:, b]))}
        output['paired_protocol_contrasts'][comparator + '_minus_crypto'] = contrast
    destination = args.output or args.results / 'reconstruction_verified_summary.json'
    destination.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'output': str(destination), 'selected_mlps': {p: output['protocols'][p]['selected_mlp_family_by_validation_mse'] for p in protocols}}, indent=2))
if __name__ == '__main__':
    main()
