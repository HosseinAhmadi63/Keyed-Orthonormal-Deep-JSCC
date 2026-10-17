from pathlib import Path
import json
import numpy as np
RESULTS = Path(__file__).resolve().parent / 'results'

def intervals(m, s, indices):
    m = np.asarray(m, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    return {'n': len(s), 'ssim': float(s.mean()), 'ssim_ci95': np.percentile(s[indices].mean(axis=1), [2.5, 97.5]).tolist(), 'psnr': float(-10 * np.log10(m.mean())), 'psnr_ci95': np.percentile(-10 * np.log10(m[indices].mean(axis=1)), [2.5, 97.5]).tolist()}

def main():
    ii = np.random.default_rng(20261024).integers(0, 3600, size=(2000, 3600))
    out = {'bootstrap_repetitions': 2000, 'bootstrap_seed': 20261024, 'unit': 'paired holdout image; conditional on one frozen codec and channel draw', 'groups': {}, 'paired_ssim_differences': {}}
    groups = [('bob', RESULTS / 'bob_holdout_per_image.npz'), ('channels', RESULTS / 'channel_control_per_image.npz'), ('known_plaintext', RESULTS / 'known_plaintext_per_image.npz'), ('archived_transfer', RESULTS / 'archived_transfer_crypto_per_image.npz')]
    for label, path in groups:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as a:
            out['groups'][label] = {k[:-5]: intervals(a[k[:-5] + '_mse'], a[k], ii) for k in a.files if k.endswith('_ssim')}
    with np.load(RESULTS / 'bob_holdout_per_image.npz') as b:
        bob = b['crypto_ssim'].astype(np.float64)
        for name in ['identity', 'crypto_wrong_key', 'crypto_nonce_bitflip']:
            delta = bob - b[name + '_ssim']
            out['paired_ssim_differences']['crypto_minus_' + name] = {'mean': float(delta.mean()), 'ci95': np.percentile(delta[ii].mean(axis=1), [2.5, 97.5]).tolist()}
    for protocol in ['crypto', 'identity', 'fixed_crypto', 'permutation_only']:
        p = RESULTS / f'classifiers_{protocol}/summary.json'
        if p.exists():
            d = json.loads(p.read_text())
            out.setdefault('classification', {})[protocol] = {'selected': d['selected_attacker'], 'selected_metrics': d['selected_test_metrics'], 'families': {k: {'accuracy': v['test']['accuracy'], 'balanced_accuracy': v['test']['balanced_accuracy'], 'accuracy_wilson_95': v['test']['accuracy_wilson_95']} for k, v in d['models'].items()}, 'runtime_seconds': d['runtime_seconds']}
    (RESULTS / 'final_verified_statistics.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'groups': list(out['groups']), 'classification_protocols': list(out.get('classification', {})), 'bob': out['groups']['bob']['crypto'], 'equivalence_delta': out['paired_ssim_differences']['crypto_minus_identity']}, indent=2))
if __name__ == '__main__':
    main()
