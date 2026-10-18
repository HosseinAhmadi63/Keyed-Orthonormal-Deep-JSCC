from pathlib import Path
import os
import hashlib, json, math, csv
import numpy as np
from scipy.stats import binomtest, chi2_contingency
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT = Path(os.environ.get('JSCC_ASSETS', str(Path(__file__).resolve().parents[1])))
BASE = ROOT / 'models/json'
OUT = ROOT / 'evaluation/figures/archive_audit'
OUT.mkdir(parents=True, exist_ok=True)
data = {p.stem: json.loads(p.read_text()) for p in BASE.glob('*.json')}
final = data['final_eval']
dist = final['distributions']
n = len(dist['bob_ssim'])
rng = np.random.default_rng(20261010)
B = 5000

def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]
keys = ['bob_ssim', 'wk_ssim', 'unscr_bob_ssim', 'unscr_eve_ssim', 'eve_base_ssim', 'eve_strong_ssim', 'eve_no_nonce_ssim', 'eve_bob_style_ssim', 'eve_base_ssim_gap', 'eve_strong_ssim_gap', 'eve_no_nonce_ssim_gap', 'eve_bob_style_ssim_gap', 'bob_mse', 'wk_mse', 'unscr_bob_mse', 'unscr_eve_mse', 'eve_base_mse', 'eve_strong_mse', 'eve_no_nonce_mse', 'eve_bob_style_mse']
arr = {k: np.asarray(dist[k], float) for k in keys}
best = final['eve_best_name']
arr['bob_minus_eve_best_ssim'] = arr['bob_ssim'] - arr[f'eve_{best}_ssim']
arr['bob_minus_unscr_bob_ssim'] = arr['bob_ssim'] - arr['unscr_bob_ssim']
boot = {k: np.empty(B) for k in arr}
for start in range(0, B, 64):
    stop = min(start + 64, B)
    idx = rng.integers(0, n, size=(stop - start, n))
    for k, x in arr.items():
        boot[k][start:stop] = x[idx].mean(axis=1)
stats = {}
for k, x in arr.items():
    ci = np.percentile(boot[k], [2.5, 97.5])
    stats[k] = {'n': n, 'mean': float(x.mean()), 'sd_images': float(x.std(ddof=1)), 'ci95_percentile_bootstrap': ci.tolist(), 'median': float(np.median(x)), 'p05': float(np.percentile(x, 5)), 'p95': float(np.percentile(x, 95)), 'max': float(x.max()), 'min': float(x.min())}
for k in keys:
    if k.endswith('_mse'):
        ci = np.percentile(-10 * np.log10(boot[k]), [2.5, 97.5])
        stats[k.replace('_mse', '_psnr')] = {'n': n, 'dataset_psnr': float(-10 * np.log10(arr[k].mean())), 'ci95_percentile_bootstrap': ci.tolist()}
cm = np.asarray(final['confusion_matrix'], int)
yt = np.asarray(dist['int_true'], int)
yp = np.asarray(dist['int_pred'], int)
assert int(cm.sum()) == n and np.array_equal(cm, np.bincount(10 * yt + yp, minlength=100).reshape(10, 10))
correct = int(np.trace(cm))
support = cm.sum(axis=1)
rec = np.diag(cm) / support
classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
perclass = [{'class_index': i, 'class_name': classes[i], 'correct': int(cm[i, i]), 'support': int(support[i]), 'recall': float(rec[i]), 'ci95_wilson': wilson(int(cm[i, i]), int(support[i]))} for i in range(10)]
class_boot = rng.binomial(support, rec, size=(B, 10)) / support
balanced_ci = np.percentile(class_boot.mean(axis=1), [2.5, 97.5]).tolist()
classification = {'n': n, 'correct': correct, 'accuracy': correct / n, 'ci95_wilson': wilson(correct, n), 'balanced_accuracy': float(rec.mean()), 'balanced_ci95_stratified_parametric_bootstrap': balanced_ci, 'binomial_two_sided_p_against_0_1': float(binomtest(correct, n, 0.1).pvalue), 'confusion_matrix_true_rows_predicted_cols': cm.tolist(), 'per_class': perclass, 'entropy_mean': float(np.mean(dist['int_entropy'])), 'confidence_mean': float(np.mean(dist['int_confidence']))}
chi_stat, chi_p, chi_df, _ = chi2_contingency(cm, correction=False)
classification['chi_square_class_prediction_independence'] = {'statistic': float(chi_stat), 'df': int(chi_df), 'p_value': float(chi_p)}
classification['prediction_marginals'] = (cm.sum(0) / cm.sum()).tolist()

def rowstats(row):
    c = np.asarray(row['confusion_matrix'], int)
    nn = int(c.sum())
    kk = int(np.trace(c))
    return {'snr_bob_db': row['snr_bob_db'], 'snr_eve_db': row['snr_eve_db'], 'n': nn, 'correct': kk, 'accuracy': kk / nn, 'ci95_wilson': wilson(kk, nn), 'balanced_accuracy': float(np.mean(np.diag(c) / c.sum(axis=1))), 'ssim_bob': row['ssim_bob'], 'ssim_eve_best': row['ssim_eve_best'], 'gap': row['ssim_bob'] - row['ssim_eve_best'], 'psnr_bob': row['psnr_bob'], 'psnr_eve_best': row['psnr_eve_best']}
sweep_stats = {'bob_snr': [rowstats(r) for r in data['snr_sweep']], 'eve_snr': [rowstats(r) for r in data['eve_snr_sweep']], 'joint_fixed_eve': [{'snr_eve_db': r.get('snr_eve_db', r.get('eve_snr_db', r.get('eve_snr_fixed_db'))), 'rows': [rowstats(x) for x in r['rows']]} for r in data['joint_snr_sweep_curves']], 'gap_delta': [{'delta_db': r['delta_db'], 'rows': [rowstats(x) for x in r['rows']]} for r in data['gap_delta_sweep']]}
for j in sweep_stats['joint_fixed_eve']:
    if j['snr_eve_db'] is None:
        j['snr_eve_db'] = j['rows'][0]['snr_eve_db']
aggdiff = {'bob_ssim_record_minus_recomputed': final['ssim_bob'] - stats['bob_ssim']['mean'], 'eve_ssim_record_minus_recomputed': final['ssim_eve_best'] - stats[f'eve_{best}_ssim']['mean'], 'wrong_ssim_record_minus_recomputed': final['ssim_wrongkey'] - stats['wk_ssim']['mean']}
report = {'canonical_source_directory': str(BASE), 'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.json')}, 'uncertainty_note': '5000 paired image bootstrap resamples conditional on chosen checkpoint, key, nonces, attacker family and archived channel draw. Not independent-run confidence intervals; test-based checkpoint/family selection limits interpretation. Sweep accuracy intervals are pointwise Wilson intervals from confusion counts.', 'bootstrap_seed': 20261010, 'bootstrap_replicates': B, 'default_operating_point': {'bob_snr_db': 12, 'eve_snr_db': 20, 'n': n, 'best_attacker': best}, 'config': data['cfg'], 'selected_main_epoch': max(data['history'], key=lambda x: x['ssim_bob'])['epoch'], 'default_statistics': stats, 'classification': classification, 'all_attacker_archived_metrics': final['eve_all'], 'aggregation_discrepancies': aggdiff, 'sweeps': sweep_stats, 'warnings': ['No verified independent seed ablations.', 'No held-out validation set; checkpoint chosen using these test images.', 'MS-SSIM has no gradient in supplied code.', 'Defense is KL(U||p), not negative entropy.', '64-bit key parameter maps through noncryptographic 32-bit effective seed.', 'Dataset nonce is fixed per image, not fresh per transmission.']}
(ROOT / 'evaluation/results/archived_verified_statistics.json').write_text(json.dumps(report, indent=2))
with (ROOT / 'evaluation/results/archived_per_class_metrics.csv').open('w') as f:
    w = csv.writer(f)
    w.writerow(['Class', 'Correct', 'Support', 'Recall', 'Wilson95_low', 'Wilson95_high'])
    for r in perclass:
        w.writerow([r['class_name'], r['correct'], r['support'], r['recall'], *r['ci95_wilson']])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 10.5, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.22, 'grid.linewidth': 0.5, 'savefig.bbox': 'tight'})
colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']

def save(fig, name):
    fig.savefig(OUT / (name + '.pdf'))
    fig.savefig(OUT / (name + '.png'), dpi=180)
    plt.close(fig)
rows = data['snr_sweep']
xx = [r['snr_bob_db'] for r in rows]
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.8), layout='constrained')
for ax, metric, label in zip(axs, ['psnr', 'ssim'], ['Dataset PSNR (dB)', 'Mean SSIM']):
    for name, k, col, mark in [('Bob', f'{metric}_bob', colors[0], 'o'), ('Eve (family maximum)', f'{metric}_eve_best', colors[1], 's'), ('Wrong key', f'{metric}_wrongkey', colors[2], '^')]:
        ax.plot(xx, [r[k] for r in rows], color=col, marker=mark, markersize=4, label=name, lw=1.5)
    ax.set_xlabel('Bob SNR (dB)')
    ax.set_ylabel(label)
    ax.set_xticks([0, 6, 12, 18, 21])
fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False)
axs[0].set_title('(a) PSNR')
axs[1].set_title('(b) SSIM')
save(fig, 'reconstruction_bob_snr')
rows = data['eve_snr_sweep']
xx = [r['snr_eve_db'] for r in rows]
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.8), layout='constrained')
for ax, metric, label in zip(axs, ['psnr', 'ssim'], ['Dataset PSNR (dB)', 'Mean SSIM']):
    ax.plot(xx, [r[f'{metric}_eve_best'] for r in rows], color=colors[1], marker='s', markersize=4, lw=1.5)
    ax.set_xlabel('Eve SNR (dB)')
    ax.set_ylabel(label)
    ax.set_xticks([0, 6, 12, 18, 24])
axs[0].set_title('(a) Eve reconstruction PSNR')
axs[1].set_title('(b) Eve reconstruction SSIM')
save(fig, 'eve_snr_sensitivity')
fig, ax = plt.subplots(figsize=(5.6, 3.0), layout='constrained')
for j, col, mark in zip(sweep_stats['joint_fixed_eve'], colors, ['o', 's', '^', 'D', 'v']):
    ax.plot([r['snr_bob_db'] for r in j['rows']], [r['gap'] for r in j['rows']], color=col, marker=mark, markersize=3.7, lw=1.2, label=f"Eve SNR = {j['snr_eve_db']:g} dB")
ax.set_xlabel('Bob SNR (dB)')
ax.set_ylabel('Mean SSIM difference (Bob - Eve)')
ax.set_xticks([0, 6, 12, 18, 21])
ax.legend(frameon=False, ncol=2, loc='lower right')
save(fig, 'ssim_gap')
fig, axs = plt.subplots(2, 3, figsize=(7.1, 4.5), layout='constrained', sharey=True)
for ax, j, col in zip(axs.flat, sweep_stats['joint_fixed_eve'], colors):
    rr = j['rows']
    xx = np.asarray([r['snr_bob_db'] for r in rr])
    yy = np.asarray([r['accuracy'] * 100 for r in rr])
    ci = np.asarray([r['ci95_wilson'] for r in rr]) * 100
    ax.fill_between(xx, ci[:, 0], ci[:, 1], color=col, alpha=0.15, lw=0)
    ax.plot(xx, yy, color=col, marker='o', markersize=3.5, lw=1.2)
    ax.axhline(10, color='#555555', ls='--', lw=0.9)
    ax.set_title(f"Eve SNR = {j['snr_eve_db']:g} dB")
    ax.set_xlabel('Bob SNR (dB)')
    ax.set_xticks([0, 12, 21])
    ax.set_ylim(8.0, 12.0)
    ax.set_yticks([8, 10, 12])
rr = sweep_stats['eve_snr']
ax = axs.flat[5]
xx = np.asarray([r['snr_eve_db'] for r in rr])
yy = np.asarray([r['accuracy'] * 100 for r in rr])
ci = np.asarray([r['ci95_wilson'] for r in rr]) * 100
ax.fill_between(xx, ci[:, 0], ci[:, 1], color=colors[0], alpha=0.15, lw=0)
ax.plot(xx, yy, color=colors[0], marker='o', markersize=3.5, lw=1.2)
ax.axhline(10, color='#555555', ls='--', lw=0.9)
ax.set_xlabel('Eve SNR (dB)')
ax.set_title('Bob SNR = 12 dB')
ax.set_xticks([0, 12, 24])
for ax in axs[:, 0]:
    ax.set_ylabel('Interception accuracy (%)')
save(fig, 'interception_sweeps')
fig, ax = plt.subplots(figsize=(5.6, 3.0), layout='constrained')
for name, k, col in [('Bob', 'bob_ssim', colors[0]), ('Eve (no nonce)', f'eve_{best}_ssim', colors[1]), ('Wrong key', 'wk_ssim', colors[2])]:
    xx = np.sort(arr[k])
    ax.plot(xx, np.arange(1, n + 1) / n, color=col, lw=1.5, label=name)
ax.set_xlabel('Per-image SSIM')
ax.set_ylabel('Empirical cumulative probability')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
save(fig, 'ssim_cdf')
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.45), layout='constrained', gridspec_kw={'width_ratios': [1.1, 1]})
ax = axs[0]
im = ax.imshow(cm / support[:, None] * 100, cmap='Blues', vmin=0, vmax=20, interpolation='nearest')
ax.grid(False)
ax.set_xticks(range(10), range(10))
ax.set_yticks(range(10), range(10))
ax.set_xlabel('Predicted class')
ax.set_ylabel('True class')
ax.set_title('(a) Row-normalized confusion (%)')
for i in range(10):
    for j in range(10):
        v = cm[i, j] / support[i] * 100
        ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=6.8, color='white' if v > 13 else '#333333')
cbar = fig.colorbar(im, ax=ax, fraction=0.043, pad=0.03)
cbar.ax.tick_params(labelsize=8)
ax = axs[1]
ci = np.asarray([r['ci95_wilson'] for r in perclass])
ax.errorbar(rec * 100, range(10), xerr=np.vstack([rec - ci[:, 0], ci[:, 1] - rec]) * 100, fmt='o', color=colors[0], capsize=2, markersize=4)
ax.axvline(10, color='#555555', ls='--', lw=0.9)
ax.set_yticks(range(10), [f'{i}: {x}' for i, x in enumerate(classes)])
ax.invert_yaxis()
ax.set_xlabel('Class recall (%)')
ax.set_title('(b) Recall and 95% Wilson interval')
ax.set_xlim(2, 19)
save(fig, 'confusion_and_per_class')
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.0), layout='constrained')
names = ['Base', 'Strong', 'No nonce', 'Bob style']
ks = ['base', 'strong', 'no_nonce', 'bob_style']
x = np.arange(4)
true = np.asarray([stats[f'eve_{k}_ssim']['mean'] for k in ks])
shuf = np.asarray([np.mean(dist[f'eve_{k}_ssim_shuf']) for k in ks])
gap = np.asarray([stats[f'eve_{k}_ssim_gap']['mean'] for k in ks])
gci = np.asarray([stats[f'eve_{k}_ssim_gap']['ci95_percentile_bootstrap'] for k in ks])
axs[0].bar(x - 0.16, true, 0.32, color=colors[1], label='True reference')
axs[0].bar(x + 0.16, shuf, 0.32, color='#BBBBBB', label='Shuffled reference')
axs[0].set_xticks(x, names, rotation=15)
axs[0].set_ylim(0, 0.4)
axs[0].set_ylabel('Mean Eve SSIM')
axs[0].legend(frameon=False, loc='upper right', fontsize=8)
axs[0].set_title('(a) Reference pairing control')
axs[1].errorbar(x, gap, yerr=np.vstack([gap - gci[:, 0], gci[:, 1] - gap]), fmt='o', color=colors[1], capsize=3)
axs[1].axhline(0, color='#555555', ls='--', lw=0.9)
axs[1].set_xticks(x, names, rotation=15)
axs[1].set_ylabel('SSIM difference (true - shuffled)')
axs[1].set_title('(b) Paired 95% image bootstrap interval')
axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
save(fig, 'shuffle_control')
print(json.dumps({'n': n, 'classification': {k: v for k, v in classification.items() if k not in ['confusion_matrix_true_rows_predicted_cols', 'per_class']}, 'default': {k: v for k, v in stats.items() if k in ['bob_ssim', f'eve_{best}_ssim', 'wk_ssim', 'bob_minus_eve_best_ssim', f'eve_{best}_ssim_gap']}, 'figures': [str(p) for p in OUT.glob('*.pdf')], 'aggregation': aggdiff}, indent=2))
