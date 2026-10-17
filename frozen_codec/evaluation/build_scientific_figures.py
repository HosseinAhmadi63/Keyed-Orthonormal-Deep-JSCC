from pathlib import Path
import argparse, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
BASE = Path(__file__).resolve().parent
RESULTS = BASE / 'results'
HIST = Path(__file__).resolve().parents[1] / 'models/json'
PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#666666', '#56B4E9']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5, 'axes.linewidth': 0.7, 'lines.linewidth': 1.5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#DADDE2', 'grid.linewidth': 0.5, 'grid.alpha': 0.7, 'axes.axisbelow': True})

def read(path):
    return json.loads(path.read_text())

def hashed(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def save(fig, out, name, data, paths):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / (name + '.pdf'), bbox_inches='tight', pad_inches=0.06)
    fig.savefig(out / (name + '.png'), dpi=220, bbox_inches='tight', pad_inches=0.06)
    (out / (name + '_plotted_values.json')).write_text(json.dumps({'source_sha256': hashed(paths), 'plotted_values': data}, indent=2) + '\n')
    plt.close(fig)
    print('Created ' + name, flush=True)

def grid(rows=2, cols=2, title='', bottom=0.2):
    fig, axes = plt.subplots(rows, cols, figsize=(6.9, 2.6 * rows), squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.985, bottom=bottom, top=0.9, wspace=0.35, hspace=0.43)
    fig.suptitle(title, fontsize=11.5, y=0.985)
    return (fig, axes.ravel())

def panel(ax, letter, title, ylabel, xlabel=None):
    ax.set_title(f'({letter}) {title}', loc='left', pad=8)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)

def cdf(ax, values, label, color, style='-'):
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    ax.step(values, np.arange(1, n + 1) / n, where='post', label=label, color=color, linestyle=style)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

def boot_intervals(mse, ssim, indices):
    m = np.asarray(mse, dtype=np.float64)
    s = np.asarray(ssim, dtype=np.float64)
    return {'psnr': float(-10 * np.log10(m.mean())), 'psnr_ci': np.percentile(-10 * np.log10(m[indices].mean(axis=1)), [2.5, 97.5]).tolist(), 'ssim': float(s.mean()), 'ssim_ci': np.percentile(s[indices].mean(axis=1), [2.5, 97.5]).tolist()}

def bars(ax, names, rows, metric, colors=None):
    vals = [v[metric] for v in rows]
    x = np.arange(len(names))
    colors = colors or PALETTE[:len(names)]
    ax.bar(x, vals, color=colors, width=0.68, zorder=3)
    if all((metric + '_ci' in row for row in rows)):
        ci = np.array([row[metric + '_ci'] for row in rows])
        v = np.asarray(vals)
        ax.errorbar(x, v, yerr=np.maximum(0, np.stack((v - ci[:, 0], ci[:, 1] - v))), fmt='none', ecolor='#222222', capsize=3, lw=0.9, zorder=4)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1 if metric == 'ssim' else max(vals) * 1.16)
    ax.grid(axis='x', visible=False)
    for j, v in enumerate(vals):
        ax.text(j, v + (0.025 if metric == 'ssim' else 0.55), f'{v:.3f}' if metric == 'ssim' else f'{v:.2f}', ha='center', fontsize=8)

def utility(out):
    p = RESULTS / 'bob_holdout_metrics.json'
    raw = RESULTS / 'bob_holdout_per_image.npz'
    if not (p.exists() and raw.exists()):
        return
    d = read(p)
    a = np.load(raw)
    keys = ['identity', 'legacy', 'crypto', 'crypto_wrong_key', 'crypto_nonce_bitflip']
    names = ['Identity', '32-bit\nmixer', 'HKDF/\nAES', 'Wrong\nkey', 'Nonce\nbit error']
    rng = np.random.default_rng(20261024)
    ii = rng.integers(0, len(a['crypto_mse']), size=(2000, len(a['crypto_mse'])))
    rows = [boot_intervals(a[k + '_mse'], a[k + '_ssim'], ii) for k in keys]
    fig, axes = grid(title='CIFAR-10: legitimate utility and inverse-mismatch controls')
    bars(axes[0], names, rows, 'ssim', [PALETTE[6], PALETTE[5], PALETTE[0], PALETTE[4], PALETTE[1]])
    panel(axes[0], 'a', 'Mean reconstruction quality', 'SSIM')
    bars(axes[1], names, rows, 'psnr', [PALETTE[6], PALETTE[5], PALETTE[0], PALETTE[4], PALETTE[1]])
    panel(axes[1], 'b', 'Dataset reconstruction PSNR', 'PSNR (dB)')
    for k, label, c, style in [('crypto', 'Correct inverse', PALETTE[0], '-'), ('crypto_wrong_key', 'Wrong key', PALETTE[4], '-'), ('crypto_nonce_bitflip', 'Nonce bit error', PALETTE[1], '--')]:
        cdf(axes[2], a[k + '_ssim'], label, c, style)
    panel(axes[2], 'c', 'Per-image fidelity distribution', 'Empirical CDF', 'SSIM')
    difference = a['crypto_ssim'] - a['identity_ssim']
    axes[3].hist(difference, bins=40, color=PALETTE[0], alpha=0.8, weights=np.ones(len(difference)) * 100 / len(difference))
    axes[3].axvline(0, color='#333333', lw=0.8, ls='--')
    panel(axes[3], 'd', 'AWGN utility comparison', 'Holdout images (%)', 'SSIM: HKDF/AES minus identity')
    fig.legend(*axes[2].get_legend_handles_labels(), loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.035), frameon=False)
    fig.text(0.5, 0.01, '3,600 holdout images; Bob 12 dB. Error bars: 2,000-resample bootstrap 95% intervals.', ha='center', fontsize=8.5)
    save(fig, out, 'cifar_utility_controls_grid', {'models': dict(zip(keys, rows)), 'paired_delta_mean': float(difference.mean()), 'paired_delta_percentiles': np.percentile(difference, [2.5, 50, 97.5]).tolist(), 'bootstrap_repetitions': 2000, 'bootstrap_seed': 20261024}, [p, raw])

def transfer(out):
    p = RESULTS / 'archived_transfer_crypto.json'
    raw = RESULTS / 'archived_transfer_crypto_per_image.npz'
    bp = RESULTS / 'bob_holdout_metrics.json'
    br = RESULTS / 'bob_holdout_per_image.npz'
    if not all((x.exists() for x in [p, raw, bp, br])):
        return
    d = read(p)
    a = np.load(raw)
    b = np.load(br)
    keys = ['base', 'strong', 'no_nonce', 'bob_style']
    names = ['Bob', 'Base', 'Strong', 'No\nnonce', 'Bob\nstyle']
    ii = np.random.default_rng(20261025).integers(0, 3600, size=(2000, 3600))
    rows = [boot_intervals(b['crypto_mse'], b['crypto_ssim'], ii)] + [boot_intervals(a[k + '_mse'], a[k + '_ssim'], ii) for k in keys]
    fig, axes = grid(title='CIFAR-10: seed-mixer-trained attacks transferred to HKDF/AES')
    bars(axes[0], names, rows, 'psnr')
    panel(axes[0], 'a', 'Pixel reconstruction error', 'PSNR (dB)')
    bars(axes[1], names, rows, 'ssim')
    panel(axes[1], 'b', 'Average structural fidelity', 'SSIM')
    cdf(axes[2], b['crypto_ssim'], 'Bob', PALETTE[0])
    for j, k in enumerate(keys):
        cdf(axes[2], a[k + '_ssim'], names[j + 1].replace('\n', ' '), PALETTE[j + 1], ['-', '--', ':', '-.'][j])
    panel(axes[2], 'c', 'Per-image reconstruction tails', 'Empirical CDF', 'SSIM')
    x = np.arange(4)
    p95 = [d['models'][k]['ssim_p95'] for k in keys]
    mx = [d['models'][k]['ssim_max'] for k in keys]
    axes[3].bar(x - 0.17, p95, 0.34, label='95th percentile', color=PALETTE[1])
    axes[3].bar(x + 0.17, mx, 0.34, label='Maximum', color=PALETTE[3])
    axes[3].set_xticks(x, names[1:])
    axes[3].set_ylim(0, 1)
    axes[3].grid(axis='x', visible=False)
    axes[3].legend(fontsize=8, frameon=False)
    panel(axes[3], 'd', 'Largest observed Eve fidelities', 'SSIM')
    fig.legend(*axes[2].get_legend_handles_labels(), loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.035), frameon=False)
    fig.text(0.5, 0.01, 'Bob 12 dB; Eve 20 dB. Frozen transfer attacks; no optimality claim.', ha='center', fontsize=8.5)
    save(fig, out, 'cifar_archived_transfer_grid', {'models': dict(zip(['bob'] + keys, rows)), 'eve_ssim_p95': dict(zip(keys, p95)), 'eve_ssim_max': dict(zip(keys, mx)), 'bootstrap_repetitions': 2000, 'bootstrap_seed': 20261025}, [p, raw, bp, br])

def kodak(out):
    p = RESULTS / 'kodak/kodak_metrics.json'
    r = RESULTS / 'kodak/kodak_per_image.json'
    if (RESULTS / 'kodak_fresh/kodak_metrics.json').exists():
        p = RESULTS / 'kodak_fresh/kodak_metrics.json'
        r = RESULTS / 'kodak_fresh/kodak_per_image.json'
    if not all((x.exists() for x in [p, r])):
        return
    d = read(p)
    rows = read(r)['images']
    keys = ['bob', 'eve_no_nonce_transfer', 'eve_bob_style_transfer', 'mean_training_image']
    labels = ['Bob', 'No-nonce Eve', 'Bob-style Eve', 'Training prior']
    colors = [PALETTE[0], PALETTE[1], PALETTE[3], PALETTE[5]]
    fresh = [name for name in d['metrics'] if name.startswith('fresh_')]
    for j, name in enumerate(fresh):
        keys.append(name)
        labels.append('CIFAR-trained MLP' + (f' {j + 1}' if len(fresh) > 1 else ''))
        colors.append(PALETTE[(j + 2) % len(PALETTE)])
    fig, axes = grid(title='Kodak: native-resolution tiled transmission on all 24 images', bottom=0.23 if fresh else 0.2)
    data = {k: {m: [row['models'][k][m] for row in rows] for m in ['ssim', 'psnr', 'mse']} for k in keys}
    x = np.arange(1, 25)
    for j, (k, label, c) in enumerate(zip(keys, labels, colors)):
        style = ['-', '-', ':', '-.', '--'][j % 5]
        axes[0].plot(x, data[k]['ssim'], color=c, label=label, marker='o', markersize=2.5, ls=style)
        axes[1].plot(x, data[k]['psnr'], color=c, label=label, marker='o', markersize=2.5, ls=style)
        cdf(axes[2], data[k]['ssim'], label, c, style)
    for ax in axes[:2]:
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_xlim(0.5, 24.5)
    axes[0].set_ylim(0, 1)
    panel(axes[0], 'a', 'Full-image structural fidelity', 'SSIM', 'Kodak image index')
    panel(axes[1], 'b', 'Full-image pixel fidelity', 'PSNR (dB)', 'Kodak image index')
    panel(axes[2], 'c', 'Variability across source images', 'Empirical CDF', 'Full-image SSIM')
    for j, (k, c) in enumerate(zip(keys, colors)):
        point = d['metrics'][k]['mean_image_ssim']
        lo, hi = d['metrics'][k]['ssim_95ci']
        axes[3].errorbar(point, len(keys) - 1 - j, xerr=np.array([[point - lo], [hi - point]]), fmt='o', color=c, capsize=4, markersize=5)
    axes[3].set_yticks(range(len(keys)), list(reversed(labels)))
    axes[3].set_ylim(-0.6, len(keys) - 0.4)
    axes[3].set_xlim(0.15, 0.85)
    panel(axes[3], 'd', 'Mean and 95% bootstrap interval', '', 'Mean full-image SSIM')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3 if fresh else 4, bbox_to_anchor=(0.5, 0.035), frameon=False)
    fig.text(0.5, 0.01, '32 x 32 tiles; Bob 12 dB; Eve 20 dB. Confidence intervals resample source images.', ha='center', fontsize=8.5)
    save(fig, out, 'kodak_native_results_grid', {'per_image': data, 'summary': d['metrics'], 'paired_ssim_gaps': d['paired_bob_ssim_gaps']}, [p, r])

def historical(out):
    bp = HIST / 'snr_sweep.json'
    ep = HIST / 'eve_snr_sweep.json'
    if not all((p.exists() for p in [bp, ep])):
        return
    b = read(bp)
    e = read(ep)
    fig, axes = grid(3, 2, '32-bit seed-mixer control: descriptive SNR sweeps', bottom=0.115)
    bx = [v['snr_bob_db'] for v in b]
    ex = [v['snr_eve_db'] for v in e]
    for j, (suffix, label, c) in enumerate([('bob', 'Bob', PALETTE[0]), ('eve_best', 'Eve envelope', PALETTE[1]), ('wrongkey', 'Wrong key', PALETTE[4])]):
        axes[0].plot(bx, [v['psnr_' + suffix] for v in b], marker='o', markersize=3, color=c, label=label)
        axes[1].plot(bx, [v['ssim_' + suffix] for v in b], marker='o', markersize=3, color=c, label=label)
    panel(axes[0], 'a', 'Bob SNR sweep', 'PSNR (dB)', 'Bob SNR (dB)')
    panel(axes[1], 'b', 'Bob SNR sweep', 'SSIM', 'Bob SNR (dB)')
    axes[1].set_ylim(0, 1)
    for j, k in enumerate(['base', 'strong', 'no_nonce', 'bob_style']):
        axes[2].plot(ex, [v['eve_all'][k]['psnr'] for v in e], color=PALETTE[j + 1], ls=['-', '--', ':', '-.'][j], marker='o', markersize=2, label=k.replace('_', ' '))
        axes[3].plot(ex, [v['eve_all'][k]['ssim'] for v in e], color=PALETTE[j + 1], ls=['-', '--', ':', '-.'][j], marker='o', markersize=2, label=k.replace('_', ' '))
    panel(axes[2], 'c', 'Eve SNR sweep', 'Eve PSNR (dB)', 'Eve SNR (dB)')
    panel(axes[3], 'd', 'Eve SNR sweep', 'Eve SSIM', 'Eve SNR (dB)')
    axes[4].plot(bx, [100 * v['intercept_acc'] for v in b], color=PALETTE[3], marker='o', markersize=3, label='Classifier')
    axes[4].axhline(10, color=PALETTE[5], ls='--', lw=1, label='Uniform chance')
    axes[4].set_ylim(8, 12)
    panel(axes[4], 'e', 'Recorded semantic interception', 'Top-1 accuracy (%)', 'Bob SNR (dB)')
    axes[5].plot(ex, [v['intercept_entropy_mean'] for v in e], color=PALETTE[2], marker='o', markersize=3)
    axes[5].axhline(np.log(10), color=PALETTE[5], ls='--', lw=1)
    axes[5].set_ylim(2.15, 2.32)
    panel(axes[5], 'f', 'Recorded classifier confidence', 'Predictive entropy (nats)', 'Eve SNR (dB)')
    axes[0].legend(fontsize=8, frameon=False, loc='lower right')
    axes[2].legend(fontsize=8, frameon=False, ncol=2)
    axes[4].legend(fontsize=8, frameon=False)
    fig.text(0.5, 0.027, 'Selection-exposed seed-mixer control; these curves do not evaluate the HKDF/AES protocol.', ha='center', fontsize=8.5)
    save(fig, out, 'historical_snr_six_panel_grid', {'bob_sweep': [{k: v[k] for k in ['snr_bob_db', 'snr_eve_db', 'psnr_bob', 'ssim_bob', 'psnr_eve_best', 'ssim_eve_best', 'psnr_wrongkey', 'ssim_wrongkey', 'intercept_acc']} for v in b], 'eve_sweep': [{k: v[k] for k in ['snr_eve_db', 'eve_all', 'intercept_entropy_mean']} for v in e]}, [bp, ep])

def classifier(out, protocol='crypto'):
    p = RESULTS / f'classifiers_{protocol}/summary.json'
    if not p.exists():
        return
    d = read(p)
    models = d['models']
    names = d['validation_ranking'][:6]
    selected = d['selected_attacker']
    s = d['selected_test_metrics']
    if not selected:
        return
    interface = {'crypto': 'unique-nonce keyed protocol', 'legacy': '32-bit seed-mixer control', 'identity': 'identity control', 'fixed_crypto': 'fixed-transform control', 'permutation_only': 'permutation-only control'}[protocol]
    fig, axes = grid(title=f'CIFAR-10: interface-fitted semantic attacks\n{interface}')
    fig.subplots_adjust(top=0.84, hspace=0.6)
    x = np.arange(len(names))
    val = [100 * models[k]['test']['accuracy'] for k in names]
    cis = np.array([models[k]['test']['accuracy_wilson_95'] for k in names]) * 100
    axes[0].bar(x, val, color=PALETTE[:len(names)])
    axes[0].errorbar(x, val, yerr=np.maximum(0, np.stack((np.asarray(val) - cis[:, 0], cis[:, 1] - np.asarray(val)))), fmt='none', color='#222222', capsize=3)
    short = {'hist_gradient_boosting_combined': 'HGB\ncomb.', 'hist_gradient_boosting_magnitude': 'HGB\nabs', 'random_forest_magnitude': 'RF\nabs', 'random_forest_combined': 'RF\ncomb.', 'logistic_raw': 'Logit\nraw', 'logistic_magnitude': 'Logit\nabs'}
    axes[0].set_xticks(x, [short.get(k, k.replace('mlp_raw_seed', 'MLP\n')) for k in names], fontsize=7.8)
    axes[0].axhline(10, color='#333333', ls='--', lw=1)
    axes[0].set_ylim(0, max(15, max(val) + 4))
    panel(axes[0], 'a', 'Validation-ranked attackers', 'Top-1 accuracy (%)')
    classes = s['per_class']
    r = [100 * q['recall'] for q in classes]
    ci = np.array([q['recall_wilson_95'] for q in classes]) * 100
    axes[1].bar(range(10), r, color=PALETTE[0])
    axes[1].errorbar(range(10), r, yerr=np.maximum(0, np.stack((np.asarray(r) - ci[:, 0], ci[:, 1] - np.asarray(r)))), fmt='none', color='#222222', capsize=2)
    axes[1].set_xticks(range(10))
    axes[1].set_ylim(0, max(25, max(r) + 8))
    panel(axes[1], 'b', 'Selected attacker: class recall', 'Recall (%)', 'CIFAR-10 class index')
    cm = np.array(s['confusion_matrix'], float)
    cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1)
    image = axes[2].pcolormesh(np.arange(11) - 0.5, np.arange(11) - 0.5, cm, cmap='Blues', vmin=0, vmax=max(0.25, float(cm.max())), shading='flat')
    axes[2].invert_yaxis()
    axes[2].set_xticks(range(10))
    axes[2].set_yticks(range(10))
    axes[2].grid(False)
    cb = fig.colorbar(image, ax=axes[2], fraction=0.035, pad=0.03)
    cb.ax.tick_params(labelsize=7.5)
    panel(axes[2], 'c', 'Class confusion: row fractions', 'True class', 'Predicted class')
    predictions = p.parent / 'selected_predictions.npz'
    if predictions.exists():
        a = np.load(predictions)
        confidence = a['probabilities'].max(axis=1)
        axes[3].hist(confidence, bins=25, color=PALETTE[3], weights=np.ones(len(confidence)) * 100 / len(confidence))
        axes[3].set_xlim(max(0, float(confidence.min()) - 0.005), min(1, float(confidence.max()) + 0.005))
    panel(axes[3], 'd', 'Selected output confidence', 'Holdout images (%)', 'Maximum predicted probability')
    fig.text(0.5, 0.03, 'Attacker selected by validation accuracy; intervals condition on one frozen model and channel draw.', ha='center', fontsize=8.5)
    save(fig, out, f'fresh_classifier_{protocol}_grid', {'displayed_validation_ranking': names, 'models': {k: models[k]['test'] for k in names}, 'selected': selected, 'selected_test_metrics': s}, [p] + ([predictions] if predictions.exists() else []))

def fresh_reconstruction(out, protocol='crypto'):
    p = RESULTS / f'reconstruction_{protocol}_final/reconstruction_metrics.json'
    if not p.exists():
        p = RESULTS / f'reconstruction_{protocol}/reconstruction_metrics.json'
    raw = p.parent / 'reconstruction_per_image.npz'
    if not (p.exists() and raw.exists()):
        return
    d = read(p)
    models = d['families']
    keys = [k for k in models if k.startswith('mlp_seed_')] + [k for k in ['ridge_raw_nonce', 'ridge_sorted_abs_nonce', 'fitting_mean_image'] if k in models]
    if not keys:
        return
    a = np.load(raw)
    keys = [k for k in keys if k + '_mse' in a and k + '_ssim' in a]
    if not keys:
        return
    n = len(a[keys[0] + '_mse'])
    ii = np.random.default_rng(20261027).integers(0, n, size=(2000, n))
    rows = [boot_intervals(a[k + '_mse'], a[k + '_ssim'], ii) for k in keys]
    names = [k.replace('mlp_seed_', 'MLP ').replace('ridge_raw_nonce', 'Ridge\nraw').replace('ridge_sorted_abs_nonce', 'Ridge\nabs').replace('fitting_mean_image', 'Image\nprior') for k in keys]
    interface = {'crypto': 'unique-nonce keyed protocol', 'legacy': '32-bit seed-mixer control', 'identity': 'identity control', 'fixed_crypto': 'fixed-transform control', 'permutation_only': 'permutation-only control'}[protocol]
    fig, axes = grid(title=f'CIFAR-10: interface-fitted reconstruction attacks\n{interface}')
    fig.subplots_adjust(top=0.84, hspace=0.6)
    bars(axes[0], names, rows, 'ssim')
    panel(axes[0], 'a', 'Reconstruction fidelity', 'Mean SSIM')
    bars(axes[1], names, rows, 'psnr')
    panel(axes[1], 'b', 'Pixel distortion', 'Dataset PSNR (dB)')
    for j, k in enumerate(keys):
        cdf(axes[2], a[k + '_ssim'], names[j].replace('\n', ' '), PALETTE[j % len(PALETTE)], ['-', '--', ':', '-.', '-', '--'][j % 6])
    panel(axes[2], 'c', 'Per-image SSIM distribution', 'Empirical CDF', 'SSIM')
    for j, k in enumerate(keys):
        epochs = models[k].get('epochs', [])
        if isinstance(epochs, list) and epochs:
            axes[3].plot([r['epoch'] for r in epochs], [r['validation_mse'] for r in epochs], color=PALETTE[j], marker='o', markersize=3, label=names[j])
    panel(axes[3], 'd', 'MLP validation curves', 'Validation MSE', 'Training epoch')
    fig.legend(*axes[2].get_legend_handles_labels(), loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.018), frameon=False)
    save(fig, out, f'fresh_reconstruction_{protocol}_grid', {'models': dict(zip(keys, rows)), 'training': d['training'], 'learning_curves': {k: models[k].get('epochs') for k in keys}, 'bootstrap_repetitions': 2000, 'bootstrap_seed': 20261027}, [p, raw])

def physical_controls(out):
    p = RESULTS / 'channel_control_metrics.json'
    raw = RESULTS / 'channel_control_per_image.npz'
    kp = RESULTS / 'known_plaintext_metrics.json'
    kr = RESULTS / 'known_plaintext_per_image.npz'
    if not all((x.exists() for x in [p, raw, kp, kr])):
        return
    d = read(p)
    k = read(kp)
    a = np.load(raw)
    ka = np.load(kr)
    required = ['AWGN_0dB', 'AWGN_6dB', 'AWGN_12dB', 'AWGN_18dB', 'AWGN_24dB', 'public_inverse_20dB', 'phase_residual_5deg', 'phase_residual_15deg', 'phase_residual_45deg', 'Rayleigh_perfect_CSI', 'Rayleigh_noisy_CSI', 'residual_ISI']
    if not all((name in d['models'] for name in required)):
        return
    ii = np.random.default_rng(20261028).integers(0, 3600, size=(2000, 3600))
    rows = {name: boot_intervals(a[name + '_mse'], a[name + '_ssim'], ii) for name in required}
    pairs = sorted(k['models'], key=int)
    known = {name: boot_intervals(ka['known_' + name + '_mse'], ka['known_' + name + '_ssim'], ii) for name in pairs}
    fig, axes = grid(3, 2, 'Frozen codec with HKDF/AES: channel residuals and misuse controls', bottom=0.12)
    snrs = [0, 6, 12, 18, 24]
    for ax, metric, letter, label in [(axes[0], 'psnr', 'a', 'PSNR (dB)'), (axes[1], 'ssim', 'b', 'SSIM')]:
        values = [rows[f'AWGN_{s}dB'][metric] for s in snrs]
        ci = np.array([rows[f'AWGN_{s}dB'][metric + '_ci'] for s in snrs])
        ax.errorbar(snrs, values, yerr=np.maximum(0, np.stack((np.asarray(values) - ci[:, 0], ci[:, 1] - np.asarray(values)))), color=PALETTE[0], marker='o', capsize=3)
        panel(ax, letter, 'Correct inverse under AWGN', label, 'Bob SNR (dB)')
    phases = [0, 5, 15, 45]
    phase_names = ['AWGN_12dB'] + [f'phase_residual_{s}deg' for s in phases[1:]]
    values = [rows[name]['ssim'] for name in phase_names]
    ci = np.array([rows[name]['ssim_ci'] for name in phase_names])
    axes[2].errorbar(phases, values, yerr=np.maximum(0, np.stack((np.asarray(values) - ci[:, 0], ci[:, 1] - np.asarray(values)))), color=PALETTE[1], marker='o', capsize=3)
    panel(axes[2], 'c', 'Unequalized phase residual at 12 dB', 'SSIM', 'I/Q rotation (degrees)')
    names = ['AWGN_12dB', 'Rayleigh_perfect_CSI', 'Rayleigh_noisy_CSI', 'residual_ISI']
    labels = ['AWGN', 'Rayleigh\nideal CSI', 'Rayleigh\nnoisy CSI', 'Residual\nISI']
    bars(axes[3], labels, [rows[name] for name in names], 'ssim')
    axes[3].tick_params(axis='x', labelsize=8)
    panel(axes[3], 'd', 'Predefined channel residuals at 12 dB', 'SSIM')
    bars(axes[4], ['Bob\n12 dB', 'Public inverse\n20 dB'], [rows['AWGN_12dB'], rows['public_inverse_20dB']], 'ssim', [PALETTE[0], PALETTE[1]])
    panel(axes[4], 'e', 'Attacker explicitly given inverse', 'SSIM')
    values = [known[name]['ssim'] for name in pairs]
    ci = np.array([known[name]['ssim_ci'] for name in pairs])
    axes[5].errorbar([int(name) for name in pairs], values, yerr=np.maximum(0, np.stack((np.asarray(values) - ci[:, 0], ci[:, 1] - np.asarray(values)))), color=PALETTE[3], marker='o', capsize=3)
    axes[5].set_xscale('log')
    axes[5].set_xticks([int(name) for name in pairs], [f'{int(name):,}' for name in pairs])
    axes[5].set_ylim(0, 1)
    panel(axes[5], 'f', 'Fixed-nonce known-plaintext recovery', 'SSIM', 'Fitting plaintext/ciphertext pairs')
    fig.text(0.5, 0.035, '3,600 holdout images; 2,000-resample bootstrap 95% intervals. Reuse recovery uses 20 dB observations.', ha='center', fontsize=8.5)
    fig.text(0.5, 0.012, 'Rayleigh equalizer floor 0.1; gain-estimate error SD 0.1; residual ISI coefficient 0.2.', ha='center', fontsize=8.5)
    save(fig, out, 'channel_misuse_six_panel_grid', {'channels': rows, 'known_plaintext': known, 'channel_specifications': d, 'known_plaintext_specifications': k, 'bootstrap_repetitions': 2000, 'bootstrap_seed': 20261028}, [p, raw, kp, kr])

def classifier_comparison(out):
    protocols = ['crypto', 'identity', 'fixed_crypto']
    paths = [RESULTS / f'classifiers_{name}/summary.json' for name in protocols]
    if not all((path.exists() for path in paths)):
        return
    for name in ['permutation_only', 'legacy']:
        path = RESULTS / f'classifiers_{name}/summary.json'
        if path.exists():
            protocols.append(name)
            paths.append(path)
    summaries = {name: read(path) for name, path in zip(protocols, paths)}
    if not all((v.get('selected_attacker') for v in summaries.values())):
        return
    labels = {'crypto': 'HKDF/\nAES', 'legacy': '32-bit\nmixer', 'identity': 'Identity', 'fixed_crypto': 'Fixed\nnonce', 'permutation_only': 'Permutation\nonly'}
    names = [labels[name] for name in protocols]
    fig, axes = grid(title='Interface-fitted semantic attacks: transform and nonce controls')
    values = [100 * summaries[name]['selected_test_metrics']['accuracy'] for name in protocols]
    cis = np.array([summaries[name]['selected_test_metrics']['accuracy_wilson_95'] for name in protocols]) * 100
    x = np.arange(len(names))
    axes[0].bar(x, values, color=PALETTE[:len(names)])
    axes[0].errorbar(x, values, yerr=np.maximum(0, np.stack((np.asarray(values) - cis[:, 0], cis[:, 1] - np.asarray(values)))), fmt='none', ecolor='#222222', capsize=3)
    axes[0].set_xticks(x, names)
    axes[0].axhline(10, color='#444444', ls='--', lw=1)
    axes[0].set_ylim(0, max(values) * 1.15)
    panel(axes[0], 'a', 'Validation-selected attacker', 'Holdout top-1 accuracy (%)')
    validation = [100 * summaries[name]['models'][summaries[name]['selected_attacker']]['validation']['accuracy'] for name in protocols]
    axes[1].plot([0, 100], [0, 100], ls='--', lw=0.8, color='#777777')
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 100)
    offsets = {'crypto': (6, 14), 'identity': (-42, -19), 'fixed_crypto': (8, 10), 'permutation_only': (35, 0), 'legacy': (8, -10)}
    for j, (protocol, name, xv, yv) in enumerate(zip(protocols, names, validation, values)):
        axes[1].plot(xv, yv, 'o', color=PALETTE[j])
        axes[1].annotate(name.replace('\n', ' '), (xv, yv), xytext=offsets[protocol], textcoords='offset points', fontsize=7.8, arrowprops={'arrowstyle': '-', 'color': PALETTE[j], 'lw': 0.7})
    panel(axes[1], 'b', 'Selection and holdout consistency', 'Holdout top-1 accuracy (%)', 'Validation top-1 accuracy (%)')
    for j, name in enumerate(protocols):
        recall = [100 * r['recall'] for r in summaries[name]['selected_test_metrics']['per_class']]
        axes[2].plot(range(10), recall, marker='o', markersize=3, color=PALETTE[j], ls=['-', '--', ':', '-.', '-'][j], label=names[j].replace('\n', ' '))
    axes[2].set_xticks(range(10))
    axes[2].set_ylim(0, 100)
    panel(axes[2], 'c', 'Class-specific behavior', 'Recall (%)', 'CIFAR-10 class index')
    for j, name in enumerate(protocols):
        models = summaries[name]['models']
        accuracies = sorted((100 * m['test']['accuracy'] for m in models.values() if 'test' in m))
        axes[3].scatter(np.full(len(accuracies), j) + np.linspace(-0.06, 0.06, len(accuracies)), accuracies, color=PALETTE[j], s=16, alpha=0.65)
    axes[3].set_xticks(x, names)
    axes[3].set_ylim(0, 100)
    axes[3].axhline(10, color='#444444', ls='--', lw=1)
    panel(axes[3], 'd', 'Finite fitted attacker family', 'Holdout top-1 accuracy (%)')
    fig.legend(*axes[2].get_legend_handles_labels(), loc='lower center', ncol=len(protocols), bbox_to_anchor=(0.5, 0.035), frameon=False)
    fig.text(0.5, 0.01, 'Same frozen encoder; Eve 20 dB. Selection uses validation data; point clouds are descriptive.', ha='center', fontsize=8.5)
    save(fig, out, 'semantic_transform_ablation_grid', {'summaries': {name: {'selected_attacker': v['selected_attacker'], 'validation': v['models'][v['selected_attacker']]['validation'], 'holdout': v['selected_test_metrics']} for name, v in summaries.items()}}, paths)

def reconstruction_comparison(out):
    protocols = ['crypto', 'identity', 'fixed_crypto']
    paths = [RESULTS / f'reconstruction_{name}_final/reconstruction_metrics.json' for name in protocols]
    raws = [path.parent / 'reconstruction_per_image.npz' for path in paths]
    if not all((path.exists() for path in paths + raws)):
        return
    summaries = {name: read(path) for name, path in zip(protocols, paths)}
    selected = {}
    arrays = {}
    for name, raw in zip(protocols, raws):
        families = summaries[name]['families']
        candidates = [k for k in families if k.startswith('mlp_seed_')]
        if not candidates:
            return
        selected[name] = min(candidates, key=lambda k: (families[k]['selected_validation_mse'], k))
        arrays[name] = np.load(raw)
        if selected[name] + '_mse' not in arrays[name]:
            return
    ii = np.random.default_rng(20261029).integers(0, 3600, size=(2000, 3600))
    rows = []
    for name in protocols:
        key = selected[name]
        rows.append(boot_intervals(arrays[name][key + '_mse'], arrays[name][key + '_ssim'], ii))
    labels = ['Unique\nnonce', 'Identity', 'Fixed\nnonce']
    displayed = protocols.copy()
    prior = 'fitting_mean_image'
    if prior + '_mse' in arrays['crypto']:
        rows.append(boot_intervals(arrays['crypto'][prior + '_mse'], arrays['crypto'][prior + '_ssim'], ii))
        labels.append('Image\nprior')
        displayed.append('training_prior')
    fig, axes = grid(title='Interface-fitted reconstruction attacks: nonce controls')
    bars(axes[0], labels, rows, 'ssim')
    panel(axes[0], 'a', 'Validation-selected direct MLP', 'Mean SSIM')
    bars(axes[1], labels, rows, 'psnr')
    panel(axes[1], 'b', 'Validation-selected direct MLP', 'Dataset PSNR (dB)')
    for j, name in enumerate(protocols):
        key = selected[name]
        style = ['-', '--', ':'][j]
        cdf(axes[2], arrays[name][key + '_ssim'], labels[j].replace('\n', ' '), PALETTE[j], style)
        epochs = summaries[name]['families'][key].get('epochs', [])
        if epochs:
            axes[3].plot([r['epoch'] for r in epochs], [r['validation_mse'] for r in epochs], marker='o', markersize=3, color=PALETTE[j], ls=style, label=labels[j].replace('\n', ' '))
    if 'training_prior' in displayed:
        cdf(axes[2], arrays['crypto'][prior + '_ssim'], 'Training prior', PALETTE[3], '--')
    panel(axes[2], 'c', 'Holdout image fidelity distribution', 'Empirical CDF', 'SSIM')
    panel(axes[3], 'd', 'Selected model validation curves', 'Validation MSE', 'Attacker training epoch')
    fig.legend(*axes[2].get_legend_handles_labels(), loc='lower center', ncol=len(displayed), bbox_to_anchor=(0.5, 0.035), frameon=False)
    fig.text(0.5, 0.01, 'Eve 20 dB; 45,000 fitting, 5,000 validation, 3,600 holdout images; 2,000 bootstrap resamples.', ha='center', fontsize=8.5)
    save(fig, out, 'reconstruction_transform_ablation_grid', {'selected_mlp': selected, 'models': dict(zip(displayed, rows)), 'selected_validation_mse': {name: summaries[name]['families'][key]['selected_validation_mse'] for name, key in selected.items()}, 'learning_curves': {name: summaries[name]['families'][key].get('epochs', []) for name, key in selected.items()}, 'bootstrap_repetitions': 2000, 'bootstrap_seed': 20261029}, paths + raws)

def signed_exploratory(out):
    protocols = ['crypto', 'identity', 'fixed_crypto', 'permutation_only']
    paths = [RESULTS / f'classifiers_signed_{name}/summary.json' for name in protocols]
    primaries = [RESULTS / f'classifiers_{name}/summary.json' for name in protocols]
    if not all((path.exists() for path in paths + primaries)):
        return
    signed = {name: read(path) for name, path in zip(protocols, paths)}
    primary = {name: read(path) for name, path in zip(protocols, primaries)}
    if not all((row.get('selected_attacker') for row in signed.values())):
        return
    labels = ['HKDF/\nAES', 'Identity', 'Fixed\nnonce', 'Permutation\nonly']
    x = np.arange(4)
    fig, axes = grid(title='Exploratory signed-invariant attacks: separate two-model suite')
    values = [100 * signed[name]['selected_test_metrics']['accuracy'] for name in protocols]
    ci = np.array([signed[name]['selected_test_metrics']['accuracy_wilson_95'] for name in protocols]) * 100
    axes[0].bar(x, values, color=PALETTE[:4])
    axes[0].errorbar(x, values, yerr=np.maximum(0, np.stack((np.asarray(values) - ci[:, 0], ci[:, 1] - np.asarray(values)))), fmt='none', ecolor='#222222', capsize=3)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, max(20, max(values) * 1.2))
    axes[0].axhline(10, color='#444444', ls='--', lw=1)
    panel(axes[0], 'a', 'Validation-selected signed attacker', 'Holdout top-1 accuracy (%)')
    for j, (model, label) in enumerate([('logistic_signed', 'Logistic'), ('random_forest_signed', 'Random forest')]):
        y = [100 * signed[name]['models'][model]['test']['accuracy'] for name in protocols]
        ci = np.array([signed[name]['models'][model]['test']['accuracy_wilson_95'] for name in protocols]) * 100
        pos = x + (j - 0.5) * 0.34
        axes[1].bar(pos, y, width=0.34, color=PALETTE[j], label=label)
        axes[1].errorbar(pos, y, yerr=np.maximum(0, np.stack((np.asarray(y) - ci[:, 0], ci[:, 1] - np.asarray(y)))), fmt='none', ecolor='#222222', capsize=2)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 20)
    axes[1].axhline(10, color='#444444', ls='--', lw=1)
    axes[1].legend(fontsize=8, frameon=False, loc='upper left')
    panel(axes[1], 'b', 'Both signed-feature estimators', 'Holdout top-1 accuracy (%)')
    for j, (summaries, label) in enumerate([(primary, 'Primary 11'), (signed, 'Signed 2')]):
        y = [100 * summaries[name]['selected_test_metrics']['accuracy'] for name in protocols]
        ci = np.array([summaries[name]['selected_test_metrics']['accuracy_wilson_95'] for name in protocols]) * 100
        pos = x + (j - 0.5) * 0.34
        axes[2].bar(pos, y, width=0.34, color=PALETTE[j + 2], label=label)
        axes[2].errorbar(pos, y, yerr=np.maximum(0, np.stack((np.asarray(y) - ci[:, 0], ci[:, 1] - np.asarray(y)))), fmt='none', ecolor='#222222', capsize=2)
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, 70)
    axes[2].axhline(10, color='#444444', ls='--', lw=1)
    axes[2].legend(fontsize=7.5, frameon=False, loc='upper left', ncol=2)
    panel(axes[2], 'c', 'Selections remain separate', 'Holdout top-1 accuracy (%)')
    for j, name in enumerate(protocols):
        recall = [100 * row['recall'] for row in signed[name]['selected_test_metrics']['per_class']]
        axes[3].plot(range(10), recall, marker='o', markersize=3, color=PALETTE[j], ls=['-', '--', ':', '-.'][j], label=labels[j].replace('\n', ' '))
    axes[3].set_xticks(range(10))
    axes[3].set_ylim(0, 30)
    panel(axes[3], 'd', 'Selected signed attacker class recall', 'Recall (%)', 'CIFAR-10 class index')
    fig.legend(*axes[3].get_legend_handles_labels(), loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.035), frameon=False)
    fig.text(0.5, 0.01, 'Eve 20 dB; same fitting/validation split; sorted signed symbols and odd moments; Wilson 95% intervals.', ha='center', fontsize=8.5)
    save(fig, out, 'exploratory_signed_invariant_grid', {'signed': {name: {'selected_attacker': row['selected_attacker'], 'selected_test_metrics': row['selected_test_metrics'], 'model_metrics': {k: v['test'] for k, v in row['models'].items()}} for name, row in signed.items()}, 'primary_selected': {name: row['selected_attacker'] for name, row in primary.items()}}, paths + primaries)

def qualitative(out):
    p = RESULTS / 'kodak/kodak_per_image.json'
    if (RESULTS / 'kodak_fresh/kodak_per_image.json').exists():
        p = RESULTS / 'kodak_fresh/kodak_per_image.json'
    if not p.exists():
        return
    rows = read(p)['images']
    fresh = [name for name in rows[0]['models'] if name.startswith('fresh_')]
    eve = fresh[0] if fresh else 'eve_no_nonce_transfer'
    eve_label = 'CIFAR-trained MLP' if fresh else 'Eve transfer'
    models = [('Source', None), ('Bob', 'bob'), (eve_label, eve)]
    for group, indices in enumerate([[0, 4, 8], [12, 16, 20]], 1):
        fig, axes = plt.subplots(3, 3, figsize=(6.9, 6.0))
        fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.04, wspace=0.07, hspace=0.23)
        fig.suptitle('Native Kodak examples: predefined source indices', fontsize=11, y=0.985)
        paths = [p]
        for column, index in enumerate(indices):
            record = rows[index]
            for row, (label, model) in enumerate(models):
                folder = 'kodak_fresh' if model and model.startswith('fresh_') else 'kodak'
                path = BASE / 'data/kodak' / record['name'] if model is None else RESULTS / folder / 'reconstructions' / model / record['name']
                paths.append(path)
                axes[row, column].imshow(Image.open(path), interpolation='nearest')
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                for spine in axes[row, column].spines.values():
                    spine.set_visible(False)
                if row == 0:
                    axes[row, column].set_title(record['name'], fontsize=9)
                else:
                    m = record['models'][model]
                    axes[row, column].set_title(f"{m['psnr']:.2f} dB; SSIM {m['ssim']:.3f}", fontsize=9)
                if column == 0:
                    axes[row, column].set_ylabel(label, fontsize=9)
        save(fig, out, f'kodak_examples_triplet_{group}', {'source_indices': [i + 1 for i in indices], 'metrics': {rows[i]['name']: rows[i]['models'] for i in indices}, 'images_resized_for_inference': False}, list(dict.fromkeys(paths)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=BASE / 'figures')
    args = parser.parse_args()
    utility(args.output)
    transfer(args.output)
    kodak(args.output)
    historical(args.output)
    physical_controls(args.output)
    classifier_comparison(args.output)
    reconstruction_comparison(args.output)
    signed_exploratory(args.output)
    for protocol in ['crypto', 'legacy', 'identity', 'fixed_crypto', 'permutation_only']:
        classifier(args.output, protocol)
        fresh_reconstruction(args.output, protocol)
    qualitative(args.output)
