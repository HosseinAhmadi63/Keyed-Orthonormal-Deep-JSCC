from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
LABELS = {'invariant_mlp_cached_nonce': 'Cached MLP + nonce', 'invariant_mlp_cached_no_nonce': 'Cached MLP; no nonce', 'invariant_mlp_surrogate_nonce': 'Resampled MLP + nonce', 'invariant_mlp_surrogate_no_nonce': 'Resampled MLP; no nonce', 'deepsets_surrogate_no_nonce': 'Resampled DeepSets'}
SHORT = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
PALETTE = ['#24557A', '#4F80A0', '#168A8A', '#6BB5AD', '#9465A5']
ORANGE = '#C46A2B'
GRAY = '#6B7280'

def read(p):
    return json.loads(Path(p).read_text())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--crypto', type=Path, required=True)
    p.add_argument('--positive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    d = read(a.crypto / 'evaluation_summary.json')
    pos = read(a.positive / 'evaluation_summary.json')
    models = d['models']
    names = list(LABELS)
    selected = d['selected']['class']
    pm = next(iter(pos['models'].values()))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.2, 'axes.titlesize': 9, 'axes.labelsize': 8.2, 'xtick.labelsize': 7.4, 'ytick.labelsize': 7.4, 'legend.fontsize': 6.7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(2, 2, figsize=(6.95, 6.35))
    fig.subplots_adjust(left=0.19, right=0.978, top=0.88, bottom=0.13, wspace=0.44, hspace=0.54)
    fig.suptitle('Interface-fitted invariant-aware classifiers', fontsize=10.3, y=0.985)
    fig.text(0.5, 0.949, 'One frozen encoder; HKDF-SHA256 / AES-256-CTR keyed protocol; Eve SNR = 20 dB', ha='center', fontsize=7.8, color=GRAY)
    yy = np.arange(6)
    vals = np.array([models[n]['accuracy'] for n in names] + [pm['accuracy']]) * 100
    ci = np.array([models[n]['accuracy_wilson_95'] for n in names] + [pm['accuracy_wilson_95']]) * 100
    for j in range(6):
        color = PALETTE[j] if j < 5 else ORANGE
        marker = '*' if j < 5 and names[j] == selected else 'o'
        ax[0, 0].errorbar(vals[j], j, xerr=[[vals[j] - ci[j, 0]], [ci[j, 1] - vals[j]]], fmt=marker, color=color, ms=7 if marker == '*' else 4, capsize=2, lw=0.9)
    ax[0, 0].set_yticks(yy, ['Cached MLP\n+ nonce', 'Cached MLP\nno nonce', 'Resampled MLP\n+ nonce', 'Resampled MLP\nno nonce', 'DeepSets\nresampled', 'Permutation only\nMLP'])
    ax[0, 0].invert_yaxis()
    ax[0, 0].axhline(4.5, color='#D1D5DB', lw=0.8)
    ax[0, 0].axvline(10, color=GRAY, ls='--', lw=0.9)
    ax[0, 0].set(xlabel='Holdout accuracy (%)', title='(a) Locked outcomes; Wilson 95% intervals', xlim=(8, 13.5))
    for n, color in [('invariant_mlp_cached_no_nonce', PALETTE[1]), ('invariant_mlp_surrogate_no_nonce', PALETTE[3]), ('deepsets_surrogate_no_nonce', PALETTE[4])]:
        h = models[n]['fit']['history']
        xx = [z['epoch'] for z in h]
        ax[0, 1].plot(xx, [z['fitting_loss'] for z in h], color=color, lw=1, ls='--')
        ax[0, 1].plot(xx, [z['validation_loss'] for z in h], color=color, lw=1.3, label=LABELS[n].replace('; no nonce', ''))
    ax[0, 1].axhline(np.log(10), color=GRAY, lw=0.8, ls=':')
    ax[0, 1].set(xlabel='Completed fitting epoch', ylabel='Cross-entropy (nats)', title='(b) Generalization and training budget')
    ax[0, 1].legend(frameon=False, loc='lower left', fontsize=6.2)
    ax[0, 1].text(0.03, 0.94, 'Solid: validation\nDashed: fitting', transform=ax[0, 1].transAxes, ha='left', va='top', fontsize=6.3, color=GRAY)
    for j, n in enumerate(names):
        h = models[n]['fit']['history']
        x = np.array([z['epoch'] * 45000 / 1000000.0 for z in h])
        y = np.array([z['validation_accuracy'] * 100 for z in h])
        ax[1, 0].plot(x, y, color=PALETTE[j], lw=0.9, alpha=0.85)
        ep = models[n]['fit']['selected_validation_epoch']
        ax[1, 0].scatter(ep * 45000 / 1000000.0, y[ep - 1], marker='*' if n == selected else 'o', s=27 if n == selected else 13, c=PALETTE[j], zorder=4)
    ax[1, 0].axhline(10, color=GRAY, lw=0.8, ls='--')
    ax[1, 0].set(xlabel='Fitting presentations (millions)', ylabel='Validation accuracy (%)', title='(c) Checkpoints chosen before holdout')
    sel = models[selected]
    xx = np.arange(10)
    for rr, offset, color, label in [(sel, -0.12, PALETTE[3], 'Selected keyed protocol'), (pm, 0.12, ORANGE, 'Permutation only')]:
        rows = rr['per_class']
        mean = np.array([z['recall'] for z in rows]) * 100
        ci = np.array([z['recall_wilson_95'] for z in rows]) * 100
        ax[1, 1].errorbar(xx + offset, mean, yerr=np.stack([mean - ci[:, 0], ci[:, 1] - mean]), fmt='o', ms=2.8, color=color, capsize=1.2, lw=0.7, label=label)
    ax[1, 1].axhline(10, color=GRAY, lw=0.8, ls='--')
    ax[1, 1].set_xticks(xx, SHORT, rotation=45, ha='right')
    ax[1, 1].set(ylabel='Holdout class recall (%)', title='(d) Class-specific outcomes', xlim=(-0.5, 9.5), ylim=(-1, 38))
    ax[1, 1].legend(frameon=False, loc='upper right', fontsize=5.8)
    for axis in ax.flat:
        axis.grid(axis='y', alpha=0.18, lw=0.5)
        axis.set_axisbelow(True)
    fig.text(0.5, 0.039, 'Classifier split: 45,000 fitting / 5,000 stratified validation images. Holdout: 3,600 images.\nAll five keyed-protocol fits and the positive control were completed before holdout evaluation.\nFinite tested attackers do not establish a bound on recoverable information.', ha='center', va='center', fontsize=7, color=GRAY)
    stem = a.output / 'fresh_invariant_classifier_grid'
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.png'), dpi=220)
    plt.close(fig)
    metadata = {'sources': {str(a.crypto / 'evaluation_summary.json'): hashlib.sha256((a.crypto / 'evaluation_summary.json').read_bytes()).hexdigest(), str(a.positive / 'evaluation_summary.json'): hashlib.sha256((a.positive / 'evaluation_summary.json').read_bytes()).hexdigest()}, 'selected_validation_family': selected, 'holdout_values': {n: {k: models[n][k] for k in ['accuracy', 'accuracy_wilson_95', 'cross_entropy', 'predictive_entropy', 'per_class']} for n in names}, 'permutation_only_positive_control': {k: pm[k] for k in ['accuracy', 'accuracy_wilson_95', 'cross_entropy', 'per_class']}}
    stem.with_suffix('.plotted_values.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(stem.with_suffix('.pdf'))
if __name__ == '__main__':
    main()
