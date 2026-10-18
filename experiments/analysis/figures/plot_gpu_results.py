from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter
BASE = Path(__file__).resolve().parent
WIDTH = 18.46 / 2.54
COLORS = {1001: '#0072B2', 1002: '#D55E00', 1003: '#009E73'}
CASES = [f'seed{s}_lambda{w:g}' for s in [1001, 1002, 1003] for w in [0.0, 0.2]]

def save(fig, name):
    fig.savefig(BASE / 'figures' / f'{name}.pdf', metadata={'Title': name, 'Creator': 'Matplotlib'}, dpi=300)
    fig.savefig(BASE / 'figures' / f'{name}.png', dpi=300)
    plt.close(fig)

def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.18, lw=0.5)
    ax.tick_params(labelsize=7)

def convergence(data):
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.5), sharex=True, sharey=True, gridspec_kw={'hspace': 0.3, 'wspace': 0.15})
    for row, weight in enumerate([0.0, 0.2]):
        for col, seed in enumerate([1001, 1002, 1003]):
            ax = axes[row, col]
            name = f'seed{seed}_lambda{weight:g}'
            history = [r for r in data['codec_history'] if r['case'] == name]
            x = np.array([r['epoch'] for r in history])
            y = np.array([r['validation_mse'] for r in history])
            selected = history[0]['selected_epoch']
            ax.plot(x, [r['fitting_mse'] for r in history], color=COLORS[seed], ls='--', lw=1.2)
            ax.plot(x, y, color=COLORS[seed], lw=1.2)
            ax.plot(selected, y[selected - 1], marker='*', ms=7, color='black', zorder=5)
            ax.axvline(selected, color='.5', ls=':', lw=0.65)
            ax.set_title(f'Seed {seed}, $\\lambda={weight:g}$', fontsize=8.5)
            ax.set_yscale('log')
            ax.set_xticks([1, 10, 20, 30])
            clean(ax)
            if row == 1:
                ax.set_xlabel('Codec epoch')
            if col == 0:
                ax.set_ylabel('Reconstruction MSE')
    fig.legend([Line2D([], [], color='.25', ls='--'), Line2D([], [], color='.25'), Line2D([], [], color='black', marker='*', ls='None', ms=7)], ['Fitting distortion', 'Validation distortion', 'Validation-selected checkpoint'], loc='lower center', ncol=3, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.94, bottom=0.14)
    save(fig, 'gpu_codec_convergence')

def matched(data):
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.75))
    configs = [('bob_psnr', None, '(a) Bob utility', 'Dataset PSNR (dB)', 1), ('bob_valid', 'bob_legacy', '(b) Bob SSIM', 'Mean RGB SSIM', 1), ('accuracy', None, '(c) Selected class attacker', 'Accuracy (%)', 100), ('bob_eve_valid', 'bob_eve_legacy', '(d) Bob-Eve SSIM gap', 'Paired SSIM gap', 1), ('eve_prior_valid', 'eve_prior_legacy', '(e) Eve SSIM gain over prior', 'Paired SSIM difference', 1), ('eve_prior_mse', None, '(f) Reconstruction MSE gain', 'Paired MSE reduction', 1)]
    for ax, (primary, legacy, title, ylabel, scale) in zip(axes.flat, configs):
        for si, seed in enumerate([1001, 1002, 1003]):
            offset = (si - 1) * 0.06
            xx = np.array([0.0, 1.0]) + offset
            for field, style, marker in [(primary, '-', 'o')] + ([] if legacy is None else [(legacy, '--', 'o')]):
                means = []
                for arm, weight in enumerate([0.0, 0.2]):
                    case = f'seed{seed}_lambda{weight:g}'
                    cells = [r for r in data['key_noise_cells'] if r['case'] == case]
                    yy = np.array([r[field] for r in cells]) * scale
                    ax.scatter(xx[arm] + np.linspace(-0.018, 0.018, 9), yy, s=7, color=COLORS[seed], alpha=0.3, marker='.' if style == '-' else '_', zorder=2)
                    means.append(next((r['aggregate'][field] for r in data['cases'] if r['case'] == case)) * scale)
                ax.plot(xx, means, color=COLORS[seed], ls=style, marker=marker, ms=3.8, lw=1.1, markerfacecolor=COLORS[seed] if style == '-' else 'white', zorder=3)
        if primary == 'accuracy':
            ax.axhline(10, color='.4', lw=0.7, ls=':')
        if primary in ['eve_prior_valid', 'eve_prior_mse']:
            ax.axhline(0, color='.4', lw=0.7, ls=':')
        if primary in ['eve_prior_valid', 'eve_prior_mse']:
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(fmt)
        ax.set_title(title, fontsize=8.3, loc='left')
        ax.set_ylabel(ylabel, fontsize=7.4)
        ax.set_xticks([0, 1], ['0', '0.2'])
        ax.set_xlabel('Defense weight $\\lambda$', fontsize=7.5)
        ax.set_xlim(-0.2, 1.2)
        clean(ax)
    handles = [Line2D([], [], color=COLORS[s], marker='o', ms=3.5, label=f'Codec seed {s}') for s in COLORS]
    handles += [Line2D([], [], color='.2', label='Valid-window SSIM (primary)'), Line2D([], [], color='.2', ls='--', marker='o', markerfacecolor='white', ms=3.5, label='Zero-padded SSIM'), Line2D([], [], color='.4', ls='None', marker='.', ms=3, label='Individual key/noise evaluations')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=6.8, frameon=False, columnspacing=1.2)
    fig.tight_layout(rect=[0.005, 0.11, 0.995, 0.99], h_pad=1.5, w_pad=1.0)
    save(fig, 'gpu_matched_summary')

def outlined(ax, row, col, color, inner=False):
    m = 0.13 if inner else 0.035
    ax.add_patch(Rectangle((col - 0.5 + m, row - 0.5 + m), 1 - 2 * m, 1 - 2 * m, fill=False, edgecolor=color, linewidth=1.15, zorder=4))

def overview(data):
    classifiers = [f'raw_classifier_{s}' for s in [51, 52, 53, 54, 55]] + ['invariant_cached_61', 'invariant_cached_62', 'invariant_surrogate_61', 'invariant_surrogate_62', 'deepsets_surrogate']
    reconstructors = [f'mlp_reconstruction_{s}' for s in [71, 72, 73]] + ['cnn_cached_nonce', 'cnn_cached_no_nonce', 'cnn_surrogate_no_nonce', 'cnn_large_surrogate_no_nonce', 'image_prior']
    clslabels = ['Raw\n51', 'Raw\n52', 'Raw\n53', 'Raw\n54', 'Raw\n55', 'Inv-c\n61', 'Inv-c\n62', 'Inv-s\n61', 'Inv-s\n62', 'Deep\nSets']
    reclabels = ['MLP\n71', 'MLP\n72', 'MLP\n73', 'CNN\nN+', 'CNN\nN-', 'CNN\nS32', 'CNN\nS256', 'Image\nprior']
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 6.75))
    panels = [('accuracy', classifiers, clslabels, 100, '(a) Classification: all 10 learned candidates', 'Accuracy (%)'), ('valid_gain', reconstructors, reclabels, 1000, '(b) Reconstruction: valid SSIM gain over prior', 'SSIM difference ($\\times10^{-3}$)'), ('mse_gain', reconstructors, reclabels, 100000, '(c) Reconstruction: MSE reduction from prior', 'MSE reduction ($\\times10^{-5}$)')]
    for ax, (metric, names, labels, scale, title, cbarlabel) in zip(axes, panels):
        matrix = []
        selected = []
        for case in CASES:
            caseinfo = next((r for r in data['cases'] if r['case'] == case))
            rows = {r['candidate']: r for r in data['candidates'] if r['case'] == case}
            matrix.append([0.0 if n == 'image_prior' else rows[n][metric] * scale for n in names])
            selected.append(names.index(caseinfo['selected_classifier'] if metric == 'accuracy' else caseinfo['selected_reconstructor']))
        matrix = np.array(matrix)
        if metric == 'accuracy':
            half = max(1.5, float(np.max(np.abs(matrix - 10))))
            norm = TwoSlopeNorm(vmin=10 - half, vcenter=10, vmax=10 + half)
        else:
            limit = float(np.max(np.abs(matrix))) * 1.05
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        im = ax.imshow(matrix, cmap='RdBu', norm=norm, aspect='auto', interpolation='nearest')
        for i in range(6):
            best = np.flatnonzero(np.isclose(matrix[i], matrix[i].max(), atol=1e-12, rtol=0))
            for j in range(len(names)):
                dark = abs(norm(matrix[i, j]) - 0.5) > 0.32
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=7.2, color='white' if dark else 'black')
            outlined(ax, i, selected[i], 'black')
            for b in best:
                outlined(ax, i, int(b), '#E69F00', inner=True)
        ax.set_xticks(np.arange(len(names)), labels, fontsize=7)
        ax.set_yticks(np.arange(6), [f'{s} / {w:g}' for s in [1001, 1002, 1003] for w in [0.0, 0.2]], fontsize=7)
        ax.set_ylabel('Codec seed / $\\lambda$', fontsize=7.6)
        ax.set_title(title, fontsize=8.5, loc='left', pad=6)
        for y in [1.5, 3.5]:
            ax.axhline(y, color='white', lw=1)
        ax.tick_params(length=0)
        bar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.018)
        bar.ax.tick_params(labelsize=6.7)
        bar.set_label(cbarlabel, fontsize=7)
    fig.legend([Patch(facecolor='none', edgecolor='black', lw=1.2), Patch(facecolor='none', edgecolor='#E69F00', lw=1.2)], ['Validation-selected learned attack', 'Strongest observed holdout value (descriptive)'], loc='lower center', ncol=2, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.13, right=0.95, top=0.965, bottom=0.09, hspace=0.55)
    save(fig, 'gpu_attack_family_overview')

def budgets(data):
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.75))
    reps = [(['raw_classifier_51', 'invariant_surrogate_61', 'deepsets_surrogate'], ['Raw MLP', 'Invariant MLP + resampling', 'DeepSets + resampling'], '(a) Classifier fitting and validation', 'Cross entropy'), (['mlp_reconstruction_71', 'cnn_cached_nonce', 'cnn_large_surrogate_no_nonce'], ['Direct MLP', 'CNN, cached + nonce', 'CNN256 + resampling'], '(b) Reconstructor fitting and validation', 'Image MSE')]
    palette = ['#0072B2', '#D55E00', '#009E73']
    for ax, (names, labels, title, ylabel) in zip(axes[0], reps):
        for name, label, color in zip(names, labels, palette):
            rows = [r for r in data['attacker_history'] if r['case'] == 'seed1001_lambda0.2' and r['candidate'] == name]
            x = [r['epoch'] for r in rows]
            val = [r['validation_loss'] for r in rows]
            sel = rows[0]['selected_epoch']
            ax.plot(x, [r['fitting_loss'] for r in rows], color=color, ls='--', lw=1)
            ax.plot(x, val, color=color, lw=1, label=label)
            ax.plot(sel, val[sel - 1], marker='o', color=color, ms=3)
        ax.set_title(title, fontsize=8.4, loc='left')
        ax.set_xlabel('Offline attacker epoch')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 30)
        ax.legend(fontsize=6.3, loc='upper right', framealpha=0.9)
        clean(ax)
    updates = {'class': [], 'rec': []}
    fresh = {'class': [], 'rec': []}
    for case in CASES:
        rows = [r for r in data['candidates'] if r['case'] == case]
        for group, kinds in [('class', ['class', 'sets']), ('rec', ['mlp', 'cnn'])]:
            updates[group].append(sum((r['updates'] for r in rows if r['kind'] in kinds)) / 1000)
            fresh[group].append(sum((r['fresh_presentations'] for r in rows if r['kind'] in kinds)) / 1000000.0)
    for ax, values, title, ylabel in [(axes[1, 0], updates, '(c) Updates across 17 attack candidates', 'Optimizer updates (thousands)'), (axes[1, 1], fresh, '(d) Public-family training augmentation', 'Resampled presentations (millions)')]:
        x = np.arange(6)
        ax.bar(x, values['class'], color='#4477AA', width=0.68, label='Classifiers (10 candidates)')
        ax.bar(x, values['rec'], bottom=values['class'], color='#CC6677', width=0.68, label='Reconstructors (7 candidates)')
        ax.set_xticks(x, [f'{s}\n{w:g}' for s in [1001, 1002, 1003] for w in [0.0, 0.2]])
        ax.set_xlabel('Codec seed (top) / defense weight (bottom)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8.3, loc='left')
        clean(ax)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    handles += [Line2D([], [], color='.2'), Line2D([], [], color='.2', ls='--'), Line2D([], [], color='.2', marker='o', ls='None', ms=3)]
    labels += ['Validation loss', 'Fitting loss', 'Selected epoch']
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=6.8, frameon=False)
    fig.tight_layout(rect=[0.005, 0.12, 0.995, 0.99], h_pad=1.5, w_pad=1.7)
    save(fig, 'gpu_attacker_budget_convergence')

def qualitative():
    with np.load(BASE / 'data/qualitative_examples.npz') as a:
        images = {k: a[k] for k in a.files}
    fig = plt.figure(figsize=(WIDTH, 6.2))
    gs = fig.add_gridspec(8, 9, width_ratios=[1.8] + [1] * 8, left=0.015, right=0.99, top=0.93, bottom=0.025, hspace=0.08, wspace=0.08)
    labels = ['Source'] + [f'Seed {s}\n$\\lambda={w:g}$' for s in [1001, 1002, 1003] for w in [0.0, 0.2]] + ['Fitting\nimage prior']
    for row in range(8):
        labelax = fig.add_subplot(gs[row, 0])
        labelax.axis('off')
        labelax.text(0.95, 0.5, labels[row], ha='right', va='center', fontsize=7.3)
        for sample in range(4):
            for side in range(2):
                ax = fig.add_subplot(gs[row, 1 + 2 * sample + side])
                ax.axis('off')
                if row == 0:
                    image = images['truth'][sample]
                elif row == 7:
                    image = images[CASES[0] + '__prior'][sample]
                else:
                    image = images[CASES[row - 1] + ('__bob' if side == 0 else '__eve')][sample]
                ax.imshow(np.clip(image.transpose(1, 2, 0), 0, 1), interpolation='nearest', vmin=0, vmax=1)
                if row == 0:
                    ax.set_title('Bob' if side == 0 else 'Eve', fontsize=6.8, pad=2)
    for sample, index in enumerate([6400, 6401, 6402, 6403]):
        bbox = gs[0, 1 + 2 * sample:3 + 2 * sample].get_position(fig)
        fig.text((bbox.x0 + bbox.x1) / 2, 0.974, f'Test image {index}', ha='center', va='top', fontsize=8)
    save(fig, 'gpu_six_codec_qualitative')

def main():
    (BASE / 'figures').mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 7.5, 'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'axes.linewidth': 0.65})
    data = json.loads((BASE / 'data/plotdata.json').read_text())
    convergence(data)
    matched(data)
    overview(data)
    budgets(data)
    qualitative()
if __name__ == '__main__':
    main()
