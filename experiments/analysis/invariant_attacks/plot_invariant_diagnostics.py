from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS = {'navy': '#24557A', 'teal': '#168A8A', 'orange': '#C46A2B', 'gray': '#6B7280', 'red': '#B4424A'}
SHORT = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    d = json.loads((a.input / 'invariant_diagnostics.json').read_text())
    v = np.load(a.input / 'invariant_diagnostics_per_image.npz', allow_pickle=False)
    s = d['statistics']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.2, 'axes.titlesize': 9.0, 'axes.labelsize': 8.2, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.0, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlepad': 7})
    fig, ax = plt.subplots(3, 2, figsize=(6.95, 8.2))
    fig.subplots_adjust(left=0.12, right=0.975, top=0.91, bottom=0.125, wspace=0.38, hspace=0.65)
    fig.suptitle('Learned-latent invariants and their observable moment contrasts', fontsize=10.0, y=0.99)
    fig.text(0.5, 0.958, 'Exploratory diagnostics on 5,000 classifier-validation images; N = 256', ha='center', fontsize=8.1, color=COLORS['gray'])
    ipr = np.sort(v['latent_participation_ratio'])
    xx = np.arange(1, len(ipr) + 1) / len(ipr)
    ax[0, 0].plot(ipr, xx, color=COLORS['navy'], lw=1.8, label='Learned latents')
    ax[0, 0].axvline(1, color=COLORS['orange'], lw=0.7, ls=':')
    ax[0, 0].axvline(2, color=COLORS['orange'], lw=0.9, ls='--')
    ax[0, 0].annotate('Algebraic examples:\nparticipation ratios 1 and 2', xy=(2, 0.05), xytext=(15, 0.45), arrowprops={'arrowstyle': '->', 'lw': 0.7, 'color': COLORS['orange']}, fontsize=7.1, color=COLORS['orange'])
    ax[0, 0].text(0.98, 0.08, f'Median = {np.median(ipr):.2f}', transform=ax[0, 0].transAxes, ha='right', fontsize=7.7)
    ax[0, 0].set(xlabel='Latent participation ratio', ylabel='Empirical cumulative probability', xlim=(0, 130), ylim=(0, 1.03), title='(a) Observed latent participation ratios')

    def class_error(axis, records, centre=0, scale=1, color=COLORS['navy']):
        means = np.array([z['mean'] for z in records])
        ci = np.array([z['mean_percentile_95'] for z in records])
        axis.errorbar(np.arange(10), (means - centre) * scale, yerr=np.stack([means - ci[:, 0], ci[:, 1] - means]) * scale, fmt='o', ms=3.4, lw=0.9, capsize=2, color=color)
        axis.set_xticks(np.arange(10), SHORT, rotation=45, ha='right')
        axis.set_xlim(-0.5, 9.5)
    class_error(ax[0, 1], s['latent_fourth']['class_means'])
    ax[0, 1].set(ylabel='Mean latent fourth moment', title='(b) Class-dependent latent fourth moment')
    expected = d['conditional_expected_fourth_20dB']
    class_error(ax[1, 0], expected['class_means'], expected['mean'], 1000, COLORS['teal'])
    ax[1, 0].axhline(0, color=COLORS['gray'], lw=0.7, ls=':')
    ax[1, 0].set(ylabel='Fourth-moment contrast ($\\times 10^{-3}$)', title='(c) Expected observable contrast (20 dB)')
    ax[1, 0].text(0.03, 0.05, 'Contrast scale: $-2/N=-1/128$', transform=ax[1, 0].transAxes, va='bottom', fontsize=7.3)
    class_error(ax[1, 1], s['observed_fourth']['class_means'], s['observed_fourth']['mean'], 1000, COLORS['orange'])
    ax[1, 1].axhline(0, color=COLORS['gray'], lw=0.7, ls=':')
    ax[1, 1].set(ylabel='Fourth-moment contrast ($\\times 10^{-3}$)', title='(d) Realized packet contrast (20 dB)')
    names = ['latent_fourth', 'observed_fourth', 'observed_signed_third', 'observed_signed_mean', 'observed_negative_fraction', 'observed_max_absolute']
    labels = ['Latent 4th', 'Packet 4th', 'Packet 3rd', 'Packet mean', 'Sign fraction', 'Packet max.']
    means = np.array([s[k]['class_variance_fraction_eta2'] for k in names])
    lohi = np.array([s[k]['permuted_label_eta2_95'] for k in names])
    yy = np.arange(len(names))
    ax[2, 0].hlines(yy, 100 * lohi[:, 0], 100 * lohi[:, 1], color='#CED4DA', lw=5, label='Random-label 95% interval')
    ax[2, 0].scatter(100 * means, yy, c=[COLORS['navy']] + [COLORS['orange']] * 5, s=22, zorder=3, label='Actual class labels')
    ax[2, 0].set_yticks(yy, labels)
    ax[2, 0].invert_yaxis()
    ax[2, 0].set(xlabel='Between-class variance $\\eta^2$ (%)', title='(e) Class fractions of selected statistics')
    keys = ['None', '30.0', '20.0', '12.0', '6.0', '0.0']
    vals = [100 * np.asarray(d['simulations'][k]['eta2_fourth']) for k in keys]
    boxes = ax[2, 1].boxplot(vals, widths=0.5, showfliers=False, patch_artist=True, medianprops={'color': COLORS['navy'], 'lw': 1.1}, boxprops={'facecolor': '#D9EEEE', 'edgecolor': COLORS['teal']}, whiskerprops={'color': COLORS['teal']}, capprops={'color': COLORS['teal']})
    for j, values in enumerate(vals):
        ax[2, 1].scatter(np.full(len(values), j + 1), values, s=5, alpha=0.45, c=COLORS['teal'])
    ax[2, 1].axhline(100 * d['exchangeable_label_expected_eta2'], color=COLORS['gray'], ls='--', lw=0.9, label='Random-label expected value')
    ax[2, 1].scatter([3], [100 * s['observed_fourth']['class_variance_fraction_eta2']], marker='D', s=25, color=COLORS['orange'], label='Captured 20 dB packets', zorder=4)
    ax[2, 1].set_xticks(range(1, 7), ['Noiseless', '30', '20', '12', '6', '0'])
    ax[2, 1].set(xlabel='Eve SNR (dB)\nIndependent ideal-family draws', ylabel='Fourth-moment $\\eta^2$ (%)', title='(f) Noise sensitivity of fourth moment')
    ax[2, 1].legend(frameon=False, loc='upper left', fontsize=6.1)
    for axis in ax.flat:
        axis.grid(axis='y', alpha=0.18, lw=0.5)
        axis.set_axisbelow(True)
    fig.text(0.5, 0.026, 'Class means: 2,000 image bootstrap resamples. Gray ranges: 2,000 label permutations.\nPanel (f): 16 paired simulator draws per SNR. These diagnostics do not bound all recoverable information.', ha='center', va='center', fontsize=7, color=COLORS['gray'])
    path = a.output / 'latent_invariant_diagnostics_grid'
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=220)
    plt.close(fig)
    metadata = {'source_json_sha256': hashlib.sha256((a.input / 'invariant_diagnostics.json').read_bytes()).hexdigest(), 'source_per_image_sha256': hashlib.sha256((a.input / 'invariant_diagnostics_per_image.npz').read_bytes()).hexdigest(), 'panels': {'a': 'latent participation-ratio ECDF; toy positions1,2', 'b': 'class mean latent fourth with95%percentilebootstrapCI', 'c': 'class mean conditional expected observedfourth minusgrandmean, ×1000; exactideal-sign moments', 'd': 'class mean actualcaptured observedfourth minusgrandmean, ×1000; capturedchallengepackets', 'e': dict(zip(names, means.tolist())), 'f': {k: d['simulations'][k]['eta2_fourth'] for k in keys}}}
    path.with_suffix('.plotted_values.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(path.with_suffix('.pdf'))
if __name__ == '__main__':
    main()
