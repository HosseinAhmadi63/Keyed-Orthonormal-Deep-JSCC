from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE = Path(__file__).resolve().parent
RESULTS = BASE / 'results'
FIGURES = BASE / 'figures'
TABLES = BASE / 'tables'
LABELS = {'bob': 'Authorized decoder', 'mean_training_image': 'Image prior', 'fitting_mean_image': 'Image prior', 'fresh_mlp_seed_3': 'Transferred direct MLP', 'eve_no_nonce_transfer': 'Transferred no-nonce decoder', 'eve_bob_style_transfer': 'Transferred Bob-style decoder', 'crypto_selected_mlp': 'Keyed-interface MLP', 'crypto_raw_ridge': 'Keyed-interface ridge', 'identity_selected_mlp': 'Identity-interface MLP', 'fixed_crypto_selected_mlp': 'Fixed-transform MLP'}
COLORS = {'legacy_zero_clipped': '#4477AA', 'reference_valid_unclipped': '#CC6677'}

def read(path):
    return json.loads(path.read_text())

def ci_text(value, interval, places=4):
    return f'{value:.{places}f} [{interval[0]:.{places}f}, {interval[1]:.{places}f}]'

def save(fig, name):
    fig.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIGURES / f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

def kodak_plot(data):
    names = ['bob', 'mean_training_image', 'fresh_mlp_seed_3', 'eve_no_nonce_transfer', 'eve_bob_style_transfer']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), gridspec_kw={'width_ratios': [1, 1.02]})
    yy = np.arange(len(names))
    for variant, offset in [('legacy_zero_clipped', -0.09), ('reference_valid_unclipped', 0.09)]:
        vals = [data['models'][n][variant]['mean'] for n in names]
        ci = np.array([data['models'][n][variant]['ci95'] for n in names])
        axes[0].errorbar(vals, yy + offset, xerr=np.array([np.array(vals) - ci[:, 0], ci[:, 1] - np.array(vals)]), fmt='o', ms=4, capsize=2, color=COLORS[variant], label='Zero padded, clipped' if variant == 'legacy_zero_clipped' else 'Valid window, unclipped')
        gains = [data['models'][n][variant]['minus_prior'] for n in names]
        ci = np.array([data['models'][n][variant]['minus_prior_ci95'] for n in names])
        axes[1].errorbar(gains, yy + offset, xerr=np.array([np.array(gains) - ci[:, 0], ci[:, 1] - np.array(gains)]), fmt='o', ms=4, capsize=2, color=COLORS[variant])
    axes[0].set_yticks(yy, [LABELS[n] for n in names])
    axes[0].invert_yaxis()
    axes[1].set_yticks(yy, [])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color='.45', lw=0.8)
    axes[0].set_xlabel('Mean full-frame RGB SSIM')
    axes[1].set_xlabel('Paired SSIM difference from image prior')
    axes[0].set_title('(a) Strongest tested non-Bob baseline is the prior', fontsize=10)
    axes[1].set_title('(b) Comparison against that strongest baseline', fontsize=10)
    axes[0].legend(loc='lower right', fontsize=8)
    for ax in axes:
        ax.grid(axis='x', alpha=0.2)
    fig.tight_layout(w_pad=2)
    save(fig, 'kodak_prior_and_paired_contrasts')

def ssim_plot(cifar, kodak):
    scopes = [('CIFAR-10: 3,600 images', cifar, ['bob', 'fitting_mean_image', 'crypto_selected_mlp', 'crypto_raw_ridge', 'identity_selected_mlp', 'fixed_crypto_selected_mlp']), ('Kodak: 24 native frames', kodak, ['bob', 'mean_training_image', 'fresh_mlp_seed_3', 'eve_no_nonce_transfer', 'eve_bob_style_transfer'])]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    for ax, (title, data, names) in zip(axes, scopes):
        yy = np.arange(len(names))
        for variant, offset in [('legacy_zero_clipped', -0.1), ('reference_valid_unclipped', 0.1)]:
            vals = [data['models'][n][variant]['mean'] for n in names]
            ci = np.array([data['models'][n][variant]['ci95'] for n in names])
            ax.errorbar(vals, yy + offset, xerr=np.array([np.array(vals) - ci[:, 0], ci[:, 1] - np.array(vals)]), fmt='o', ms=4, capsize=2, color=COLORS[variant], label='Zero padded, clipped' if variant == 'legacy_zero_clipped' else 'Valid window, unclipped')
        ax.set_yticks(yy, [LABELS[n] for n in names])
        ax.invert_yaxis()
        ax.set_xlabel('Mean RGB SSIM')
        ax.set_title(title, fontsize=10)
        ax.grid(axis='x', alpha=0.2)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.tight_layout(w_pad=2.4)
    save(fig, 'ssim_padding_sensitivity')

def resource_plot(resources, metric, suffix):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3), sharex=True)
    for ax, (snr, data) in zip(axes, resources.items()):
        profiles = list(data['profiles'].values())
        x = np.arange(len(profiles))
        for who, color, marker, label in [('reference', '#666666', 's', 'DCT utility reference'), ('proposed', '#0072B2', 'o', 'Keyed payload + coded header')]:
            y = np.array([p[who][metric]['mean'] for p in profiles])
            ci = np.array([p[who][metric]['ci95'] for p in profiles])
            ax.errorbar(x, y, yerr=np.array([y - ci[:, 0], ci[:, 1] - y]), fmt=marker + '-', color=color, ms=4, lw=1.2, capsize=3, label=label)
        ax.set_xticks(x, [str(p['group_profile']) for p in profiles])
        ax.set_xlabel('Payload packets per header (B)')
        ax.set_title(f'Anchor SNR {snr} dB', fontsize=10)
        ax.grid(axis='y', alpha=0.2)
        for xx, p in zip(x, profiles):
            ax.annotate(f"{p['detected_failed_groups']}/{p['groups']} failed groups", (xx, p['proposed'][metric]['mean']), xytext=(7, 10) if xx == 0 else (0, -18), textcoords='offset points', ha='left' if xx == 0 else 'center', va='bottom' if xx == 0 else 'top', fontsize=6.7)
    axes[0].set_ylabel('Mean zero-padded RGB SSIM' if metric == 'legacy_ssim' else 'Mean valid-window RGB SSIM')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, 'equal_resource_utility' + suffix)

def header_plot(header):
    rows = list(header['monte_carlo'].values())
    fig, ax = plt.subplots(figsize=(5.3, 3.1))
    for row in rows:
        x = row['header_real_symbol_snr_db']
        p = row['decoded_header_error_rate']
        if p:
            ci = row['decoded_header_error_wilson95']
            ax.errorbar([x], [p], yerr=[[p - ci[0]], [ci[1] - p]], fmt='o', color='#0072B2', capsize=3)
        else:
            upper = row['zero_error_one_sided95_upper']
            ax.plot(x, upper, 'v', color='#0072B2', ms=6)
            ax.annotate('0 / 10,000', (x, upper), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
    ax.set_yscale('log')
    ax.set_ylim(0.0001, 1.4)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xlabel('Coded-header real-symbol SNR (dB)')
    ax.set_ylabel('Decoded header error probability')
    ax.grid(which='both', alpha=0.15)
    ax.text(0.03, 0.05, 'Bars: pointwise 95% Wilson intervals\nTriangles: one-sided 95% upper limit after zero errors', transform=ax.transAxes, fontsize=7.5, ha='left', va='bottom')
    fig.tight_layout()
    save(fig, 'coded_header_reliability')

def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    cifar = read(RESULTS / 'ssim_cifar/ssim_sensitivity.json')
    kodak = read(RESULTS / 'ssim_kodak/ssim_sensitivity.json')
    resources = {snr: read(RESULTS / f'equal_resource_{snr}dB/equal_resource_utility.json') for snr in [12, 6]}
    header = read(RESULTS / 'header_resource/header_reliability.json')
    kodak_plot(kodak)
    ssim_plot(cifar, kodak)
    resource_plot(resources, 'legacy_ssim', '')
    resource_plot(resources, 'valid_ssim', '_valid')
    header_plot(header)
if __name__ == '__main__':
    main()
