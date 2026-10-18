import os
from typing import Optional, List, Dict, Any
import torch
PLOTS_AVAILABLE = True
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception:
    PLOTS_AVAILABLE = False
from config import Cfg

def _plot_if_available(func):

    def wrapper(*args, **kwargs):
        if not PLOTS_AVAILABLE:
            print('[plots] matplotlib not available; skipping plots.')
            return None
        return func(*args, **kwargs)
    return wrapper

def _safe_list(x, default=0.0):
    try:
        return list(x)
    except Exception:
        return [default]

def _as_float_list(history: List[Dict[str, Any]], key: str, default: float=0.0) -> List[float]:
    out = []
    for r in history:
        v = r.get(key, default)
        try:
            out.append(float(v))
        except Exception:
            out.append(float(default))
    return out

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _save_fig(cfg: Cfg, fig, name_base: str, pdf: Optional[PdfPages]=None):
    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight')
    if cfg.SAVE_PLOTS_PNG:
        out = os.path.join(cfg.OUT_DIR, f'{name_base}.png')
        fig.savefig(out, dpi=cfg.PLOTS_DPI, bbox_inches='tight')

def _norm_confusion(cm: List[List[int]]) -> torch.Tensor:
    t = torch.tensor(cm, dtype=torch.float32)
    row_sum = t.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return t / row_sum

def _cdf(values: List[float]):
    v = torch.tensor(values, dtype=torch.float32)
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return ([0.0], [0.0])
    v_sorted, _ = torch.sort(v)
    y = torch.linspace(0.0, 1.0, steps=v_sorted.numel())
    return (v_sorted.tolist(), y.tolist())

@_plot_if_available
def make_plots(cfg: Cfg, history: List[Dict[str, Any]], snr_rows: List[Dict[str, Any]], br_metrics: Optional[Dict[str, Any]]=None, eve_snr_rows: Optional[List[Dict[str, Any]]]=None, joint_entries: Optional[List[Dict[str, Any]]]=None, gap_entries: Optional[List[Dict[str, Any]]]=None, final_eval: Optional[Dict[str, Any]]=None):
    _ensure_dir(cfg.OUT_DIR)
    pdf_path = os.path.join(cfg.OUT_DIR, 'plots_report.pdf')
    pdf = PdfPages(pdf_path) if cfg.SAVE_PLOTS_PDF else None
    epochs = _as_float_list(history, 'epoch', 0.0)
    loss_main = _as_float_list(history, 'loss_main', 0.0)
    loss_bob = _as_float_list(history, 'loss_bob', 0.0)
    loss_def = _as_float_list(history, 'loss_def', 0.0)
    loss_eve = _as_float_list(history, 'loss_eve', 0.0)
    loss_int = _as_float_list(history, 'loss_int', 0.0)
    bob_ssim = _as_float_list(history, 'ssim_bob', 0.0)
    bob_psnr = _as_float_list(history, 'psnr_bob', 0.0)
    eve_ssim = _as_float_list(history, 'ssim_eve_best', 0.0)
    eve_psnr = _as_float_list(history, 'psnr_eve_best', 0.0)
    int_acc = _as_float_list(history, 'intercept_acc', 0.0)
    wk_ssim = _as_float_list(history, 'ssim_wrongkey', 0.0) if cfg.DO_WRONG_KEY_EVAL else None
    wk_psnr = _as_float_list(history, 'psnr_wrongkey', 0.0) if cfg.DO_WRONG_KEY_EVAL else None
    fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
    plt.plot(epochs, loss_main, label='Main loss')
    plt.plot(epochs, loss_bob, label='Bob loss')
    plt.plot(epochs, loss_def, label='Defense KL loss')
    plt.plot(epochs, loss_eve, label='Eve recon loss')
    plt.plot(epochs, loss_int, label='Intercept loss')
    plt.title('Training losses over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_fig(cfg, fig, 'train_losses_over_epochs', pdf)
    plt.close(fig)
    fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
    plt.plot(epochs, bob_ssim, label='Bob SSIM')
    plt.plot(epochs, eve_ssim, label='Eve best SSIM')
    if wk_ssim is not None:
        plt.plot(epochs, wk_ssim, label='Wrong key SSIM')
    plt.title('Structural similarity over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_fig(cfg, fig, 'train_ssim_over_epochs', pdf)
    plt.close(fig)
    fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
    plt.plot(epochs, bob_psnr, label='Bob PSNR')
    plt.plot(epochs, eve_psnr, label='Eve best PSNR')
    if wk_psnr is not None:
        plt.plot(epochs, wk_psnr, label='Wrong key PSNR')
    plt.title('Peak signal to noise ratio over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_fig(cfg, fig, 'train_psnr_over_epochs', pdf)
    plt.close(fig)
    fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
    plt.plot(epochs, int_acc, label='Intercept accuracy')
    plt.axhline(0.1, linestyle='--', linewidth=1.0, label='Chance level (10 classes)')
    plt.title('Intercept accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_fig(cfg, fig, 'train_intercept_accuracy_over_epochs', pdf)
    plt.close(fig)
    fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
    plt.scatter(bob_ssim, eve_ssim, s=18)
    plt.title('Tradeoff: Bob SSIM versus Eve best SSIM across epochs')
    plt.xlabel('Bob SSIM')
    plt.ylabel('Eve best SSIM')
    plt.grid(True, alpha=0.3)
    _save_fig(cfg, fig, 'tradeoff_bob_ssim_vs_eve_ssim_epochs', pdf)
    plt.close(fig)
    fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
    plt.scatter(bob_ssim, int_acc, s=18)
    plt.title('Tradeoff: Bob SSIM versus intercept accuracy across epochs')
    plt.xlabel('Bob SSIM')
    plt.ylabel('Intercept accuracy')
    plt.grid(True, alpha=0.3)
    _save_fig(cfg, fig, 'tradeoff_bob_ssim_vs_intercept_accuracy_epochs', pdf)
    plt.close(fig)
    fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
    gap = [bs - es for bs, es in zip(bob_ssim, eve_ssim)]
    plt.plot(epochs, gap, label='SSIM gap (Bob minus Eve best)')
    plt.title('Security gap over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM gap')
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_fig(cfg, fig, 'security_gap_over_epochs', pdf)
    plt.close(fig)
    snr_x = [float(r['snr_bob_db']) for r in snr_rows]
    snr_bob_ssim = [float(r['ssim_bob']) for r in snr_rows]
    snr_bob_psnr = [float(r['psnr_bob']) for r in snr_rows]
    snr_eve_ssim = [float(r['ssim_eve_best']) for r in snr_rows]
    snr_eve_psnr = [float(r['psnr_eve_best']) for r in snr_rows]
    snr_int_acc = [float(r['intercept_acc']) for r in snr_rows]
    snr_wk_ssim = [float(r.get('ssim_wrongkey', 0.0)) for r in snr_rows] if cfg.DO_WRONG_KEY_EVAL else None
    snr_wk_psnr = [float(r.get('psnr_wrongkey', 0.0)) for r in snr_rows] if cfg.DO_WRONG_KEY_EVAL else None
    fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
    plt.plot(snr_x, snr_bob_ssim, label='Bob SSIM')
    plt.plot(snr_x, snr_eve_ssim, label='Eve best SSIM')
    if snr_wk_ssim is not None:
        plt.plot(snr_x, snr_wk_ssim, label='Wrong key SSIM')
    plt.title(f'SNR sweep: structural similarity versus Bob SNR (Eve SNR fixed at {cfg.SNR_EVE_DB:g} dB)')
    plt.xlabel('Bob SNR (dB)')
    plt.ylabel('SSIM')
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_fig(cfg, fig, 'snr_sweep_ssim_vs_bob_snr', pdf)
    plt.close(fig)
    fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
    plt.plot(snr_x, snr_bob_psnr, label='Bob PSNR')
    plt.plot(snr_x, snr_eve_psnr, label='Eve best PSNR')
    if snr_wk_psnr is not None:
        plt.plot(snr_x, snr_wk_psnr, label='Wrong key PSNR')
    plt.title(f'SNR sweep: peak signal to noise ratio versus Bob SNR (Eve SNR fixed at {cfg.SNR_EVE_DB:g} dB)')
    plt.xlabel('Bob SNR (dB)')
    plt.ylabel('PSNR (dB)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_fig(cfg, fig, 'snr_sweep_psnr_vs_bob_snr', pdf)
    plt.close(fig)
    fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
    plt.plot(snr_x, snr_int_acc, label='Intercept accuracy')
    plt.axhline(0.1, linestyle='--', linewidth=1.0, label='Chance level (10 classes)')
    plt.title(f'SNR sweep: intercept accuracy versus Bob SNR (Eve SNR fixed at {cfg.SNR_EVE_DB:g} dB)')
    plt.xlabel('Bob SNR (dB)')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_fig(cfg, fig, 'snr_sweep_intercept_accuracy_vs_bob_snr', pdf)
    plt.close(fig)
    fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
    ssimgap = [b - e for b, e in zip(snr_bob_ssim, snr_eve_ssim)]
    plt.plot(snr_x, ssimgap, label='SSIM gap (Bob minus Eve best)')
    plt.title('SNR sweep: security gap versus Bob SNR')
    plt.xlabel('Bob SNR (dB)')
    plt.ylabel('SSIM gap')
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_fig(cfg, fig, 'snr_sweep_security_gap_vs_bob_snr', pdf)
    plt.close(fig)
    if eve_snr_rows is not None:
        ex = [float(r['snr_eve_db']) for r in eve_snr_rows]
        ess = [float(r['ssim_eve_best']) for r in eve_snr_rows]
        eps = [float(r['psnr_eve_best']) for r in eve_snr_rows]
        ia = [float(r['intercept_acc']) for r in eve_snr_rows]
        fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
        plt.plot(ex, ess, label='Eve best SSIM')
        plt.title(f'Eve SNR sweep: Eve best SSIM versus Eve SNR (Bob SNR fixed at {cfg.BOB_SNR_FIXED_FOR_EVE_SWEEP:g} dB)')
        plt.xlabel('Eve SNR (dB)')
        plt.ylabel('Eve best SSIM')
        plt.grid(True, alpha=0.3)
        plt.legend()
        _save_fig(cfg, fig, 'eve_snr_sweep_eve_ssim_vs_eve_snr', pdf)
        plt.close(fig)
        fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
        plt.plot(ex, eps, label='Eve best PSNR')
        plt.title(f'Eve SNR sweep: Eve best PSNR versus Eve SNR (Bob SNR fixed at {cfg.BOB_SNR_FIXED_FOR_EVE_SWEEP:g} dB)')
        plt.xlabel('Eve SNR (dB)')
        plt.ylabel('Eve best PSNR (dB)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        _save_fig(cfg, fig, 'eve_snr_sweep_eve_psnr_vs_eve_snr', pdf)
        plt.close(fig)
        fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
        plt.plot(ex, ia, label='Intercept accuracy')
        plt.axhline(0.1, linestyle='--', linewidth=1.0, label='Chance level (10 classes)')
        plt.title(f'Eve SNR sweep: intercept accuracy versus Eve SNR (Bob SNR fixed at {cfg.BOB_SNR_FIXED_FOR_EVE_SWEEP:g} dB)')
        plt.xlabel('Eve SNR (dB)')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        _save_fig(cfg, fig, 'eve_snr_sweep_intercept_accuracy_vs_eve_snr', pdf)
        plt.close(fig)
    if joint_entries is not None:
        fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
        for entry in joint_entries:
            eve_snr = entry['eve_snr_db']
            rows = entry['rows']
            x = [float(r['snr_bob_db']) for r in rows]
            y = [float(r['ssim_bob']) for r in rows]
            plt.plot(x, y, label=f'Eve SNR {eve_snr:g} dB')
        plt.title('Joint sweep: Bob SSIM versus Bob SNR for multiple fixed Eve SNR values')
        plt.xlabel('Bob SNR (dB)')
        plt.ylabel('Bob SSIM')
        plt.grid(True, alpha=0.3)
        plt.legend()
        _save_fig(cfg, fig, 'joint_sweep_bob_ssim_curves', pdf)
        plt.close(fig)
        fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
        for entry in joint_entries:
            eve_snr = entry['eve_snr_db']
            rows = entry['rows']
            x = [float(r['snr_bob_db']) for r in rows]
            y = [float(r['ssim_eve_best']) for r in rows]
            plt.plot(x, y, label=f'Eve SNR {eve_snr:g} dB')
        plt.title('Joint sweep: Eve best SSIM versus Bob SNR for multiple fixed Eve SNR values')
        plt.xlabel('Bob SNR (dB)')
        plt.ylabel('Eve best SSIM')
        plt.grid(True, alpha=0.3)
        plt.legend()
        _save_fig(cfg, fig, 'joint_sweep_eve_ssim_curves', pdf)
        plt.close(fig)
        fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
        for entry in joint_entries:
            eve_snr = entry['eve_snr_db']
            rows = entry['rows']
            x = [float(r['snr_bob_db']) for r in rows]
            y = [float(r['intercept_acc']) for r in rows]
            plt.plot(x, y, label=f'Eve SNR {eve_snr:g} dB')
        plt.axhline(0.1, linestyle='--', linewidth=1.0, label='Chance level (10 classes)')
        plt.title('Joint sweep: intercept accuracy versus Bob SNR for multiple fixed Eve SNR values')
        plt.xlabel('Bob SNR (dB)')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        _save_fig(cfg, fig, 'joint_sweep_intercept_accuracy_curves', pdf)
        plt.close(fig)
        if cfg.MAKE_HEATMAPS:
            bob_vals = list(cfg.SNR_SWEEP)
            eve_vals = [e['eve_snr_db'] for e in joint_entries]
            bob_ssim_grid = torch.zeros((len(eve_vals), len(bob_vals)))
            eve_ssim_grid = torch.zeros((len(eve_vals), len(bob_vals)))
            gap_grid = torch.zeros((len(eve_vals), len(bob_vals)))
            for yi, entry in enumerate(joint_entries):
                rows = entry['rows']
                for xi, r in enumerate(rows):
                    bob_ssim_grid[yi, xi] = float(r['ssim_bob'])
                    eve_ssim_grid[yi, xi] = float(r['ssim_eve_best'])
                    gap_grid[yi, xi] = float(r['ssim_bob']) - float(r['ssim_eve_best'])

            def heatmap(mat, title, name):
                fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
                plt.imshow(mat.numpy(), aspect='auto', origin='lower')
                plt.colorbar()
                plt.xticks(range(len(bob_vals)), [str(b) for b in bob_vals])
                plt.yticks(range(len(eve_vals)), [str(e) for e in eve_vals])
                plt.xlabel('Bob SNR (dB)')
                plt.ylabel('Eve SNR (dB)')
                plt.title(title)
                _save_fig(cfg, fig, name, pdf)
                plt.close(fig)
            heatmap(bob_ssim_grid, 'Joint sweep heatmap: Bob SSIM', 'joint_heatmap_bob_ssim')
            heatmap(eve_ssim_grid, 'Joint sweep heatmap: Eve best SSIM', 'joint_heatmap_eve_ssim')
            heatmap(gap_grid, 'Joint sweep heatmap: SSIM gap (Bob minus Eve best)', 'joint_heatmap_ssim_gap')
    if gap_entries is not None:
        fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
        for entry in gap_entries:
            delta = entry['delta_db']
            rows = entry['rows']
            x = [float(r['snr_bob_db']) for r in rows]
            y = [float(r['ssim_bob']) - float(r['ssim_eve_best']) for r in rows]
            plt.plot(x, y, label=f'Delta {delta:g} dB')
        plt.title('Gap sweep: SSIM gap (Bob minus Eve best) versus Bob SNR')
        plt.xlabel('Bob SNR (dB)')
        plt.ylabel('SSIM gap')
        plt.grid(True, alpha=0.3)
        plt.legend()
        _save_fig(cfg, fig, 'gap_sweep_ssim_gap_curves', pdf)
        plt.close(fig)
        fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
        for entry in gap_entries:
            delta = entry['delta_db']
            rows = entry['rows']
            x = [float(r['snr_bob_db']) for r in rows]
            y = [float(r['intercept_acc']) for r in rows]
            plt.plot(x, y, label=f'Delta {delta:g} dB')
        plt.axhline(0.1, linestyle='--', linewidth=1.0, label='Chance level (10 classes)')
        plt.title('Gap sweep: intercept accuracy versus Bob SNR')
        plt.xlabel('Bob SNR (dB)')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        _save_fig(cfg, fig, 'gap_sweep_intercept_accuracy_curves', pdf)
        plt.close(fig)
    dist_source = None
    source_name = None
    if br_metrics is not None and isinstance(br_metrics, dict) and ('distributions' in br_metrics):
        dist_source = br_metrics
        source_name = 'best response evaluation'
    elif final_eval is not None and isinstance(final_eval, dict) and ('distributions' in final_eval):
        dist_source = final_eval
        source_name = 'final evaluation'
    if dist_source is not None:
        d = dist_source['distributions']
        bob_ssim_vals = _safe_list(d.get('bob_ssim', []), 0.0)
        int_ent_vals = _safe_list(d.get('int_entropy', []), 0.0)
        int_conf_vals = _safe_list(d.get('int_confidence', []), 0.0)
        pow_vals = _safe_list(d.get('cipher_power', []), 0.0)
        eve_best_name = dist_source.get('eve_best_name', None)
        eve_best_ssim_vals = []
        if eve_best_name is not None:
            eve_best_ssim_vals = _safe_list(d.get(f'eve_{eve_best_name}_ssim', []), 0.0)
        wk_ssim_vals = _safe_list(d.get('wk_ssim', []), 0.0) if cfg.DO_WRONG_KEY_EVAL else []

        def hist_plot(values, title, xlabel, name, bins=40):
            fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
            v = torch.tensor(values, dtype=torch.float32)
            v = v[torch.isfinite(v)]
            if v.numel() > 0:
                plt.hist(v.numpy(), bins=bins)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            _save_fig(cfg, fig, name, pdf)
            plt.close(fig)
        hist_plot(bob_ssim_vals, f'Distribution of Bob SSIM ({source_name})', 'Bob SSIM', 'dist_bob_ssim')
        if len(eve_best_ssim_vals) > 0:
            hist_plot(eve_best_ssim_vals, f'Distribution of Eve best SSIM ({source_name})', 'Eve best SSIM', 'dist_eve_best_ssim')
        if cfg.DO_WRONG_KEY_EVAL and len(wk_ssim_vals) > 0:
            hist_plot(wk_ssim_vals, f'Distribution of wrong key SSIM ({source_name})', 'Wrong key SSIM', 'dist_wrong_key_ssim')
        hist_plot(int_ent_vals, f'Distribution of intercept entropy ({source_name})', 'Entropy (nats)', 'dist_intercept_entropy')
        hist_plot(int_conf_vals, f'Distribution of intercept confidence ({source_name})', 'Max class probability', 'dist_intercept_confidence')
        hist_plot(pow_vals, f'Distribution of ciphertext power after normalization ({source_name})', 'Ciphertext power', 'dist_cipher_power')

        def cdf_plot(values_list, labels, title, xlabel, name):
            fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
            for values, lab in zip(values_list, labels):
                xs, ys = _cdf(values)
                plt.plot(xs, ys, label=lab)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel('CDF')
            plt.grid(True, alpha=0.3)
            plt.legend()
            _save_fig(cfg, fig, name, pdf)
            plt.close(fig)
        cdf_plot([bob_ssim_vals, eve_best_ssim_vals if len(eve_best_ssim_vals) else [0.0], wk_ssim_vals if len(wk_ssim_vals) else [0.0]], ['Bob SSIM', 'Eve best SSIM', 'Wrong key SSIM' if cfg.DO_WRONG_KEY_EVAL else 'Wrong key SSIM (missing)'], f'CDF of SSIM values ({source_name})', 'SSIM', 'cdf_ssim_bob_eve_wrongkey')
        cdf_plot([int_ent_vals, int_conf_vals], ['Intercept entropy', 'Intercept confidence'], f'CDF of intercept uncertainty metrics ({source_name})', 'Value', 'cdf_intercept_uncertainty')
    cm_src = None
    cm_title = None
    if final_eval is not None and final_eval.get('confusion_matrix', None) is not None:
        cm_src = final_eval['confusion_matrix']
        cm_title = 'Confusion matrix from final evaluation'
    elif br_metrics is not None and br_metrics.get('confusion_matrix', None) is not None:
        cm_src = br_metrics['confusion_matrix']
        cm_title = 'Confusion matrix from best response evaluation'
    if cm_src is not None:
        cm_t = torch.tensor(cm_src, dtype=torch.float32)
        cm_n = _norm_confusion(cm_src)
        fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
        plt.imshow(cm_t.numpy(), aspect='auto')
        plt.colorbar()
        plt.title(cm_title + ' (raw counts)')
        plt.xlabel('Predicted class')
        plt.ylabel('True class')
        _save_fig(cfg, fig, 'confusion_matrix_raw', pdf)
        plt.close(fig)
        fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
        plt.imshow(cm_n.numpy(), aspect='auto', vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.title(cm_title + ' (row-normalized)')
        plt.xlabel('Predicted class')
        plt.ylabel('True class')
        _save_fig(cfg, fig, 'confusion_matrix_normalized', pdf)
        plt.close(fig)
    fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
    plt.scatter(snr_bob_ssim, snr_eve_ssim, s=22)
    for i, snr in enumerate(snr_x):
        plt.text(snr_bob_ssim[i], snr_eve_ssim[i], f'{snr:g}', fontsize=8)
    plt.title('SNR sweep scatter: Bob SSIM versus Eve best SSIM (annotated by Bob SNR)')
    plt.xlabel('Bob SSIM')
    plt.ylabel('Eve best SSIM')
    plt.grid(True, alpha=0.3)
    _save_fig(cfg, fig, 'snr_sweep_scatter_bob_ssim_vs_eve_ssim', pdf)
    plt.close(fig)
    fig = plt.figure(figsize=(cfg.PLOTS_FIG_W, cfg.PLOTS_FIG_H))
    plt.scatter(snr_bob_ssim, snr_int_acc, s=22)
    for i, snr in enumerate(snr_x):
        plt.text(snr_bob_ssim[i], snr_int_acc[i], f'{snr:g}', fontsize=8)
    plt.title('SNR sweep scatter: Bob SSIM versus intercept accuracy (annotated by Bob SNR)')
    plt.xlabel('Bob SSIM')
    plt.ylabel('Intercept accuracy')
    plt.grid(True, alpha=0.3)
    _save_fig(cfg, fig, 'snr_sweep_scatter_bob_ssim_vs_intercept_accuracy', pdf)
    plt.close(fig)
    if pdf is not None:
        pdf.close()
        print('[plots] wrote:', pdf_path)
    print('[plots] wrote png figures into:', cfg.OUT_DIR)
