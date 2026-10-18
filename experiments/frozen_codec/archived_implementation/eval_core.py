import os
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
except Exception as e:
    raise ImportError('torchvision is required for this project.') from e
PLOTS_AVAILABLE = True
try:
    import matplotlib.pyplot as plt
except Exception:
    PLOTS_AVAILABLE = False
from config import Cfg
from utils import make_image_grid, psnr_from_mse, _randperm_no_fixed_points, _summary_stats
from ssim_ms import ssim_per_image
from scrambler import power_normalize_per_sample, per_sample_power
from models.attackers import EveReconNoNonce, EveBobStyleDecoder, EveRecon, EveReconStrong, InterceptClf

def _save_labeled_grid_png(grid_tensor: torch.Tensor, row_labels: List[str], n: int, out_path: str):
    if not PLOTS_AVAILABLE:
        return
    img = grid_tensor.detach().cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(min(18, 1.1 * n), 6.5))
    plt.imshow(img)
    plt.axis('off')
    H = img.shape[0]
    num_rows = len(row_labels)
    for i, lab in enumerate(row_labels):
        y = (i + 0.5) * (H / num_rows)
        plt.text(5, y, lab, fontsize=12, va='center', bbox=dict(facecolor='white', alpha=0.65, edgecolor='none'))
    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, dpi=200)
    plt.close()

@torch.no_grad()
def eval_once(cfg: Cfg, enc, dec, scrambler, chan, eve_models: Dict[str, nn.Module], int_clf: nn.Module, loader, device, snr_bob_db: Optional[float]=None, snr_eve_db: Optional[float]=None, collect_distributions: bool=False, save_samples: Optional[bool]=None) -> Dict[str, Any]:
    enc.eval()
    dec.eval()
    int_clf.eval()
    for m in eve_models.values():
        m.eval()
    snr_bob = cfg.SNR_BOB_DB if snr_bob_db is None else float(snr_bob_db)
    snr_eve = cfg.SNR_EVE_DB if snr_eve_db is None else float(snr_eve_db)
    do_save = cfg.SAVE_SAMPLE_GRIDS if save_samples is None else bool(save_samples)
    mse_bob_sum = 0.0
    ssim_bob_sum = 0.0
    eve_mse_sum = {k: 0.0 for k in eve_models}
    eve_ssim_sum = {k: 0.0 for k in eve_models}
    int_correct = 0
    int_total = 0
    cm = None
    mse_wk_sum = 0.0
    ssim_wk_sum = 0.0
    avg_entropy_sum = 0.0
    avg_conf_sum = 0.0
    cipher_power_sum = 0.0
    do_sanity = bool(cfg.RUN_PAPER_SANITY_CHECKS)
    do_shuffle = do_sanity and bool(cfg.SANITY_SHUFFLE_TEST)
    do_unscr = do_sanity and bool(cfg.SANITY_UNSCRAMBLED_BASELINE)
    eve_ssim_shuf_sum = {k: 0.0 for k in eve_models}
    eve_gap_sum = {k: 0.0 for k in eve_models}
    shuffle_batches_count = 0
    unscr_bob_mse_sum = 0.0
    unscr_bob_ssim_sum = 0.0
    unscr_eve_mse_sum = 0.0
    unscr_eve_ssim_sum = 0.0
    unscr_batches_count = 0
    dist = {'bob_mse': [], 'bob_ssim': [], 'wk_mse': [], 'wk_ssim': [], 'int_pred': [], 'int_true': [], 'int_entropy': [], 'int_confidence': [], 'cipher_power': [], 'unscr_bob_mse': [], 'unscr_bob_ssim': [], 'unscr_eve_mse': [], 'unscr_eve_ssim': []}
    for k in eve_models:
        dist[f'eve_{k}_mse'] = []
        dist[f'eve_{k}_ssim'] = []
        dist[f'eve_{k}_ssim_shuf'] = []
        dist[f'eve_{k}_ssim_gap'] = []
    n_batches = 0
    sample_saved = False
    for x, y, nonce_i64 in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        nonce_i64 = nonce_i64.to(device, non_blocking=True)
        z = enc(x)
        z_scr = scrambler.scramble(z, nonce_i64)
        z_scr = power_normalize_per_sample(z_scr, cfg.TARGET_CIPHER_POWER)
        cp = per_sample_power(z_scr)
        cipher_power_sum += cp.mean().item()
        y_eve = chan(z_scr, snr_eve)
        y_bob = chan(z_scr, snr_bob)
        z_hat = scrambler.descramble(y_bob, nonce_i64)
        x_hat = dec(z_hat)
        eve_out: Dict[str, torch.Tensor] = {}
        for name, m in eve_models.items():
            if isinstance(m, (EveReconNoNonce, EveBobStyleDecoder)):
                eve_out[name] = m(y_eve)
            else:
                eve_out[name] = m(y_eve, nonce_i64)
        logits = int_clf(y_eve, nonce_i64)
        pred = logits.argmax(dim=-1)
        p = F.softmax(logits.float(), dim=-1).clamp_min(1e-12)
        ent = (-p * torch.log(p)).sum(dim=-1)
        conf = p.max(dim=-1).values
        avg_entropy_sum += ent.mean().item()
        avg_conf_sum += conf.mean().item()
        int_correct += (pred == y).sum().item()
        int_total += y.numel()
        if cm is None:
            num_classes = logits.shape[-1]
            cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        for t, p_ in zip(y.view(-1).cpu(), pred.view(-1).cpu()):
            cm[t.long(), p_.long()] += 1
        bob_mse = F.mse_loss(x_hat.float(), x.float(), reduction='none').mean(dim=(1, 2, 3))
        bob_ssim = ssim_per_image(x_hat, x)
        mse_bob_sum += bob_mse.mean().item()
        ssim_bob_sum += bob_ssim.mean().item()
        for name in eve_models:
            xe = eve_out[name]
            emse = F.mse_loss(xe.float(), x.float(), reduction='none').mean(dim=(1, 2, 3))
            essim = ssim_per_image(xe, x)
            eve_mse_sum[name] += emse.mean().item()
            eve_ssim_sum[name] += essim.mean().item()
        if do_shuffle:
            B = x.shape[0]
            perm = _randperm_no_fixed_points(B, x.device)
            if perm is not None:
                x_shuf = x[perm]
                for name in eve_models:
                    xe = eve_out[name]
                    essim_shuf = ssim_per_image(xe, x_shuf)
                    essim_true = ssim_per_image(xe, x)
                    gap = essim_true - essim_shuf
                    eve_ssim_shuf_sum[name] += essim_shuf.mean().item()
                    eve_gap_sum[name] += gap.mean().item()
                    if collect_distributions:
                        dist[f'eve_{name}_ssim_shuf'].append(essim_shuf.detach().cpu())
                        dist[f'eve_{name}_ssim_gap'].append(gap.detach().cpu())
                shuffle_batches_count += 1
        if do_unscr:
            z_plain = power_normalize_per_sample(z, cfg.TARGET_CIPHER_POWER)
            y_eve_plain = chan(z_plain, snr_eve)
            y_bob_plain = chan(z_plain, snr_bob)
            x_bob_plain = dec(y_bob_plain)
            x_eve_plain = dec(y_eve_plain)
            ub_mse = F.mse_loss(x_bob_plain.float(), x.float(), reduction='none').mean(dim=(1, 2, 3))
            ub_ssim = ssim_per_image(x_bob_plain, x)
            ue_mse = F.mse_loss(x_eve_plain.float(), x.float(), reduction='none').mean(dim=(1, 2, 3))
            ue_ssim = ssim_per_image(x_eve_plain, x)
            unscr_bob_mse_sum += ub_mse.mean().item()
            unscr_bob_ssim_sum += ub_ssim.mean().item()
            unscr_eve_mse_sum += ue_mse.mean().item()
            unscr_eve_ssim_sum += ue_ssim.mean().item()
            unscr_batches_count += 1
            if collect_distributions:
                dist['unscr_bob_mse'].append(ub_mse.detach().cpu())
                dist['unscr_bob_ssim'].append(ub_ssim.detach().cpu())
                dist['unscr_eve_mse'].append(ue_mse.detach().cpu())
                dist['unscr_eve_ssim'].append(ue_ssim.detach().cpu())
        if cfg.DO_WRONG_KEY_EVAL:
            z_wk = scrambler.descramble(y_bob, nonce_i64, key_override_pyint=cfg.WRONG_KEY_U64)
            x_wk = dec(z_wk)
            wk_mse = F.mse_loss(x_wk.float(), x.float(), reduction='none').mean(dim=(1, 2, 3))
            wk_ssim = ssim_per_image(x_wk, x)
            mse_wk_sum += wk_mse.mean().item()
            ssim_wk_sum += wk_ssim.mean().item()
        else:
            x_wk = None
        if do_save and (not sample_saved):
            os.makedirs(cfg.OUT_DIR, exist_ok=True)
            n = min(cfg.SAMPLE_GRID_N, x.shape[0])
            scores = {}
            for k in eve_models.keys():
                s = ssim_per_image(eve_out[k][:n], x[:n]).mean()
                if not torch.isfinite(s):
                    s = torch.tensor(-1.0, device=s.device)
                scores[k] = float(s.item())
            best_eve_name_batch = max(scores.keys(), key=lambda k: scores[k])
            x_eve_best = eve_out[best_eve_name_batch]
            if x_wk is None:
                grid = make_image_grid(torch.cat([x[:n], x_hat[:n], x_eve_best[:n]], dim=0), nrow=n)
                name = f'samples_gt_bob_evebest_{best_eve_name_batch}.png'
            else:
                grid = make_image_grid(torch.cat([x[:n], x_hat[:n], x_eve_best[:n], x_wk[:n]], dim=0), nrow=n)
                name = f'samples_gt_bob_evebest_{best_eve_name_batch}_wrongkey.png'
            torchvision.utils.save_image(grid, os.path.join(cfg.OUT_DIR, name))
            want = ['base', 'strong', 'no_nonce', 'bob_style']
            has_all = all((k in eve_out for k in want))
            if has_all:
                rows = [x[:n], x_hat[:n], eve_out['base'][:n], eve_out['strong'][:n], eve_out['no_nonce'][:n], eve_out['bob_style'][:n]]
                labels = ['Ground truth', 'Bob', 'Eve base', 'Eve strong', 'Eve no nonce', 'Eve Bob Style']
                if x_wk is not None:
                    rows.append(x_wk[:n])
                    labels.append('Wrong key')
                grid2 = make_image_grid(torch.cat(rows, dim=0), nrow=n)
                fn_base = f'samples_compare_eves_B{snr_bob:g}_E{snr_eve:g}.png'
                out_png = os.path.join(cfg.OUT_DIR, fn_base)
                torchvision.utils.save_image(grid2, out_png)
                if PLOTS_AVAILABLE:
                    fn_lab = f'samples_compare_eves_B{snr_bob:g}_E{snr_eve:g}_labeled.png'
                    out_lab = os.path.join(cfg.OUT_DIR, fn_lab)
                    _save_labeled_grid_png(grid2, labels, n, out_lab)
            sample_saved = True
        if collect_distributions:
            dist['bob_mse'].append(bob_mse.detach().cpu())
            dist['bob_ssim'].append(bob_ssim.detach().cpu())
            dist['int_entropy'].append(ent.detach().cpu())
            dist['int_confidence'].append(conf.detach().cpu())
            dist['cipher_power'].append(cp.detach().cpu())
            if cfg.DO_WRONG_KEY_EVAL:
                dist['wk_mse'].append(wk_mse.detach().cpu())
                dist['wk_ssim'].append(wk_ssim.detach().cpu())
            dist['int_pred'].append(pred.detach().cpu())
            dist['int_true'].append(y.detach().cpu())
            for name in eve_models:
                xe = eve_out[name]
                emse = F.mse_loss(xe.float(), x.float(), reduction='none').mean(dim=(1, 2, 3))
                essim = ssim_per_image(xe, x)
                dist[f'eve_{name}_mse'].append(emse.detach().cpu())
                dist[f'eve_{name}_ssim'].append(essim.detach().cpu())
        n_batches += 1
        if n_batches >= cfg.EVAL_MAX_BATCHES:
            break
    div = max(1, n_batches)
    eve_mse = {k: eve_mse_sum[k] / div for k in eve_models}
    eve_ssim = {k: eve_ssim_sum[k] / div for k in eve_models}
    best_eve_name = max(eve_models.keys(), key=lambda k: eve_ssim[k])
    best_eve_mse = eve_mse[best_eve_name]
    best_eve_ssim = eve_ssim[best_eve_name]
    mse_bob = mse_bob_sum / div
    ssim_bob = ssim_bob_sum / div
    psnr_bob = psnr_from_mse(torch.tensor(mse_bob)).item()
    psnr_eve_best = psnr_from_mse(torch.tensor(best_eve_mse)).item()
    out = {'snr_bob_db': snr_bob, 'snr_eve_db': snr_eve, 'batches': n_batches, 'mse_bob': mse_bob, 'psnr_bob': psnr_bob, 'ssim_bob': ssim_bob, 'eve_best_name': best_eve_name, 'mse_eve_best': best_eve_mse, 'psnr_eve_best': psnr_eve_best, 'ssim_eve_best': best_eve_ssim, 'intercept_acc': float(int_correct) / max(1, int_total), 'confusion_matrix': cm.tolist() if cm is not None else None, 'eve_all': {k: {'mse': eve_mse[k], 'psnr': psnr_from_mse(torch.tensor(eve_mse[k])).item(), 'ssim': eve_ssim[k]} for k in eve_models}, 'intercept_entropy_mean': avg_entropy_sum / div, 'intercept_confidence_mean': avg_conf_sum / div, 'cipher_power_mean': cipher_power_sum / div}
    if cfg.DO_WRONG_KEY_EVAL:
        mse_wk = mse_wk_sum / div
        ssim_wk = ssim_wk_sum / div
        out.update({'mse_wrongkey': mse_wk, 'psnr_wrongkey': psnr_from_mse(torch.tensor(mse_wk)).item(), 'ssim_wrongkey': ssim_wk})
    if do_sanity:
        sanity = {}
        if do_shuffle and shuffle_batches_count > 0:
            sanity['shuffle_test_mean'] = {k: {'ssim_true_mean': float(eve_ssim[k]), 'ssim_shuf_mean': float(eve_ssim_shuf_sum[k] / max(1, shuffle_batches_count)), 'gap_mean': float(eve_gap_sum[k] / max(1, shuffle_batches_count))} for k in eve_models}
        if do_unscr and unscr_batches_count > 0:
            ub_mse = float(unscr_bob_mse_sum / max(1, unscr_batches_count))
            ub_ssim = float(unscr_bob_ssim_sum / max(1, unscr_batches_count))
            ue_mse = float(unscr_eve_mse_sum / max(1, unscr_batches_count))
            ue_ssim = float(unscr_eve_ssim_sum / max(1, unscr_batches_count))
            sanity['unscrambled_baseline_mean'] = {'bob': {'mse': ub_mse, 'psnr': float(psnr_from_mse(torch.tensor(ub_mse)).item()), 'ssim': ub_ssim}, 'eve_with_bob_decoder': {'mse': ue_mse, 'psnr': float(psnr_from_mse(torch.tensor(ue_mse)).item()), 'ssim': ue_ssim}}
        out['paper_sanity'] = sanity
    if collect_distributions:
        for k in list(dist.keys()):
            if len(dist[k]) > 0 and isinstance(dist[k][0], torch.Tensor):
                dist[k] = torch.cat(dist[k], dim=0).detach().cpu().numpy().tolist()
            elif hasattr(dist[k], 'tolist'):
                dist[k] = dist[k].tolist()
        out['distributions'] = dist
        if do_sanity and bool(cfg.SANITY_WORSTCASE_LEAKAGE):
            pctl = float(cfg.LEAKAGE_PCTL)
            wc = {}
            for name in eve_models:
                vals = dist.get(f'eve_{name}_ssim', [])
                wc[name] = _summary_stats(vals, pctl=pctl)
            best_vals = dist.get(f'eve_{best_eve_name}_ssim', [])
            wc['eve_best'] = _summary_stats(best_vals, pctl=pctl)
            if cfg.DO_WRONG_KEY_EVAL:
                wc['wrongkey'] = _summary_stats(dist.get('wk_ssim', []), pctl=pctl)
            if do_shuffle:
                wc_shuf = {}
                wc_gap = {}
                for name in eve_models:
                    wc_shuf[name] = _summary_stats(dist.get(f'eve_{name}_ssim_shuf', []), pctl=pctl)
                    wc_gap[name] = _summary_stats(dist.get(f'eve_{name}_ssim_gap', []), pctl=pctl)
                out.setdefault('paper_sanity', {})
                out['paper_sanity']['shuffle_test_worstcase'] = {'ssim_shuf': wc_shuf, 'gap_true_minus_shuf': wc_gap}
            out.setdefault('paper_sanity', {})
            out['paper_sanity']['worstcase_leakage_ssim'] = wc
            if do_unscr:
                out['paper_sanity']['unscrambled_baseline_worstcase'] = {'bob': _summary_stats(dist.get('unscr_bob_ssim', []), pctl=pctl), 'eve_with_bob_decoder': _summary_stats(dist.get('unscr_eve_ssim', []), pctl=pctl)}
    return out
