from typing import Dict, Any, List
import torch
import torch.nn as nn
from config import Cfg
from eval_core import eval_once

@torch.no_grad()
def snr_sweep(cfg: Cfg, enc, dec, scrambler, chan, eve_models: Dict[str, nn.Module], int_clf: nn.Module, test_loader, device) -> List[Dict[str, Any]]:
    rows = []
    for snr in cfg.SNR_SWEEP:
        r = eval_once(cfg, enc, dec, scrambler, chan, eve_models, int_clf, test_loader, device, snr_bob_db=float(snr), snr_eve_db=cfg.SNR_EVE_DB, collect_distributions=False, save_samples=False)
        rows.append(r)
        print(f"  [SNR sweep] Bob SNR={snr:>4.1f} | Bob SSIM and PSNR={r['ssim_bob']:.3f}/{r['psnr_bob']:.2f} | Eve best ({r['eve_best_name']}) SSIM and PSNR={r['ssim_eve_best']:.3f}/{r['psnr_eve_best']:.2f} | Intercept accuracy={r['intercept_acc']:.3f}")
    return rows

@torch.no_grad()
def eve_snr_sweep_fixed_bob(cfg: Cfg, enc, dec, scrambler, chan, eve_models: Dict[str, nn.Module], int_clf: nn.Module, test_loader, device) -> List[Dict[str, Any]]:
    rows = []
    bob_fixed = float(cfg.BOB_SNR_FIXED_FOR_EVE_SWEEP)
    for snr_e in cfg.EVE_SNR_SWEEP:
        r = eval_once(cfg, enc, dec, scrambler, chan, eve_models, int_clf, test_loader, device, snr_bob_db=bob_fixed, snr_eve_db=float(snr_e), collect_distributions=False, save_samples=False)
        rows.append(r)
        print(f"  [Eve SNR sweep] Eve SNR={snr_e:>4.1f} (Bob fixed {bob_fixed:.1f}) | Eve best ({r['eve_best_name']}) SSIM and PSNR={r['ssim_eve_best']:.3f}/{r['psnr_eve_best']:.2f} | Intercept accuracy={r['intercept_acc']:.3f}")
    return rows

@torch.no_grad()
def joint_snr_sweep_curves(cfg: Cfg, enc, dec, scrambler, chan, eve_models: Dict[str, nn.Module], int_clf: nn.Module, test_loader, device) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for eve_snr in cfg.JOINT_FIXED_EVE_SNRS:
        rows = []
        for bob_snr in cfg.SNR_SWEEP:
            r = eval_once(cfg, enc, dec, scrambler, chan, eve_models, int_clf, test_loader, device, snr_bob_db=float(bob_snr), snr_eve_db=float(eve_snr), collect_distributions=False, save_samples=False)
            rows.append(r)
        entries.append({'eve_snr_db': float(eve_snr), 'rows': rows})
        print(f'  [Joint sweep] fixed Eve SNR={eve_snr:.1f} done.')
    return entries

@torch.no_grad()
def gap_delta_sweep(cfg: Cfg, enc, dec, scrambler, chan, eve_models: Dict[str, nn.Module], int_clf: nn.Module, test_loader, device) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for delta in cfg.GAP_DELTAS_DB:
        rows = []
        for bob_snr in cfg.SNR_SWEEP:
            eve_snr = float(bob_snr) + float(delta)
            r = eval_once(cfg, enc, dec, scrambler, chan, eve_models, int_clf, test_loader, device, snr_bob_db=float(bob_snr), snr_eve_db=float(eve_snr), collect_distributions=False, save_samples=False)
            rows.append(r)
        entries.append({'delta_db': float(delta), 'rows': rows})
        print(f'  [Gap sweep] delta={delta:.1f} done.')
    return entries
