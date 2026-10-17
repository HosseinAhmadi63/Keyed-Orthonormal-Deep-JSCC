import json
import os
import torch
from config import Cfg
from utils import is_power_of_two
from data import build_data
from train_core import train_main
from eval_core import eval_once
from sweeps import snr_sweep, eve_snr_sweep_fixed_bob, joint_snr_sweep_curves, gap_delta_sweep
from plots_pack import make_plots
from models.attackers import EveRecon, EveReconStrong, EveReconNoNonce, EveBobStyleDecoder, InterceptClf

def attacker_best_response(cfg: Cfg, enc, dec, scrambler, chan, device: str, train_loader, test_loader, num_classes: int):
    print('\n[Best-response attackers] Finetuning Eve and intercept against frozen system...')
    enc.eval()
    dec.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    for p in dec.parameters():
        p.requires_grad_(False)
    eve_models = {'base': EveRecon(cfg.LATENT_N, nonce_emb_dim=32).to(device), 'strong': EveReconStrong(cfg.LATENT_N, nonce_emb_dim=64).to(device), 'no_nonce': EveReconNoNonce(cfg.LATENT_N).to(device), 'bob_style': EveBobStyleDecoder(cfg.LATENT_N).to(device)}
    int_br = InterceptClf(cfg.LATENT_N, num_classes=num_classes, nonce_emb_dim=32).to(device)
    opt_eves = {k: torch.optim.AdamW(m.parameters(), lr=cfg.ATTACKER_LR, weight_decay=cfg.WEIGHT_DECAY) for k, m in eve_models.items()}
    opt_int = torch.optim.AdamW(int_br.parameters(), lr=cfg.ATTACKER_LR, weight_decay=cfg.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.AMP and device == 'cuda')

    def _scaler_step(opt, loss, params, clip):
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            scaler.update()
            return False
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=clip)
        scaler.step(opt)
        opt.zero_grad(set_to_none=True)
        scaler.update()
        return True
    for ep in range(1, cfg.ATTACKER_EPOCHS + 1):
        for m in eve_models.values():
            m.train()
        int_br.train()
        eve_losses = {k: 0.0 for k in eve_models}
        int_loss_sum = 0.0
        steps = 0
        for x, y, nonce_i64 in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            nonce_i64 = nonce_i64.to(device, non_blocking=True)
            with torch.no_grad():
                z = enc(x)
                z_scr = scrambler.scramble(z, nonce_i64)
                from scrambler import power_normalize_per_sample
                z_scr = power_normalize_per_sample(z_scr, cfg.TARGET_CIPHER_POWER)
                y_eve = chan(z_scr, cfg.SNR_EVE_DB)
            with torch.amp.autocast('cuda', enabled=cfg.AMP and device == 'cuda'):
                logits = int_br(y_eve, nonce_i64)
                li = torch.nn.functional.cross_entropy(logits, y)
            _scaler_step(opt_int, li, list(int_br.parameters()), cfg.GRAD_CLIP_NORM)
            for name, m in eve_models.items():
                opt = opt_eves[name]
                with torch.amp.autocast('cuda', enabled=cfg.AMP and device == 'cuda'):
                    if isinstance(m, (EveReconNoNonce, EveBobStyleDecoder)):
                        xe = m(y_eve)
                    else:
                        xe = m(y_eve, nonce_i64)
                    le = torch.nn.functional.mse_loss(xe.float(), x.float())
                _scaler_step(opt, le, list(m.parameters()), cfg.GRAD_CLIP_NORM)
                eve_losses[name] += float(le.detach().cpu())
            int_loss_sum += float(li.detach().cpu())
            steps += 1
        msg = f'  [BR Ep {ep:02d}/{cfg.ATTACKER_EPOCHS}] | Intercept loss={int_loss_sum / max(1, steps):.4f}'
        for k in eve_models:
            msg += f' | Eve ({k}) MSE={eve_losses[k] / max(1, steps):.4f}'
        print(msg)
    metrics = eval_once(cfg, enc, dec, scrambler, chan, eve_models, int_br, test_loader, device, collect_distributions=True)
    torch.save({'cfg': cfg.__dict__, 'int_br': int_br.state_dict(), 'eve_models': {k: m.state_dict() for k, m in eve_models.items()}, 'metrics': metrics}, os.path.join(cfg.OUT_DIR, 'best_response_attackers.pt'))
    return metrics

def main():
    cfg = Cfg()
    assert is_power_of_two(cfg.LATENT_N), 'LATENT_N must be power of 2.'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | AMP={cfg.AMP and device == 'cuda'}")
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    enc, dec, scrambler, chan, int_clf, eve_train, history, best = train_main(cfg, device)
    ckpt_path = os.path.join(cfg.OUT_DIR, 'best_main.pt')
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        enc.load_state_dict(ckpt['enc'])
        dec.load_state_dict(ckpt['dec'])
        int_clf.load_state_dict(ckpt['int_clf'])
        eve_train.load_state_dict(ckpt['eve_train'])
        print('\nBest (main):', ckpt['best'])
    train_loader, test_loader, num_classes = build_data(cfg)
    br_metrics = None
    if cfg.RUN_ATTACKER_BEST_RESPONSE:
        br_metrics = attacker_best_response(cfg, enc, dec, scrambler, chan, device, train_loader, test_loader, num_classes)
        with open(os.path.join(cfg.OUT_DIR, 'best_response_metrics.json'), 'w') as f:
            json.dump(br_metrics, f, indent=2)
    eve_eval = {'base': EveRecon(cfg.LATENT_N, nonce_emb_dim=32).to(device), 'strong': EveReconStrong(cfg.LATENT_N, nonce_emb_dim=64).to(device), 'no_nonce': EveReconNoNonce(cfg.LATENT_N).to(device), 'bob_style': EveBobStyleDecoder(cfg.LATENT_N).to(device)}
    eve_eval['base'].load_state_dict(eve_train.state_dict())
    if cfg.RUN_ATTACKER_BEST_RESPONSE:
        br = torch.load(os.path.join(cfg.OUT_DIR, 'best_response_attackers.pt'), map_location=device)
        for k in eve_eval:
            if k in br['eve_models']:
                eve_eval[k].load_state_dict(br['eve_models'][k])
        int_sweep = InterceptClf(cfg.LATENT_N, num_classes=num_classes, nonce_emb_dim=32).to(device)
        int_sweep.load_state_dict(br['int_br'])
    else:
        int_sweep = int_clf
    print('\nRunning SNR sweep (Bob swept, Eve fixed)...')
    snr_rows = snr_sweep(cfg, enc, dec, scrambler, chan, eve_eval, int_sweep, test_loader, device)
    with open(os.path.join(cfg.OUT_DIR, 'snr_sweep.json'), 'w') as f:
        json.dump(snr_rows, f, indent=2)
    print('\nRunning Eve SNR sweep (Eve swept, Bob fixed)...')
    eve_snr_rows = eve_snr_sweep_fixed_bob(cfg, enc, dec, scrambler, chan, eve_eval, int_sweep, test_loader, device)
    with open(os.path.join(cfg.OUT_DIR, 'eve_snr_sweep.json'), 'w') as f:
        json.dump(eve_snr_rows, f, indent=2)
    print('\nRunning joint sweep curves (Bob swept, multiple fixed Eve SNR values)...')
    joint_entries = joint_snr_sweep_curves(cfg, enc, dec, scrambler, chan, eve_eval, int_sweep, test_loader, device)
    with open(os.path.join(cfg.OUT_DIR, 'joint_snr_sweep_curves.json'), 'w') as f:
        json.dump(joint_entries, f, indent=2)
    print('\nRunning gap sweep (Eve SNR equals Bob SNR plus delta)...')
    gap_entries = gap_delta_sweep(cfg, enc, dec, scrambler, chan, eve_eval, int_sweep, test_loader, device)
    with open(os.path.join(cfg.OUT_DIR, 'gap_delta_sweep.json'), 'w') as f:
        json.dump(gap_entries, f, indent=2)
    final_eval = eval_once(cfg, enc, dec, scrambler, chan, eve_eval, int_sweep, test_loader, device, collect_distributions=True, save_samples=True)
    with open(os.path.join(cfg.OUT_DIR, 'final_eval.json'), 'w') as f:
        json.dump(final_eval, f, indent=2)
    print('\nFinal eval summary:', {k: final_eval.get(k) for k in ['ssim_bob', 'psnr_bob', 'ssim_eve_best', 'psnr_eve_best', 'intercept_acc', 'eve_best_name', 'ssim_wrongkey', 'psnr_wrongkey', 'intercept_entropy_mean', 'intercept_confidence_mean', 'cipher_power_mean', 'paper_sanity']})
    make_plots(cfg, history, snr_rows, br_metrics=br_metrics, eve_snr_rows=eve_snr_rows, joint_entries=joint_entries, gap_entries=gap_entries, final_eval=final_eval)
    print('\nDone. Outputs written to:', cfg.OUT_DIR)
if __name__ == '__main__':
    main()
