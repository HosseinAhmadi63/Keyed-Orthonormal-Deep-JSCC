import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List
import torch
import torch.nn.functional as F
from config import Cfg
from data import build_data
from utils import set_seed, is_power_of_two
from ssim_ms import ms_ssim_mean
from channel import AWGN
from scrambler import KeyedScrambler, power_normalize_per_sample
from models.autoencoder import Encoder, Decoder
from models.attackers import EveRecon, InterceptClf, EveReconStrong, EveReconNoNonce, EveBobStyleDecoder
from eval_core import eval_once

def _scaler_step(cfg: Cfg, scaler, opt, loss, params, clip: float):
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

def train_main(cfg: Cfg, device: str):
    assert is_power_of_two(cfg.LATENT_N), 'LATENT_N must be power of 2.'
    set_seed(cfg.SEED)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUT_DIR, 'cfg.json'), 'w') as f:
        json.dump(asdict(cfg), f, indent=2)
    train_loader, test_loader, num_classes = build_data(cfg)
    enc = Encoder(cfg.LATENT_N).to(device)
    dec = Decoder(cfg.LATENT_N).to(device)
    scrambler = KeyedScrambler(cfg.LATENT_N, cfg.KEY_U64, use_perm=cfg.USE_PER_SAMPLE_PERM).to(device)
    chan = AWGN().to(device)
    eve_train = EveRecon(cfg.LATENT_N, nonce_emb_dim=32).to(device)
    int_clf = InterceptClf(cfg.LATENT_N, num_classes=num_classes, nonce_emb_dim=32).to(device)
    opt_main = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=cfg.LR_MAIN, weight_decay=cfg.WEIGHT_DECAY)
    opt_eve = torch.optim.AdamW(eve_train.parameters(), lr=cfg.LR_EVE, weight_decay=cfg.WEIGHT_DECAY)
    opt_int = torch.optim.AdamW(int_clf.parameters(), lr=cfg.LR_INT, weight_decay=cfg.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.AMP and device == 'cuda')
    history: List[Dict[str, Any]] = []
    best = None
    for ep in range(1, cfg.EPOCHS + 1):
        enc.train()
        dec.train()
        eve_train.train()
        int_clf.train()
        t0 = time.time()
        sums = dict(loss_main=0.0, loss_bob=0.0, loss_def=0.0, loss_eve=0.0, loss_int=0.0)
        steps = 0
        for it, (x, y, nonce_i64) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            nonce_i64 = nonce_i64.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=cfg.AMP and device == 'cuda'):
                z = enc(x)
                z_scr = scrambler.scramble(z, nonce_i64)
                z_scr = power_normalize_per_sample(z_scr, cfg.TARGET_CIPHER_POWER)
                y_eve = chan(z_scr, cfg.SNR_EVE_DB)
                y_bob = chan(z_scr, cfg.SNR_BOB_DB)
            for _ in range(max(1, int(cfg.ATTACKER_STEPS_PER_MAIN))):
                with torch.amp.autocast('cuda', enabled=cfg.AMP and device == 'cuda'):
                    logits_det = int_clf(y_eve.detach(), nonce_i64)
                    loss_int = F.cross_entropy(logits_det, y)
                _scaler_step(cfg, scaler, opt_int, loss_int, list(int_clf.parameters()), cfg.GRAD_CLIP_NORM)
                with torch.amp.autocast('cuda', enabled=cfg.AMP and device == 'cuda'):
                    x_eve = eve_train(y_eve.detach(), nonce_i64)
                    loss_eve = F.mse_loss(x_eve.float(), x.float())
                _scaler_step(cfg, scaler, opt_eve, loss_eve, list(eve_train.parameters()), cfg.GRAD_CLIP_NORM)
            for p in int_clf.parameters():
                p.requires_grad_(False)
            for p in eve_train.parameters():
                p.requires_grad_(False)
            with torch.amp.autocast('cuda', enabled=cfg.AMP and device == 'cuda'):
                z_hat = scrambler.descramble(y_bob, nonce_i64)
                x_hat = dec(z_hat)
                mse = F.mse_loss(x_hat.float(), x.float())
                mss = ms_ssim_mean(x_hat, x)
                loss_bob = mse + cfg.LAMBDA_MS_SSIM * (1.0 - mss)
                logits_adv = int_clf(y_eve, nonce_i64).float()
                log_p = F.log_softmax(logits_adv, dim=-1)
                log_p = torch.clamp(log_p, min=-30.0, max=0.0)
                u = torch.full_like(log_p, 1.0 / log_p.shape[-1])
                loss_def = F.kl_div(log_p, u, reduction='batchmean', log_target=False)
                if cfg.LAMBDA_EVE > 0.0:
                    x_eve_adv = eve_train(y_eve, nonce_i64)
                    loss_eve_adv = -F.mse_loss(x_eve_adv.float(), x.float())
                else:
                    loss_eve_adv = 0.0
                loss_main = loss_bob + cfg.LAMBDA_DEF * loss_def + cfg.LAMBDA_EVE * loss_eve_adv
            ok = _scaler_step(cfg, scaler, opt_main, loss_main, list(enc.parameters()) + list(dec.parameters()), cfg.GRAD_CLIP_NORM)
            for p in int_clf.parameters():
                p.requires_grad_(True)
            for p in eve_train.parameters():
                p.requires_grad_(True)
            if not ok:
                print('Non-finite main loss detected; stopping training.')
                break
            sums['loss_main'] += float(loss_main.detach().cpu())
            sums['loss_bob'] += float(loss_bob.detach().cpu())
            sums['loss_def'] += float(loss_def.detach().cpu())
            sums['loss_eve'] += float(loss_eve.detach().cpu())
            sums['loss_int'] += float(loss_int.detach().cpu())
            steps += 1
            if cfg.PRINT_EVERY > 0 and it % cfg.PRINT_EVERY == 0:
                print(f"  ep{ep:03d} it{it:04d} | bob={sums['loss_bob'] / steps:.4f} def(KL)={sums['loss_def'] / steps:.4f} eve={sums['loss_eve'] / steps:.4f} int={sums['loss_int'] / steps:.4f}")
        eve_models = {'base': EveRecon(cfg.LATENT_N, nonce_emb_dim=32).to(device), 'strong': EveReconStrong(cfg.LATENT_N, nonce_emb_dim=64).to(device), 'no_nonce': EveReconNoNonce(cfg.LATENT_N).to(device), 'bob_style': EveBobStyleDecoder(cfg.LATENT_N).to(device)}
        eve_models['base'].load_state_dict(eve_train.state_dict())
        metrics = eval_once(cfg, enc, dec, scrambler, chan, eve_models, int_clf, test_loader, device, collect_distributions=False)
        dt = time.time() - t0
        row = {'epoch': ep, 'time_s': dt, 'loss_main': sums['loss_main'] / max(1, steps), 'loss_bob': sums['loss_bob'] / max(1, steps), 'loss_def': sums['loss_def'] / max(1, steps), 'loss_eve': sums['loss_eve'] / max(1, steps), 'loss_int': sums['loss_int'] / max(1, steps), **metrics}
        history.append(row)
        msg = f"[Ep {ep:03d}/{cfg.EPOCHS}] {dt:.1f}s | Bob SSIM and PSNR={row['ssim_bob']:.3f}/{row['psnr_bob']:.2f} | Eve best ({row['eve_best_name']}) SSIM and PSNR={row['ssim_eve_best']:.3f}/{row['psnr_eve_best']:.2f} | Intercept accuracy={row['intercept_acc']:.3f}"
        if cfg.DO_WRONG_KEY_EVAL:
            msg += f" | Wrong key SSIM and PSNR={row['ssim_wrongkey']:.3f}/{row['psnr_wrongkey']:.2f}"
        print(msg)
        if best is None or row['ssim_bob'] > best['ssim_bob']:
            best = row
            torch.save({'cfg': asdict(cfg), 'enc': enc.state_dict(), 'dec': dec.state_dict(), 'int_clf': int_clf.state_dict(), 'eve_train': eve_train.state_dict(), 'best': best}, os.path.join(cfg.OUT_DIR, 'best_main.pt'))
        with open(os.path.join(cfg.OUT_DIR, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    return (enc, dec, scrambler, chan, int_clf, eve_train, history, best)
