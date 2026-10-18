from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import convolve1d
import torch
import torch.nn.functional as F
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
ASSETS = Path(os.environ.get('JSCC_ASSETS', str(ROOT / 'frozen_codec')))
INPUTS = Path(os.environ.get('JSCC_INPUTS', str(ROOT / 'data/interface_controls')))
sys.path.insert(0, str(ROOT / 'frozen_codec/archived_implementation'))
sys.path.insert(0, str(ROOT / 'frozen_codec/evaluation'))
from models.autoencoder import Decoder
from models.attackers import EveReconNoNonce, EveBobStyleDecoder
from ssim_ms import ssim_per_image

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

class DirectMLPRecon(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(262, 512), torch.nn.SiLU(), torch.nn.Linear(512, 1024), torch.nn.SiLU(), torch.nn.Linear(1024, 3072), torch.nn.Sigmoid())

    def forward(self, obs, nonce):
        return self.mlp(torch.cat((obs, nonce), -1)).view(-1, 3, 32, 32)

def load_decoder():
    path = ASSETS / 'models/pt/best_main.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = Decoder(256).eval()
    model.load_state_dict(checkpoint['dec'])
    return (model, path)

def load_mlp(protocol):
    family = {'crypto': 'mlp_seed_3', 'identity': 'mlp_seed_3', 'fixed_crypto': 'mlp_seed_1'}[protocol]
    path = ASSETS / f'evaluation/results/reconstruction_{protocol}_final/{family}_attacker.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['architecture'] == [262, 512, 1024, 3072]
    model = DirectMLPRecon().eval()
    model.load_state_dict(checkpoint['state_dict'])
    return (model, path)

@torch.inference_mode()
def predict(model, observations, nonce=None, batch=128):
    parts = []
    for start in range(0, len(observations), batch):
        x = torch.from_numpy(np.ascontiguousarray(observations[start:start + batch])).float()
        if nonce is not None:
            n = torch.from_numpy(np.ascontiguousarray(nonce[start:start + batch])).float()
            out = model(x, n)
        else:
            out = model(x)
        parts.append(out.numpy())
    return np.concatenate(parts)

def assemble(tiles, height, width):
    return tiles.reshape(height // 32, width // 32, 3, 32, 32).transpose(0, 3, 1, 4, 2).reshape(height, width, 3)

def reference_valid_ssim(pred, target):
    x = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    t = np.arange(-5, 6, dtype=np.float64)
    g = np.exp(-t * t / (2 * 1.5 ** 2))
    g /= g.sum()

    def filt(a):
        return convolve1d(convolve1d(a, g, axis=-1, mode='constant'), g, axis=-2, mode='constant')[..., 5:-5, 5:-5]
    ux, uy = (filt(x), filt(y))
    vx, vy, vxy = (filt(x * x) - ux * ux, filt(y * y) - uy * uy, filt(x * y) - ux * uy)
    score = (2 * ux * uy + 0.01 ** 2) * (2 * vxy + 0.03 ** 2) / ((ux * ux + uy * uy + 0.01 ** 2) * (vx + vy + 0.03 ** 2))
    return score.mean(axis=(1, 2, 3))

@torch.inference_mode()
def metric_arrays(pred, target, batch=64):
    if target.dtype == np.uint8:
        target = target.astype(np.float32) / 255
    output = {k: [] for k in ['mse', 'legacy_zero_clipped', 'zero_unclipped', 'valid_same_guards_unclipped', 'reference_valid_unclipped']}
    t = torch.arange(-5, 6, dtype=torch.float32)
    g = torch.exp(-t * t / (2 * 1.5 ** 2))
    g /= g.sum()
    window = (g[:, None] @ g[None, :]).view(1, 1, 11, 11).repeat(3, 1, 1, 1)
    for start in range(0, len(pred), batch):
        xp = np.ascontiguousarray(pred[start:start + batch], dtype=np.float32)
        yp = np.ascontiguousarray(target[start:start + batch], dtype=np.float32)
        x, y = (torch.from_numpy(xp), torch.from_numpy(yp))
        filt = lambda a: F.conv2d(a, window, padding=5, groups=3)
        ux, uy = (filt(x), filt(y))
        vx, vy = ((filt(x * x) - ux * ux).clamp_min(0), (filt(y * y) - uy * uy).clamp_min(0))
        cov = filt(x * y) - ux * uy
        smap = (2 * ux * uy + 0.01 ** 2) * (2 * cov + 0.03 ** 2) / ((ux * ux + uy * uy + 0.01 ** 2) * (vx + vy + 0.03 ** 2) + 1e-08)
        unclip = smap.mean((1, 2, 3))
        legacy = ssim_per_image(x, y)
        if not torch.allclose(legacy, unclip.clamp(0, 1), atol=2e-06, rtol=0):
            raise AssertionError('Legacy SSIM reconstruction mismatch')
        output['mse'].append((x - y).square().mean((1, 2, 3)).numpy())
        output['legacy_zero_clipped'].append(legacy.numpy())
        output['zero_unclipped'].append(unclip.numpy())
        output['valid_same_guards_unclipped'].append(smap[..., 5:-5, 5:-5].mean((1, 2, 3)).numpy())
        output['reference_valid_unclipped'].append(reference_valid_ssim(xp, yp))
    return {k: np.concatenate(v) for k, v in output.items()}

def bootstrap_summary(arrays, prior=None, draws=10000, seed=20261015):
    names = list(arrays)
    n = len(arrays[names[0]]['mse'])
    indices = np.random.default_rng(seed).integers(0, n, size=(draws, n))
    variants = [k for k in arrays[names[0]] if k != 'mse']
    out = {}
    ci = lambda x: np.quantile(x, [0.025, 0.975]).tolist()
    for name in names:
        row = {}
        m = arrays[name]['mse'].astype(np.float64)
        bm = m[indices].mean(1)
        row['dataset_psnr_db'] = float(-10 * np.log10(m.mean()))
        row['dataset_psnr_ci95'] = ci(-10 * np.log10(bm))
        for variant in variants:
            value = arrays[name][variant].astype(np.float64)
            bs = value[indices].mean(1)
            stats = {'mean': float(value.mean()), 'ci95': ci(bs), 'min': float(value.min()), 'max': float(value.max()), 'negative_images': int((value < 0).sum()), 'above_one_images': int((value > 1).sum())}
            if prior is not None:
                difference = value - arrays[prior][variant]
                stats['minus_prior'] = float(difference.mean())
                stats['minus_prior_ci95'] = ci(difference[indices].mean(1))
            row[variant] = stats
        row['legacy_clipping_changed_images'] = int(np.count_nonzero(np.abs(arrays[name]['legacy_zero_clipped'] - arrays[name]['zero_unclipped']) > 1e-07))
        out[name] = row
    return {'n': n, 'draws': draws, 'seed': seed, 'models': out}
