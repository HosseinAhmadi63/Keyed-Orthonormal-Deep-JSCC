from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import requests
import torch
from keyed_jscc.crypto_scrambler import CryptoScrambler, nonce_features
from keyed_jscc.experiment_utils import normalize, valid_ssim
from keyed_jscc.legacy_ssim import ssim_per_image
from keyed_jscc.public_sampling import sample_packets

_namespace_labels = {}

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).digest()[:4], 'big')

def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Non-finite scientific value requires an explicit interpretation before JSON serialization')
    return value

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(_json_value(obj), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def fit_signature(cfg, extra=None):
    return hashlib.sha256(json.dumps([cfg, extra], sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()

def torch_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

def _packet_material(case, domain, draw, key_index, repetition, n):
    label = json.dumps([case['name'], str(domain), int(draw), int(key_index), int(repetition)])
    context = int.from_bytes(hashlib.sha256(('revision-v2-packet-namespace:'+label).encode()).digest()[:8],'big')
    if context in _namespace_labels and _namespace_labels[context] != label:
        raise RuntimeError('Experiment nonce namespace collision; change the domain registry')
    _namespace_labels[context] = label
    if n >= 2**32:
        raise ValueError('Packet allocation exceeds namespace')
    prefix = context << 32
    order = np.random.default_rng(stable_seed(label, 'nonce-order')).permutation(n)
    nonces = [prefix + int(i) for i in order]
    base = bytes.fromhex(case['key_hex'])
    key = base if key_index == 0 else hashlib.sha256(base + b'revision-key' + int(key_index).to_bytes(4, 'big')).digest()
    params = CryptoScrambler(key, 256).parameters_batch(nonces)
    return params, nonce_features(nonces)

def actual_observations(u, case, domain, snr_db, draw=0, key_index=0, repetitions=1):
    u = np.asarray(u, dtype=np.float32)
    obs, nf = [], []
    for rep in range(repetitions):
        params, features = _packet_material(case, domain, draw, key_index, rep, len(u))
        rng = np.random.default_rng(stable_seed(case['name'], domain, snr_db, draw, key_index, rep, 'noise'))
        sigma = 0.0 if snr_db is None or np.isinf(snr_db) else 10.0**(-float(snr_db)/20)
        obs.append(params.forward(u) + rng.normal(0, sigma, u.shape).astype(np.float32))
        nf.append(features)
    return np.stack(obs, 1).astype(np.float32), np.stack(nf, 1).astype(np.float32)

def bob_observations(u, case, domain, snr_db, draw=0, key_index=0):
    u = np.asarray(u, dtype=np.float32)
    params, _ = _packet_material(case, domain, draw, key_index, 0, len(u))
    rng = np.random.default_rng(stable_seed(case['name'], domain, snr_db, draw, key_index, 'bob-noise'))
    sigma = 0.0 if snr_db is None or np.isinf(snr_db) else 10.0**(-float(snr_db)/20)
    return params.inverse(params.forward(u) + rng.normal(0, sigma, u.shape).astype(np.float32)).astype(np.float32)

def public_observations(u, rng, snr_db, repetitions=1):
    snr = float('inf') if snr_db is None else snr_db
    return np.stack([sample_packets(np.asarray(u, dtype=np.float32), rng, snr) for _ in range(repetitions)], 1).astype(np.float32)

@torch.no_grad()
def image_rows(pred, truth, device='cpu', batch=128):
    pred = np.asarray(pred, dtype=np.float32)
    truth = np.asarray(truth)
    if pred.shape != truth.shape or pred.ndim != 4 or pred.shape[1] != 3 or not np.isfinite(pred).all():
        raise ValueError('Predictions must be finite NCHW RGB with the reference shape')
    target = truth.astype(np.float32) / 255 if truth.dtype == np.uint8 else truth.astype(np.float32)
    rows = {k: [] for k in ('mse', 'ssim_valid', 'ssim_legacy')}
    for start in range(0, len(pred), batch):
        p = torch.as_tensor(pred[start:start+batch], device=device).clamp(0, 1)
        y = torch.as_tensor(target[start:start+batch], device=device)
        rows['mse'].append((p-y).square().mean((1,2,3)).cpu().numpy().astype(np.float64))
        rows['ssim_valid'].append(valid_ssim(p,y).cpu().numpy().astype(np.float64))
        rows['ssim_legacy'].append(ssim_per_image(p,y).cpu().numpy().astype(np.float64))
    return {k: np.concatenate(v) for k,v in rows.items()}

def image_summary(rows):
    mse = float(np.mean(rows['mse'], dtype=np.float64))
    return dict(n=len(rows['mse']), mse=mse, dataset_psnr=None if mse == 0 else -10*math.log10(mse), mean_ssim_valid=float(np.mean(rows['ssim_valid'])), mean_ssim_legacy=float(np.mean(rows['ssim_legacy'])))

def classification_rows(labels, probs):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if not np.isfinite(probs).all() or len(probs) != len(labels) or not np.allclose(probs.sum(1), 1, atol=1e-4):
        raise ValueError('Invalid class probabilities')
    correct = probs.argmax(1) == labels
    classes = np.unique(labels)
    ce = -np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-12))
    return dict(n=len(labels), accuracy=float(correct.mean()), balanced_accuracy=float(np.mean([correct[labels==c].mean() for c in classes])), cross_entropy=float(ce.mean()), correct=correct, negative_log_probability=ce)

@torch.no_grad()
def encode_images(enc, images, device, batch=256):
    enc.eval()
    result = []
    for start in range(0, len(images), batch):
        x = torch.as_tensor(np.array(images[start:start+batch]), device=device).float()/255
        result.append(normalize(enc(x)).float().cpu().numpy())
    return np.concatenate(result)

def download_file(url, path, expected_sha256=None, expected_md5=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def valid(p):
        if expected_sha256 and sha256_file(p) != expected_sha256:
            return False
        if expected_md5:
            h = hashlib.md5()
            with p.open('rb') as stream:
                for block in iter(lambda: stream.read(2**20), b''):
                    h.update(block)
            return h.hexdigest() == expected_md5
        return True
    if path.exists():
        if valid(path):
            return path
        raise RuntimeError(f'Checksum failed for {path}; move that file aside before retrying')
    part = path.with_name(path.name+'.part')
    for attempt in range(1,5):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with requests.get(url, headers=headers, timeout=(30,90), stream=True) as response:
                response.raise_for_status()
                append = offset and response.status_code == 206
                if not append:
                    offset = 0
                total = int(response.headers.get('Content-Length', 0))+offset
                done, last = offset, 0.0
                with part.open('ab' if append else 'wb') as stream:
                    for block in response.iter_content(2**20):
                        stream.write(block)
                        done += len(block)
                        if time.monotonic()-last > 5:
                            pct = f'{100*done/total:.1f}%' if total else 'total size unknown'
                            print(f'Download {path.name}: {done/2**20:.1f} MiB, {pct}', flush=True)
                            last = time.monotonic()
            if not valid(part):
                part.unlink()
                raise RuntimeError('Downloaded file checksum mismatch')
            os.replace(part,path)
            return path
        except (requests.RequestException, OSError) as exc:
            if attempt == 4:
                raise RuntimeError(f'Download failed: {url}; restarting resumes partial files. {exc}') from exc
            print(f'Download retry {attempt}/4: {exc}', flush=True)
            time.sleep(2*attempt)

def load_kodak_frames(cfg=None):
    cfg = cfg or {}
    assets = Path(cfg.get('assets_dir', Path(__file__).resolve().parents[1]/'assets'))
    with np.load(assets/'kodak_packets.npz', allow_pickle=False) as data:
        print('Kodak cache fields: '+', '.join(data.files), flush=True)
        tiles = data['images']
        index = data['image_index']
    records = json.loads((assets/'DATASETS.json').read_text())['kodak']['images']
    frames = []
    for i,record in enumerate(records):
        h,w = record['height'], record['width']
        selected = tiles[index == i]
        if selected.shape != (h*w//1024,3,32,32) or selected.dtype != np.uint8:
            raise RuntimeError('Kodak tile inventory differs from the recorded original frames')
        frames.append(selected.reshape(h//32,w//32,3,32,32).transpose(0,3,1,4,2).reshape(h,w,3))
    return frames
