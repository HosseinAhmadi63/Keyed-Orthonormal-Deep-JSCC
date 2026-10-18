from __future__ import annotations
import argparse
import gc
import hashlib
import json
import marshal
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
SOURCE = Path(__file__).resolve().parents[1] / 'archived_implementation'
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from models.attackers import EveRecon, EveReconStrong, EveReconNoNonce, EveBobStyleDecoder
from ssim_ms import ssim_per_image
FAMILIES = ('base', 'strong', 'no_nonce', 'bob_style')
SEED_OFFSETS = {'base': 11, 'strong': 23, 'no_nonce': 37, 'bob_style': 53}

def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)

def _default_device() -> str:
    return 'mps' if torch.backends.mps.is_available() else 'cpu'

class CachedPackets(Dataset):

    def __init__(self, observations: np.ndarray, nonce_features: np.ndarray, images: np.ndarray, indices: np.ndarray | None=None):
        self.observations = torch.from_numpy(np.ascontiguousarray(observations, dtype=np.float32))
        self.nonce_features = torch.from_numpy(np.ascontiguousarray(nonce_features, dtype=np.float32))
        self.images = torch.from_numpy(np.ascontiguousarray(images))
        self.indices = np.arange(len(images), dtype=np.int64) if indices is None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        j = int(self.indices[idx])
        return (self.observations[j], self.nonce_features[j], self.images[j])

class NonceFeatureRecon(nn.Module):

    def __init__(self, family: str, latent_n: int, nonce_dim: int):
        super().__init__()
        if family == 'base':
            self.archived = EveRecon(latent_n, nonce_emb_dim=32)
        elif family == 'strong':
            self.archived = EveReconStrong(latent_n, nonce_emb_dim=64)
        else:
            raise ValueError(f'Nonce-aware wrapper cannot build family {family!r}')
        first = self.archived.nonce_emb.mlp[0]
        self.archived.nonce_emb.mlp[0] = nn.Linear(nonce_dim, first.out_features)
        self.family = family

    def forward(self, observations: torch.Tensor, nonce_features: torch.Tensor):
        a = self.archived
        embedding = a.nonce_emb.mlp(nonce_features)
        h = a.fc(torch.cat((observations, embedding), dim=-1))
        h = h.view(observations.shape[0], 256, 4, 4)
        if self.family == 'strong':
            h = h + a.res(h)
        return a.deconv(h)

def _build_model(family: str, latent_n: int, nonce_dim: int) -> nn.Module:
    if family in ('base', 'strong'):
        return NonceFeatureRecon(family, latent_n, nonce_dim)
    if family == 'no_nonce':
        return EveReconNoNonce(latent_n)
    if family == 'bob_style':
        return EveBobStyleDecoder(latent_n)
    raise ValueError(f'Unknown family {family!r}; choose among {FAMILIES}')

def _predict(model: nn.Module, family: str, obs: torch.Tensor, nonce: torch.Tensor) -> torch.Tensor:
    return model(obs, nonce) if family in ('base', 'strong') or family.startswith('mlp_') else model(obs)

def _move(batch, device: torch.device):
    obs, nonce, images = batch
    return (obs.to(device), nonce.to(device), images.to(device).float().div_(255.0))

@torch.no_grad()
def _validation_mse(model: nn.Module, family: str, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = (0.0, 0)
    for batch in loader:
        obs, nonce, images = _move(batch, device)
        recon = _predict(model, family, obs, nonce)
        mse = (recon.float() - images).square().mean((1, 2, 3))
        total += float(mse.sum().cpu())
        count += len(images)
    return total / count

@torch.no_grad()
def _evaluate_holdout(model: nn.Module, family: str, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    mse_list, ssim_list, examples = ([], [], [])
    for batch in loader:
        obs, nonce, images = _move(batch, device)
        recon = _predict(model, family, obs, nonce)
        mse_list.append((recon.float() - images).square().mean((1, 2, 3)).cpu().numpy())
        ssim_list.append(ssim_per_image(recon, images).cpu().numpy())
        if not examples:
            examples.append(recon[:16].cpu().numpy())
    return (np.concatenate(mse_list), np.concatenate(ssim_list), examples[0])

def _source_hashes() -> dict[str, str]:
    relpaths = ('models/attackers.py', 'models/autoencoder.py', 'ssim_ms.py')
    return {p: hashlib.sha256((SOURCE / p).read_bytes()).hexdigest() for p in relpaths}

def _loaded_implementation_fingerprints() -> dict[str, Any]:
    funcs = {'DirectMLPRecon.__init__': DirectMLPRecon.__init__, 'DirectMLPRecon.forward': DirectMLPRecon.forward, 'ssim_per_image': getattr(ssim_per_image, '__wrapped__', ssim_per_image), 'ridge_features': _ridge_features, 'pixels': _pixels}
    return {'kind': 'SHA256 of Python marshal code objects; Python-version/path dependent', 'functions': {name: hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest() for name, fn in funcs.items()}, 'source_files_separately_supplied': ['models/attackers.py', 'models/autoencoder.py', 'ssim_ms.py', 'reconstruction_attacks.py']}

def _cache_metadata(cache) -> Any:
    for k in ('metadata_json', 'meta_json', 'metadata'):
        if k in cache.files:
            val = cache[k]
            if val.size == 1:
                v = val.item()
                if isinstance(v, bytes):
                    v = v.decode()
                if isinstance(v, str):
                    try:
                        return json.loads(v)
                    except json.JSONDecodeError:
                        return {'raw': v}
    return None

def _atomic_json(path: Path, obj: Any):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2))
    temp.replace(path)

def _run_convolutional_attacks(cachepath: str | Path, outputdir: str | Path, *, epochs: int=8, batch_size: int=128, validation_size: int=5000, seed: int=20261010, families: Iterable[str]=FAMILIES, device: str | None=None, num_threads: int=2, require_holdout_size: int | None=3600, protocol: str | None=None) -> dict[str, Any]:
    cachepath, outputdir = (Path(cachepath).resolve(), Path(outputdir).resolve())
    outputdir.mkdir(parents=True, exist_ok=True)
    families = tuple(families)
    if not families or any((f not in FAMILIES for f in families)) or len(set(families)) != len(families):
        raise ValueError(f'Families must be a nonempty unique subset of {FAMILIES}')
    if epochs < 1 or batch_size < 1 or validation_size < 1:
        raise ValueError('Positive epochs, batch size and training-validation size are required')
    torch.set_num_threads(num_threads)
    resolved_device = torch.device(device or _default_device())
    if resolved_device.type == 'mps' and (not torch.backends.mps.is_available()):
        raise RuntimeError('MPS was requested but is unavailable')
    with np.load(cachepath, allow_pickle=False) as cache:
        required = ('train_obs', 'train_nonce_features', 'train_images', 'test_obs', 'test_nonce_features', 'test_images')
        missing = [k for k in required if k not in cache.files]
        if missing:
            raise ValueError(f'Cache lacks {missing}')
        a = {k: cache[k] for k in required}
        metadata = _cache_metadata(cache)
        test_labels = np.array(cache['y_test']) if 'y_test' in cache.files else None
        test_indices = np.array(cache['test_indices']) if 'test_indices' in cache.files else None
    train_n, test_n = (len(a['train_images']), len(a['test_images']))
    if train_n <= validation_size + batch_size:
        raise ValueError('Cache is too small for the requested disjoint fitting/validation split')
    if require_holdout_size is not None and test_n != require_holdout_size:
        raise ValueError(f'Expected untouched holdout size {require_holdout_size}, found {test_n}')
    for scope in ('train', 'test'):
        images, obs, nonce = (a[f'{scope}_images'], a[f'{scope}_obs'], a[f'{scope}_nonce_features'])
        if images.dtype != np.uint8 or images.shape[1:] != (3, 32, 32):
            raise ValueError(f'{scope}_images must be uint8[N,3,32,32], got {images.dtype}/{images.shape}')
        if obs.ndim != 2 or nonce.ndim != 2 or len(obs) != len(images) or (len(nonce) != len(images)):
            raise ValueError(f'Inconsistent {scope} observation/nonce/image shapes')
        if not np.isfinite(obs).all() or not np.isfinite(nonce).all():
            raise ValueError(f'Nonfinite {scope} observations or nonce features')
    latent_n = a['train_obs'].shape[1]
    nonce_dim = a['train_nonce_features'].shape[1]
    if a['test_obs'].shape[1] != latent_n or a['test_nonce_features'].shape[1] != nonce_dim:
        raise ValueError('Training and holdout latent/nonce dimensions differ')
    if nonce_dim not in (4, 6):
        raise ValueError(f'Expected 4 or 6 normalized public nonce features, got {nonce_dim}')
    split_rng = np.random.default_rng(seed)
    order = split_rng.permutation(train_n)
    val_indices, fit_indices = (order[:validation_size], order[validation_size:])
    np.savez(outputdir / 'attacker_split_indices.npz', fitting=fit_indices, validation=val_indices)
    fit_data = CachedPackets(a['train_obs'], a['train_nonce_features'], a['train_images'], fit_indices)
    val_data = CachedPackets(a['train_obs'], a['train_nonce_features'], a['train_images'], val_indices)
    test_data = CachedPackets(a['test_obs'], a['test_nonce_features'], a['test_images'])
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    updates_per_epoch = len(fit_indices) // batch_size
    report: dict[str, Any] = {'cache_path': str(cachepath), 'protocol': protocol, 'cache_metadata': metadata, 'source_sha256': _source_hashes(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'training': {'seed': seed, 'fresh_initialization_seed_offsets': SEED_OFFSETS, 'encoder_retrained': False, 'cached_packets_per_image': 1, 'cached_awgn_resampled_during_fitting': False, 'attacker_pool_n': train_n, 'fitting_n': len(fit_indices), 'validation_n': len(val_indices), 'holdout_n': test_n, 'latent_n': latent_n, 'public_nonce_feature_dim': nonce_dim, 'epochs': epochs, 'batch_size': batch_size, 'drop_last': True, 'updates_per_epoch': updates_per_epoch, 'updates_per_attacker': epochs * updates_per_epoch, 'optimizer': 'AdamW', 'learning_rate': 0.0002, 'weight_decay': 0.0001, 'gradient_clip_norm': 1.0, 'loss': 'MSE', 'device': str(resolved_device), 'cpu_threads': num_threads, 'checkpoint_selection': 'minimum image-weighted validation MSE', 'holdout_used_for_selection': False, 'family_selection_on_holdout': False, 'amp': False}, 'families': {}}
    _atomic_json(outputdir / 'reconstruction_metrics.json', report)
    per_image: dict[str, np.ndarray] = {}
    if test_labels is not None:
        per_image['y_true'] = test_labels
    if test_indices is not None:
        per_image['test_indices'] = test_indices
    total_started = time.monotonic()
    for family in families:
        model_seed = seed + SEED_OFFSETS[family]
        _seed(model_seed)
        generator = torch.Generator(device='cpu').manual_seed(seed + 101)
        fit_loader = DataLoader(fit_data, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator, num_workers=0)
        model = _build_model(family, latent_n, nonce_dim).to(resolved_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0001)
        best_mse, best_epoch, best_state = (float('inf'), None, None)
        history = []
        family_started = time.monotonic()
        for ep in range(1, epochs + 1):
            started = time.monotonic()
            model.train()
            loss_sum, observed, steps = (0.0, 0, 0)
            for batch in fit_loader:
                obs, nonce, images = _move(batch, resolved_device)
                optimizer.zero_grad(set_to_none=True)
                recon = _predict(model, family, obs, nonce)
                loss = (recon.float() - images).square().mean()
                if not bool(torch.isfinite(loss).cpu()):
                    raise RuntimeError(f'Nonfinite fitting loss for {family} at epoch {ep}')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * len(images)
                observed += len(images)
                steps += 1
            val_mse = _validation_mse(model, family, val_loader, resolved_device)
            if not np.isfinite(val_mse):
                raise RuntimeError(f'Nonfinite validation MSE for {family} at epoch {ep}')
            if val_mse < best_mse:
                best_mse, best_epoch = (val_mse, ep)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = {'epoch': ep, 'fitting_mse': loss_sum / observed, 'fitting_images_seen': observed, 'updates': steps, 'validation_mse': val_mse, 'elapsed_s': time.monotonic() - started}
            history.append(row)
            print(f"[{protocol or cachepath.stem}] recon {family} epoch {ep}/{epochs} fit_mse={row['fitting_mse']:.6f} val_mse={val_mse:.6f} {row['elapsed_s']:.1f}s", flush=True)
            _atomic_json(outputdir / f'{family}_training_history.json', history)
        assert best_state is not None
        model.load_state_dict(best_state)
        torch.save({'family': family, 'state_dict': best_state, 'seed': model_seed, 'best_validation_epoch': best_epoch, 'best_validation_mse': best_mse, 'latent_n': latent_n, 'nonce_feature_dim': nonce_dim, 'training': report['training']}, outputdir / f'{family}_attacker.pt')
        mse, ssim, examples = _evaluate_holdout(model, family, test_loader, resolved_device)
        assert len(mse) == test_n and len(ssim) == test_n
        metrics = {'initialization_seed': model_seed, 'parameter_count': sum((p.numel() for p in model.parameters())), 'selected_validation_epoch': best_epoch, 'selected_validation_mse': best_mse, 'holdout_n': test_n, 'holdout_mse': float(mse.astype(np.float64).mean()), 'holdout_dataset_psnr': float(-10 * np.log10(mse.astype(np.float64).mean())), 'holdout_ssim': float(ssim.astype(np.float64).mean()), 'holdout_ssim_p95': float(np.percentile(ssim, 95)), 'holdout_ssim_max': float(ssim.max()), 'elapsed_s': time.monotonic() - family_started, 'epochs': history}
        report['families'][family] = metrics
        per_image[f'{family}_mse'] = mse
        per_image[f'{family}_ssim'] = ssim
        per_image[f'{family}_first16_reconstructions'] = examples
        np.savez_compressed(outputdir / 'reconstruction_per_image.npz', **per_image)
        _atomic_json(outputdir / 'reconstruction_metrics.json', report)
        print(f"[{protocol or cachepath.stem}] recon {family} holdout PSNR={metrics['holdout_dataset_psnr']:.4f} SSIM={metrics['holdout_ssim']:.6f} (validation-selected epoch {best_epoch})", flush=True)
        del model, optimizer, best_state
        gc.collect()
        if resolved_device.type == 'mps':
            torch.mps.empty_cache()
    report['total_elapsed_s'] = time.monotonic() - total_started
    report['per_image_file'] = 'reconstruction_per_image.npz'
    _atomic_json(outputdir / 'reconstruction_metrics.json', report)
    return report

class DirectMLPRecon(nn.Module):

    def __init__(self, latent_n: int, nonce_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(latent_n + nonce_dim, 512), nn.SiLU(), nn.Linear(512, 1024), nn.SiLU(), nn.Linear(1024, 3 * 32 * 32), nn.Sigmoid())

    def forward(self, observations: torch.Tensor, nonce_features: torch.Tensor):
        return self.mlp(torch.cat((observations, nonce_features), dim=-1)).view(-1, 3, 32, 32)

def _metrics(mse: np.ndarray, ssim: np.ndarray) -> dict[str, Any]:
    return {'holdout_n': len(mse), 'holdout_mse': float(mse.astype(np.float64).mean()), 'holdout_dataset_psnr': float(-10 * np.log10(mse.astype(np.float64).mean())), 'holdout_ssim': float(ssim.astype(np.float64).mean()), 'holdout_ssim_p95': float(np.percentile(ssim, 95)), 'holdout_ssim_max': float(ssim.max())}

def _ridge_features(obs: np.ndarray, nonce: np.ndarray, sorted_absolute: bool) -> np.ndarray:
    observations = np.sort(np.abs(obs), axis=1) if sorted_absolute else obs
    return np.concatenate((observations, nonce), axis=1).astype(np.float64)

def _pixels(images: np.ndarray) -> np.ndarray:
    return images.reshape(len(images), -1).astype(np.float64) / 255.0

@torch.no_grad()
def _evaluate_array_predictions(predict, images: np.ndarray, features: np.ndarray | None, batch_size: int, device: torch.device):
    mses, ssims, examples = ([], [], [])
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        pred = predict(None if features is None else features[start:stop], stop - start)
        pred = np.clip(pred, 0, 1).reshape(-1, 3, 32, 32).astype(np.float32)
        target = torch.from_numpy(np.ascontiguousarray(images[start:stop])).to(device).float() / 255.0
        output = torch.from_numpy(np.ascontiguousarray(pred)).to(device)
        mses.append((output - target).square().mean((1, 2, 3)).cpu().numpy())
        ssims.append(ssim_per_image(output, target).cpu().numpy())
        if not examples:
            examples.append(pred[:16])
    return (np.concatenate(mses), np.concatenate(ssims), examples[0])

def run_reconstruction_attacks(cachepath: str | Path, outputdir: str | Path, *, epochs: int=8, batch_size: int=256, validation_size: int=5000, seed: int=20261010, mlp_restarts: int=3, ridge_alphas: Iterable[float]=(1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0), device: str | None=None, num_threads: int=2, require_holdout_size: int | None=3600, protocol: str | None=None, extended: bool=False) -> dict[str, Any]:
    cachepath, outputdir = (Path(cachepath).resolve(), Path(outputdir).resolve())
    outputdir.mkdir(parents=True, exist_ok=True)
    if epochs < 1 or batch_size < 1 or validation_size < 1 or (mlp_restarts < 0):
        raise ValueError('Positive epoch, batch and validation counts; nonnegative restart count required')
    torch.set_num_threads(num_threads)
    resolved_device = torch.device(device or _default_device())
    with np.load(cachepath, allow_pickle=False) as cache:
        required = ('train_obs', 'train_nonce_features', 'train_images', 'test_obs', 'test_nonce_features', 'test_images')
        missing = [k for k in required if k not in cache.files]
        if missing:
            raise ValueError(f'Cache lacks {missing}')
        a = {k: cache[k] for k in required}
        metadata = _cache_metadata(cache)
        extras = {k: np.array(cache[k]) for k in ('y_test', 'test_indices') if k in cache.files}
    train_n, test_n = (len(a['train_images']), len(a['test_images']))
    if train_n <= validation_size + batch_size:
        raise ValueError('Insufficient disjoint fitting/validation observations')
    if require_holdout_size is not None and test_n != require_holdout_size:
        raise ValueError(f'Expected untouched holdout size {require_holdout_size}, found {test_n}')
    for scope in ('train', 'test'):
        im, obs, nonce = (a[f'{scope}_images'], a[f'{scope}_obs'], a[f'{scope}_nonce_features'])
        if im.dtype != np.uint8 or im.shape[1:] != (3, 32, 32):
            raise ValueError(f'{scope}_images must be uint8[N,3,32,32]')
        if obs.ndim != 2 or nonce.ndim != 2 or len(obs) != len(im) or (len(nonce) != len(im)):
            raise ValueError(f'Inconsistent {scope} shapes')
        if not np.isfinite(obs).all() or not np.isfinite(nonce).all():
            raise ValueError(f'Nonfinite {scope} observations/features')
    latent_n, nonce_dim = (a['train_obs'].shape[1], a['train_nonce_features'].shape[1])
    if a['test_obs'].shape[1] != latent_n or a['test_nonce_features'].shape[1] != nonce_dim:
        raise ValueError('Train/holdout dimensions differ')
    if nonce_dim not in (4, 6):
        raise ValueError('Nonce features must be four or six normalized public chunks')
    order = np.random.default_rng(seed).permutation(train_n)
    val_indices, fit_indices = (order[:validation_size], order[validation_size:])
    np.savez(outputdir / 'attacker_split_indices.npz', fitting=fit_indices, validation=val_indices)
    fit_data = CachedPackets(a['train_obs'], a['train_nonce_features'], a['train_images'], fit_indices)
    val_data = CachedPackets(a['train_obs'], a['train_nonce_features'], a['train_images'], val_indices)
    test_data = CachedPackets(a['test_obs'], a['test_nonce_features'], a['test_images'])
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    report: dict[str, Any] = {'cache_path': str(cachepath), 'protocol': protocol, 'cache_metadata': metadata, 'loaded_implementation_fingerprints': _loaded_implementation_fingerprints(), 'runtime_versions': {'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__)}, 'training': {'seed': seed, 'encoder_retrained': False, 'cached_packets_per_image': 1, 'cached_awgn_resampled_during_fitting': False, 'attacker_pool_n': train_n, 'fitting_n': len(fit_indices), 'validation_n': len(val_indices), 'holdout_n': test_n, 'latent_n': latent_n, 'public_nonce_feature_dim': nonce_dim, 'epochs': epochs, 'batch_size': batch_size, 'drop_last': True, 'updates_per_epoch': len(fit_indices) // batch_size, 'updates_per_mlp': epochs * (len(fit_indices) // batch_size), 'optimizer': 'AdamW', 'learning_rate': 0.0002, 'weight_decay': 0.0001, 'gradient_clip_norm': 1.0, 'loss': 'MSE', 'device': str(resolved_device), 'cpu_threads': num_threads, 'amp': False, 'checkpoint_selection': 'minimum image-weighted validation MSE', 'holdout_used_for_selection': False, 'family_selection_on_holdout': False, 'mlp_architecture': [latent_n + nonce_dim, 512, 1024, 3072], 'mlp_hidden_activation': 'SiLU', 'mlp_output_activation': 'sigmoid', 'mlp_restarts': mlp_restarts, 'ridge_alpha_scaling': '(X.T@X)/N + alpha*I', 'ridge_feature_scaling': 'fitting-only mean and standard deviation', 'ridge_target': 'RGB image pixels in [0,1], centered by fitting-only mean image', 'ridge_prediction_clipped_to_unit_interval': True, 'ridge_alpha_selected_on_validation': True}, 'families': {}}
    per_image = {'y_true': extras['y_test']} if 'y_test' in extras else {}
    if 'test_indices' in extras:
        per_image['test_indices'] = extras['test_indices']
    per_image['first16_ground_truth'] = a['test_images'][:16]
    total_started = time.monotonic()
    for restart in range(mlp_restarts):
        family = f'mlp_seed_{restart + 1}'
        model_seed = seed + 101 + 106 * restart
        _seed(model_seed)
        loader = DataLoader(fit_data, batch_size=batch_size, shuffle=True, drop_last=True, generator=torch.Generator(device='cpu').manual_seed(seed + 101), num_workers=0)
        model = DirectMLPRecon(latent_n, nonce_dim).to(resolved_device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.0001)
        best_mse, best_epoch, best_state = (float('inf'), None, None)
        history = []
        started_family = time.monotonic()
        for ep in range(1, epochs + 1):
            started = time.monotonic()
            model.train()
            loss_sum, seen, steps = (0.0, 0, 0)
            for batch in loader:
                obs, nonce, images = _move(batch, resolved_device)
                opt.zero_grad(set_to_none=True)
                recon = model(obs, nonce)
                loss = (recon - images).square().mean()
                if not bool(torch.isfinite(loss).cpu()):
                    raise RuntimeError(f'Nonfinite {family} fitting loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                loss_sum += float(loss.detach().cpu()) * len(images)
                seen += len(images)
                steps += 1
            val_mse = _validation_mse(model, family, val_loader, resolved_device)
            if val_mse < best_mse:
                best_mse, best_epoch = (val_mse, ep)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = {'epoch': ep, 'fitting_mse': loss_sum / seen, 'fitting_images_seen': seen, 'updates': steps, 'validation_mse': val_mse, 'elapsed_s': time.monotonic() - started}
            history.append(row)
            print(f"[{protocol or cachepath.stem}] {family} epoch {ep}/{epochs} fit_mse={row['fitting_mse']:.6f} val_mse={val_mse:.6f} {row['elapsed_s']:.1f}s", flush=True)
            _atomic_json(outputdir / f'{family}_training_history.json', history)
        if best_state is None:
            raise RuntimeError('No finite validation checkpoint')
        model.load_state_dict(best_state)
        torch.save({'state_dict': best_state, 'seed': model_seed, 'selected_validation_epoch': best_epoch, 'selected_validation_mse': best_mse, 'architecture': report['training']['mlp_architecture']}, outputdir / f'{family}_attacker.pt')
        mse, ssim, examples = _evaluate_holdout(model, family, test_loader, resolved_device)
        metrics = _metrics(mse, ssim)
        metrics.update({'initialization_seed': model_seed, 'parameter_count': sum((p.numel() for p in model.parameters())), 'selected_validation_epoch': best_epoch, 'selected_validation_mse': best_mse, 'epochs': history, 'elapsed_s': time.monotonic() - started_family})
        report['families'][family] = metrics
        for suffix, value in (('mse', mse), ('ssim', ssim), ('first16_reconstructions', examples)):
            per_image[f'{family}_{suffix}'] = value
        np.savez_compressed(outputdir / 'reconstruction_per_image.npz', **per_image)
        _atomic_json(outputdir / 'reconstruction_metrics.json', report)
        print(f"[{protocol or cachepath.stem}] {family} holdout PSNR={metrics['holdout_dataset_psnr']:.4f} SSIM={metrics['holdout_ssim']:.6f}", flush=True)
        del model, opt, best_state
        gc.collect()
        if resolved_device.type == 'mps':
            torch.mps.empty_cache()
    ridge_alphas = tuple((float(x) for x in ridge_alphas))
    if not ridge_alphas or any((x <= 0 for x in ridge_alphas)):
        raise ValueError('Positive ridge alphas required')
    mean_image = a['train_images'][fit_indices].mean(axis=0, dtype=np.float64).reshape(1, -1) / 255.0
    for sorted_absolute in (False, True):
        family = 'ridge_sorted_abs_nonce' if sorted_absolute else 'ridge_raw_nonce'
        started = time.monotonic()
        features = _ridge_features(a['train_obs'], a['train_nonce_features'], sorted_absolute)
        xmean = features[fit_indices].mean(axis=0)
        xscale = features[fit_indices].std(axis=0).clip(1e-08)
        features = (features - xmean) / xscale
        p = features.shape[1]
        gram = np.zeros((p, p))
        cross = np.zeros((p, 3072))
        for start in range(0, len(fit_indices), 1024):
            ii = fit_indices[start:start + 1024]
            xx = features[ii]
            yy = _pixels(a['train_images'][ii]) - mean_image
            gram += xx.T @ xx
            cross += xx.T @ yy
        gram /= len(fit_indices)
        cross /= len(fit_indices)
        evals, evecs = np.linalg.eigh(gram)
        projected = evecs.T @ cross
        best_val, best_alpha, best_coef = (float('inf'), None, None)
        grid = []
        for alpha in ridge_alphas:
            coef = evecs @ (projected / (evals[:, None] + alpha))
            err, count = (0.0, 0)
            for start in range(0, len(val_indices), 1024):
                ii = val_indices[start:start + 1024]
                pred = np.clip(features[ii] @ coef + mean_image, 0, 1)
                yy = _pixels(a['train_images'][ii])
                err += float(np.square(pred - yy).sum())
                count += pred.size
            val_mse = err / count
            grid.append({'alpha': alpha, 'validation_mse': val_mse})
            if val_mse < best_val:
                best_val, best_alpha, best_coef = (val_mse, alpha, coef)
        test_features = (_ridge_features(a['test_obs'], a['test_nonce_features'], sorted_absolute) - xmean) / xscale
        predict = lambda xx, nn: xx @ best_coef + mean_image
        mse, ssim, examples = _evaluate_array_predictions(predict, a['test_images'], test_features, batch_size, resolved_device)
        metrics = _metrics(mse, ssim)
        metrics.update({'selected_validation_alpha': best_alpha, 'selected_validation_mse': best_val, 'validation_alpha_grid': grid, 'feature_transform': 'sorted absolute intercepted symbols plus public nonce' if sorted_absolute else 'raw intercepted symbols plus public nonce', 'coefficient_count': int(best_coef.size), 'elapsed_s': time.monotonic() - started})
        report['families'][family] = metrics
        np.savez_compressed(outputdir / f'{family}_coefficients.npz', coefficients=best_coef, feature_mean=xmean, feature_scale=xscale, mean_image=mean_image, alpha=best_alpha)
        for suffix, value in (('mse', mse), ('ssim', ssim), ('first16_reconstructions', examples)):
            per_image[f'{family}_{suffix}'] = value
        np.savez_compressed(outputdir / 'reconstruction_per_image.npz', **per_image)
        _atomic_json(outputdir / 'reconstruction_metrics.json', report)
        print(f"[{protocol or cachepath.stem}] {family} alpha={best_alpha:g} holdout PSNR={metrics['holdout_dataset_psnr']:.4f} SSIM={metrics['holdout_ssim']:.6f}", flush=True)
        del features, gram, cross, evecs, projected, best_coef, test_features
        gc.collect()
    mse, ssim, examples = _evaluate_array_predictions(lambda xx, nn: np.repeat(mean_image, nn, axis=0), a['test_images'], None, batch_size, resolved_device)
    report['families']['fitting_mean_image'] = _metrics(mse, ssim)
    for suffix, value in (('mse', mse), ('ssim', ssim), ('first16_reconstructions', examples)):
        per_image[f'fitting_mean_image_{suffix}'] = value
    np.savez_compressed(outputdir / 'reconstruction_per_image.npz', **per_image)
    report['total_elapsed_s'] = time.monotonic() - total_started
    report['per_image_file'] = 'reconstruction_per_image.npz'
    if extended:
        report['extended_convolutional_report'] = _run_convolutional_attacks(cachepath, outputdir / 'extended_conv', epochs=epochs, batch_size=128, validation_size=validation_size, seed=seed, device=device, num_threads=num_threads, require_holdout_size=require_holdout_size, protocol=protocol)
    _atomic_json(outputdir / 'reconstruction_metrics.json', report)
    return report

def main():
    p = argparse.ArgumentParser()
    p.add_argument('cachepath', type=Path)
    p.add_argument('outputdir', type=Path)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--validation-size', type=int, default=5000)
    p.add_argument('--seed', type=int, default=20261010)
    p.add_argument('--mlp-restarts', type=int, default=3)
    p.add_argument('--extended', action='store_true')
    p.add_argument('--device', choices=('mps', 'cpu'), default=None)
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--holdout-size', type=int, default=3600)
    p.add_argument('--protocol', default=None)
    args = p.parse_args()
    r = run_reconstruction_attacks(args.cachepath, args.outputdir, epochs=args.epochs, batch_size=args.batch_size, validation_size=args.validation_size, seed=args.seed, mlp_restarts=args.mlp_restarts, device=args.device, num_threads=args.cpu_threads, require_holdout_size=args.holdout_size, protocol=args.protocol, extended=args.extended)
    print(json.dumps({f: {k: v for k, v in m.items() if k != 'epochs'} for f, m in r['families'].items()}, indent=2))
if __name__ == '__main__':
    main()
