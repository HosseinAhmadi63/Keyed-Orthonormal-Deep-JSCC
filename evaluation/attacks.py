from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from keyed_jscc.attack_models import ConvReconstructor, DeepSetsClassifier, ResidualMLPClassifier
from keyed_jscc.experiment_utils import ImageMLP, save_torch, valid_ssim
from keyed_jscc.fit_attackers import specs as original_specs
from keyed_jscc.public_sampling import invariant_features
from evaluation.common import (
    actual_observations, classification_rows, fit_signature, image_rows,
    image_summary, public_observations, save_npz, stable_seed, torch_seed, write_json,
)


def defaults():
    return {
        'tasks': ['metric_alignment', 'budget', 'high_snr', 'multi_observation', 'attributes'],
        'primary_epochs': 30,
        'batch_size': 256,
        'validation_batch_size': 128,
        'primary_patience': 0,
        'extended_cases': ['seed1001_lambda0', 'seed1001_lambda0.2'],
        'budget_epochs': [30, 60, 120],
        'budget_seeds': [811, 812],
        'high_snr_values': [20.0, 30.0, 40.0, 'noiseless'],
        'high_snr_epochs': 60,
        'high_snr_seeds': [821],
        'high_snr_keys': 3,
        'high_snr_noise_draws': 2,
        'multi_observation_counts': [1, 2, 4, 8, 16],
        'multi_observation_epochs': 60,
        'multi_observation_seeds': [831],
        'attribute_epochs': 60,
        'attribute_seeds': [841, 842],
        'standardization_max_images': 12000,
        'save_full_predictions': True,
        'cnn_channels': 256,
        'deepsets_width': 64,
        'classifier_width': 512,
        'primary_spec_names': None,
        'control_fit_epochs': 30,
        'controls': ['identity', 'fixed_transform'],
        'record_training_timings': True,
    }


def source_signature():
    root = Path(__file__).resolve().parents[1]
    names = ['evaluation/attacks.py', 'evaluation/common.py',
             'src/keyed_jscc/attack_models.py', 'src/keyed_jscc/fit_attackers.py',
             'src/keyed_jscc/experiment_utils.py', 'src/keyed_jscc/public_sampling.py',
             'src/keyed_jscc/legacy_ssim.py', 'src/keyed_jscc/crypto_scrambler.py']
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode('utf-8'))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _settings(cfg, smoke):
    out = defaults()
    out.update(cfg.get('attacks', {}))
    unknown = set(cfg.get('attacks', {})) - set(defaults())
    if unknown:
        raise ValueError('Unknown attacks configuration keys: ' + ', '.join(sorted(unknown)))
    if smoke:
        out.update(primary_epochs=2, budget_epochs=[1, 2], budget_seeds=[811],
                   high_snr_values=[20.0, 'noiseless'], high_snr_epochs=2,
                   high_snr_seeds=[821], high_snr_keys=1, high_snr_noise_draws=1,
                   multi_observation_counts=[1, 2], multi_observation_epochs=2,
                   multi_observation_seeds=[831], attribute_epochs=2,
                   attribute_seeds=[841], batch_size=8, validation_batch_size=8,
                   cnn_channels=16, deepsets_width=8, classifier_width=32,
                   standardization_max_images=24, control_fit_epochs=2,
                   extended_cases=['*'])
    if out['primary_patience'] != 0:
        raise ValueError('Use full predetermined epoch budgets; primary_patience must be zero')
    for key in ('primary_epochs', 'batch_size', 'validation_batch_size', 'high_snr_epochs',
                'high_snr_keys', 'high_snr_noise_draws', 'multi_observation_epochs',
                'attribute_epochs', 'standardization_max_images'):
        if not isinstance(out[key], int) or out[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if set(out['tasks']) - {'metric_alignment', 'budget', 'high_snr', 'multi_observation', 'attributes'}:
        raise ValueError('Unknown attacker task')
    if not out['budget_epochs'] or sorted(set(out['budget_epochs'])) != out['budget_epochs']:
        raise ValueError('budget_epochs must be strictly increasing positive integers')
    if min(out['budget_epochs']) < 1:
        raise ValueError('budget_epochs must be positive')
    if not out['multi_observation_counts'] or min(out['multi_observation_counts']) < 1:
        raise ValueError('multi_observation_counts must be positive')
    return out


def _number_snr(value):
    return float('inf') if value == 'noiseless' else float(value)


def _snr_name(value):
    return 'noiseless' if value == 'noiseless' else ('snr' + format(float(value), 'g').replace('.', 'p'))


def primary_specs(settings, smoke=False):
    old_cfg = {'classifier_raw_seeds': [51, 52, 53, 54, 55],
               'invariant_seeds': [61, 62], 'reconstruction_mlp_seeds': [71, 72, 73]}
    candidates = original_specs(old_cfg, smoke)
    for spec in candidates:
        spec['task'] = 'classification' if spec['kind'] in ('class', 'sets') else 'reconstruction'
        spec['epochs'] = settings['primary_epochs']
        if smoke and spec['kind'] == 'cnn':
            spec['channels'] = settings['cnn_channels']
    if settings['primary_spec_names'] is not None:
        keep = set(settings['primary_spec_names'])
        unknown = keep - {s['name'] for s in candidates}
        if unknown:
            raise ValueError('Unknown primary spec names: ' + ', '.join(sorted(unknown)))
        candidates = [s for s in candidates if s['name'] in keep]
    return candidates


class ExpressiveDeepSets(nn.Module):
    def __init__(self, width=64, out_dim=10):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(5, width), nn.LayerNorm(width), nn.GELU(),
                                 nn.Linear(width, width * 2), nn.GELU(),
                                 nn.Linear(width * 2, width * 2), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(6 * width + 4, 256), nn.LayerNorm(256),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128),
                                 nn.GELU(), nn.Linear(128, out_dim))

    def forward(self, x):
        features = torch.stack((x, x.abs(), x.square(), x.sign(), torch.log1p(x.abs())), -1)
        h = self.phi(features)
        return self.rho(torch.cat((h.mean(1), h.amax(1), h.std(1, unbiased=False),
                                   x.mean(1, keepdim=True), x.square().mean(1, keepdim=True),
                                   x.pow(4).mean(1, keepdim=True),
                                   x.pow(6).mean(1, keepdim=True)), 1))


class FlexibleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, width=512):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(),
                          nn.Dropout(0.1), nn.Linear(width, width)) for _ in range(3)
        ])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Linear(width, output_dim))

    def forward(self, x):
        h = self.stem(x)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h)


class PacketSetAttacker(nn.Module):
    def __init__(self, input_dim, task='classification', width=256, channels=64):
        super().__init__()
        self.packet = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU(),
                                    nn.Linear(width, width), nn.GELU(),
                                    nn.Linear(width, width), nn.GELU())
        self.task = task
        if task == 'reconstruction':
            self.head = ConvReconstructor(3 * width + 1, channels=channels)
        else:
            self.head = FlexibleMLP(3 * width + 1, 10, width=width)

    def forward(self, x):
        h = self.packet(x)
        count = torch.full((len(x), 1), math.log1p(x.shape[1]), dtype=h.dtype, device=h.device)
        pooled = torch.cat((h.mean(1), h.amax(1), h.std(1, unbiased=False), count), 1)
        return self.head(pooled)


def build_model(spec, input_dim, settings):
    kind = spec['kind']
    if kind == 'class':
        return ResidualMLPClassifier(input_dim)
    if kind == 'sets':
        return DeepSetsClassifier()
    if kind == 'cnn':
        return ConvReconstructor(input_dim, channels=spec.get('channels', 32))
    if kind == 'mlp':
        return ImageMLP(input_dim)
    if kind == 'expressive_sets':
        return ExpressiveDeepSets(settings['deepsets_width'])
    if kind in ('expressive_class', 'binary', 'attributes'):
        output_dim = {'expressive_class': 10, 'binary': 2, 'attributes': 3}[kind]
        return FlexibleMLP(input_dim, output_dim, width=settings['classifier_width'])
    if kind == 'packet_set':
        return PacketSetAttacker(input_dim, spec['task'],
                                 width=min(settings['classifier_width'], 256),
                                 channels=min(settings['cnn_channels'], 128))
    raise ValueError('Unknown model kind ' + kind)


def attack_features(observations, nonce, spec):
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim == 2:
        obs = obs[:, None, :]
    if obs.ndim != 3:
        raise ValueError('observations must have shape images x repetitions x latent_dimension')
    n, repetitions, dimension = obs.shape
    if spec['kind'] in ('sets', 'expressive_sets'):
        if repetitions != 1:
            raise ValueError('Coordinate DeepSets expects one packet per image')
        return obs[:, 0, :]
    values = invariant_features(obs.reshape(n * repetitions, dimension), spec.get('view', 'raw_fusion'))
    if spec.get('nonce', False):
        nf = np.asarray(nonce, dtype=np.float32)
        if nf.ndim == 2:
            nf = nf[:, None, :]
        if nf.shape[:2] != (n, repetitions):
            raise ValueError('nonce features do not match observations')
        values = np.concatenate((values, nf.reshape(n * repetitions, -1)), 1)
    if spec['kind'] == 'packet_set':
        return values.reshape(n, repetitions, -1)
    if repetitions != 1:
        raise ValueError('A single-packet architecture cannot consume repeated observations')
    return values


def animal_vehicle(labels):
    labels = np.asarray(labels, dtype=np.int64)
    if ((labels < 0) | (labels > 9)).any():
        raise ValueError('Expected CIFAR-10 class identifiers')
    return np.isin(labels, [0, 1, 8, 9]).astype(np.int64)


def image_attributes(images):
    x = np.asarray(images, dtype=np.float32) / np.float32(255)
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError('Expected NCHW RGB uint8 images')
    lum = np.float32(0.299) * x[:, 0] + np.float32(0.587) * x[:, 1] + np.float32(0.114) * x[:, 2]
    brightness = lum.mean((1, 2))
    chroma = (x.max(1) - x.min(1)).mean((1, 2))
    edge = 0.5 * (np.square(np.diff(lum, axis=1)).mean((1, 2)) +
                  np.square(np.diff(lum, axis=2)).mean((1, 2)))
    return np.column_stack((brightness, chroma, edge)).astype(np.float32)


def _task_target(case, spec, part):
    if spec['task'] == 'classification':
        return np.asarray(case[part + '_labels'], dtype=np.int64)
    if spec['task'] == 'binary':
        return animal_vehicle(case[part + '_labels'])
    if spec['task'] == 'attributes':
        return image_attributes(case[part + '_images'])
    return case[part + '_images']


def _scoring_metric(spec):
    return {'classification': 'accuracy', 'binary': 'balanced_accuracy',
            'reconstruction': 'ssim_valid', 'attributes': 'negative_normalized_mse'}[spec['task']]


def select_validation_checkpoints(history, metric):
    if not history:
        raise ValueError('Cannot select from an empty validation history')
    loss_key = 'validation_loss'
    metric_key = 'validation_' + metric
    if any(not np.isfinite(row[loss_key]) or not np.isfinite(row[metric_key]) for row in history):
        raise ValueError('Selection metrics must be finite')
    by_loss = min(history, key=lambda r: (r[loss_key], r['epoch']))
    by_metric = min(history, key=lambda r: (-r[metric_key], r[loss_key], r['epoch']))
    return {'loss_epoch': by_loss['epoch'], 'metric_epoch': by_metric['epoch']}


def _array_sha(array):
    x = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(x.shape).encode())
    h.update(x.dtype.str.encode())
    h.update(memoryview(x).cast('B'))
    return h.hexdigest()


def _check_split(case):
    fit = np.asarray(case['fit'], dtype=np.int64)
    val = np.asarray(case['val'], dtype=np.int64)
    size = len(case['train_u'])
    if not len(fit) or not len(val) or np.intersect1d(fit, val).size:
        raise ValueError('Fitting and validation image indices must be nonempty and disjoint')
    for indices in (fit, val):
        if len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= size:
            raise ValueError('Invalid fitting or validation image indices')
    if len(case['train_images']) != size or len(case['train_labels']) != size:
        raise ValueError('Training inputs are not aligned')
    return fit, val


def _lock(folder, signature, details):
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'manifest.json'
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding='utf-8'))
        if existing['signature'] != signature:
            raise RuntimeError('Configuration/source/input signature mismatch at ' + str(folder) +
                               '; use a new output directory')
    else:
        write_json(manifest, dict(signature=signature, **details))


def _torch_cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _stat_features(case, spec, cfg, settings):
    fit = np.asarray(case['fit'])
    rng = np.random.default_rng(stable_seed(case['name'], spec['name'], 'feature_statistics'))
    ii = rng.choice(fit, min(len(fit), settings['standardization_max_images']), replace=False)
    if spec.get('aug', False):
        obs = public_observations(case['train_u'][ii], rng, _number_snr(spec['snr']), repetitions=1)
        nonce = np.zeros((len(ii), 1, 6), dtype=np.float32)
    else:
        obs, nonce = actual_observations(case['train_u'][ii], case,
                                         'attacks/' + spec['packet_domain'] + '/statistics',
                                         _number_snr(spec['snr']), repetitions=1)
    features = attack_features(obs, nonce, spec)
    flat = features.reshape(-1, features.shape[-1])
    if spec['kind'] in ('sets', 'expressive_sets'):
        return np.zeros(flat.shape[1], np.float32), np.ones(flat.shape[1], np.float32)
    mean = flat.mean(0, dtype=np.float64).astype(np.float32)
    scale = flat.std(0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-5] = 1.0
    return mean, scale


def _target_tensor(target, task, device, target_mean=None, target_scale=None):
    if task in ('classification', 'binary'):
        return torch.as_tensor(target, dtype=torch.long, device=device)
    if task == 'reconstruction':
        return torch.as_tensor(np.array(target, copy=True), device=device).float() / 255.0
    return (torch.as_tensor(target, dtype=torch.float32, device=device) - target_mean) / target_scale


def _validation(model, vf, targets, spec, settings, device, target_mean=None, target_scale=None):
    model.eval()
    n = len(targets)
    if not n:
        raise ValueError('Validation set is empty')
    loss_sum = 0.0
    metric_sum = 0.0
    correct = np.zeros(2, dtype=np.int64)
    counts = np.zeros(2, dtype=np.int64)
    bs = settings['validation_batch_size']
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = torch.as_tensor(vf[i:i + bs], device=device)
            yb = _target_tensor(targets[i:i + bs], spec['task'], device, target_mean, target_scale)
            pred = model(xb)
            if not torch.isfinite(pred).all():
                raise RuntimeError('Nonfinite validation prediction')
            if spec['task'] in ('classification', 'binary'):
                loss = F.cross_entropy(pred, yb)
                hit = pred.argmax(1).eq(yb)
                metric_sum += float(hit.sum())
                if spec['task'] == 'binary':
                    for cls in range(2):
                        counts[cls] += int(yb.eq(cls).sum())
                        correct[cls] += int((hit & yb.eq(cls)).sum())
            else:
                loss = F.mse_loss(pred, yb)
                if spec['task'] == 'reconstruction':
                    metric_sum += float(valid_ssim(pred, yb).sum())
                else:
                    metric_sum -= float((pred - yb).square().mean(1).sum())
            loss_sum += float(loss) * len(yb)
    score = metric_sum / n
    if not np.isfinite(loss_sum) or not np.isfinite(score):
        raise RuntimeError('Nonfinite validation metric')
    if spec['task'] == 'binary':
        if (counts == 0).any():
            raise ValueError('Binary validation must contain animals and vehicles')
        score = float(np.mean(correct / counts))
    return float(loss_sum / n), float(score)


def _get_cached(case, spec):
    obs, nonce = actual_observations(case['train_u'], case,
                                     'attacks/' + spec['packet_domain'] + '/training',
                                     _number_snr(spec['snr']), repetitions=1)
    if spec.get('control') == 'identity':
        rng = np.random.default_rng(stable_seed(case['name'], spec['packet_domain'], 'control_noise'))
        obs = case['train_u'][:, None, :].copy()
        if np.isfinite(_number_snr(spec['snr'])):
            obs += rng.normal(0, 10 ** (-_number_snr(spec['snr']) / 20), obs.shape).astype(np.float32)
    elif spec.get('control') == 'fixed_transform':
        from keyed_jscc.crypto_scrambler import CryptoScrambler
        params = CryptoScrambler(bytes.fromhex(case['key_hex']), case['train_u'].shape[1]).parameters(0)
        rng = np.random.default_rng(stable_seed(case['name'], spec['packet_domain'], 'control_noise'))
        obs = params.forward(case['train_u'])[:, None, :]
        if np.isfinite(_number_snr(spec['snr'])):
            obs += rng.normal(0, 10 ** (-_number_snr(spec['snr']) / 20), obs.shape).astype(np.float32)
    return obs, nonce


def fit_one(case, spec, cfg, settings, folder, device, smoke=False):
    folder = Path(folder)
    fit, val = _check_split(case)
    metric = _scoring_metric(spec)
    signature = fit_signature(cfg, extra={'module': source_signature(),
                                          'spec': spec, 'case': case['name'],
                                          'fit': _array_sha(fit), 'val': _array_sha(val),
                                          'train_u': _array_sha(case['train_u'])})
    _lock(folder, signature, {'case': case['name'], 'spec': spec,
                             'selection_metric': metric, 'test_used_for_fit_or_selection': False})
    finished = folder / 'fit_complete.json'
    if finished.exists():
        return json.loads(finished.read_text(encoding='utf-8'))
    torch_seed(spec['seed'])
    if not spec.get('aug', False):
        obs, nonce = _get_cached(case, spec)
        cached = attack_features(obs, nonce, spec)
        mean = cached[fit].reshape(-1, cached.shape[-1]).mean(0, dtype=np.float64).astype(np.float32)
        scale = cached[fit].reshape(-1, cached.shape[-1]).std(0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-5] = 1.0
        if spec['kind'] in ('sets', 'expressive_sets'):
            mean = np.zeros_like(mean)
            scale = np.ones_like(scale)
        cached = (cached - mean) / scale
        if spec.get('control'):
            val_features = {1: cached[val]}
        else:
            vo, vn = actual_observations(case['train_u'][val], case,
                                         'attacks/' + spec['packet_domain'] + '/validation',
                                         _number_snr(spec['snr']), repetitions=1)
            val_features = {1: (attack_features(vo, vn, spec) - mean) / scale}
    else:
        cached = None
        mean, scale = _stat_features(case, spec, cfg, settings)
        max_rep = max(spec.get('counts', [1]))
        vo, vn = actual_observations(case['train_u'][val], case,
                                     'attacks/' + spec['packet_domain'] + '/validation',
                                     _number_snr(spec['snr']), repetitions=max_rep)
        val_features = {}
        for count in spec.get('counts', [1]):
            val_features[count] = (attack_features(vo[:, :count], vn[:, :count], spec) - mean) / scale
        del vo, vn
    input_dim = len(mean)
    target = _task_target(case, spec, 'train')
    target_mean_np, target_scale_np = np.zeros(3, np.float32), np.ones(3, np.float32)
    if spec['task'] == 'attributes':
        target_mean_np = target[fit].mean(0, dtype=np.float64).astype(np.float32)
        target_scale_np = target[fit].std(0, dtype=np.float64).astype(np.float32)
        target_scale_np[target_scale_np < 1e-6] = 1
    target_mean = torch.as_tensor(target_mean_np, device=device)
    target_scale = torch.as_tensor(target_scale_np, device=device)
    model = build_model(spec, input_dim, settings).to(device)
    lr = 0.0002 if spec['task'] == 'reconstruction' else 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    epochs = spec['epochs']
    history = []
    best_loss = float('inf')
    best_metric = -float('inf')
    start_epoch = 1
    counts = spec.get('counts', [1])
    best_by_count = {str(count): -float('inf') for count in counts}
    last = folder / 'resume.pt'
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state['signature'] != signature:
            raise RuntimeError('Resume signature mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history = state['history']
        best_loss, best_metric = state['best_loss'], state['best_metric']
        best_by_count = state['best_by_count']
        start_epoch = state['epoch'] + 1
    target_val = target[val]
    class_weights = None
    if spec['task'] == 'binary':
        class_counts = np.bincount(target[fit], minlength=2)
        if not class_counts.all():
            raise ValueError('Binary fitting must contain animals and vehicles')
        class_weights = torch.as_tensor(len(fit) / (2.0 * class_counts), device=device, dtype=torch.float32)
    snapshots = set(spec.get('budget_snapshots', []))
    batch_size = settings['batch_size']
    for epoch in range(start_epoch, epochs + 1):
        started = time.monotonic()
        epoch_seed = stable_seed(case['name'], spec['name'], spec['seed'], epoch)
        torch_seed(epoch_seed)
        rng = np.random.default_rng(epoch_seed)
        order = rng.permutation(fit)
        model.train()
        loss_sum, seen, fresh_packets, updates = 0.0, 0, 0, 0
        for k in range(0, len(order), batch_size):
            ii = order[k:k + batch_size]
            repetitions = int(rng.choice(counts))
            if spec.get('aug', False):
                simulated = public_observations(case['train_u'][ii], rng,
                                                 _number_snr(spec['snr']), repetitions=repetitions)
                ns = np.zeros((len(ii), repetitions, 6), np.float32)
                fb = (attack_features(simulated, ns, spec) - mean) / scale
                fresh_packets += len(ii) * repetitions
            else:
                fb = cached[ii]
            xb = torch.as_tensor(fb, device=device)
            yb = _target_tensor(target[ii], spec['task'], device, target_mean, target_scale)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = F.cross_entropy(prediction, yb, weight=class_weights) if spec['task'] in ('classification', 'binary') else F.mse_loss(prediction, yb)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss at ' + spec['name'])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ii)
            seen += len(ii)
            updates += 1
        losses, metrics, detail = [], [], {}
        for count in counts:
            value_loss, value_metric = _validation(model, val_features[count], target_val, spec,
                                                   settings, device, target_mean, target_scale)
            losses.append(value_loss)
            metrics.append(value_metric)
            detail[str(count)] = {'loss': value_loss, metric: value_metric}
        val_loss = float(np.mean(losses))
        val_metric = float(np.mean(metrics))
        row = {'epoch': epoch, 'fitting_loss': loss_sum / seen,
               'validation_loss': val_loss, 'validation_' + metric: val_metric,
               'validation_by_count': detail, 'optimizer_updates': updates,
               'fitting_images': seen, 'fresh_simulated_packets': fresh_packets,
               'seconds': time.monotonic() - started, 'epoch_seed': epoch_seed}
        history.append(row)
        selected_state = {
            'state_dict': _torch_cpu_state(model), 'spec': spec,
            'input_dim': input_dim, 'settings': settings,
            'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale),
            'target_mean': torch.from_numpy(target_mean_np),
            'target_scale': torch.from_numpy(target_scale_np),
            'epoch': epoch, 'validation_loss': val_loss, 'validation_metric': val_metric,
            'validation_by_count': detail, 'selection_metric': metric,
            'signature': signature, 'test_used_for_selection': False,
        }
        if val_loss < best_loss:
            best_loss = val_loss
            save_torch(folder / 'best_loss.pt', selected_state)
        if val_metric > best_metric or (val_metric == best_metric and val_loss == best_loss):
            best_metric = val_metric
            save_torch(folder / 'best_metric.pt', selected_state)
        if len(counts) > 1:
            for count in counts:
                count_metric = detail[str(count)][metric]
                if count_metric > best_by_count[str(count)]:
                    best_by_count[str(count)] = count_metric
                    save_torch(folder / ('best_metric_count' + str(count) + '.pt'), selected_state)
        if epoch in snapshots:
            save_torch(folder / ('epoch_' + str(epoch) + '.pt'), selected_state)
            for criterion in ('loss', 'metric'):
                current = torch.load(folder / ('best_' + criterion + '.pt'), map_location='cpu', weights_only=False)
                save_torch(folder / ('budget_' + str(epoch) + '_best_' + criterion + '.pt'), current)
        save_torch(last, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'history': history, 'best_loss': best_loss, 'best_metric': best_metric,
                          'best_by_count': best_by_count, 'epoch': epoch, 'signature': signature,
                          'torch_rng_state': torch.get_rng_state(),
                          'cuda_rng_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                          'numpy_rng_state': rng.bit_generator.state,
                          'resume_policy': 'epoch boundary with deterministic per-epoch Torch/NumPy seeds'})
        write_json(folder / 'history.json', history)
        print('  attacker ' + spec['name'] + ' epoch ' + str(epoch) + '/' + str(epochs) +
              ': validation loss=' + format(val_loss, '.6f') + ' ' + metric + '=' +
              format(val_metric, '.6f') + ' (' + format(row['seconds'], '.1f') + ' s)', flush=True)
    loss_selected = torch.load(folder / 'best_loss.pt', map_location='cpu', weights_only=False)
    metric_selected = torch.load(folder / 'best_metric.pt', map_location='cpu', weights_only=False)
    report = {'spec': spec, 'completed_epochs': len(history),
              'fitting_n': len(fit), 'validation_n': len(val), 'batch_size': batch_size,
              'training_loss': 'class-weighted cross entropy' if spec['task'] == 'binary' else
                               ('cross entropy' if spec['task'] == 'classification' else
                                ('normalized attribute MSE' if spec['task'] == 'attributes' else 'pixel MSE')),
              'selection': {'metric': metric, 'metric_epoch': metric_selected['epoch'],
                            'metric_value': metric_selected['validation_metric'],
                            'loss_epoch': loss_selected['epoch'],
                            'loss_value': loss_selected['validation_loss']},
              'optimizer_updates': sum(r['optimizer_updates'] for r in history),
              'fresh_simulated_packets': sum(r['fresh_simulated_packets'] for r in history),
              'training_seconds': sum(r['seconds'] for r in history),
              'test_used_for_fit_or_selection': False,
              'observation_generator': 'independent public transform family' if spec.get('aug') else
                                       ('stationary control' if spec.get('control') else 'recorded-key unique nonces'),
              'validation_generator': 'actual recorded key; disjoint fitting/validation nonce domains',
              'metric_aggregation_for_variable_count': 'unweighted mean over predetermined counts' if len(counts) > 1 else None,
              'signature': signature}
    write_json(finished, report)
    del model, optimizer, cached, val_features
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return report


def predict_checkpoint(path, observations, nonce, device, batch=128):
    state = torch.load(path, map_location='cpu', weights_only=False)
    spec, settings = state['spec'], state['settings']
    model = build_model(spec, state['input_dim'], settings).to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    pred = []
    mean, scale = state['mean'].numpy(), state['scale'].numpy()
    with torch.no_grad():
        for i in range(0, len(observations), batch):
            features = attack_features(observations[i:i + batch], nonce[i:i + batch], spec)
            xb = torch.as_tensor((features - mean) / scale, device=device)
            values = model(xb)
            if spec['task'] in ('classification', 'binary'):
                values = values.softmax(1)
            elif spec['task'] == 'attributes':
                values = values * state['target_scale'].to(device) + state['target_mean'].to(device)
            if not torch.isfinite(values).all():
                raise RuntimeError('Nonfinite held-out prediction')
            pred.append(values.cpu().numpy())
    del model
    return np.concatenate(pred).astype(np.float32), state


def _class_summary(labels, probabilities):
    labels = np.asarray(labels, np.int64)
    p = np.asarray(probabilities, np.float64)
    if p.ndim != 2 or len(p) != len(labels) or not np.isfinite(p).all():
        raise ValueError('Invalid class probability array')
    prediction = p.argmax(1)
    classes = p.shape[1]
    confusion = np.bincount(classes * labels + prediction, minlength=classes * classes).reshape(classes, classes)
    counts = confusion.sum(1)
    recall = np.diag(confusion) / np.maximum(counts, 1)
    if classes == 10:
        summary = classification_rows(labels, p)
        if isinstance(summary, dict) and isinstance(summary.get('accuracy'), (float, int, np.floating)):
            summary = {key: value for key, value in summary.items() if np.isscalar(value) or value is None}
        else:
            summary = {}
    else:
        summary = {}
    summary.update(n=len(labels), accuracy=float(np.mean(prediction == labels)),
                   balanced_accuracy=float(recall[counts > 0].mean()),
                   cross_entropy=float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
                   confusion=confusion.tolist(), recall=recall.tolist())
    if classes == 2:
        scores = p[:, 1]
        order = np.argsort(scores, kind='stable')
        ranks = np.empty(len(scores), np.float64)
        sorted_scores = scores[order]
        starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
        ends = np.r_[starts[1:], len(scores)]
        for start, end in zip(starts, ends):
            ranks[order[start:end]] = (start + end + 1) / 2
        positive = labels == 1
        n1, n0 = int(positive.sum()), int((~positive).sum())
        summary['auroc'] = float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None
    return summary


def evaluate_one(case, checkpoint, folder, cfg, settings, device, count=1,
                 key_index=0, draw=0, snr=None, save_predictions=True):
    folder = Path(folder)
    completed = folder / 'metrics.json'
    if completed.exists():
        return json.loads(completed.read_text(encoding='utf-8'))
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    spec = state['spec']
    use_snr = spec['snr'] if snr is None else snr
    obs, nonce = actual_observations(case['test_u'], case,
                                     'attacks/' + spec['packet_domain'] + '/test',
                                     _number_snr(use_snr), draw=draw, key_index=key_index,
                                     repetitions=count)
    if spec.get('control'):
        rng = np.random.default_rng(stable_seed(case['name'], spec['packet_domain'], 'control_test_noise', key_index, draw))
        if spec['control'] == 'fixed_transform':
            from keyed_jscc.crypto_scrambler import CryptoScrambler
            params = CryptoScrambler(bytes.fromhex(case['key_hex']), case['test_u'].shape[1]).parameters(0)
            obs = params.forward(case['test_u'])[:, None, :]
        else:
            obs = case['test_u'][:, None, :].copy()
        if np.isfinite(_number_snr(use_snr)):
            obs += rng.normal(0, 10 ** (-_number_snr(use_snr) / 20), obs.shape).astype(np.float32)
    prediction, state = predict_checkpoint(checkpoint, obs, nonce, device,
                                           batch=settings['validation_batch_size'])
    truth = _task_target(case, spec, 'test')
    arrays = {'test_index': np.arange(len(truth), dtype=np.int64),
              'selected_epoch': np.asarray(state['epoch'], np.int64)}
    if spec['task'] in ('classification', 'binary'):
        summary = _class_summary(truth, prediction)
        arrays.update(y_true=truth, probabilities=prediction, predicted_label=prediction.argmax(1),
                      correct=prediction.argmax(1) == truth,
                      negative_log_probability=-np.log(np.clip(prediction[np.arange(len(truth)), truth], 1e-12, 1)))
    elif spec['task'] == 'reconstruction':
        rows = image_rows(prediction, truth, device=device)
        summary = image_summary(rows)
        arrays.update({key: value for key, value in rows.items() if np.asarray(value).shape == (len(truth),)})
        if save_predictions and settings['save_full_predictions']:
            arrays['predictions_float32'] = prediction
        else:
            arrays['first_predictions_float32'] = prediction[:16]
    else:
        train_target = image_attributes(case['train_images'])
        prior = train_target[case['fit']].mean(0, dtype=np.float64)
        errors = (prediction - truth).astype(np.float64) ** 2
        baseline = (truth - prior).astype(np.float64) ** 2
        target_variance = np.square(truth - truth.mean(0, dtype=np.float64)).sum(0)
        r2 = 1 - errors.sum(0) / np.maximum(target_variance, 1e-15)
        summary = {'n': len(truth), 'attribute_names': ['mean_luminance', 'mean_chroma_range', 'finite_difference_luminance_energy'],
                   'mse': errors.mean(0).tolist(), 'prior_mse': baseline.mean(0).tolist(),
                   'r_squared': r2.tolist(), 'mse_improvement_over_training_mean': (baseline.mean(0) - errors.mean(0)).tolist(),
                   'fitting_only_prior_mean': prior.tolist()}
        arrays.update(y_true=truth, predictions=prediction, squared_error=errors,
                      prior_prediction=np.broadcast_to(prior, truth.shape), prior_squared_error=baseline)
    arrays['test_labels'] = np.asarray(case['test_labels'])
    folder.mkdir(parents=True, exist_ok=True)
    save_npz(folder / 'per_image.npz', **arrays)
    summary.update(spec_name=spec['name'], task=spec['task'], selected_epoch=state['epoch'],
                   selection_metric=state['selection_metric'], count=count, snr_eve=use_snr,
                   key_index=key_index, noise_draw=draw, selection_completed_before_test=True,
                   test_used_for_fit_or_selection=False)
    write_json(completed, summary)
    return summary


def _fit_and_evaluate(case, spec, cfg, settings, folder, device, smoke):
    report = fit_one(case, spec, cfg, settings, folder, device, smoke)
    results = {}
    for criterion in ('loss', 'metric'):
        results[criterion] = evaluate_one(case, folder / ('best_' + criterion + '.pt'),
                                          folder / ('test_selected_' + criterion), cfg,
                                          settings, device)
    return {'fit': report, 'holdout': results}


def _strong_specs(settings, seed, prefix, epochs, snr=20.0):
    shared = {'nonce': False, 'aug': True, 'seed': seed, 'epochs': epochs,
              'snr': snr, 'packet_domain': prefix}
    return [dict(shared, name=prefix + '_cnn256_seed' + str(seed), kind='cnn',
                 channels=settings['cnn_channels'], task='reconstruction', view='raw_fusion'),
            dict(shared, name=prefix + '_invariant_seed' + str(seed), kind='expressive_class',
                 task='classification', view='fusion'),
            dict(shared, name=prefix + '_deepsets_seed' + str(seed), kind='expressive_sets',
                 task='classification', view='raw')]


def _prior_baselines(case, settings, folder, device):
    folder = Path(folder)
    fit, val = _check_split(case)
    mean_image = np.asarray(case['train_images'][fit], dtype=np.float64).mean(0).astype(np.float32) / 255
    mean_rgb = mean_image.mean((1, 2), keepdims=True)
    candidates = {'fitting_mean_image': mean_image,
                  'fitting_mean_rgb': np.broadcast_to(mean_rgb, mean_image.shape).copy()}
    for gray in np.linspace(0.1, 0.9, 9):
        candidates['constant_gray_' + format(gray, '.1f')] = np.full_like(mean_image, gray)
    validation = {}
    for name, image in candidates.items():
        prediction = np.broadcast_to(image, (len(val),) + image.shape).copy()
        validation[name] = image_summary(image_rows(prediction, case['train_images'][val], device=device))
    loss_name = min(validation, key=lambda name: (validation[name]['mse'], name))
    metric_name = min(validation, key=lambda name: (-validation[name]['mean_ssim_valid'],
                                                    validation[name]['mse'], name))
    selection = {'by_validation_mse': loss_name, 'by_validation_ssim_valid': metric_name,
                 'validation': validation, 'test_used_for_selection': False}
    write_json(folder / 'locked_prior_selection.json', selection)
    holdout = {}
    for name in sorted(set([loss_name, metric_name])):
        image = candidates[name]
        prediction = np.broadcast_to(image, (len(case['test_images']),) + image.shape).copy()
        rows = image_rows(prediction, case['test_images'], device=device)
        holdout[name] = image_summary(rows)
        save_npz(folder / (name + '_per_image.npz'), **rows,
                 image_independent_prediction=image)
    frequencies = np.bincount(np.asarray(case['train_labels'])[fit], minlength=10).astype(np.float64)
    frequencies /= frequencies.sum()
    probabilities = np.broadcast_to(frequencies, (len(case['test_labels']), 10)).copy()
    holdout['fitting_class_prior'] = _class_summary(case['test_labels'], probabilities)
    save_npz(folder / 'fitting_class_prior_per_image.npz',
             y_true=case['test_labels'], probabilities=probabilities)
    result = {'selection': selection, 'holdout': holdout,
              'interpretation': 'image-independent fitting-set estimates and fixed constants; validation-selected before test',
              'test_used_for_fit_or_selection': False}
    write_json(folder / 'summary.json', result)
    return result


def _metric_alignment(case, cfg, settings, folder, device, smoke):
    candidates = primary_specs(settings, smoke)
    fits = {}
    for spec in candidates:
        spec.update(snr=20.0, packet_domain='primary')
        fits[spec['name']] = fit_one(case, spec, cfg, settings, folder / spec['name'], device, smoke)
    locked = {}
    for task in ('classification', 'reconstruction'):
        group = [name for name, report in fits.items() if report['spec']['task'] == task]
        if not group:
            continue
        metric_best = min(group, key=lambda n: (-fits[n]['selection']['metric_value'],
                                                fits[n]['selection']['metric_epoch'], n))
        loss_best = min(group, key=lambda n: (fits[n]['selection']['loss_value'],
                                              fits[n]['selection']['loss_epoch'], n))
        locked[task] = {'metric_aligned': metric_best, 'loss_aligned': loss_best,
                        'metric_name': _scoring_metric(fits[metric_best]['spec'])}
    write_json(folder / 'locked_suite_selection.json', {
        'selections': locked, 'selection_source': 'training-derived validation only',
        'all_fits_completed_before_test': True, 'test_used_for_selection': False,
    })
    results = {}
    for name in fits:
        results[name] = {}
        for criterion in ('loss', 'metric'):
            results[name][criterion] = evaluate_one(case, folder / name / ('best_' + criterion + '.pt'),
                                                    folder / name / ('test_selected_' + criterion),
                                                    cfg, settings, device)
    control_results = {}
    for control in settings['controls']:
        for task, kind in [('classification', 'expressive_class'), ('reconstruction', 'cnn')]:
            spec = dict(name=control + '_' + task, kind=kind, task=task,
                        view='raw', nonce=False, aug=False, seed=891, epochs=settings['control_fit_epochs'],
                        snr=20.0, packet_domain='control/' + control, control=control,
                        channels=min(settings['cnn_channels'], 64))
            control_results[spec['name']] = _fit_and_evaluate(case, spec, cfg, settings,
                                                             folder / 'controls' / spec['name'], device, smoke)
    priors = _prior_baselines(case, settings, folder / 'image_independent_priors', device)
    result = {'candidate_count': len(candidates), 'fits': fits, 'selections': locked,
              'image_independent_priors': priors,
              'holdout': results, 'stationary_controls': control_results,
              'primary_ssim': 'valid-window SSIM', 'secondary_ssim': 'legacy zero-padded SSIM',
              'test_used_for_fit_or_selection': False}
    write_json(folder / 'summary.json', result)
    return result


def _budget(case, cfg, settings, folder, device, smoke):
    records = {}
    budgets = settings['budget_epochs']
    for seed in settings['budget_seeds']:
        for spec in _strong_specs(settings, seed, 'budget', max(budgets)):
            spec['budget_snapshots'] = budgets
            destination = folder / spec['name']
            fit_report = fit_one(case, spec, cfg, settings, destination, device, smoke)
            results = {}
            for budget in budgets:
                results[str(budget)] = {}
                for criterion in ('loss', 'metric'):
                    checkpoint = destination / ('budget_' + str(budget) + '_best_' + criterion + '.pt')
                    results[str(budget)][criterion] = evaluate_one(
                        case, checkpoint, destination / ('test_budget' + str(budget) + '_' + criterion),
                        cfg, settings, device, save_predictions=(budget == max(budgets)))
            records[spec['name']] = {'fit': fit_report, 'budget_results': results}
    result = {'runs': records, 'budget_epochs': budgets,
              'training_seeds': settings['budget_seeds'],
              'comparison': 'same optimization trajectory; best validation checkpoint available within each budget',
              'test_used_for_fit_or_selection': False}
    write_json(folder / 'summary.json', result)
    return result


def _high_snr(case, cfg, settings, folder, device, smoke):
    records = {}
    for snr in settings['high_snr_values']:
        prefix = 'adaptive_' + _snr_name(snr)
        for seed in settings['high_snr_seeds']:
            for spec in _strong_specs(settings, seed, prefix, settings['high_snr_epochs'], snr):
                destination = folder / spec['name']
                fit_report = fit_one(case, spec, cfg, settings, destination, device, smoke)
                results = {}
                draws = 1 if snr == 'noiseless' else settings['high_snr_noise_draws']
                for key_index in range(settings['high_snr_keys']):
                    for draw in range(draws):
                        name = 'key' + str(key_index) + '_draw' + str(draw)
                        results[name] = evaluate_one(
                            case, destination / 'best_metric.pt', destination / ('test_' + name),
                            cfg, settings, device, key_index=key_index, draw=draw,
                            save_predictions=(key_index == 0 and draw == 0))
                records[spec['name']] = {'fit': fit_report, 'holdout': results}
    result = {'runs': records, 'snr_values': settings['high_snr_values'],
              'adaptation': 'independent fresh fitting from random initialization at every SNR',
              'selection_key_index': 0, 'test_keys': settings['high_snr_keys'],
              'test_used_for_fit_or_selection': False}
    write_json(folder / 'summary.json', result)
    return result


def _multi_observation(case, cfg, settings, folder, device, smoke):
    records = {}
    counts = settings['multi_observation_counts']
    for seed in settings['multi_observation_seeds']:
        for task in ('classification', 'reconstruction'):
            spec = dict(name='packet_set_' + task + '_seed' + str(seed), task=task,
                        kind='packet_set', view='raw_fusion', nonce=False, aug=True,
                        seed=seed, epochs=settings['multi_observation_epochs'], counts=counts,
                        snr=20.0, packet_domain='same_image_independent_packets')
            destination = folder / spec['name']
            report = fit_one(case, spec, cfg, settings, destination, device, smoke)
            heldout = {}
            for count in counts:
                checkpoint = destination / ('best_metric_count' + str(count) + '.pt') if len(counts) > 1 else destination / 'best_metric.pt'
                heldout[str(count)] = evaluate_one(case, checkpoint, destination / ('test_count' + str(count)),
                                                   cfg, settings, device, count=count)
            records[spec['name']] = {'fit': report, 'holdout': heldout}
    result = {'runs': records, 'counts': counts,
              'sampling_unit': 'one source image, repeated under independently sampled fresh transforms and noise',
              'training': 'repetition count sampled uniformly per minibatch from predetermined counts',
              'selection': 'separate validation-best checkpoint for each predetermined count',
              'aggregation': 'shared packet encoder followed by permutation-invariant mean/max/std pooling',
              'test_used_for_fit_or_selection': False}
    write_json(folder / 'summary.json', result)
    return result


def _attributes(case, cfg, settings, folder, device, smoke):
    records = {}
    for seed in settings['attribute_seeds']:
        for task in ('binary', 'attributes'):
            spec = dict(name='direct_' + task + '_seed' + str(seed), task=task,
                        kind=task, view='raw_fusion', nonce=False, aug=True,
                        seed=seed, epochs=settings['attribute_epochs'], snr=20.0,
                        packet_domain='additional_inference')
            records[spec['name']] = _fit_and_evaluate(case, spec, cfg, settings,
                                                       folder / spec['name'], device, smoke)
    result = {'runs': records,
              'binary_labels': {'0': 'animal: bird, cat, deer, dog, frog, horse',
                                '1': 'vehicle: airplane, automobile, ship, truck'},
              'binary_training': 'direct class-weighted binary cross-entropy; not a relabelled ten-class prediction',
              'binary_selection': 'validation balanced accuracy',
              'attributes': {'mean_luminance': 'mean(0.299 R + 0.587 G + 0.114 B)',
                             'mean_chroma_range': 'mean(max(R,G,B)-min(R,G,B))',
                             'finite_difference_luminance_energy': '0.5*(mean(vertical difference squared)+mean(horizontal difference squared))'},
              'attribute_scaling_and_priors': 'estimated from fitting images only',
              'test_used_for_fit_or_selection': False}
    write_json(folder / 'summary.json', result)
    return result


def run(case, cfg, outdir, device, smoke=False):
    outdir = Path(outdir)
    device = torch.device(device)
    settings = _settings(cfg, smoke)
    fit, val = _check_split(case)
    signature = fit_signature(cfg, extra={
        'stage': 'attacks', 'module_sha256': source_signature(),
        'case': case['name'], 'fit_sha256': _array_sha(fit), 'val_sha256': _array_sha(val),
        'train_latents_sha256': _array_sha(case['train_u']),
        'train_labels_sha256': _array_sha(case['train_labels']),
        'key_sha256': hashlib.sha256(case['key_hex'].encode()).hexdigest(), 'smoke': smoke,
    })
    _lock(outdir, signature, {'case': case['name'], 'settings': settings,
                             'fitting_n': len(fit), 'validation_n': len(val),
                             'test_used_for_fit_or_selection': False})
    extension_allowed = '*' in settings['extended_cases'] or case['name'] in settings['extended_cases']
    functions = {'metric_alignment': _metric_alignment, 'budget': _budget,
                 'high_snr': _high_snr, 'multi_observation': _multi_observation,
                 'attributes': _attributes}
    coverage = {'case': case['name'], 'requested_tasks': settings['tasks'],
                'completed': [], 'not_requested': [], 'not_scheduled_for_case': [],
                'results': {}, 'test_used_for_fit_or_selection': False,
                'smoke_test': smoke, 'scientific_results': not smoke,
                'signature': signature}
    for name, fn in functions.items():
        if name not in settings['tasks']:
            coverage['not_requested'].append(name)
        elif name != 'metric_alignment' and not extension_allowed:
            coverage['not_scheduled_for_case'].append(name)
        else:
            print('Attacker stage ' + case['name'] + ': ' + name, flush=True)
            result = fn(case, cfg, settings, outdir / name, device, smoke)
            coverage['results'][name] = result
            coverage['completed'].append(name)
            write_json(outdir / 'coverage_summary.json', coverage)
    coverage['completed_all_tasks_scheduled_for_case'] = (
        set(coverage['completed']) == set(settings['tasks']) - set(coverage['not_scheduled_for_case']))
    write_json(outdir / 'coverage_summary.json', coverage)
    return coverage
