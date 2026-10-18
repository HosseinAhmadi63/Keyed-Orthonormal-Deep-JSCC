from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, math, os, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from attack_models import ResidualMLPClassifier, DeepSetsClassifier, ConvReconstructor
from public_sampling import invariant_features, sample_packets

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()

def wilson(correct, n):
    z = 1.959963984540054
    p = correct / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0, c - r), min(1, c + r)]

def class_statistics(y, prob):
    pred = prob.argmax(1)
    cm = np.bincount(10 * y + pred, minlength=100).reshape(10, 10)
    n = cm.sum(1)
    hit = np.diag(cm)
    recall = hit / n
    return {'n': len(y), 'correct': int(hit.sum()), 'accuracy': float((pred == y).mean()), 'accuracy_wilson_95': wilson(int(hit.sum()), len(y)), 'balanced_accuracy': float(recall.mean()), 'confusion_matrix': cm.tolist(), 'cross_entropy': float(-np.log(np.maximum(prob[np.arange(len(y)), y], 1e-12)).mean()), 'predictive_entropy': float(-(prob * np.log(np.maximum(prob, 1e-12))).sum(1).mean()), 'per_class': [{'class': k, 'n': int(n[k]), 'correct': int(hit[k]), 'recall': float(recall[k]), 'recall_wilson_95': wilson(int(hit[k]), int(n[k]))} for k in range(10)]}

def specifications(suite, seed, channels=32):
    out = []
    if suite in ('classifiers', 'all', 'positive_control'):
        cases = [('cached_nonce', False, True), ('cached_no_nonce', False, False), ('surrogate_nonce', True, True), ('surrogate_no_nonce', True, False)] if suite != 'positive_control' else [('cached_no_nonce', False, False)]
        for label, aug, nonce in cases:
            out.append({'name': 'invariant_mlp_' + label, 'task': 'class', 'kind': 'invariant_mlp', 'view': 'fusion', 'nonce': nonce, 'surrogate': aug, 'seed': seed, 'max_epochs': 60, 'min_epochs': 20, 'patience': 12, 'max_seconds': 480, 'batch_size': 512, 'lr': 0.0003, 'weight_decay': 0.0001, 'width': 256})
        if suite != 'positive_control':
            out.append({'name': 'deepsets_surrogate_no_nonce', 'task': 'class', 'kind': 'deepsets', 'view': 'raw', 'nonce': False, 'surrogate': True, 'seed': seed, 'max_epochs': 60, 'min_epochs': 20, 'patience': 12, 'max_seconds': 600, 'batch_size': 256, 'lr': 0.0003, 'weight_decay': 0.0001, 'width': 16})
    if suite in ('reconstructors', 'all'):
        for label, aug, nonce in [('cached_nonce', False, True), ('cached_no_nonce', False, False), ('surrogate_no_nonce', True, False)]:
            out.append({'name': 'conv_' + label, 'task': 'recon', 'kind': 'conv', 'view': 'raw_fusion', 'nonce': nonce, 'surrogate': aug, 'seed': seed, 'max_epochs': 30, 'min_epochs': 15, 'patience': 10, 'max_seconds': 1200, 'batch_size': 64, 'lr': 0.0002, 'weight_decay': 0.0001, 'channels': channels})
    return out

def build_model(spec, input_dim, nonce_dim):
    if spec['kind'] == 'invariant_mlp':
        return ResidualMLPClassifier(input_dim, spec['width'])
    if spec['kind'] == 'deepsets':
        return DeepSetsClassifier(nonce_dim if spec['nonce'] else 0, spec['width'])
    if spec['kind'] == 'conv':
        return ConvReconstructor(input_dim, spec['channels'])
    raise ValueError(spec['kind'])

def concatenate_features(obs, nonce, spec):
    features = invariant_features(obs, spec['view'])
    header = nonce if spec['nonce'] else np.zeros_like(nonce)
    return np.concatenate([features, header], axis=1)

@torch.no_grad()
def validation(model, x, nonce, labels, images, spec, mean, scale, batch_size=256):
    model.eval()
    total = 0.0
    count = 0
    correct = 0
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size]))
        nt = torch.from_numpy(np.ascontiguousarray(nonce[start:start + batch_size]))
        pred = model(batch, nt)
        if spec['task'] == 'class':
            truth = torch.from_numpy(labels[start:start + batch_size])
            loss = nn.functional.cross_entropy(pred, truth, reduction='sum')
            correct += int((pred.argmax(1) == truth).sum())
        else:
            truth = torch.from_numpy(images[start:start + batch_size].copy()).float() / 255.0
            loss = (pred - truth).square().mean((1, 2, 3)).sum()
        total += float(loss)
        count += len(batch)
    return {'loss': total / count, 'accuracy': correct / count if spec['task'] == 'class' else None}

def fit_one(spec, data, split, output, protocol, source_hashes):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'model.pt').exists() and (output / 'fit_metrics.json').exists():
        return json.loads((output / 'fit_metrics.json').read_text())
    fit, val = (split['fit'], split['validation'])
    torch.manual_seed(spec['seed'])
    rng = np.random.default_rng(spec['seed'])
    deep = spec['kind'] == 'deepsets'
    features = data['obs'] if deep else concatenate_features(data['obs'], data['nonce'], spec)
    if deep:
        mean = np.zeros(features.shape[1], np.float32)
        scale = np.ones(features.shape[1], np.float32)
    else:
        mean = features[fit].mean(0, dtype=np.float64).astype(np.float32)
        scale = features[fit].std(0, dtype=np.float64).astype(np.float32)
        scale = np.where(scale > 1e-05, scale, 1).astype(np.float32)
    xv = np.ascontiguousarray((features[val] - mean) / scale)
    nv = np.ascontiguousarray(data['nonce'][val])
    yv = data['labels'][val]
    iv = data['images'][val] if spec['task'] == 'recon' else None
    xf = None if spec['surrogate'] else torch.from_numpy(np.ascontiguousarray((features[fit] - mean) / scale))
    nf = torch.from_numpy(np.ascontiguousarray(data['nonce'][fit]))
    yf = torch.from_numpy(data['labels'][fit])
    imf = data['images'][fit] if spec['task'] == 'recon' else None
    ufit = data['latents'][fit]
    nonce_pool = data['nonce'][fit]
    model = build_model(spec, features.shape[1], data['nonce'].shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, threshold=1e-05, min_lr=1e-06)
    history = []
    best_loss = math.inf
    best_score = None
    best_state = None
    best_epoch = None
    stale = 0
    beg = time.monotonic()
    stop = 'maximum_epochs'
    sim_counter = 65536
    for epoch in range(1, spec['max_epochs'] + 1):
        model.train()
        order = rng.permutation(len(fit))
        loss_sum = 0.0
        seen = 0
        updates = 0
        epoch_start = time.monotonic()
        for start in range(0, len(order), spec['batch_size']):
            ii = order[start:start + spec['batch_size']]
            if spec['surrogate']:
                family = 'permutation_only' if protocol == 'permutation_only' else 'full'
                obs = sample_packets(ufit[ii], rng, 20.0, family)
                counters = np.arange(sim_counter, sim_counter + len(ii), dtype=np.uint64)
                sim_counter += len(ii)
                nonce = np.broadcast_to(nonce_pool[0], (len(ii), nonce_pool.shape[1])).copy()
                nonce[:, -2] = (counters >> 16 & 65535).astype(np.float32) / 65535.0
                nonce[:, -1] = (counters & 65535).astype(np.float32) / 65535.0
                xx = obs if deep else concatenate_features(obs, nonce, spec)
                xx = torch.from_numpy(np.ascontiguousarray((xx - mean) / scale))
                nnc = torch.from_numpy(np.ascontiguousarray(nonce))
            else:
                xx = xf[ii]
                nnc = nf[ii]
            optimizer.zero_grad(set_to_none=True)
            pred = model(xx, nnc)
            truth = yf[ii] if spec['task'] == 'class' else torch.from_numpy(imf[ii].copy()).float() / 255.0
            loss = nn.functional.cross_entropy(pred, truth) if spec['task'] == 'class' else (pred - truth).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0 if spec['task'] == 'class' else 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ii)
            seen += len(ii)
            updates += 1
        vm = validation(model, xv, nv, yv, iv, spec, mean, scale)
        if not math.isfinite(vm['loss']):
            raise RuntimeError('Nonfinite validation loss')
        score = (vm['accuracy'], -vm['loss']) if spec['task'] == 'class' else (-vm['loss'],)
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if vm['loss'] < best_loss - 1e-07:
            best_loss = vm['loss']
            stale = 0
        else:
            stale += 1
        scheduler.step(vm['loss'])
        row = {'epoch': epoch, 'fitting_loss': loss_sum / seen, 'validation_loss': vm['loss'], 'validation_accuracy': vm['accuracy'], 'images_seen': seen, 'updates': updates, 'new_surrogate_packets': seen if spec['surrogate'] else 0, 'learning_rate': optimizer.param_groups[0]['lr'], 'epoch_seconds': time.monotonic() - epoch_start, 'total_seconds': time.monotonic() - beg}
        history.append(row)
        write_json(output / 'history.json', history)
        write_json(output / 'progress.json', {'specification': spec, 'history': history, 'selected_validation_epoch': best_epoch, 'minimum_validation_loss': best_loss})
        print(f"{spec['name']} epoch {epoch}: train={row['fitting_loss']:.7f} val={vm['loss']:.7f} acc={vm['accuracy']} seconds={row['epoch_seconds']:.2f}", flush=True)
        if epoch >= spec['min_epochs'] and stale >= spec['patience']:
            stop = 'validation_loss_plateau'
            break
        if time.monotonic() - beg >= spec['max_seconds']:
            stop = 'soft_time_budget'
            break
    checkpoint = {'state_dict': best_state, 'specification': spec, 'input_dim': features.shape[1], 'nonce_dim': data['nonce'].shape[1], 'feature_mean': torch.from_numpy(mean), 'feature_scale': torch.from_numpy(scale), 'selected_validation_epoch': best_epoch, 'selected_validation_score': list(best_score), 'source_sha256': source_hashes, 'protocol': protocol}
    torch.save(checkpoint, output / 'model.pt')
    report = {'specification': spec, 'parameter_count': sum((p.numel() for p in model.parameters())), 'selected_validation_epoch': best_epoch, 'selected_validation_accuracy': best_score[0] if spec['task'] == 'class' else None, 'selected_validation_loss': -best_score[1] if spec['task'] == 'class' else -best_score[0], 'epochs_completed': len(history), 'updates': sum((h['updates'] for h in history)), 'fitting_presentations': sum((h['images_seen'] for h in history)), 'stop_reason': stop, 'elapsed_seconds': time.monotonic() - beg, 'source_sha256': source_hashes, 'checkpoint_sha256': file_hash(output / 'model.pt'), 'fit_indices_sha256': array_hash(fit), 'validation_indices_sha256': array_hash(val), 'history': history, 'holdout_used_for_fitting_or_selection': False}
    write_json(output / 'fit_metrics.json', report)
    return report

class LoadedAttack:

    def __init__(self, path):
        self.path = Path(path)
        c = torch.load(path, map_location='cpu', weights_only=True)
        self.spec = c['specification']
        self.task = self.spec['task']
        self.mean = c['feature_mean'].numpy()
        self.scale = c['feature_scale'].numpy()
        self.model = build_model(self.spec, c['input_dim'], c['nonce_dim']).eval()
        self.model.load_state_dict(c['state_dict'])

    @torch.inference_mode()
    def __call__(self, observations, nonce_features, batch_size=128):
        chunks = []
        for start in range(0, len(observations), batch_size):
            obs = observations[start:start + batch_size]
            nonce = nonce_features[start:start + batch_size]
            x = obs if self.spec['kind'] == 'deepsets' else concatenate_features(obs, nonce, self.spec)
            x = torch.from_numpy(np.ascontiguousarray((x - self.mean) / self.scale))
            n = torch.from_numpy(np.ascontiguousarray(nonce))
            pred = self.model(x, n)
            if self.task == 'class':
                pred = pred.softmax(1)
            chunks.append(pred.numpy())
        return np.concatenate(chunks)

def load_attacker(checkpoint_path):
    return LoadedAttack(checkpoint_path)

def evaluate(cachepath, output, ssim_path):
    plan = json.loads((output / 'fit_summary.json').read_text())
    reports = {}
    selected = {}
    for task in ['class', 'recon']:
        names = [k for k, v in plan['models'].items() if v['specification']['task'] == task]
        if names:
            selected[task] = min(names, key=lambda k: (-plan['models'][k]['selected_validation_accuracy'], plan['models'][k]['selected_validation_loss'], k)) if task == 'class' else min(names, key=lambda k: (plan['models'][k]['selected_validation_loss'], k))
    write_json(output / 'locked_selection.json', {'selected': selected, 'rule': 'validation only; all fits complete before holdout array access', 'models': plan['models']})
    with np.load(cachepath, allow_pickle=False) as d:
        obs = d['test_obs']
        nonce = d['test_nonce_features']
        labels = d['y_test']
        images = d['test_images']
        test_indices = d['test_indices']
    assert len(obs) == 3600 and np.array_equal(test_indices, np.arange(6400, 10000))
    if any((v['specification']['task'] == 'recon' for v in plan['models'].values())):
        module_spec = importlib.util.spec_from_file_location('jscc_ssim', ssim_path)
        ssim_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(ssim_module)
    for name, record in plan['models'].items():
        destination = output / name
        predictor = load_attacker(destination / 'model.pt')
        pred = predictor(obs, nonce)
        if predictor.task == 'class':
            report = class_statistics(labels, pred)
            np.savez_compressed(destination / 'predictions.npz', probabilities=pred, predictions=pred.argmax(1), y_test=labels, test_indices=test_indices)
        else:
            mse = []
            ssim = []
            with torch.no_grad():
                for start in range(0, len(pred), 128):
                    p = torch.from_numpy(pred[start:start + 128])
                    truth = torch.from_numpy(images[start:start + 128].copy()).float() / 255.0
                    mse.append((p - truth).square().mean((1, 2, 3)).numpy())
                    ssim.append(ssim_module.ssim_per_image(p, truth).numpy())
            mse = np.concatenate(mse)
            ssim = np.concatenate(ssim)
            report = {'n': len(mse), 'mse': float(mse.mean(dtype=np.float64)), 'dataset_psnr': float(-10 * np.log10(mse.mean(dtype=np.float64))), 'ssim': float(ssim.mean(dtype=np.float64)), 'ssim_p95': float(np.percentile(ssim, 95)), 'ssim_max': float(ssim.max())}
            np.savez_compressed(destination / 'predictions.npz', reconstructions=pred, mse=mse, ssim=ssim, test_indices=test_indices)
        report.update({'checkpoint_sha256': file_hash(destination / 'model.pt'), 'selected_on_validation': name == selected[predictor.task], 'fit': record})
        write_json(destination / 'metrics.json', report)
        reports[name] = report
        write_json(output / 'evaluation_summary.json', {'protocol': plan['protocol'], 'selected': selected, 'models': reports})
        print('Holdout evaluated ' + name + ': ' + json.dumps({k: v for k, v in report.items() if k in ['accuracy', 'cross_entropy', 'mse', 'dataset_psnr', 'ssim']}), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ssim-module', type=Path, required=True)
    p.add_argument('--suite', choices=['all', 'classifiers', 'reconstructors', 'positive_control'], default='all')
    p.add_argument('--phase', choices=['fit', 'evaluate', 'all'], default='all')
    p.add_argument('--protocol', default='crypto')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--channels', type=int, default=32)
    p.add_argument('--only', nargs='*')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.phase in ('fit', 'all'):
        specs = specifications(a.suite, a.seed, a.channels)
        if a.only:
            specs = [s for s in specs if s['name'] in a.only]
        if not specs:
            raise ValueError('No selected model specifications')
        source_hashes = {name: file_hash(Path(__file__).resolve().parent / name) for name in ['train_attacks.py', 'attack_models.py', 'public_sampling.py']}
        with np.load(a.split, allow_pickle=False) as d:
            fit_key = next((k for k in ['fit', 'fitting', 'fit_indices', 'train_indices', 'train'] if k in d))
            val_key = next((k for k in ['validation', 'validation_indices', 'val_indices', 'val'] if k in d))
            split = {'fit': d[fit_key], 'validation': d[val_key]}
        assert len(split['fit']) == 45000 and len(split['validation']) == 5000 and (len(np.unique(np.concatenate(list(split.values())))) == 50000)
        with np.load(a.cache, allow_pickle=False) as d:
            data = {'obs': d['train_obs'], 'nonce': d['train_nonce_features'], 'labels': d['y_train'], 'latents': d['train_latents'], 'images': d['train_images'] if any((s['task'] == 'recon' for s in specs)) else None}
        manifest = {'protocol': a.protocol, 'planned_specifications': specs, 'train_n': 50000, 'fitting_n': 45000, 'validation_n': 5000, 'holdout_n': 3600, 'fit_indices_sha256': array_hash(split['fit']), 'validation_indices_sha256': array_hash(split['validation']), 'source_sha256': source_hashes, 'train_obs_sha256': array_hash(data['obs']), 'train_labels_sha256': array_hash(data['labels']), 'train_latents_sha256': array_hash(data['latents']), 'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(), 'device': 'cpu', 'threads': a.threads, 'test_arrays_loaded_during_fit': False, 'surrogate': 'independent uniform signs and unbiased uniform permutations; freshAWGN20dB every presentation; no experimental key read', 'models': {}}
        write_json(a.output / 'fit_summary.json', manifest)
        for spec in specs:
            manifest['models'][spec['name']] = fit_one(spec, data, split, a.output / spec['name'], a.protocol, source_hashes)
            write_json(a.output / 'fit_summary.json', manifest)
    if a.phase in ('evaluate', 'all'):
        evaluate(a.cache, a.output, a.ssim_module)
if __name__ == '__main__':
    main()
