from pathlib import Path
import gc, json, math, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .attack_models import ResidualMLPClassifier, DeepSetsClassifier, ConvReconstructor
from .public_sampling import sample_packets, invariant_features
from .experiment_utils import ImageMLP, save_torch, write_json, image_metrics, summarize_images, class_metrics

def specs(cfg, smoke):
    if smoke:
        return [dict(name='raw_classifier_51', kind='class', view='raw', nonce=True, aug=False, seed=51), dict(name='cnn_surrogate_no_nonce', kind='cnn', view='raw_fusion', nonce=False, aug=True, seed=81)]
    out = []
    for seed in cfg['classifier_raw_seeds']:
        out.append(dict(name=f'raw_classifier_{seed}', kind='class', view='raw', nonce=True, aug=False, seed=seed))
    for seed in cfg['invariant_seeds']:
        for aug in [False, True]:
            out.append(dict(name=f"invariant_{('surrogate' if aug else 'cached')}_{seed}", kind='class', view='fusion', nonce=False, aug=aug, seed=seed))
    out.append(dict(name='deepsets_surrogate', kind='sets', view='raw', nonce=False, aug=True, seed=63))
    for seed in cfg['reconstruction_mlp_seeds']:
        out.append(dict(name=f'mlp_reconstruction_{seed}', kind='mlp', view='raw', nonce=True, aug=False, seed=seed))
    for nonce, aug in [(True, False), (False, False), (False, True)]:
        out.append(dict(name=f"cnn_{('surrogate' if aug else 'cached')}_{('nonce' if nonce else 'no_nonce')}", kind='cnn', view='raw_fusion', nonce=nonce, aug=aug, seed=81))
    out.append(dict(name='cnn_large_surrogate_no_nonce', kind='cnn', view='raw_fusion', nonce=False, aug=True, seed=82, channels=256))
    return out

def features(obs, nonce, spec):
    f = invariant_features(obs, spec['view'])
    if spec['nonce']:
        f = np.concatenate((f, nonce), axis=1)
    return f.astype(np.float32)

def make_model(spec, dim):
    if spec['kind'] == 'sets':
        return DeepSetsClassifier()
    if spec['kind'] == 'class':
        return ResidualMLPClassifier(dim)
    if spec['kind'] == 'cnn':
        return ConvReconstructor(dim, channels=spec.get('channels', 32))
    if spec['kind'] == 'mlp':
        return ImageMLP(dim)
    raise ValueError(spec)

def is_class(spec):
    return spec['kind'] in ['class', 'sets']

def standardize(f, mean, scale):
    return (f - mean) / scale

def fit_candidate(spec, obs, nonce, u, images, labels, fit, val, cfg, folder, device, smoke):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'fit_complete.json').exists():
        return json.loads((folder / 'fit_complete.json').read_text())
    f = features(obs, nonce, spec)
    mean = f[fit].mean(0, dtype=np.float64).astype(np.float32)
    scale = f[fit].std(0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-05] = 1.0
    if spec['kind'] == 'sets':
        mean = np.zeros_like(mean)
        scale = np.ones_like(scale)
    xval = torch.as_tensor(standardize(f[val], mean, scale), device=device)
    target_val = torch.as_tensor(labels[val] if is_class(spec) else images[val].copy(), device=device)
    if not is_class(spec):
        target_val = target_val.float() / 255
    torch.manual_seed(spec['seed'])
    model = make_model(spec, f.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001 if is_class(spec) else 0.0002, weight_decay=0.0001)
    limit = 2 if smoke else cfg['attacker_epochs']
    batch = min(8, cfg['attacker_batch']) if smoke else cfg['attacker_batch']
    epoch_start = 1
    history = []
    best = float('inf')
    stale = 0
    last = folder / 'attack_last.pt'
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history = state['history']
        best = state['best']
        stale = state['stale']
        epoch_start = state['epoch'] + 1
        if stale >= cfg['attacker_patience']:
            epoch_start = limit + 1
    bestpath = folder / 'best.pt'
    for epoch in range(epoch_start, limit + 1):
        start = time.monotonic()
        torch.manual_seed(spec['seed'] + 1000 * epoch)
        rng = np.random.default_rng(spec['seed'] + 10000 * epoch)
        order = rng.permutation(fit)
        model.train()
        loss_sum = 0.0
        seen = 0
        for start_i in range(0, len(order), batch):
            ii = order[start_i:start_i + batch]
            if spec['aug']:
                sim = sample_packets(u[ii], rng, cfg['snr_eve_db'])
                nf = nonce[rng.choice(fit, len(ii), replace=True)]
                fb = features(sim, nf, spec)
            else:
                fb = f[ii]
            xb = torch.as_tensor(standardize(fb, mean, scale), device=device)
            target = torch.as_tensor(labels[ii] if is_class(spec) else images[ii].copy(), device=device)
            if not is_class(spec):
                target = target.float() / 255
            pred = model(xb)
            loss = F.cross_entropy(pred, target) if is_class(spec) else F.mse_loss(pred, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite attacker loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ii)
            seen += len(ii)
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for i in range(0, len(val), batch):
                pred = model(xval[i:i + batch])
                target = target_val[i:i + batch]
                loss = F.cross_entropy(pred, target) if is_class(spec) else F.mse_loss(pred, target)
                total += float(loss) * len(target)
                count += len(target)
        score = total / count
        row = {'epoch': epoch, 'fitting_loss': loss_sum / seen, 'validation_loss': score, 'updates': math.ceil(len(fit) / batch), 'seconds': time.monotonic() - start, 'fresh_simulated_presentations': seen if spec['aug'] else 0}
        history.append(row)
        if score < best:
            best = score
            stale = 0
            save_torch(bestpath, {'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'spec': spec, 'input_dim': f.shape[1], 'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale), 'selected_epoch': epoch, 'selected_validation_loss': score})
        else:
            stale += 1
        save_torch(last, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'history': history, 'best': best, 'stale': stale, 'epoch': epoch})
        write_json(folder / 'history.json', history)
        print(f"  attack {spec['name']} epoch{epoch}/{limit}: val={score:.7f} ({row['seconds']:.1f}s)", flush=True)
        if stale >= cfg['attacker_patience']:
            break
    selected = torch.load(bestpath, map_location='cpu', weights_only=True)
    report = {'spec': spec, 'completed_epochs': len(history), 'selected_epoch': selected['selected_epoch'], 'selected_validation_loss': selected['selected_validation_loss'], 'stopping_reason': 'validation_patience' if stale >= cfg['attacker_patience'] else 'epoch_budget', 'updates': sum((r['updates'] for r in history)), 'fresh_simulated_presentations': sum((r['fresh_simulated_presentations'] for r in history)), 'fitting_n': len(fit), 'validation_n': len(val), 'test_used_for_fit_or_selection': False}
    write_json(folder / 'fit_complete.json', report)
    del model, optimizer, xval, target_val, f
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return report

def load_candidate(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    spec = saved['spec']
    model = make_model(spec, saved['input_dim']).to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    mean = saved['mean'].numpy()
    scale = saved['scale'].numpy()

    @torch.no_grad()
    def predict(obs, nonce):
        f = standardize(features(obs, nonce, spec), mean, scale)
        out = []
        for i in range(0, len(f), 128):
            pred = model(torch.as_tensor(f[i:i + 128], device=device))
            out.append((pred.softmax(1) if is_class(spec) else pred).cpu().numpy())
        return np.concatenate(out)
    return (predict, spec)

def fit_all_and_lock(obs, nonce, u, images, labels, fit, val, cfg, folder, device, smoke):
    reports = []
    for spec in specs(cfg, smoke):
        reports.append(fit_candidate(spec, obs, nonce, u, images, labels, fit, val, cfg, folder / spec['name'], device, smoke))
    cls = min((r for r in reports if is_class(r['spec'])), key=lambda r: r['selected_validation_loss'])
    rec = min((r for r in reports if not is_class(r['spec'])), key=lambda r: r['selected_validation_loss'])
    locked = {'classifier': cls['spec']['name'], 'reconstructor': rec['spec']['name'], 'criterion': {'classifier': 'minimum validation cross entropy', 'reconstructor': 'minimum validation MSE'}, 'reports': reports, 'test_arrays_accessed_by_fitting': False}
    write_json(folder / 'locked_selection.json', locked)
    return locked

def evaluate_all(obs, nonce, images, labels, cfg, folder, device, smoke):
    locked = json.loads((folder / 'locked_selection.json').read_text())
    all_metrics = {}
    for spec in specs(cfg, smoke):
        name = spec['name']
        predict, _ = load_candidate(folder / name / 'best.pt', device)
        if is_class(spec):
            probs = predict(obs, nonce)
            metrics = class_metrics(labels, probs)
            np.savez_compressed(folder / name / 'holdout.npz', y_true=labels, probabilities=probs)
        else:
            combined = np.concatenate((obs, nonce), axis=1)

            def image_predict(x, n):
                return torch.as_tensor(predict(x[:, :256], x[:, 256:]), device=device)
            rows = image_metrics(image_predict, images, combined, device)
            metrics = summarize_images(rows)
            np.savez_compressed(folder / name / 'holdout.npz', **rows)
        all_metrics[name] = metrics
        write_json(folder / name / 'holdout_metrics.json', metrics)
        print(f"  holdout {name}: {metrics.get('accuracy', metrics.get('mean_ssim_legacy')):.6f}", flush=True)
    write_json(folder / 'all_holdout_metrics.json', all_metrics)
    return all_metrics
