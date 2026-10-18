from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import platform
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np

@dataclass
class AttackConfig:
    protocol: str = 'unspecified'
    n_classes: int = 10
    validation_size: int = 5000
    split_seed: int = 20261010
    estimator_seed: int = 20261011
    threads: int = 2
    logistic_C: float = 1.0
    logistic_max_iter: int = 300
    tree_seconds: float = 180.0
    rf_trees: int = 100
    rf_max_depth: int = 16
    hgb_iterations: int = 100
    hgb_max_leaf_nodes: int = 15
    hgb_max_depth: int = 6
    hgb_learning_rate: float = 0.1
    mlp_seeds: tuple[int, ...] = (51, 52, 53, 54, 55)
    mlp_views: tuple[str, ...] = ('raw',)
    mlp_hidden: tuple[int, int] = (256, 256)
    mlp_batch: int = 128
    mlp_epochs: int = 15
    mlp_patience: int = 5
    mlp_learning_rate: float = 0.001
    mlp_weight_decay: float = 0.0001
    mlp_seconds: float = 180.0
    device: str = 'cpu'
    save_models: bool = True

def _jsonwrite(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')

def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()

def wilson(correct: int, total: int, z: float=1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    p = correct / total
    den = 1 + z * z / total
    ctr = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total ** 2)) / den
    return [max(0.0, ctr - half), min(1.0, ctr + half)]

def classification_statistics(y: np.ndarray, pred: np.ndarray, classes: int) -> dict:
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    cm = np.bincount(classes * y + pred, minlength=classes ** 2).reshape(classes, classes)
    totals = cm.sum(1)
    correct = np.diag(cm)
    recalls = np.divide(correct, totals, out=np.zeros(classes, float), where=totals != 0)
    return {'n': int(len(y)), 'correct': int(correct.sum()), 'accuracy': float(correct.sum() / len(y)), 'accuracy_wilson_95': wilson(int(correct.sum()), len(y)), 'balanced_accuracy': float(recalls[totals > 0].mean()), 'confusion_matrix': cm.tolist(), 'per_class': [{'class': k, 'n': int(totals[k]), 'correct': int(correct[k]), 'recall': float(recalls[k]), 'recall_wilson_95': wilson(int(correct[k]), int(totals[k]))} for k in range(classes)], 'interval_scope': 'Binomial sampling uncertainty conditional on this frozen model, attacker selection, observed test cases and single cached channel draw; does not include encoder-training or channel-run uncertainty.'}

def magnitude_features(obs: np.ndarray) -> np.ndarray:
    a = np.abs(np.asarray(obs, dtype=np.float64))
    moments = np.column_stack([a.sum(1), np.linalg.norm(a, axis=1), np.sum(a ** 4, axis=1) ** 0.25, np.sum(a ** 6, axis=1) ** (1 / 6), a.max(1), a.min(1), a.mean(1), a.std(1), np.mean(a ** 2, axis=1), np.mean(a ** 4, axis=1), np.mean(a ** 6, axis=1), np.quantile(a, 0.25, axis=1), np.quantile(a, 0.5, axis=1), np.quantile(a, 0.75, axis=1), np.quantile(a, 0.9, axis=1)])
    return np.ascontiguousarray(np.column_stack([np.sort(a, axis=1), moments]), dtype=np.float32)

def _input(obs: Any, nonce: Any, labels: Any, n_classes: int, label: str):
    obs = np.asarray(obs, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if obs.ndim != 2 or len(obs) != len(y):
        raise ValueError(f'{label}: obs must be n-by-d and labels must have length n')
    if nonce is None:
        nonce = np.empty((len(obs), 0), np.float32)
    nonce = np.asarray(nonce, dtype=np.float32)
    if nonce.ndim == 1:
        nonce = nonce[:, None]
    if nonce.ndim != 2 or len(nonce) != len(obs):
        raise ValueError(f'{label}: nonce features must be n-by-q')
    if not np.isfinite(obs).all() or not np.isfinite(nonce).all():
        raise ValueError(f'{label}: non-finite observation or nonce features')
    if not np.all((y >= 0) & (y < n_classes)):
        raise ValueError(f'{label}: labels must be integers in 0,...,n_classes-1')
    return (obs, nonce, y)

def _fit_tree_budget(model, X, y, target: int, attr: str, seconds: float):
    start = time.perf_counter()
    trained = 0
    for count in range(10, target + 10, 10):
        count = min(count, target)
        model.set_params(**{attr: count})
        model.fit(X, y)
        trained = count
        if count == target or time.perf_counter() - start >= seconds:
            break
    return (model, trained)

def _fit_mlp(X_fit, y_fit, X_val, y_val, X_test, cfg: AttackConfig, seed: int, path: Path):
    import torch
    from torch import nn
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(seed)
    if cfg.device == 'cpu':
        torch.use_deterministic_algorithms(True)
    device = torch.device(cfg.device)
    model = nn.Sequential(nn.Linear(X_fit.shape[1], cfg.mlp_hidden[0]), nn.ReLU(), nn.Linear(cfg.mlp_hidden[0], cfg.mlp_hidden[1]), nn.ReLU(), nn.Linear(cfg.mlp_hidden[1], cfg.n_classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.mlp_learning_rate, weight_decay=cfg.mlp_weight_decay)
    lossfun = nn.CrossEntropyLoss()
    xf = torch.as_tensor(X_fit, dtype=torch.float32)
    yf = torch.as_tensor(y_fit, dtype=torch.long)
    xv = torch.as_tensor(X_val, dtype=torch.float32)
    yv = torch.as_tensor(y_val, dtype=torch.long)
    xt = torch.as_tensor(X_test, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed + 10000)

    def predict_tensor(x):
        model.eval()
        probs = []
        with torch.no_grad():
            for batch in x.split(1024):
                probs.append(model(batch.to(device)).softmax(1).cpu())
        return torch.cat(probs).numpy()
    history = []
    best_loss = math.inf
    best_state = None
    best_epoch = None
    stale = 0
    reason = 'maximum_epochs'
    start = time.perf_counter()
    for epoch in range(1, cfg.mlp_epochs + 1):
        model.train()
        order = torch.randperm(len(xf), generator=gen)
        running = 0.0
        for idx in order.split(cfg.mlp_batch):
            xbatch, ybatch = (xf[idx].to(device), yf[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss = lossfun(model(xbatch), ybatch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.detach().cpu()) * len(idx)
        vp = predict_tensor(xv)
        vloss = float(-np.log(np.clip(vp[np.arange(len(y_val)), y_val], 1e-12, 1)).mean())
        vac = float((vp.argmax(1) == y_val).mean())
        history.append({'epoch': epoch, 'train_cross_entropy': running / len(xf), 'validation_cross_entropy': vloss, 'validation_accuracy': vac, 'elapsed_seconds': time.perf_counter() - start})
        if vloss < best_loss - 1e-06:
            best_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= cfg.mlp_patience:
            reason = 'validation_patience'
            break
        if time.perf_counter() - start >= cfg.mlp_seconds:
            reason = 'soft_time_budget_checked_at_epoch_end'
            break
    if best_state is None:
        raise RuntimeError('MLP did not complete one epoch')
    model.load_state_dict(best_state)
    val_probs = predict_tensor(xv)
    test_probs = predict_tensor(xt)
    if cfg.save_models:
        torch.save({'state_dict': best_state, 'input_dimension': X_fit.shape[1], 'hidden': cfg.mlp_hidden, 'classes': cfg.n_classes, 'seed': seed, 'selected_epoch': best_epoch}, path / 'model.pt')
    _jsonwrite(path / 'history.json', history)
    return (val_probs, test_probs, {'seed': seed, 'selected_epoch': best_epoch, 'completed_epochs': len(history), 'stopping_reason': reason, 'selection': 'minimum validation cross entropy', 'optimizer': 'AdamW', 'hidden': list(cfg.mlp_hidden), 'activation': 'ReLU', 'learning_rate': cfg.mlp_learning_rate, 'weight_decay': cfg.mlp_weight_decay, 'batch_size': cfg.mlp_batch, 'gradient_norm_clip': 5.0, 'dropout': 0.0, 'updates': len(history) * math.ceil(len(X_fit) / cfg.mlp_batch), 'torch_version': torch.__version__, 'device': str(device)})

def run_classifier_attacks(train_obs, train_nonce_features, y_train, test_obs, test_nonce_features, y_test, outputdir, config: AttackConfig | dict | None=None):
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    cfg = config if isinstance(config, AttackConfig) else AttackConfig(**config or {})
    os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(cfg.threads))
    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    tr, tn, yt = _input(train_obs, train_nonce_features, y_train, cfg.n_classes, 'train')
    te, en, ye = _input(test_obs, test_nonce_features, y_test, cfg.n_classes, 'test')
    if tr.shape[1] != te.shape[1] or tn.shape[1] != en.shape[1]:
        raise ValueError('Train/test feature dimensions differ')
    if cfg.validation_size >= len(tr):
        raise ValueError('validation_size must be smaller than training size')
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=cfg.validation_size, random_state=cfg.split_seed)
    fit_idx, val_idx = next(splitter.split(np.zeros(len(yt)), yt))
    np.savez_compressed(outputdir / 'split_indices.npz', fit=fit_idx, validation=val_idx)
    mag_tr, mag_te = (magnitude_features(tr), magnitude_features(te))
    feature_sets = {'raw': (np.column_stack([tr, tn]), np.column_stack([te, en])), 'magnitude': (np.column_stack([mag_tr, tn]), np.column_stack([mag_te, en])), 'combined': (np.column_stack([tr, mag_tr, tn]), np.column_stack([te, mag_te, en]))}
    candidates = []
    for view in ('raw', 'magnitude'):
        candidates.append((f'logistic_{view}', view, 'logistic'))
    for view in ('magnitude', 'combined'):
        candidates.append((f'random_forest_{view}', view, 'rf'))
        candidates.append((f'hist_gradient_boosting_{view}', view, 'hgb'))
    results = {}
    probabilities = {}
    failures = {}

    def persist(name, view, vp, tp, detail, elapsed):
        path = outputdir / name
        path.mkdir(exist_ok=True)
        np.savez_compressed(path / 'predictions.npz', y_test=ye, predictions=tp.argmax(1), probabilities=tp.astype(np.float32), validation_predictions=vp.argmax(1), validation_probabilities=vp.astype(np.float32), y_validation=yt[val_idx])
        row = {'name': name, 'feature_view': view, 'input_dimension': int(feature_sets[view][0].shape[1]), 'validation': classification_statistics(yt[val_idx], vp.argmax(1), cfg.n_classes), 'runtime_seconds': float(elapsed), 'actual_configuration': detail}
        _jsonwrite(path / 'metrics.json', row)
        results[name] = row
        probabilities[name] = tp.astype(np.float32)
        print(f"{cfg.protocol} {name}: validation={row['validation']['accuracy']:.4f} time={elapsed:.1f}s (test metrics deferred)", flush=True)
        _jsonwrite(outputdir / 'progress.json', {'models': results, 'failures': failures})
    with threadpool_limits(limits=cfg.threads):
        for name, view, kind in candidates:
            start = time.perf_counter()
            path = outputdir / name
            path.mkdir(exist_ok=True)
            xtrain, xtest = feature_sets[view]
            scaler = StandardScaler().fit(xtrain[fit_idx])
            xf = scaler.transform(xtrain[fit_idx]).astype(np.float32)
            xv = scaler.transform(xtrain[val_idx]).astype(np.float32)
            xe = scaler.transform(xtest).astype(np.float32)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    if kind == 'logistic':
                        model = LogisticRegression(C=cfg.logistic_C, solver='lbfgs', max_iter=cfg.logistic_max_iter, random_state=cfg.estimator_seed, tol=0.0001)
                        model.fit(xf, yt[fit_idx])
                        trained = int(np.max(model.n_iter_))
                    elif kind == 'rf':
                        model = RandomForestClassifier(n_estimators=10, warm_start=True, max_depth=cfg.rf_max_depth, max_features='sqrt', min_samples_leaf=5, n_jobs=cfg.threads, random_state=cfg.estimator_seed)
                        model, trained = _fit_tree_budget(model, xf, yt[fit_idx], cfg.rf_trees, 'n_estimators', cfg.tree_seconds)
                    else:
                        model = HistGradientBoostingClassifier(max_iter=10, warm_start=True, early_stopping=False, max_leaf_nodes=cfg.hgb_max_leaf_nodes, max_depth=cfg.hgb_max_depth, learning_rate=cfg.hgb_learning_rate, l2_regularization=1.0, min_samples_leaf=20, random_state=cfg.estimator_seed)
                        model, trained = _fit_tree_budget(model, xf, yt[fit_idx], cfg.hgb_iterations, 'max_iter', cfg.tree_seconds)
                vp, tp = (model.predict_proba(xv), model.predict_proba(xe))
                if not np.array_equal(model.classes_, np.arange(cfg.n_classes)):
                    raise ValueError('Missing training class; probability-column order differs')
                detail = {'estimator': type(model).__name__, 'parameters': model.get_params(), 'completed_iterations_or_trees': trained, 'warnings': [str(w.message) for w in caught], 'preprocessing': 'StandardScaler fitted on fit split only', 'time_cap': 'Soft tree cap checked after ten trees/boosting rounds'}
                if cfg.save_models:
                    import joblib
                    joblib.dump({'scaler': scaler, 'model': model}, path / 'model.joblib', compress=3)
                persist(name, view, vp, tp, detail, time.perf_counter() - start)
            except Exception as exc:
                failures[name] = {'type': type(exc).__name__, 'message': str(exc), 'elapsed_seconds': time.perf_counter() - start}
                _jsonwrite(path / 'failure.json', failures[name])
                print(f'{cfg.protocol} {name} failed: {exc}', flush=True)
        for view in cfg.mlp_views:
            if view not in feature_sets:
                raise ValueError(f'Unknown MLP view: {view}')
            xtrain, xtest = feature_sets[view]
            scaler = StandardScaler().fit(xtrain[fit_idx])
            xf = scaler.transform(xtrain[fit_idx]).astype(np.float32)
            xv = scaler.transform(xtrain[val_idx]).astype(np.float32)
            xe = scaler.transform(xtest).astype(np.float32)
            for seed in cfg.mlp_seeds:
                name = f'mlp_{view}_seed{seed}'
                start = time.perf_counter()
                path = outputdir / name
                path.mkdir(exist_ok=True)
                try:
                    vp, tp, detail = _fit_mlp(xf, yt[fit_idx], xv, yt[val_idx], xe, cfg, seed, path)
                    detail['preprocessing'] = 'StandardScaler fitted on fit split only'
                    if cfg.save_models:
                        import joblib
                        joblib.dump(scaler, path / 'scaler.joblib', compress=3)
                    persist(name, view, vp, tp, detail, time.perf_counter() - start)
                except Exception as exc:
                    failures[name] = {'type': type(exc).__name__, 'message': str(exc), 'elapsed_seconds': time.perf_counter() - start}
                    _jsonwrite(path / 'failure.json', failures[name])
                    print(f'{cfg.protocol} {name} failed: {exc}', flush=True)
    ranking = sorted(results, key=lambda name: (-results[name]['validation']['accuracy'], name))
    selected = ranking[0] if ranking else None
    for name, row in results.items():
        row['test'] = classification_statistics(ye, probabilities[name].argmax(1), cfg.n_classes)
        _jsonwrite(outputdir / name / 'metrics.json', row)
    if selected is not None:
        np.savez_compressed(outputdir / 'selected_predictions.npz', y_test=ye, predictions=probabilities[selected].argmax(1), probabilities=probabilities[selected])
    summary = {'protocol': cfg.protocol, 'config': asdict(cfg), 'fit_size': len(fit_idx), 'validation_size': len(val_idx), 'test_size': len(ye), 'selection_rule': 'highest validation accuracy; lexicographic model-name tie-break', 'selected_attacker': selected, 'validation_ranking': ranking, 'selected_test_metrics': results[selected]['test'] if selected else None, 'models': results, 'failures': failures, 'train_observations_sha256': array_sha256(tr), 'test_observations_sha256': array_sha256(te), 'train_labels_sha256': array_sha256(yt), 'test_labels_sha256': array_sha256(ye), 'train_nonce_features_sha256': array_sha256(tn), 'test_nonce_features_sha256': array_sha256(en), 'fit_indices_sha256': array_sha256(fit_idx), 'validation_indices_sha256': array_sha256(val_idx), 'nonce_feature_dimension': int(tn.shape[1]), 'feature_description': 'Raw noisy symbols; ascending sorted absolute symbols; L1,L2,L4,L6 norms,max,min,mean,std,2nd/4th/6th absolute moments,0.25/0.5/0.75/0.9 quantiles; same supplied public nonce features appended to every view.', 'numpy_version': np.__version__, 'sklearn_version': sklearn.__version__, 'python_version': platform.python_version(), 'platform': platform.platform(), 'runtime_seconds': time.perf_counter() - total_start}
    _jsonwrite(outputdir / 'summary.json', summary)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--outputdir', required=True)
    parser.add_argument('--protocol', default='unspecified')
    parser.add_argument('--mlp-seeds', default='51,52,53,54,55')
    parser.add_argument('--mlp-views', default='raw')
    parser.add_argument('--mlp-epochs', type=int, default=15)
    parser.add_argument('--tree-seconds', type=float, default=180)
    parser.add_argument('--mlp-seconds', type=float, default=180)
    parser.add_argument('--validation-size', type=int, default=5000)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--no-models', action='store_true')
    args = parser.parse_args()
    cfg = AttackConfig(protocol=args.protocol, mlp_seeds=tuple((int(x) for x in args.mlp_seeds.split(',') if x)), mlp_views=tuple((x for x in args.mlp_views.split(',') if x)), mlp_epochs=args.mlp_epochs, tree_seconds=args.tree_seconds, mlp_seconds=args.mlp_seconds, validation_size=args.validation_size, threads=args.threads, device=args.device, save_models=not args.no_models)
    with np.load(args.cache, allow_pickle=False) as data:
        summary = run_classifier_attacks(data['train_obs'], data['train_nonce_features'] if 'train_nonce_features' in data else None, data['y_train'], data['test_obs'], data['test_nonce_features'] if 'test_nonce_features' in data else None, data['y_test'], args.outputdir, cfg)
    print(json.dumps({'selected_attacker': summary['selected_attacker'], 'selected_test_metrics': summary['selected_test_metrics']}), flush=True)
if __name__ == '__main__':
    main()
