from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import warnings
import numpy as np
from classifier_attacks import AttackConfig, array_sha256, classification_statistics, _fit_tree_budget

def signed_features(obs):
    x = np.asarray(obs, dtype=np.float64)
    moments = np.column_stack([x.mean(1), x.std(1), x.min(1), x.max(1), np.mean(x ** 3, axis=1), np.mean(x ** 5, axis=1), np.mean(x ** 7, axis=1), np.mean(x < 0, axis=1), np.mean(x > 0, axis=1), np.quantile(x, 0.1, axis=1), np.quantile(x, 0.25, axis=1), np.quantile(x, 0.5, axis=1), np.quantile(x, 0.75, axis=1), np.quantile(x, 0.9, axis=1)])
    return np.column_stack([np.sort(x, axis=1), moments]).astype(np.float32)

def run(protocol, base):
    import scipy
    import sklearn
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    cfg = AttackConfig(protocol=protocol, threads=2)
    os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
    cache = base / 'data' / f'cache_{protocol}.npz'
    primary = base / 'results' / f'classifiers_{protocol}'
    out = base / 'results' / f'classifiers_signed_{protocol}'
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with np.load(cache, allow_pickle=False) as d:
        tr, te = (d['train_obs'], d['test_obs'])
        tn, en = (d['train_nonce_features'], d['test_nonce_features'])
        yt, ye = (d['y_train'].astype(np.int64), d['y_test'].astype(np.int64))
    with np.load(primary / 'split_indices.npz', allow_pickle=False) as d:
        fit, val = (d['fit'], d['validation'])
    if len(fit) != 45000 or len(val) != 5000 or set(fit).intersection(set(val)):
        raise ValueError('Primary training-validation split does not match frozen design')
    orig = json.loads((primary / 'summary.json').read_text())
    if array_sha256(fit) != orig['fit_indices_sha256'] or array_sha256(val) != orig['validation_indices_sha256']:
        raise ValueError('Split hashes differ from original suite')
    if array_sha256(yt) != orig['train_labels_sha256']:
        raise ValueError('Training labels differ from original suite')
    xtr = np.column_stack([signed_features(tr), tn])
    xte = np.column_stack([signed_features(te), en])
    scaler = StandardScaler().fit(xtr[fit])
    xf, xv, xe = (scaler.transform(x).astype(np.float32) for x in (xtr[fit], xtr[val], xte))
    models, probabilities = ({}, {})
    with threadpool_limits(limits=2):
        for name in ['logistic_signed', 'random_forest_signed']:
            t0 = time.perf_counter()
            p = out / name
            p.mkdir(exist_ok=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if name == 'logistic_signed':
                    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=300, tol=0.0001, random_state=cfg.estimator_seed)
                    model.fit(xf, yt[fit])
                    count = int(np.max(model.n_iter_))
                else:
                    model = RandomForestClassifier(n_estimators=10, warm_start=True, max_depth=16, max_features='sqrt', min_samples_leaf=5, n_jobs=2, random_state=cfg.estimator_seed)
                    model, count = _fit_tree_budget(model, xf, yt[fit], 100, 'n_estimators', 180.0)
            vp, tp = (model.predict_proba(xv), model.predict_proba(xe))
            if not np.array_equal(model.classes_, np.arange(10)):
                raise ValueError('Probability column ordering differs from fixed class indices')
            valstats = classification_statistics(yt[val], vp.argmax(1), 10)
            models[name] = {'name': name, 'feature_view': 'signed_invariants', 'input_dimension': int(xf.shape[1]), 'validation': valstats, 'runtime_seconds': time.perf_counter() - t0, 'actual_configuration': {'estimator': type(model).__name__, 'parameters': model.get_params(), 'completed_iterations_or_trees': count, 'warnings': [str(w.message) for w in caught], 'preprocessing': 'StandardScaler fitted on original45k fit split only', 'soft_tree_seconds': 180.0}}
            probabilities[name] = tp.astype(np.float32)
            np.savez_compressed(p / 'predictions.npz', y_test=ye, predictions=tp.argmax(1), probabilities=tp.astype(np.float32), y_validation=yt[val], validation_predictions=vp.argmax(1), validation_probabilities=vp.astype(np.float32))
            joblib.dump({'scaler': scaler, 'model': model}, p / 'model.joblib', compress=3)
            print(f"SIGNED {protocol} {name} validation={valstats['accuracy']:.4f} time={models[name]['runtime_seconds']:.1f}s (test metrics deferred)", flush=True)
    ranking = sorted(models, key=lambda n: (-models[n]['validation']['accuracy'], n))
    selected = ranking[0]
    for name, row in models.items():
        row['test'] = classification_statistics(ye, probabilities[name].argmax(1), 10)
        (out / name / 'metrics.json').write_text(json.dumps(row, indent=2) + '\n')
    np.savez_compressed(out / 'selected_predictions.npz', y_test=ye, predictions=probabilities[selected].argmax(1), probabilities=probabilities[selected])
    summary = {'suite': 'exploratory_signed_invariants', 'protocol': protocol, 'design_basis': 'Validation-selected signed-feature classifiers.', 'feature_description': 'Sorted signed noisy observations; signed mean,std,min,max; third,fifth,seventh moments; negative and positive fractions; signed10th,25th,50th,75th,90th percentiles; all six public nonce features.', 'fit_size': len(fit), 'validation_size': len(val), 'test_size': len(ye), 'selection_rule': 'highest validation accuracy, lexicographic name tie-break', 'selected_attacker': selected, 'validation_ranking': ranking, 'selected_test_metrics': models[selected]['test'], 'models': models, 'fit_indices_sha256': array_sha256(fit), 'validation_indices_sha256': array_sha256(val), 'train_observations_sha256': array_sha256(tr), 'test_observations_sha256': array_sha256(te), 'train_labels_sha256': array_sha256(yt), 'test_labels_sha256': array_sha256(ye), 'numpy_version': np.__version__, 'sklearn_version': sklearn.__version__, 'scipy_version': scipy.__version__, 'joblib_version': joblib.__version__, 'runtime_seconds': time.perf_counter() - start}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(f'SIGNED_COMPLETE {protocol} ' + json.dumps({'selected': selected, 'accuracy': summary['selected_test_metrics']['accuracy'], 'balanced_accuracy': summary['selected_test_metrics']['balanced_accuracy'], 'wilson': summary['selected_test_metrics']['accuracy_wilson_95'], 'runtime_seconds': summary['runtime_seconds']}), flush=True)
    return summary
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--protocols', default='crypto,identity,fixed_crypto,permutation_only')
    args = parser.parse_args()
    for protocol in args.protocols.split(','):
        run(protocol, Path(args.base))
