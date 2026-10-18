from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import warnings
import numpy as np
from classifier_attacks import array_sha256, classification_statistics
from signed_invariant_attacks import signed_features

def extend(protocol, base):
    import joblib
    import scipy
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    directory = base / 'results' / f'classifiers_signed_{protocol}'
    original_path = directory / 'summary_initial300.json'
    if not original_path.exists():
        shutil.copy2(directory / 'summary.json', original_path)
        shutil.copy2(directory / 'selected_predictions.npz', directory / 'selected_predictions_initial300.npz')
    original = json.loads(original_path.read_text())
    row = original['models']['logistic_signed']
    if not row['actual_configuration']['warnings']:
        raise ValueError(f'{protocol}: no original training solver warning to resolve')
    config = dict(row['actual_configuration']['parameters'])
    assert config['max_iter'] == 300 and config['solver'] == 'lbfgs'
    config['max_iter'] = 3000
    start = time.perf_counter()
    out = directory / 'extended_solver' / 'logistic_signed'
    out.mkdir(parents=True, exist_ok=True)
    cache = base / 'data' / f'cache_{protocol}.npz'
    with np.load(cache, allow_pickle=False) as a:
        tr, te = (a['train_obs'], a['test_obs'])
        tn, en = (a['train_nonce_features'], a['test_nonce_features'])
        yt, ye = (a['y_train'].astype(np.int64), a['y_test'].astype(np.int64))
    for label, array in [('train_observations', tr), ('test_observations', te), ('train_labels', yt), ('test_labels', ye)]:
        assert array_sha256(array) == original[label + '_sha256']
    with np.load(base / 'results' / f'classifiers_{protocol}' / 'split_indices.npz', allow_pickle=False) as a:
        fit, val = (a['fit'], a['validation'])
    assert array_sha256(fit) == original['fit_indices_sha256']
    assert array_sha256(val) == original['validation_indices_sha256']
    scaler = joblib.load(directory / 'logistic_signed/model.joblib')['scaler']
    xtr = np.column_stack([signed_features(tr), tn])
    xte = np.column_stack([signed_features(te), en])
    xf, xv, xe = (scaler.transform(x).astype(np.float32) for x in (xtr[fit], xtr[val], xte))
    fit_started = time.perf_counter()
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = LogisticRegression(**config)
        model.fit(xf, yt[fit])
    converged = not any((issubclass(w.category, ConvergenceWarning) for w in caught))
    vp = model.predict_proba(xv).astype(np.float32)
    tp = model.predict_proba(xe).astype(np.float32)
    assert np.array_equal(model.classes_, np.arange(10))
    new = {'name': 'logistic_signed', 'feature_view': 'signed_invariants', 'input_dimension': int(xf.shape[1]), 'validation': classification_statistics(yt[val], vp.argmax(1), 10), 'runtime_seconds': time.perf_counter() - fit_started, 'actual_configuration': {'estimator': 'LogisticRegression', 'parameters': model.get_params(), 'completed_iterations_or_trees': int(np.max(model.n_iter_)), 'converged': converged, 'warnings': [str(w.message) for w in caught], 'preprocessing': 'Original StandardScaler, fitted on the same45k fit split only'}, 'artifacts_directory': 'extended_solver/logistic_signed'}
    summary = json.loads(original_path.read_text())
    summary['models']['logistic_signed'] = new
    ranking = sorted(summary['models'], key=lambda n: (-summary['models'][n]['validation']['accuracy'], n))
    selected = ranking[0]
    new['test'] = classification_statistics(ye, tp.argmax(1), 10)
    np.savez_compressed(out / 'predictions.npz', y_test=ye, predictions=tp.argmax(1), probabilities=tp, y_validation=yt[val], validation_predictions=vp.argmax(1), validation_probabilities=vp)
    joblib.dump({'scaler': scaler, 'model': model}, out / 'model.joblib', compress=3)
    (out / 'metrics.json').write_text(json.dumps(new, indent=2) + '\n')
    selected_file = (out if selected == 'logistic_signed' else directory / selected) / 'predictions.npz'
    with np.load(selected_file, allow_pickle=False) as a:
        np.savez_compressed(directory / 'selected_predictions.npz', y_test=a['y_test'], predictions=a['predictions'], probabilities=a['probabilities'])
    elapsed = time.perf_counter() - start
    extension = {'protocol': protocol, 'change': 'max_iter300 to3000 only', 'reason': 'Training-side convergence warning; no holdout hyperparameter tuning', 'original300_summary_sha256': hashlib.sha256(original_path.read_bytes()).hexdigest(), 'original300_logistic': row, 'extended_logistic': new, 'RF_unchanged': True, 'selected_attacker': selected, 'selected_test_metrics': summary['models'][selected]['test'], 'runtime_seconds': elapsed, 'numpy_version': np.__version__, 'sklearn_version': sklearn.__version__, 'scipy_version': scipy.__version__, 'joblib_version': joblib.__version__}
    (directory / 'extended_solver/result.json').write_text(json.dumps(extension, indent=2) + '\n')
    summary.update({'validation_ranking': ranking, 'selected_attacker': selected, 'selected_test_metrics': summary['models'][selected]['test'], 'runtime_seconds': original['runtime_seconds'] + elapsed, 'solver_extension': {'original300_summary': 'summary_initial300.json', 'report': 'extended_solver/result.json', 'max_iter': 3000, 'converged': converged, 'reason': extension['reason'], 'only_changed_parameter': 'max_iter'}})
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print('EXTENDED_SOLVER_COMPLETE ' + json.dumps({'protocol': protocol, 'converged': converged, 'n_iter': new['actual_configuration']['completed_iterations_or_trees'], 'validation': new['validation']['accuracy'], 'logistic_test': new['test']['accuracy'], 'selected': selected, 'selected_test': summary['selected_test_metrics']['accuracy'], 'selected_wilson': summary['selected_test_metrics']['accuracy_wilson_95'], 'runtime_seconds': elapsed}), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--protocols', default='crypto,fixed_crypto')
    args = parser.parse_args()
    for protocol in args.protocols.split(','):
        extend(protocol, args.base)
