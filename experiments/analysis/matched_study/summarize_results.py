from __future__ import annotations
import argparse, csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for v in iter(lambda: f.read(8 << 20), b''):
            h.update(v)
    return h.hexdigest()

def write(p, v):
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')

def wilson(hit, n):
    z = 1.959963984540054
    p = hit / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    t = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - t, c + t]

def cls(y, p):
    pred = p.argmax(1)
    cm = np.bincount(y * 10 + pred, minlength=100).reshape(10, 10)
    hits = int(np.trace(cm))
    return {'n': len(y), 'correct': hits, 'accuracy': hits / len(y), 'wilson95': wilson(hits, len(y)), 'balanced_accuracy': float(np.mean(np.diag(cm) / cm.sum(1))), 'confusion': cm.tolist(), 'recall': (np.diag(cm) / cm.sum(1)).tolist(), 'cross_entropy': float(-np.log(np.maximum(p[np.arange(len(y)), y].astype(np.float64), 1e-15)).mean()), 'predictive_entropy': float(-(p.astype(np.float64) * np.log(np.maximum(p.astype(np.float64), 1e-15))).sum(1).mean()), 'predicted_class_counts': np.bincount(pred, minlength=10).tolist()}

def image(a, prefix=''):
    m = a[prefix + 'mse'].astype(np.float64)
    return {'n': len(m), 'mse': float(m.mean()), 'dataset_psnr': float(-10 * np.log10(m.mean())), 'mean_ssim_legacy': float(a[prefix + 'ssim_legacy'].mean(dtype=np.float64)), 'mean_ssim_valid': float(a[prefix + 'ssim_valid'].mean(dtype=np.float64))}

def compare(a, b, path, errors):
    for k, v in a.items():
        if k in b:
            x = np.asarray(v)
            z = np.asarray(b[k])
            error = float(np.max(np.abs(x - z))) if x.size else 0
            if error > 1e-10:
                errors.append({'path': path, 'metric': k, 'maximum_absolute_difference': error})

def loadnp(p):
    with np.load(p, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    r = a.results
    cfg = read(r / 'RUN_LOCK.json')['config']
    keys = read(r / 'EXPERIMENT_KEYS_PUBLIC_FIXTURES.json')
    provenance = {}
    split = loadnp(r / 'split_manifest.npz')
    fit, val, hold = [split[k] for k in ['fitting', 'validation', 'holdout_indices']]
    assert np.array_equal(hold, np.arange(6400, 10000))
    assert len(fit) == 45000 and len(val) == 5000 and (len(np.union1d(fit, val)) == 50000) and (not np.intersect1d(fit, val).size)
    with np.load(a.cache) as d:
        y = d['y_train']
        yt = d['y_test']
    rng = np.random.default_rng(cfg['split_seed'])
    expected = []
    for k in range(10):
        expected.extend(rng.permutation(np.flatnonzero(y == k))[:500])
    assert np.array_equal(val, np.sort(expected)) and np.array_equal(fit, np.setdiff1d(np.arange(50000), val))
    provenance['split_matches_regenerated_stratified_split'] = True
    provenance['validation_class_counts'] = np.bincount(y[val], minlength=10).tolist()
    provenance['holdout_label_counts'] = np.bincount(yt, minlength=10).tolist()
    errors = []
    cases = {}
    all_candidates = []
    effort = []
    per_image = {}
    sourceall = read(r / 'ALL_RESULTS.json')
    assert sourceall['config'] == cfg and len(sourceall['cases']) == 6
    for seed in cfg['seeds']:
        pair = []
        for lam in cfg['defense_weights']:
            name = f'seed{seed}_lambda{lam:g}'
            folder = r / name
            e = read(folder / 'evaluation_complete.json')
            assert e in sourceall['cases']
            tr = read(folder / 'training_complete.json')
            history = read(folder / 'training_history.json')
            assert len(history) == 30 and [x['epoch'] for x in history] == list(range(1, 31))
            best = min(history, key=lambda x: x['validation_mse'])
            assert tr['selected_epoch'] == best['epoch'] == e['codec_selected_epoch']
            assert abs(tr['selected_validation_mse'] - best['validation_mse']) < 1e-12
            tag = cfg['seeds'].index(seed)
            trprefix = int.from_bytes(b'TR' + tag.to_bytes(2, 'big') + bytes(8), 'big')
            acprefix = int.from_bytes(b'AC' + tag.to_bytes(2, 'big') + bytes(8), 'big')
            for ep, h in enumerate(history):
                assert int(h['nonce_first'], 16) == trprefix + ep * 45000 and int(h['nonce_last'], 16) == trprefix + (ep + 1) * 45000 - 1 and (h['new_unique_training_packets'] == 45000) and (h['codec_updates'] == 352)
            manifest = read(folder / 'packet_manifest.json')
            assert manifest['key_hex'] == keys[name]['training_key'] and manifest['additional_evaluation_keys'] == keys[name]['extra_keys']
            ns = [int(x, 16) for x in manifest['nonce_hex']]
            expected = [acprefix + int(i) for i in np.random.default_rng(cfg['split_seed'] + 200 + seed).permutation(53600)]
            assert len(ns) == 53600 and len(set(ns)) == 53600 and (ns == expected)
            assert not any((trprefix <= n < trprefix + 1350000 for n in ns))
            pair.append((manifest['key_hex'], manifest['additional_evaluation_keys'], ns))
            lat = loadnp(folder / 'latents.npz')
            normerrors = {k: float(np.max(np.abs(np.mean(v.astype(np.float64) ** 2, axis=1) - 1))) for k, v in lat.items()}
            assert lat['training'].shape == (50000, 256) and lat['holdout'].shape == (3600, 256)
            assert max(normerrors.values()) < 2e-06
            af = folder / 'attacks'
            locked = read(af / 'locked_selection.json')
            reports = locked['reports']
            assert len(reports) == 17 and len(list(af.glob('*/fit_complete.json'))) == 20
            creps = [x for x in reports if x['spec']['kind'] in ['class', 'sets']]
            rreps = [x for x in reports if x['spec']['kind'] not in ['class', 'sets']]
            assert len(creps) == 10 and len(rreps) == 7
            assert locked['classifier'] == min(creps, key=lambda x: x['selected_validation_loss'])['spec']['name']
            assert locked['reconstructor'] == min(rreps, key=lambda x: x['selected_validation_loss'])['spec']['name']
            assert e['attacker_selection'] == locked
            candidates = {}
            positive = {}
            budget = []
            for sub in sorted(af.iterdir()):
                if not sub.is_dir():
                    continue
                rep = read(sub / 'fit_complete.json')
                hh = read(sub / 'history.json')
                assert rep['completed_epochs'] == len(hh) and len(hh) <= 30
                besth = min(hh, key=lambda x: x['validation_loss'])
                assert besth['epoch'] == rep['selected_epoch'] and abs(besth['validation_loss'] - rep['selected_validation_loss']) < 1e-12
                assert rep['updates'] == 176 * len(hh)
                assert rep['fresh_simulated_presentations'] == (45000 * len(hh) if rep['spec']['aug'] else 0)
                assert not rep['test_used_for_fit_or_selection']
                arr = loadnp(sub / 'holdout.npz')
                reported = read(sub / 'holdout_metrics.json')
                isclass = 'probabilities' in arr
                if isclass:
                    assert np.array_equal(arr['y_true'], yt) and arr['probabilities'].shape == (3600, 10) and np.isfinite(arr['probabilities']).all()
                    assert np.max(np.abs(arr['probabilities'].sum(1) - 1)) < 1e-05
                    metrics = cls(yt, arr['probabilities'])
                else:
                    metrics = image(arr)
                    assert all((np.isfinite(arr[k]).all() for k in ['mse', 'ssim_legacy', 'ssim_valid']))
                    assert np.all(arr['mse'] >= 0)
                compare(reported, metrics, f'{name}/{sub.name}', errors)
                record = {'name': sub.name, 'task': 'classification' if isclass else 'reconstruction', 'spec': rep['spec'], 'fit': rep, 'metrics': metrics, 'history': hh, 'stopping_at_epoch_cap': len(hh) == 30, 'selected_epoch_is_last_completed': rep['selected_epoch'] == len(hh)}
                (candidates if sub.name in [x['spec']['name'] for x in reports] else positive)[sub.name] = record
                all_candidates.append({'case': name, **record})
                budget.append({'name': sub.name, 'epochs': len(hh), 'seconds': sum((x['seconds'] for x in hh)), 'updates': rep['updates'], 'surrogate_presentations': rep['fresh_simulated_presentations'], 'selected_epoch': rep['selected_epoch'], 'stop': rep['stopping_reason']})
            for nn, mm in e['all_attacker_metrics'].items():
                assert mm == read(af / nn / 'holdout_metrics.json')
            for nn, mm in e['positive_controls'].items():
                assert mm == read(af / nn / 'holdout_metrics.json')
            prior = loadnp(folder / 'prior_holdout.npz')
            priorstats = image(prior)
            compare(e['fitting_mean_prior'], priorstats, name + '/prior', errors)
            cells = []
            arrays = []
            for ki in range(3):
                for ni in range(3):
                    ar = loadnp(folder / f'key{ki}_noise{ni}_per_image.npz')
                    assert np.array_equal(ar['y_true'], yt)
                    b = image(ar, 'bob_')
                    v = image(ar, 'eve_')
                    c = cls(yt, ar['probabilities'])
                    reported = e['key_noise_repetitions'][ki * 3 + ni]
                    assert reported['key_index'] == ki and reported['noise_draw'] == ni
                    compare(reported['bob'], b, name + f'/key{ki}/noise{ni}/bob', errors)
                    compare(reported['selected_reconstruction'], v, name + f'/key{ki}/noise{ni}/eve', errors)
                    compare(reported['selected_classification'], c, name + f'/key{ki}/noise{ni}/class', errors)
                    cells.append({'key_index': ki, 'noise_draw': ni, 'bob': b, 'eve': v, 'classification': c})
                    arrays.append(ar)
            pp = {'bob_mse': np.mean([x['bob_mse'] for x in arrays], axis=0, dtype=np.float64), 'bob_ssim_valid': np.mean([x['bob_ssim_valid'] for x in arrays], axis=0, dtype=np.float64), 'bob_ssim_legacy': np.mean([x['bob_ssim_legacy'] for x in arrays], axis=0, dtype=np.float64), 'eve_mse': np.mean([x['eve_mse'] for x in arrays], axis=0, dtype=np.float64), 'eve_ssim_valid': np.mean([x['eve_ssim_valid'] for x in arrays], axis=0, dtype=np.float64), 'eve_ssim_legacy': np.mean([x['eve_ssim_legacy'] for x in arrays], axis=0, dtype=np.float64), 'accuracy': np.mean([x['probabilities'].argmax(1) == yt for x in arrays], axis=0, dtype=np.float64), 'prior_mse': prior['mse'].astype(np.float64), 'prior_ssim_valid': prior['ssim_valid'].astype(np.float64), 'prior_ssim_legacy': prior['ssim_legacy'].astype(np.float64)}
            per_image[name] = pp
            mean = {k: float(v.mean()) for k, v in pp.items()}
            mean['bob_dataset_psnr'] = -10 * math.log10(mean['bob_mse'])
            mean['eve_dataset_psnr'] = -10 * math.log10(mean['eve_mse'])
            mean['eve_minus_prior_valid_ssim'] = mean['eve_ssim_valid'] - mean['prior_ssim_valid']
            mean['eve_minus_prior_legacy_ssim'] = mean['eve_ssim_legacy'] - mean['prior_ssim_legacy']
            mean['eve_minus_prior_mse'] = mean['eve_mse'] - mean['prior_mse']
            maxc = max((x for x in candidates.values() if x['task'] == 'classification'), key=lambda x: x['metrics']['accuracy'])
            maxr = max((x for x in candidates.values() if x['task'] == 'reconstruction'), key=lambda x: x['metrics']['mean_ssim_valid'])
            cases[name] = {'seed': seed, 'lambda': lam, 'selected_codec_epoch': tr['selected_epoch'], 'selected_classifier': locked['classifier'], 'selected_reconstructor': locked['reconstructor'], 'repeat_cell_mean_metrics': mean, 'repeat_cells': cells, 'prior': priorstats, 'candidates': candidates, 'positive_controls': positive, 'descriptive_highest_test_accuracy_family': maxc['name'], 'descriptive_highest_test_accuracy': maxc['metrics']['accuracy'], 'descriptive_highest_test_valid_ssim_family': maxr['name'], 'descriptive_highest_test_valid_ssim': maxr['metrics']['mean_ssim_valid'], 'latent_norm_max_error': normerrors, 'codec_training_seconds': sum((x['seconds'] for x in history)), 'attacker_effort': budget}
            effort.extend(budget)
            print(name, json.dumps({'repeat_means': mean, 'selected_cls': locked['classifier'], 'max_test_cls': maxc['name'], 'max_test_acc': maxc['metrics']['accuracy']}), flush=True)
        assert pair[0] == pair[1]
    assert len(set((keys[f'seed{s}_lambda0']['training_key'] for s in cfg['seeds']))) == 3
    provenance.update({'nonce_and_key_schedules_verified': True, 'paired_lambda_fixtures_equal': True, 'six_complete_30epoch_codecs': True, 'fresh_attacker_fits': len(all_candidates), 'nominal_classification_fits': sum((x['task'] == 'classification' for x in all_candidates)), 'nominal_reconstruction_fits': sum((x['task'] == 'reconstruction' for x in all_candidates)), 'metric_reproduction_errors': errors})
    assert not errors
    measures = ['bob_dataset_psnr', 'bob_ssim_valid', 'eve_ssim_valid', 'eve_minus_prior_valid_ssim', 'accuracy']
    contrasts = {}
    tcrit = 4.302652729911275
    for m in measures:
        vv = np.array([cases[f'seed{s}_lambda0.2']['repeat_cell_mean_metrics'][m] - cases[f'seed{s}_lambda0']['repeat_cell_mean_metrics'][m] for s in cfg['seeds']])
        mean = float(vv.mean())
        sd = float(vv.std(ddof=1))
        contrasts[m] = {'lambda0.2_minus_0_by_seed': dict(zip(map(str, cfg['seeds']), vv.tolist())), 'mean_difference': mean, 'sample_sd_across_3_paired_seeds': sd, 'descriptive_t2_95_interval': [mean - tcrit * sd / math.sqrt(3), mean + tcrit * sd / math.sqrt(3)]}
    bylambda = {str(l): {m: {'mean_across_3_seeds': float(np.mean([cases[f'seed{s}_lambda{l:g}']['repeat_cell_mean_metrics'][m] for s in cfg['seeds']])), 'sd_across_3_seeds': float(np.std([cases[f'seed{s}_lambda{l:g}']['repeat_cell_mean_metrics'][m] for s in cfg['seeds']], ddof=1))} for m in measures} for l in cfg['defense_weights']}
    ci = {name: {} for name in cases}
    rng = np.random.default_rng(2026101509)
    bootstrap = 2000
    for name, pp in per_image.items():
        store = {m: [] for m in ['bob_ssim_valid', 'eve_ssim_valid', 'accuracy', 'bob_dataset_psnr', 'eve_minus_prior_valid_ssim', 'eve_minus_prior_mse']}
        for j in range(0, bootstrap, 100):
            ix = rng.integers(0, 3600, (min(100, bootstrap - j), 3600))
            for m in ['bob_ssim_valid', 'eve_ssim_valid', 'accuracy']:
                store[m].extend(pp[m][ix].mean(1))
            store['bob_dataset_psnr'].extend(-10 * np.log10(pp['bob_mse'][ix].mean(1)))
            store['eve_minus_prior_valid_ssim'].extend((pp['eve_ssim_valid'] - pp['prior_ssim_valid'])[ix].mean(1))
            store['eve_minus_prior_mse'].extend((pp['eve_mse'] - pp['prior_mse'])[ix].mean(1))
        ci[name] = {m: np.quantile(v, [0.025, 0.975]).tolist() for m, v in store.items()}
    out = {'environment': read(r / 'environment.json'), 'config': cfg, 'cases': cases, 'lambda_group_summary': bylambda, 'paired_seed_contrasts': contrasts, 'conditional_image_bootstrap': {'resamples': bootstrap, 'seed': 2026101509, 'case_intervals': ci}, 'effort': {'codec_epochs': 180, 'codec_optimizer_updates': 63360, 'online_classifier_updates': 126720, 'attacker_fits': len(effort), 'attacker_epochs_completed': sum((x['epochs'] for x in effort)), 'attacker_optimizer_updates': sum((x['updates'] for x in effort)), 'surrogate_presentations': sum((x['surrogate_presentations'] for x in effort)), 'fits_reaching30epochs': sum((x['epochs'] == 30 for x in effort)), 'attacker_training_seconds_sum': sum((x['seconds'] for x in effort)), 'codec_training_seconds_sum': sum((c['codec_training_seconds'] for c in cases.values()))}}
    write(a.output / 'gpu_aggregate_consistency.json', {'all_results_matches6individual_evaluation_records': True, 'all102main_and18control_metrics_match_individual_records': True, 'cases': list(cases)})
    write(a.output / 'gpu_numeric_summary.json', out)
    write(a.output / 'gpu_summary.json', out)
    np.savez_compressed(a.output / 'gpu_repeat_averaged_per_image.npz', **{name + '__' + k: v for name, pp in per_image.items() for k, v in pp.items()})
    with (a.output / 'gpu_case_summary.csv').open('w', newline='') as f:
        w = csv.writer(f)
        fields = ['case', 'seed', 'lambda', 'selected_codec_epoch', 'selected_classifier', 'selected_reconstructor'] + list(next(iter(cases.values()))['repeat_cell_mean_metrics'])
        w.writerow(fields)
        for name, c in cases.items():
            w.writerow([name] + [c[x] for x in fields[1:6]] + [c['repeat_cell_mean_metrics'][x] for x in fields[6:]])
    print('COMPLETE', json.dumps({'cases': 6, 'fits': len(effort), 'metric_errors': len(errors), 'contrasts': {k: v['mean_difference'] for k, v in contrasts.items()}, 'effort': out['effort']}), flush=True)
if __name__ == '__main__':
    main()
