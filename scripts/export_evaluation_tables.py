from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def scalar_fields(record):
    return {key: value for key, value in record.items() if value is None or isinstance(value, (str, bool, int, float))}


def metric_fields(record):
    keep = ('n', 'mse', 'dataset_psnr', 'mean_ssim_valid', 'mean_ssim_legacy', 'accuracy', 'balanced_accuracy', 'cross_entropy', 'auroc', 'selected_epoch', 'selection_metric', 'count', 'snr_eve', 'key_index', 'noise_draw')
    result = {key: record[key] for key in keep if key in record and not isinstance(record[key], (list, dict))}
    if 'selection_metric' in result:
        result['task_metric'] = result.pop('selection_metric')
    return result


def write_csv(output, name, rows):
    if not rows:
        return None
    fields = list(dict.fromkeys(key for row in rows for key in row))
    for row in rows:
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError('Nonfinite CSV value: ' + name)
    path = output / name
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return {'file': name, 'rows': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/evaluation'))
    parser.add_argument('--module-directory', type=Path)
    args = parser.parse_args()
    root, output = args.results, args.output
    output.mkdir(exist_ok=True, parents=True)
    tables, origins = {}, {}

    def read(rel):
        path = root / rel
        origins[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        return load(path)

    def add(name, row):
        tables.setdefault(name, []).append(row)

    for folder in sorted((root / 'attacks').iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name
        data = read(f'attacks/{case}/metric_alignment/summary.json')
        for model, fit in data['fits'].items():
            spec = fit['spec']
            add('attack_fitting.csv', {'case': case, 'suite': 'primary', 'model': model, 'task': spec['task'], 'attacker_seed': spec['seed'], 'epochs': fit['completed_epochs'], 'fitting_images': fit['fitting_n'], 'validation_images': fit['validation_n'], 'optimizer_updates': fit['optimizer_updates'], 'simulated_packets': fit['fresh_simulated_packets'], 'training_seconds': fit['training_seconds']})
            for criterion, metrics in data['holdout'][model].items():
                selected = data['selections'][spec['task']][criterion + '_aligned'] == model
                row = {'case': case, 'model': model, 'task': spec['task'], 'criterion': criterion, 'suite_selected': selected, **metric_fields(metrics)}
                add('attack_candidates.csv', row)
                if selected:
                    add('attack_selected.csv', row)
        for control, record in data['stationary_controls'].items():
            fit, spec = record['fit'], record['fit']['spec']
            add('attack_fitting.csv', {'case': case, 'suite': 'stationary_control', 'model': control, 'task': spec['task'], 'attacker_seed': spec['seed'], 'epochs': fit['completed_epochs'], 'fitting_images': fit['fitting_n'], 'validation_images': fit['validation_n'], 'optimizer_updates': fit['optimizer_updates'], 'simulated_packets': fit['fresh_simulated_packets'], 'training_seconds': fit['training_seconds']})
            for criterion, metrics in record['holdout'].items():
                add('attack_controls.csv', {'case': case, 'control': control, 'criterion': criterion, **metric_fields(metrics)})
        priors = data['image_independent_priors']
        for model, record in priors.get('holdout', {}).items():
            add('image_priors.csv', {'case': case, 'model': model, **metric_fields(record)})
        for suite in ('budget', 'high_snr', 'multi_observation', 'attributes'):
            file = folder / suite / 'summary.json'
            if not file.exists():
                continue
            data = read(file.relative_to(root).as_posix())
            for model, record in data['runs'].items():
                fit, spec = record['fit'], record['fit']['spec']
                add('attack_fitting.csv', {'case': case, 'suite': suite, 'model': model, 'task': spec['task'], 'attacker_seed': spec['seed'], 'epochs': fit['completed_epochs'], 'fitting_images': fit['fitting_n'], 'validation_images': fit['validation_n'], 'optimizer_updates': fit['optimizer_updates'], 'simulated_packets': fit['fresh_simulated_packets'], 'training_seconds': fit['training_seconds']})
                if suite == 'budget':
                    for budget, results in record['budget_results'].items():
                        for criterion, metrics in results.items():
                            add('attack_budget.csv', {'case': case, 'model': model, 'attacker_seed': spec['seed'], 'budget_epochs': int(budget), 'criterion': criterion, **metric_fields(metrics)})
                elif suite in ('high_snr', 'multi_observation'):
                    for condition, metrics in record['holdout'].items():
                        add('attack_snr.csv' if suite == 'high_snr' else 'attack_packet_sets.csv', {'case': case, 'model': model, 'attacker_seed': spec['seed'], 'condition': condition, **metric_fields(metrics)})
                else:
                    for criterion, metrics in record['holdout'].items():
                        if spec['task'] == 'binary':
                            add('attack_binary.csv', {'case': case, 'model': model, 'attacker_seed': spec['seed'], 'criterion': criterion, **metric_fields(metrics)})
                        else:
                            for index, target in enumerate(metrics['attribute_names']):
                                add('attack_attributes.csv', {'case': case, 'model': model, 'attacker_seed': spec['seed'], 'criterion': criterion, 'target': target, 'n': metrics['n'], 'selected_epoch': metrics['selected_epoch'], 'mse': metrics['mse'][index], 'prior_mse': metrics['prior_mse'][index], 'r_squared': metrics['r_squared'][index], 'mse_improvement_over_training_mean': metrics['mse_improvement_over_training_mean'][index]})
    for folder in sorted((root / 'channels').iterdir()):
        case = folder.name
        data = read(f'channels/{case}/summary.json')
        for record in data['channels']:
            for interface in ('identity', 'keyed'):
                add('channel_outcomes.csv', {'case': case, 'scenario': record['scenario'], 'snr_db': record['snr_db'], 'draw': record['draw'], 'interface': interface, **metric_fields(record[interface])})
        for record in data['known_plaintext']['diverse_known_plaintexts']:
            add('reuse_recovery.csv', {'case': case, **{key: value for key, value in scalar_fields(record).items() if key not in ('interpretation', 'test_used_for_selection')}})
        for record in data['known_plaintext']['identical_plaintext_averaging']:
            add('reuse_averaging.csv', {'case': case, **{key: value for key, value in scalar_fields(record).items() if key not in ('interpretation', 'test_used_for_selection')}})
    for folder in sorted((root / 'baselines').iterdir()):
        case = folder.name
        data = read(f'baselines/{case}/summary.json')
        for record in data['operating_points']:
            row = scalar_fields(record)
            row = {key: value for key, value in row.items() if 'checkpoint' not in key}
            add('protection_operating_points.csv', {'case': case, **row})
        for component, timing in data['timings_ms_per_image_cpu_batch_amortized']['values'].items():
            add('protection_timings.csv', {'case': case, 'component': component, 'batch': 128, 'median_ms_per_image': timing['median'], 'p10_ms_per_image': timing['p10'], 'p90_ms_per_image': timing['p90'], 'repetitions': len(timing['samples'])})
    for folder in sorted((root / 'natural').iterdir()):
        case = folder.name
        data = read(f'natural/{case}/summary.json')
        for model, selection in data['selections'].items():
            add('natural_selections.csv', {'case': case, 'model': model, 'family': selection['family'], 'attacker_seed': selection['seed'], 'selected_epoch': selection['selected_epoch'], 'validation_ssim_valid': selection['validation_ssim_valid'], 'validation_mse': selection['validation_mse'], 'model_parameters': selection['model_parameters'], 'family_selected': data['family_selected_by_validation'].get(selection['family']) == model})
        for model, record in data['summaries'].items():
            row = {'case': case, 'model': model, **metric_fields(record['metrics'])}
            for metric, interval in record['bootstrap']['intervals_95'].items():
                row[metric + '_lo95'], row[metric + '_hi95'] = interval
            add('natural_summary.csv', row)
        file = folder / 'kodak_frame_metrics.csv'
        origins[file.relative_to(root).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        with file.open(newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                add('natural_frames.csv', {'case': case, **row})
    data = read('protocol/global/summary.json')
    for record in data['header_monte_carlo']:
        row = scalar_fields(record)
        row = {key: value for key, value in row.items() if key not in ('decoder', 'noise_model')}
        row['failure_lo95'], row['failure_hi95'] = record['failure_wilson95']
        add('header_reliability.csv', row)
    for record in data['header_accounting']:
        add('header_resources.csv', {key: value for key, value in scalar_fields(record).items() if key not in ('interpretation', 'header_layout')})
    for record in data['cpu_benchmarks']:
        row = {key: value for key, value in scalar_fields(record).items() if key != 'unit_note'}
        row['median_ms_per_loop_item'] = row.pop('median_ms_per_image')
        row['p95_ms_per_loop_item'] = row.pop('p95_ms_per_image')
        row['loop_item'] = ('header' if record['component'] in ('mac_generate', 'mac_verify', 'receiver_auth_and_durable_replay', 'hamming_encode', 'hamming_soft_decode') else 'reserved_counter' if record['component'] == 'durable_nonce_reservation' else 'tile')
        add('protocol_timings.csv', row)
    for record in data['derived_processing_accounting']:
        add('protocol_processing.csv', {key: value for key, value in scalar_fields(record).items() if key != 'scope'})
    (output / 'protocol_checks.json').write_text(json.dumps(data['protocol_tests'], indent=2, sort_keys=True) + '\n')
    diagnostic = read('diagnostics/seed1001_lambda0/training_loss_summary.json')
    for case, record in diagnostic['cases'].items():
        add('codec_loss_diagnostics.csv', {'case': case, 'last_epochs': diagnostic['last_epochs'], **record})
    manifest = read('RUN_MANIFEST.json')
    config = {key: value for key, value in manifest['config'].items() if not key.endswith('_dir')}
    (output / 'experiment_config.json').write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    inventory = [write_csv(output, name, rows) for name, rows in sorted(tables.items())]
    inventory += [{'file': name, 'sha256': hashlib.sha256((output / name).read_bytes()).hexdigest()} for name in ('experiment_config.json', 'protocol_checks.json')]
    provenance = {'schema_version': 1, 'run_signature': manifest['signature'], 'input_asset_manifest_sha256': manifest['asset_manifest_sha256'], 'input_record_sha256': origins, 'tables': inventory, 'observation_units': {'cifar': '3600 source images reused across conditions; cells are conditional measurements, not independent image samples', 'natural': '24 complete Kodak frames reused across predictors and codecs', 'attack_budget': 'best validation checkpoint available along one trajectory at each epoch limit', 'channel': 'same source images; codec, draw and interface are explicit', 'timing': 'component medians and derived sums are distinct'}}
    if args.module_directory:
        provenance['executed_module_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(args.module_directory.glob('*.py'))}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'tables': inventory, 'total_bytes': sum(p.stat().st_size for p in output.iterdir() if p.is_file())}, indent=2))


if __name__ == '__main__':
    main()
