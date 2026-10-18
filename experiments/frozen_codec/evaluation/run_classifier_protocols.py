from pathlib import Path
import argparse
import json
import time
import zipfile
import numpy as np
from classifier_attacks import AttackConfig, run_classifier_attacks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocols', default='crypto,identity,fixed_crypto')
    parser.add_argument('--base', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--wait-seconds', type=float, default=1800)
    args = parser.parse_args()
    base = Path(args.base)
    for protocol in args.protocols.split(','):
        output = base / 'results' / f'classifiers_{protocol}'
        if (output / 'summary.json').exists():
            print(f'PROTOCOL_ALREADY_COMPLETE {protocol}', flush=True)
            continue
        cache = base / 'data' / f'cache_{protocol}.npz'
        start = time.monotonic()
        arrays = None
        print(f'WAIT_CACHE {protocol} {cache}', flush=True)
        while time.monotonic() - start < args.wait_seconds:
            if cache.exists():
                try:
                    with np.load(cache, allow_pickle=False) as data:
                        arrays = [data[key] for key in ('train_obs', 'train_nonce_features', 'y_train', 'test_obs', 'test_nonce_features', 'y_test')]
                        metadata = str(data['metadata_json']) if 'metadata_json' in data else '{}'
                    break
                except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
                    print(f'CACHE_NOT_COMPLETE {protocol} {type(error).__name__}', flush=True)
            time.sleep(15)
        if arrays is None:
            raise TimeoutError(f'Cache {cache} did not become readable in {args.wait_seconds}s')
        output.mkdir(parents=True, exist_ok=True)
        (output / 'source_cache_metadata.json').write_text(json.dumps(json.loads(metadata), indent=2) + '\n')
        print(f'START_PROTOCOL {protocol}', flush=True)
        cfg = AttackConfig(protocol=protocol, threads=2)
        summary = run_classifier_attacks(*arrays, output, cfg)
        print(f'PROTOCOL_COMPLETE {protocol} ' + json.dumps({'selected': summary['selected_attacker'], 'test_accuracy': summary['selected_test_metrics']['accuracy'], 'balanced_accuracy': summary['selected_test_metrics']['balanced_accuracy'], 'wilson': summary['selected_test_metrics']['accuracy_wilson_95'], 'runtime_seconds': summary['runtime_seconds']}), flush=True)
        del arrays
if __name__ == '__main__':
    main()
