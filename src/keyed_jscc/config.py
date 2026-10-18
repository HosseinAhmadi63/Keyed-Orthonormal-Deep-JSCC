import copy
import json
from pathlib import Path

CONFIG = {'experiment': 'keyed_orthonormal_jscc',
 'seeds': [1001, 1002, 1003],
 'defense_weights': [0.0, 0.2],
 'codec_epochs': 30,
 'codec_batch': 128,
 'codec_lr': 0.0002,
 'online_classifier_steps': 2,
 'attacker_epochs': 30,
 'attacker_patience': 8,
 'attacker_batch': 256,
 'classifier_raw_seeds': [51, 52, 53, 54, 55],
 'invariant_seeds': [61, 62],
 'reconstruction_mlp_seeds': [71, 72, 73],
 'reconstruction_cnn_seeds': [81],
 'snr_bob_db': 12.0,
 'snr_eve_db': 20.0,
 'repeat_keys': 3,
 'repeat_noise_draws': 3,
 'split_seed': 20261015,
 'validation_per_class': 500,
 'holdout_start': 6400,
 'latent_n': 256,
 'threads': 8,
 'amp_bfloat16': True,
 'bootstrap_samples': 2000,
 'max_return_part_bytes': 110000000}


def project_root():
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / 'configs' / 'experiment.json').is_file() else Path.cwd()


def load_config(path=None):
    selected = Path(path).expanduser().resolve() if path is not None else project_root() / 'configs' / 'experiment.json'
    if path is None and not selected.exists():
        settings = copy.deepcopy(CONFIG)
    else:
        settings = json.loads(selected.read_text(encoding='utf-8'))
    missing = set(CONFIG) - set(settings)
    unknown = set(settings) - set(CONFIG)
    if missing or unknown:
        raise ValueError(f'Configuration keys: missing={sorted(missing)}, unknown={sorted(unknown)}')
    if settings['latent_n'] != 256:
        raise ValueError('This codec requires latent_n=256')
    if not settings['seeds'] or len(set(settings['seeds'])) != len(settings['seeds']):
        raise ValueError('Training seeds must be nonempty and unique')
    if settings['repeat_keys'] < 1 or settings['repeat_noise_draws'] < 1:
        raise ValueError('Key and channel repetition counts must be positive')
    return settings
