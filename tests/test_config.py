import json
from pathlib import Path

from keyed_jscc.config import CONFIG, load_config


def test_paired_training_configuration():
    path = Path(__file__).resolve().parents[1] / 'configs' / 'experiment.json'
    settings = load_config(path)
    assert settings == json.loads(path.read_text()) == CONFIG
    assert settings['seeds'] == [1001, 1002, 1003]
    assert settings['defense_weights'] == [0.0, 0.2]
    assert settings['repeat_keys'] * settings['repeat_noise_draws'] == 9
    assert settings['codec_epochs'] == settings['attacker_epochs'] == 30
