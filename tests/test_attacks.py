import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation import attacks
from evaluation.common import public_observations


def test_metric_checkpoint_selection_is_not_loss_selection():
    history = [
        {'epoch': 1, 'validation_loss': 0.04, 'validation_ssim_valid': 0.35},
        {'epoch': 2, 'validation_loss': 0.03, 'validation_ssim_valid': 0.30},
        {'epoch': 3, 'validation_loss': 0.05, 'validation_ssim_valid': 0.40},
    ]
    selected = attacks.select_validation_checkpoints(history, 'ssim_valid')
    assert selected == {'loss_epoch': 2, 'metric_epoch': 3}
    history[2]['validation_ssim_valid'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        attacks.select_validation_checkpoints(history, 'ssim_valid')


def test_primary_rerun_preserves_all_seventeen_original_families():
    candidates = attacks.primary_specs(attacks.defaults())
    assert len(candidates) == 17
    assert sum(s['task'] == 'classification' for s in candidates) == 10
    assert sum(s['task'] == 'reconstruction' for s in candidates) == 7
    assert len({s['name'] for s in candidates}) == 17
    assert next(s for s in candidates if s['name'] == 'cnn_large_surrogate_no_nonce')['channels'] == 256


def test_repeated_observations_keep_source_image_and_change_each_transform():
    u = np.random.default_rng(21).normal(size=(6, 256)).astype(np.float32)
    u /= np.sqrt(np.mean(u * u, axis=1, keepdims=True))
    observations = public_observations(u, np.random.default_rng(41), float('inf'), repetitions=4)
    assert observations.shape == (6, 4, 256)
    np.testing.assert_allclose(np.sum(observations * observations, axis=2),
                               np.repeat(np.sum(u * u, axis=1)[:, None], 4, 1), rtol=1e-5)
    assert not np.array_equal(observations[:, 0], observations[:, 1])
    assert not np.array_equal(observations[0], observations[1])


def test_packet_set_is_invariant_to_interception_order():
    torch.manual_seed(9)
    model = attacks.PacketSetAttacker(17, task='classification', width=16).eval()
    features = torch.randn(3, 4, 17)
    with torch.no_grad():
        first = model(features)
        second = model(features[:, [2, 0, 3, 1]])
    torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-6)
    assert not torch.allclose(first, model(features[:, :1]))


def test_direct_attribute_targets_have_physical_definitions():
    images = np.zeros((3, 3, 32, 32), dtype=np.uint8)
    images[1] = 255
    images[2, 0] = 255
    target = attacks.image_attributes(images)
    np.testing.assert_allclose(target[:2, 0], [0, 1], atol=1e-6)
    np.testing.assert_allclose(target[:, 2], 0, atol=1e-8)
    np.testing.assert_allclose(target[:, 1], [0, 0, 1])
    np.testing.assert_allclose(target[2, 0], 0.299, atol=1e-6)
    np.testing.assert_array_equal(attacks.animal_vehicle(np.arange(10)), [1, 1, 0, 0, 0, 0, 0, 0, 1, 1])


def test_fit_selection_never_reads_holdout_and_resume_is_deterministic(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(412)
    u = rng.normal(size=(24, 256)).astype(np.float32)
    u /= np.sqrt(np.mean(u * u, 1, keepdims=True))
    case = dict(name='unit_case', key_hex='00' * 32, train_u=u,
                train_images=rng.integers(0, 256, (24, 3, 32, 32), dtype=np.uint8),
                train_labels=np.arange(24) % 10, fit=np.arange(16), val=np.arange(16, 24))
    cfg = {'attacks': {}}
    settings = attacks._settings(cfg, True)
    spec = dict(name='small_classifier', kind='expressive_class', task='classification',
                aug=True, nonce=False, view='fusion', seed=77, epochs=2,
                snr=20.0, packet_domain='unit_classifier')
    report = attacks.fit_one(case, spec, cfg, settings, tmp_path / 'complete', torch.device('cpu'), True)
    assert report['test_used_for_fit_or_selection'] is False
    history = json.loads((tmp_path / 'complete' / 'history.json').read_text())
    assert len(history) == 2
    expected = attacks.select_validation_checkpoints(history, 'accuracy')
    assert report['selection']['metric_epoch'] == expected['metric_epoch']
    assert report['selection']['loss_epoch'] == expected['loss_epoch']
    report_again = attacks.fit_one(case, spec, cfg, settings, tmp_path / 'complete', torch.device('cpu'), True)
    assert report_again == report
    original_validation = attacks._validation
    calls = [0]
    def interrupt_second_epoch(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError('deliberate simulated interruption')
        return original_validation(*args, **kwargs)
    attacks._validation = interrupt_second_epoch
    try:
        with pytest.raises(RuntimeError, match='simulated interruption'):
            attacks.fit_one(case, spec, cfg, settings, tmp_path / 'interrupted', torch.device('cpu'), True)
    finally:
        attacks._validation = original_validation
    resumed = attacks.fit_one(case, spec, cfg, settings, tmp_path / 'interrupted', torch.device('cpu'), True)
    assert resumed['selection'] == report['selection']
    for criterion in ('loss', 'metric'):
        uninterrupted = torch.load(tmp_path / 'complete' / ('best_' + criterion + '.pt'), weights_only=False)
        restarted = torch.load(tmp_path / 'interrupted' / ('best_' + criterion + '.pt'), weights_only=False)
        assert uninterrupted['epoch'] == restarted['epoch']
        for key in uninterrupted['state_dict']:
            torch.testing.assert_close(uninterrupted['state_dict'][key], restarted['state_dict'][key], rtol=0, atol=0)
    changed = copy.deepcopy(cfg)
    changed['attacks']['primary_epochs'] = 99
    with pytest.raises(RuntimeError, match='signature mismatch'):
        attacks.fit_one(case, spec, changed, settings, tmp_path / 'complete', torch.device('cpu'), True)
