import io
import os
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pytest
from evaluation import common, runner
from evaluation.data import load_case

def test_actual_packets_domain_separation_and_exact_inverse(tmp_path):
    rng=np.random.default_rng(5)
    u=rng.normal(size=(20,256)).astype(np.float32)
    case=dict(name='test',key_hex=bytes(range(32)).hex(),cache_dir=tmp_path)
    a,n=common.actual_observations(u,case,'one',float('inf'),repetitions=2)
    b,m=common.actual_observations(u,case,'two',float('inf'),repetitions=2)
    assert a.shape==(20,2,256) and n.shape==(20,2,6)
    assert not np.allclose(a[:,0],a[:,1]) and not np.allclose(a,b)
    np.testing.assert_allclose(common.bob_observations(u,case,'one',float('inf')),u,atol=1e-6)
    np.testing.assert_allclose(np.sum(a**2,axis=2),np.sum(u**2,axis=1)[:,None].repeat(2,1),rtol=2e-6)

def test_json_rejects_nonfinite_and_preserves_explicit_none(tmp_path):
    with pytest.raises(ValueError):
        common.write_json(tmp_path/'bad.json',dict(metric=float('nan')))
    common.write_json(tmp_path/'good.json',dict(legitimate_unbounded_value=None))
    assert json.loads((tmp_path/'good.json').read_text())['legitimate_unbounded_value'] is None

def test_source_inventory_ignores_runtime_changes(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    (tmp_path/'evaluation').mkdir();(tmp_path/'evaluation'/'a.py').write_text('x=1\n')
    before=runner.source_inventory()
    (tmp_path/'logs').mkdir();(tmp_path/'logs'/'running.txt').write_text('mutable\n')
    (tmp_path/'hardware_report.json').write_text('{}')
    assert runner.source_inventory()==before
    (tmp_path/'evaluation'/'a.py').write_text('x=2\n')
    assert runner.source_inventory()!=before

def test_single_run_lock(tmp_path):
    with runner.exclusive_run(tmp_path/'lock'):
        with pytest.raises(RuntimeError):
            with runner.exclusive_run(tmp_path/'lock'):
                pass

def _return_fixture(tmp_path,monkeypatch):
    original=runner.ROOT
    for folder in ('evaluation','src','configs','scripts','tests','assets'):
        (tmp_path/folder).mkdir()
    for name in ('ASSET_MANIFEST.json','DATASETS.json','split_manifest.npz','experiment_keys.json','legacy_experiment_keys.json','protocol_vectors.json','codec_training_curves.csv'):
        shutil.copyfile(original/'assets'/name,tmp_path/'assets'/name)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    out=tmp_path/'results'/'test';out.mkdir(parents=True)
    return out

def test_return_metrics_complete_and_original_predictions_preserved(tmp_path,monkeypatch):
    out=_return_fixture(tmp_path,monkeypatch)
    metrics=np.arange(32,dtype=np.float64)
    common.save_npz(out/'per_image.npz',mse=metrics,predictions_float32=np.ones((32,3,32,32),np.float32))
    before=common.sha256_file(out/'per_image.npz')
    (out/'best.pt').write_bytes(b'checkpoint-fixture')
    target=runner.result_archive(out,smoke=True)
    with zipfile.ZipFile(target) as z:
        with np.load(io.BytesIO(z.read('results/per_image.npz'))) as data:
            np.testing.assert_array_equal(data['mse'],metrics)
            assert data['first16_predictions_float32'].shape==(16,3,32,32)
        record=json.loads(z.read('RESULTS_INVENTORY.json'))
        assert record['retained_model_files'][0]['path']=='best.pt'
    assert common.sha256_file(out/'per_image.npz')==before

def test_completed_stage_detects_modified_outputs(tmp_path):
    (tmp_path/'result.json').write_text('{}')
    common.write_json(tmp_path/'STAGE_COMPLETE.json',dict(signature='one',files=runner._stage_files(tmp_path)))
    assert runner._already_complete(tmp_path,'one')
    (tmp_path/'result.json').write_text('{"changed":true}')
    with pytest.raises(RuntimeError):
        runner._already_complete(tmp_path,'one')

def test_model_collection_retains_selected_and_excludes_resume(tmp_path,monkeypatch):
    out=_return_fixture(tmp_path,monkeypatch)
    (out/'best.pt').write_bytes(b'selected')
    (out/'last.pt').write_bytes(b'optimizer-state')
    runner.model_archives(out)
    parts=sorted((tmp_path/'model_parts').glob('*.zip'))
    assert len(parts)==1
    with zipfile.ZipFile(parts[0]) as z:
        assert z.namelist()==['results/best.pt']

def test_metric_function_reports_reference_identity():
    image=np.random.default_rng(4).integers(0,256,(2,3,32,32),dtype=np.uint8)
    rows=common.image_rows(image.astype(np.float32)/255,image)
    np.testing.assert_allclose(rows['mse'],0,atol=1e-12)
    np.testing.assert_allclose(rows['ssim_valid'],1,atol=1e-6)

@pytest.mark.requires_assets
def test_real_saved_case_loads_frozen_archive_format(tmp_path):
    asset_dir=Path(os.environ.get('KEYED_JSCC_ASSETS', str(Path(__file__).resolve().parents[1]/'assets')))
    if not (asset_dir/'seed1001_lambda0.2/best_codec.pt').exists() or not (asset_dir/'seed1001_lambda0.2/latents.npz').exists():
        pytest.skip('Saved codec and latent assets are supplied separately')
    data=(np.broadcast_to(np.zeros((1,3,32,32),np.uint8),(50000,3,32,32)),np.arange(50000)%10,np.broadcast_to(np.zeros((1,3,32,32),np.uint8),(3600,3,32,32)),np.arange(3600)%10)
    case=load_case('seed1001_lambda0.2',dict(assets_dir=str(asset_dir),cache_dir=str(tmp_path)),data,'cpu',smoke=False)
    assert case['train_u'].shape==(50000,256) and case['test_u'].shape==(3600,256)
    assert len(case['fit'])==45000 and len(case['val'])==5000
    assert not np.intersect1d(case['fit'],case['val']).size
    assert not any(p.requires_grad for p in case['enc'].parameters())


def test_result_archive_uses_recorded_external_asset_metadata(tmp_path,monkeypatch):
    out=_return_fixture(tmp_path,monkeypatch)
    external=tmp_path/'external_assets'
    shutil.copytree(tmp_path/'assets',external)
    (external/'DATASETS.json').write_text('{"external_fixture": true}')
    common.write_json(out/'RUN_MANIFEST.json',dict(config=dict(assets_dir=str(external))))
    target=runner.result_archive(out,smoke=True)
    with zipfile.ZipFile(target) as archive:
        assert json.loads(archive.read('executed_source/assets/DATASETS.json'))=={'external_fixture':True}
