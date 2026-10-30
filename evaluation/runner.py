from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time
import traceback
import zipfile
import numpy as np
import torch
from . import attacks, channels, natural, protocol, diagnostics, baselines
from .common import sha256_file, fit_signature, write_json
from .data import verify_assets, load_cifar, load_case

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache'/'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'cache'))

def default_config():
    return dict(schema_version=2, name='full', seed=20261024, threads=8,
                stages=['attacks','diagnostics','channels','baselines','natural','protocol'],
                cases=[f'seed{s}_lambda{l}' for s in (1001,1002,1003) for l in ('0','0.2')],
                natural_cases=['frozen_seed42','seed1001_lambda0.2'],
                channel_cases=['seed1001_lambda0','seed1001_lambda0.2','seed1002_lambda0.2','seed1003_lambda0.2'],
                baseline_cases=['seed1001_lambda0','seed1001_lambda0.2'],
                attacks=attacks.defaults(),channels=channels.defaults(),natural=natural.defaults(),protocol=protocol.protocol_defaults(),baselines=baselines.defaults())

def source_inventory():
    paths=[]
    for sub in ('evaluation','src','configs','scripts','tests'):
        paths.extend(p for p in (ROOT/sub).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.txt','.ps1','.bat'))
    paths.extend(ROOT/name for name in ('requirements.txt','requirements-evaluation.txt','pyproject.toml') if (ROOT/name).exists())
    paths=sorted(paths)
    return [{'path':p.relative_to(ROOT).as_posix(),'sha256':sha256_file(p)} for p in paths]

def hardware_report(device):
    result = dict(platform=platform.platform(), python=sys.version, executable=sys.executable, torch=torch.__version__, numpy=np.__version__, cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(), threads=torch.get_num_threads(), device=str(device))
    if device.type == 'cuda':
        p=torch.cuda.get_device_properties(device)
        result.update(gpu=p.name,vram_bytes=p.total_memory,compute_capability=list(torch.cuda.get_device_capability(device)))
    return result

@contextmanager
def exclusive_run(path):
    path=Path(path)
    path.parent.mkdir(exist_ok=True,parents=True)
    stream=path.open('a+b')
    if path.stat().st_size == 0:
        stream.write(b'0'); stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError('This results folder is already running in another window. Keep one experiment window open.') from exc
    try:
        yield
    finally:
        if os.name == 'nt':
            stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        stream.close()

def _stage_files(folder):
    rows=[]
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.name != 'STAGE_COMPLETE.json' and '__pycache__' not in path.parts:
            rows.append(dict(path=path.relative_to(folder).as_posix(),bytes=path.stat().st_size,sha256=sha256_file(path)))
    return rows

def _already_complete(folder,signature):
    stamp=folder/'STAGE_COMPLETE.json'
    if not stamp.exists():
        return False
    record=json.loads(stamp.read_text())
    if record['signature'] != signature:
        raise RuntimeError(f'Stage configuration changed: {folder}. Use a new results folder.')
    for row in record['files']:
        path=folder/row['path']
        if not path.exists() or path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
            raise RuntimeError(f'Completed output is missing or modified: {path}')
    return True

def result_archive(outdir,smoke=False):
    outdir=Path(outdir)
    target=ROOT/('SMOKE_EVALUATION_RESULTS.zip' if smoke else 'EVALUATION_RESULTS.zip')
    tmp=target.with_suffix('.zip.partial')
    excluded={'crop_cache','packet_cache','__pycache__'}
    count=0;retained=[];compacted=[]
    manifest_path=outdir/'RUN_MANIFEST.json'
    recorded=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    assets=Path(recorded.get('config',{}).get('assets_dir',str(ROOT/'assets')))
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for path in sorted(outdir.rglob('*')):
            if path.is_file() and not excluded.intersection(path.relative_to(outdir).parts) and path.name != '.running.lock':
                destination=Path('results')/path.relative_to(outdir)
                if path.suffix in ('.pt','.pth','.joblib'):
                    retained.append(dict(path=path.relative_to(outdir).as_posix(),bytes=path.stat().st_size,sha256=sha256_file(path),retained_on_source_machine=True));continue
                if path.suffix == '.npz':
                    arrays={};removed=[]
                    with np.load(path,allow_pickle=False) as data:
                        for key in data.files:
                            value=data[key]
                            if 'prediction' in key.lower() and value.ndim >= 4 and len(value)>16:
                                arrays['first16_'+key]=value[:16]
                                removed.append(dict(array=key,original_shape=list(value.shape),retained_shape=list(value[:16].shape)))
                            else:
                                arrays[key]=value
                    if removed:
                        stream=io.BytesIO();np.savez_compressed(stream,**arrays)
                        archive.writestr(destination.as_posix(),stream.getvalue())
                        compacted.append(dict(path=path.relative_to(outdir).as_posix(),original_sha256=sha256_file(path),arrays=removed,full_arrays_retained_on_source_machine=True));count+=1;continue
                archive.write(path,destination);count+=1
        for sub in ('evaluation','src','configs','scripts','tests'):
            for path in sorted((ROOT/sub).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    archive.write(path,Path('executed_source')/path.relative_to(ROOT))
        for name in ('pyproject.toml','requirements.txt','requirements-evaluation.txt','README.md','docs/EVALUATION.md'):
            if (ROOT/name).exists():
                archive.write(ROOT/name,Path('executed_source')/name)
        for name in ('ASSET_MANIFEST.json','DATASETS.json','split_manifest.npz','experiment_keys.json','legacy_experiment_keys.json','protocol_vectors.json','codec_training_curves.csv'):
            archive.write(assets/name,Path('executed_source')/'assets'/name)
        archive.writestr('RESULTS_INVENTORY.json',json.dumps(dict(scope='Complete numerical metric vectors, curves, manifests and representative images; executable model weights and full image prediction arrays remain in the source results directory.',retained_model_files=retained,compacted_prediction_files=compacted),indent=2))
    os.replace(tmp,target)
    (target.with_suffix('.zip.sha256')).write_text(sha256_file(target)+'  '+target.name+'\n')
    print(f'Results archive: {target} ({target.stat().st_size/2**30:.2f} GiB; {count} result files)',flush=True)
    return target

def model_archives(outdir,limit_mib=450):
    outdir=Path(outdir)
    target=ROOT/'model_parts';target.mkdir(exist_ok=True)
    records=[];size=0;index=0;archive=None
    try:
        for path in sorted(outdir.rglob('*')):
            if not path.is_file() or path.suffix not in ('.pt','.pth','.joblib'):
                continue
            if path.name in ('resume.pt','last.pt','training_state.pt') or 'last' in path.stem or 'resume' in path.stem:
                continue
            if archive is None or size+path.stat().st_size > limit_mib*2**20:
                if archive:
                    archive.close()
                index+=1;size=0
                name=target/f'MODELS_part{index:03}.zip'
                archive=zipfile.ZipFile(name,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True)
                records.append(name)
            archive.write(path,Path('results')/path.relative_to(outdir));size+=path.stat().st_size
    finally:
        if archive:
            archive.close()
    write_json(target/'MODEL_PARTS.json',dict(parts=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha256_file(p)) for p in records],scope='Selected model checkpoints, excluding optimizer-only resume states; each ZIP opens independently'))
    print(f'Wrote {len(records)} independent model ZIP files to {target}',flush=True)

def execute(cfg,outdir,device,smoke=False,selected_stages=None):
    outdir=Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(json.dumps(cfg))
    cfg.update(assets_dir=str(Path(cfg.get('assets_dir', ROOT/'assets')).expanduser().resolve()),dataset_dir=str(ROOT/'data'),cache_dir=str(ROOT/'cache'))
    if smoke:
        cfg['cases']=['seed1001_lambda0.2']
        cfg['natural_cases']=cfg['cases']
        cfg['channel_cases']=cfg['cases']
        cfg['baseline_cases']=cfg['cases']
    scientific_source=source_inventory()
    asset_manifest_sha256=sha256_file(Path(cfg['assets_dir'])/'ASSET_MANIFEST.json')
    signature=fit_signature(cfg,dict(source=scientific_source,asset_manifest_sha256=asset_manifest_sha256,smoke=smoke))
    lock=outdir/'RUN_MANIFEST.json'
    if lock.exists() and json.loads(lock.read_text())['signature'] != signature:
        raise RuntimeError('Configuration or executable source changed. Preserve the current results and choose a new --output directory.')
    if not lock.exists():
        write_json(lock,dict(signature=signature,config=cfg,source=scientific_source,asset_manifest_sha256=asset_manifest_sha256,created_utc=datetime.now(timezone.utc).isoformat(),scientific_results=not smoke,selection_data='synthetic 16-image validation split' if smoke else 'original 5000-image training-derived validation split',test_data='synthetic software fixture' if smoke else 'CIFAR-10 official test indices 6400..9999 and Kodak frames; excluded from fitting/selection'))
    environment=hardware_report(device)
    environment_path=outdir/'hardware_report.json'
    if environment_path.exists():
        initial=json.loads(environment_path.read_text())
        for key in ('torch','numpy','cuda_runtime','cudnn'):
            if initial.get(key) != environment.get(key):
                raise RuntimeError(f'Runtime library changed since the original run: {key}. Use a new results folder.')
    else:
        write_json(environment_path,environment)
    sessions=outdir/'execution_sessions.json'
    history=json.loads(sessions.read_text()) if sessions.exists() else []
    history.append(dict(started_utc=datetime.now(timezone.utc).isoformat(),environment=environment))
    write_json(sessions,history)
    verify_assets(ROOT, cfg['assets_dir'])
    data=load_cifar(cfg,smoke)
    stages=selected_stages or cfg['stages']
    tasks=[]
    for stage in stages:
        if stage == 'protocol':
            tasks.append((stage,None))
        elif stage == 'natural':
            tasks.extend((stage,name) for name in cfg['natural_cases'])
        elif stage == 'channels':
            tasks.extend((stage,name) for name in cfg['channel_cases'])
        elif stage == 'baselines':
            tasks.extend((stage,name) for name in cfg['baseline_cases'])
        elif stage == 'attacks':
            tasks.extend((stage,name) for name in cfg['cases'])
        elif stage == 'diagnostics':
            tasks.extend((stage,name) for name in cfg['cases'])
        else:
            raise ValueError('Unknown stage '+stage)
    status=dict(scientific_results=not smoke,requested_tasks=[f'{s}/{n}' if n else s for s,n in tasks],completed=[],elapsed_seconds=0)
    started=time.monotonic()
    try:
        for index,(stage,name) in enumerate(tasks):
            label=stage+'/'+name if name else stage
            status.update(current=label,state='running',elapsed_seconds=time.monotonic()-started)
            write_json(outdir/'STATUS.json',status)
            folder=outdir/stage/(name or 'global')
            job_signature=fit_signature(signature,label)
            print(f'\nSTAGE {index+1}/{len(tasks)}: {label}',flush=True)
            if _already_complete(folder,job_signature):
                print('Saved stage verified; continuing.',flush=True)
            else:
                folder.mkdir(parents=True,exist_ok=True)
                begun=time.monotonic()
                if stage == 'protocol':
                    protocol.run_protocol(cfg,folder,smoke)
                else:
                    case=load_case(name,cfg,data,device,smoke)
                    if stage == 'baselines':
                        from . import baselines
                        module=baselines
                    else:
                        module={'attacks':attacks,'channels':channels,'natural':natural,'diagnostics':diagnostics}[stage]
                    module.run(case,cfg,folder,device,smoke)
                    if stage == 'diagnostics':
                        diagnostics.training_summary(cfg,folder)
                    del case
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                write_json(folder/'STAGE_COMPLETE.json',dict(signature=job_signature,scientific_results=not smoke,elapsed_seconds=time.monotonic()-begun,files=_stage_files(folder)))
            status['completed'].append(label)
            print(f'Completed {label}; {len(status["completed"])}/{len(tasks)} stages finished.',flush=True)
        status.update(state='complete',current=None,elapsed_seconds=time.monotonic()-started)
        write_json(outdir/'STATUS.json',status)
    except BaseException as exc:
        status.update(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),elapsed_seconds=time.monotonic()-started)
        write_json(outdir/'STATUS.json',status)
        (outdir/'failure.txt').write_text(traceback.format_exc())
        raise
    result_archive(outdir,smoke)
    print('\nSMOKE TEST COMPLETE - synthetic results, not scientific evidence.' if smoke else '\nALL REQUESTED EXPERIMENTS COMPLETE. The numerical archive and checksum are saved in the project directory.',flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs'/'evaluation.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--assets-dir',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--stages',nargs='+',choices=['attacks','channels','baselines','natural','protocol','diagnostics'])
    parser.add_argument('--archive-only',action='store_true')
    parser.add_argument('--collect-models',action='store_true')
    parser.add_argument('--collect-full-results',action='store_true')
    parser.add_argument('--write-default-config',action='store_true')
    args=parser.parse_args()
    if args.write_default_config:
        write_json(args.config,default_config());return
    cfg=json.loads(args.config.read_text())
    if args.assets_dir is not None:
        cfg['assets_dir']=str(args.assets_dir.expanduser().resolve())
    torch.set_num_threads(cfg.get('threads',8))
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device)
    collection=args.archive_only or args.collect_models or args.collect_full_results
    if not args.smoke and device.type != 'cuda' and not collection:
        raise RuntimeError('The full training run requires a CUDA-capable GPU. Use --smoke --device cpu only for a software check.')
    if device.type == 'cuda' and not torch.cuda.is_available() and not collection:
        raise RuntimeError('CUDA GPU is unavailable. Check the installed PyTorch CUDA build and GPU driver.')
    outdir=args.output or ROOT/'outputs'/('evaluation_smoke' if args.smoke else cfg.get('name','evaluation'))
    if args.archive_only:
        result_archive(outdir,args.smoke);return
    if args.collect_models:
        model_archives(outdir);return
    if args.collect_full_results:
        target=ROOT/'COMPLETE_RESULTS.zip'
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
            for path in sorted(outdir.rglob('*')):
                if path.is_file() and not {'crop_cache','packet_cache','__pycache__'}.intersection(path.relative_to(outdir).parts) and path.name != '.running.lock':
                    archive.write(path,Path('results')/path.relative_to(outdir))
        target.with_suffix('.zip.sha256').write_text(sha256_file(target)+'  '+target.name+'\n')
        print(f'Complete scientific outputs: {target}',flush=True);return
    with exclusive_run(outdir/'.running.lock'):
        execute(cfg,outdir,device,args.smoke,args.stages)
