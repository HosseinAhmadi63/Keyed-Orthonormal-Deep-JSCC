from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pickle
import tarfile
import numpy as np
import torch
from keyed_jscc.codec_models import Encoder, Decoder
from keyed_jscc.experiment_utils import split_indices
from .common import download_file, encode_images, save_npz, sha256_file, write_json

def verify_assets(root, assets_dir=None):
    root = Path(root)
    assets = Path(assets_dir) if assets_dir is not None else root/'assets'
    manifest = json.loads((assets/'ASSET_MANIFEST.json').read_text())
    for row in manifest['files']:
        path = assets/row['path']
        if not path.exists() or path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
            raise RuntimeError(f'Saved experimental asset failed its checksum: {row["path"]}')
    print(f'Verified {len(manifest["files"])} saved experimental assets.', flush=True)

def load_cifar(cfg, smoke=False):
    if smoke:
        rng = np.random.default_rng(42)
        images = rng.integers(0,256,(64,3,32,32),dtype=np.uint8)
        labels = np.arange(64,dtype=np.int64)%10
        return images[:48],labels[:48],images[48:],labels[48:]
    folder = Path(cfg['dataset_dir'])
    folder.mkdir(exist_ok=True, parents=True)
    record = json.loads((Path(cfg['assets_dir'])/'DATASETS.json').read_text())['cifar10']
    path = folder/'cifar-10-python.tar.gz'
    for candidate in (folder/'cifar-10-python.tar',folder/'cifar-10-python.tar.gz'):
        if candidate.exists() and candidate.stat().st_size == record['bytes'] and sha256_file(candidate) == record['sha256']:
            path = candidate
            break
    if not path.exists():
        for url in record['urls']:
            try:
                download_file(url,path,expected_sha256=record['sha256'])
                break
            except RuntimeError:
                if url == record['urls'][-1]:
                    raise
    if sha256_file(path) != record['sha256']:
        raise RuntimeError('CIFAR-10 archive differs from the original official dataset')
    xs,ys=[],[]
    with tarfile.open(path,'r:*') as archive:
        for i in range(1,6):
            member = archive.extractfile(f'cifar-10-batches-py/data_batch_{i}')
            obj = pickle.load(member,encoding='bytes')
            xs.append(obj[b'data'])
            ys.extend(obj[b'labels'])
        obj = pickle.load(archive.extractfile('cifar-10-batches-py/test_batch'),encoding='bytes')
    return np.concatenate(xs).reshape(-1,3,32,32),np.asarray(ys,dtype=np.int64),obj[b'data'].reshape(-1,3,32,32)[6400:].copy(),np.asarray(obj[b'labels'],dtype=np.int64)[6400:]

def load_case(name, cfg, data, device, smoke=False):
    assets = Path(cfg['assets_dir'])
    x,y,tx,ty = data
    legacy = name == 'frozen_seed42'
    source = assets/name/'best_codec.pt'
    checkpoint = torch.load(source,map_location='cpu',weights_only=False)
    enc,dec=Encoder(256).to(device),Decoder(256).to(device)
    enc.load_state_dict(checkpoint['enc'],strict=True)
    dec.load_state_dict(checkpoint['dec'],strict=True)
    enc.eval();dec.eval()
    for module in (enc,dec):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    if smoke:
        fit,val=np.arange(32),np.arange(32,48)
        u,tu=encode_images(enc,x,device,8),encode_images(enc,tx,device,8)
    elif legacy:
        fit,val=split_indices(y,dict(split_seed=20261015,validation_per_class=500))
        cache = Path(cfg['cache_dir'])/name/'latents.npz'
        meta=cache.with_suffix('.json')
        fingerprint=dict(codec_sha256=sha256_file(source),train_images_sha256=hashlib.sha256(np.asarray(x).tobytes()).hexdigest(),test_images_sha256=hashlib.sha256(np.asarray(tx).tobytes()).hexdigest())
        if cache.exists():
            stored=json.loads(meta.read_text()) if meta.exists() else {}
            if stored.get('inputs') != fingerprint or stored.get('sha256') != sha256_file(cache):
                raise RuntimeError('Legacy latent cache does not match the codec and source images. Remove the derived cache and restart.')
            with np.load(cache) as z:
                u,tu=z['train'],z['holdout']
        else:
            print(f'Encoding public CIFAR images for {name}.',flush=True)
            u,tu=encode_images(enc,x,device),encode_images(enc,tx,device)
            save_npz(cache,train=u,holdout=tu)
            write_json(meta,dict(inputs=fingerprint,sha256=sha256_file(cache),train_shape=list(u.shape),test_shape=list(tu.shape)))
    else:
        with np.load(assets/'split_manifest.npz') as z:
            fit,val=z['fitting'],z['validation']
        with np.load(assets/name/'latents.npz') as z:
            u,tu=z['training'],z['holdout']
        if u.shape != (50000,256) or tu.shape != (3600,256):
            raise RuntimeError('Latent archive dimensions differ from the frozen study')
    keys=json.loads((assets/'experiment_keys.json').read_text())
    if legacy:
        key=json.loads((assets/'legacy_experiment_keys.json').read_text())['key_hex']
    else:
        keyrecord=keys[name]
        key=keyrecord if isinstance(keyrecord,str) else keyrecord['training_key']
    seed = 42 if legacy else int(name.split('_')[0][4:])
    weight = 0.2 if legacy else float(name.split('lambda')[1])
    return dict(name=name,seed=seed,weight=weight,case_index=0 if legacy else 1+2*(seed-1001)+int(weight>0),train_u=u,test_u=tu,train_images=x,test_images=tx,train_labels=y,test_labels=ty,fit=fit,val=val,enc=enc,dec=dec,key_hex=key,cache_dir=Path(cfg['cache_dir'])/name,codec_sha256=sha256_file(source),codec_selected_epoch=checkpoint.get('selected_epoch'),legacy=legacy)
