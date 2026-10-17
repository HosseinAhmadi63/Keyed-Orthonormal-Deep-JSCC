from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from kodak_evaluation import EXPERIMENTS, assemble_tiles, full_image_metrics, aggregate_image_metrics, write_json
from ssim_ms import ssim_per_image

class DirectMLPRecon(torch.nn.Module):

    def __init__(self, latent_n: int, nonce_dim: int):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(latent_n + nonce_dim, 512), torch.nn.SiLU(), torch.nn.Linear(512, 1024), torch.nn.SiLU(), torch.nn.Linear(1024, 3072), torch.nn.Sigmoid())

    def forward(self, observations, public_nonce):
        return self.mlp(torch.cat((observations, public_nonce), dim=-1)).view(-1, 3, 32, 32)

@torch.inference_mode()
def predict(model, obs, nonce, images, batch_size):
    outputs = []
    mse = []
    ssim = []
    for start in range(0, len(obs), batch_size):
        observed = torch.from_numpy(np.ascontiguousarray(obs[start:start + batch_size]))
        public = torch.from_numpy(np.ascontiguousarray(nonce[start:start + batch_size]))
        target = torch.from_numpy(images[start:start + batch_size].copy()).float() / 255.0
        decoded = model(observed, public)
        outputs.append(decoded.numpy())
        mse.append((decoded - target).square().mean((1, 2, 3)).numpy())
        ssim.append(ssim_per_image(decoded, target).numpy())
    return (np.concatenate(outputs), np.concatenate(mse), np.concatenate(ssim))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', type=Path, nargs='+')
    parser.add_argument('--cache', type=Path, default=EXPERIMENTS / 'data/kodak_packet_cache.npz')
    parser.add_argument('--prior-results', type=Path, default=EXPERIMENTS / 'results/kodak/kodak_per_image.json')
    parser.add_argument('--data', type=Path, default=EXPERIMENTS / 'data/kodak')
    parser.add_argument('--output', type=Path, default=EXPERIMENTS / 'results/kodak_fresh')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    prior = json.loads(args.prior_results.read_text())
    rows = prior['images']
    cache = np.load(args.cache, allow_pickle=False)
    obs = cache['eve_observations']
    nonce = cache['nonce_features']
    images = cache['images']
    image_index = cache['image_index']
    metadata = json.loads(str(cache['metadata_json'].item()))
    assert len(rows) == 24, 'publication Kodak evaluation requires all24 source images'
    assert np.array_equal(np.unique(image_index), np.arange(24))
    model_metadata = {}
    per_tile = {'image_index': image_index}
    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        architecture = checkpoint.get('architecture')
        if architecture != [obs.shape[1] + nonce.shape[1], 512, 1024, 3072]:
            raise ValueError(f'unsupported saved architecture: {architecture}')
        label = checkpoint_path.stem
        if label.endswith('_attacker'):
            label = label[:-9]
        label = 'fresh_' + label
        if label in model_metadata:
            raise ValueError('checkpoint labels must be unique')
        model = DirectMLPRecon(obs.shape[1], nonce.shape[1]).eval()
        model.load_state_dict(checkpoint['state_dict'])
        predicted, mse, ssim = predict(model, obs, nonce, images, args.batch_size)
        model_metadata[label] = {'checkpoint': str(checkpoint_path), 'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), 'initialization_seed': checkpoint.get('seed'), 'selected_validation_epoch': checkpoint.get('selected_validation_epoch'), 'selected_validation_mse': checkpoint.get('selected_validation_mse'), 'architecture': architecture, 'fit_on_kodak': False, 'selection_on_kodak': False}
        destination = args.output / 'reconstructions' / label
        destination.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows):
            selected = np.flatnonzero(image_index == index)
            full = assemble_tiles(predicted[selected], row['height'], row['width'])
            reference = np.asarray(Image.open(args.data / row['name']).convert('RGB'))
            metrics = full_image_metrics(full, reference)
            metrics.update({'mean_tile_mse': float(mse[selected].mean()), 'mean_tile_ssim': float(ssim[selected].mean()), 'tile_ssim_p95': float(np.percentile(ssim[selected], 95))})
            row['models'][label] = metrics
            Image.fromarray(np.rint(np.clip(full, 0, 1) * 255).astype(np.uint8)).save(destination / row['name'])
        per_tile[label + '_mse'] = mse
        per_tile[label + '_ssim'] = ssim
        metadata['fresh_attack_models'] = model_metadata
        metadata['fresh_attack_transfer_scope'] = 'CIFAR-10 fitting/validation only; Kodak patches evaluated at original nativeimage scale; all96publicnonce bits supplied'
        write_json(args.output / 'kodak_per_image.json', {'metadata': metadata, 'images': rows})
        print(f"Completed{label}: mean native-image SSIM={np.mean([row['models'][label]['ssim'] for row in rows]):.6f}; elapsed{time.monotonic() - started:.1f}s", flush=True)
    aggregate = aggregate_image_metrics(rows, 20261020)
    write_json(args.output / 'kodak_metrics.json', {'metadata': metadata, **aggregate})
    np.savez(args.output / 'kodak_fresh_per_tile.npz', **per_tile)
    print(json.dumps({'complete': True, 'source_images': 24, 'tiles': len(image_index), 'elapsed_s': time.monotonic() - started}), flush=True)
if __name__ == '__main__':
    main()
