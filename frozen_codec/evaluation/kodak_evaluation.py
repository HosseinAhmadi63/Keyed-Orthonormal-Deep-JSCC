from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tarfile
import time
import numpy as np
from PIL import Image
import torch
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'archived_implementation'
sys.path.insert(0, str(SOURCE))
from models.autoencoder import Encoder, Decoder
from models.attackers import EveReconNoNonce, EveBobStyleDecoder
from ssim_ms import ssim_per_image
from crypto_scrambler import CryptoScrambler, NonceCounter, nonce_features, PROTOCOL
ARCHIVED = ROOT / 'models'
EXPERIMENTS = ROOT / 'evaluation'
PATCH = 32
NONCE_START = 23307203198626192505389973504

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def tile_image(image: np.ndarray) -> np.ndarray:
    height, width, channels = image.shape
    if channels != 3 or height % PATCH or width % PATCH:
        raise ValueError('native RGB image dimensions must be divisible by32')
    return np.ascontiguousarray(image.reshape(height // PATCH, PATCH, width // PATCH, PATCH, 3).transpose(0, 2, 4, 1, 3).reshape(-1, 3, PATCH, PATCH))

def assemble_tiles(tiles: np.ndarray, height: int, width: int) -> np.ndarray:
    expected = height // PATCH * (width // PATCH)
    if tiles.shape != (expected, 3, PATCH, PATCH):
        raise ValueError('patch count/shape differs from source image')
    return tiles.reshape(height // PATCH, width // PATCH, 3, PATCH, PATCH).transpose(0, 3, 1, 4, 2).reshape(height, width, 3)

def training_mean_image(archive: Path) -> np.ndarray:
    sums = np.zeros((3, PATCH, PATCH), dtype=np.float64)
    count = 0
    with tarfile.open(archive, 'r:gz') as file:
        for part in range(1, 6):
            data = pickle.load(file.extractfile(f'cifar-10-batches-py/data_batch_{part}'), encoding='bytes')
            images = data[b'data'].reshape(-1, 3, PATCH, PATCH)
            sums += images.astype(np.float64).sum(axis=0)
            count += len(images)
    return (sums / (count * 255.0)).astype(np.float32)

@torch.inference_mode()
def predict_patches(model, inputs: np.ndarray, batch_size: int, images: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    outputs, mse_parts, ssim_parts = ([], [], [])
    for start in range(0, len(inputs), batch_size):
        observed = torch.from_numpy(np.ascontiguousarray(inputs[start:start + batch_size]))
        pred = model(observed)
        outputs.append(pred.numpy())
        if images is not None:
            reference = torch.from_numpy(images[start:start + batch_size].copy()).float() / 255.0
            mse_parts.append((pred - reference).square().mean((1, 2, 3)).numpy())
            ssim_parts.append(ssim_per_image(pred, reference).numpy())
    return (np.concatenate(outputs), np.concatenate(mse_parts) if mse_parts else None, np.concatenate(ssim_parts) if ssim_parts else None)

@torch.inference_mode()
def full_image_metrics(pred: np.ndarray, reference: np.ndarray) -> dict:
    estimate = torch.from_numpy(np.ascontiguousarray(pred.transpose(2, 0, 1)))[None]
    target = torch.from_numpy(np.ascontiguousarray(reference.transpose(2, 0, 1)))[None].float() / 255.0
    mse = float((estimate - target).square().mean())
    ssim = float(ssim_per_image(estimate, target)[0])
    return {'mse': mse, 'psnr': float(-10 * np.log10(max(mse, 1e-12))), 'ssim': ssim}

def aggregate_image_metrics(rows: list[dict], bootstrap_seed: int, repetitions: int=10000) -> dict:
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(rows), size=(repetitions, len(rows)))
    models = list(rows[0]['models'])
    report = {'source_images': len(rows), 'bootstrap_repetitions': repetitions, 'bootstrap_seed': bootstrap_seed, 'bootstrap_unit': 'source image', 'ci_method': 'paired nonparametric image-cluster percentile bootstrap (95%)', 'metrics': {}, 'paired_bob_ssim_gaps': {}}
    for model in models:
        mse = np.asarray([row['models'][model]['mse'] for row in rows], np.float64)
        ssim = np.asarray([row['models'][model]['ssim'] for row in rows], np.float64)
        boot_mse = mse[indices].mean(axis=1)
        boot_ssim = ssim[indices].mean(axis=1)

        def ci(array):
            return np.percentile(array, [2.5, 97.5]).tolist()
        report['metrics'][model] = {'mean_image_mse': float(mse.mean()), 'mse_95ci': ci(boot_mse), 'dataset_psnr': float(-10 * np.log10(mse.mean())), 'dataset_psnr_95ci': ci(-10 * np.log10(boot_mse)), 'mean_image_psnr': float(np.mean(-10 * np.log10(mse))), 'mean_image_ssim': float(ssim.mean()), 'ssim_95ci': ci(boot_ssim), 'image_ssim_min': float(ssim.min()), 'image_ssim_max': float(ssim.max()), 'image_ssim_std': float(ssim.std(ddof=1))}
    bob = np.asarray([row['models']['bob']['ssim'] for row in rows], np.float64)
    for model in models:
        if model == 'bob':
            continue
        other = np.asarray([row['models'][model]['ssim'] for row in rows], np.float64)
        gap = bob - other
        report['paired_bob_ssim_gaps'][model] = {'mean': float(gap.mean()), '95ci': np.percentile(gap[indices].mean(axis=1), [2.5, 97.5]).tolist(), 'positive_images': int(np.sum(gap > 0)), 'total_images': len(gap)}
    return report

def scientific_grid(rows: list[dict], image_paths: list[Path], reconstructed: Path, destination: Path) -> None:
    from PIL import ImageDraw, ImageFont, ImageOps
    indices = [0, 4, 8, 12, 16, 20]
    if len(rows) < 24:
        indices = list(range(min(6, len(rows))))
    columns = len(indices)
    labels = [('ground_truth', 'Source'), ('bob', 'Bob'), ('eve_no_nonce_transfer', 'Eve: archived no-nonce transfer'), ('eve_bob_style_transfer', 'Eve: archived Bob-style transfer'), ('mean_training_image', 'CIFAR training prior')]
    fontpath = '/System/Library/Fonts/Supplemental/Arial.ttf'
    regular = ImageFont.truetype(fontpath, 30)
    small = ImageFont.truetype(fontpath, 25)
    title = ImageFont.truetype(fontpath, 40)
    cellw, cellh, labelw, top = (400, 330, 330, 130)
    canvas = Image.new('RGB', (labelw + columns * cellw, len(labels) * cellh + top), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((labelw, 15), 'Native Kodak images through independent 32 × 32 packets', font=title, fill='black')
    draw.text((labelw, 72), 'Bob: 12 dB; Eve: 20 dB; HKDF-SHA256/AES-256-CTR parameter generation', font=regular, fill='black')
    for column, index in enumerate(indices):
        filename = image_paths[index].name
        for row, (model, label) in enumerate(labels):
            path = image_paths[index] if model == 'ground_truth' else reconstructed / model / filename
            picture = ImageOps.contain(Image.open(path).convert('RGB'), (cellw - 18, cellh - 62), Image.Resampling.LANCZOS)
            x = labelw + column * cellw + (cellw - picture.width) // 2
            y = top + row * cellh + (cellh - 62 - picture.height) // 2
            canvas.paste(picture, (x, y))
            if row == 0:
                text = filename
            else:
                metric = rows[index]['models'][model]
                text = f"{metric['psnr']:.2f} dB; SSIM {metric['ssim']:.3f}"
            draw.text((labelw + column * cellw + 15, top + (row + 1) * cellh - 50), text, font=small, fill='black')
            if column == 0:
                draw.text((15, top + row * cellh + 125), label, font=small, fill='black')
    canvas.save(destination.with_suffix('.png'))
    canvas.save(destination.with_suffix('.pdf'), resolution=180)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=EXPERIMENTS / 'data/kodak')
    parser.add_argument('--keys', type=Path, default=EXPERIMENTS / 'experiment_keys.json')
    parser.add_argument('--checkpoint', type=Path, default=ARCHIVED / 'pt/best_main.pt')
    parser.add_argument('--attackers', type=Path, default=ARCHIVED / 'pt/best_response_attackers.pt')
    parser.add_argument('--output', type=Path, default=EXPERIMENTS / 'results/kodak')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--limit', type=int, default=24)
    parser.add_argument('--noise-seed', type=int, default=20261019)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    images = [args.data / f'kodim{index:02d}.png' for index in range(1, args.limit + 1)]
    if any((not path.exists() for path in images)):
        raise FileNotFoundError('all requested Kodak PNG images must be downloaded before evaluation')
    keydata = json.loads(args.keys.read_text()) if args.keys.is_file() else {'key_hex': 'aa47c63ebedee13713bbd1e4ed31f1849fed3be178b9fa04ef79da31fdc4a212', 'wrong_key_hex': 'eb47ecb539e8f5f3e365eafa8213f19aa7979fb1fea47f2f63abccff2f17193e'}
    key = bytes.fromhex(keydata['key_hex'])
    scrambler = CryptoScrambler(key, 256)
    counter = NonceCounter(NONCE_START)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    encoder = Encoder(256).eval()
    encoder.load_state_dict(checkpoint['enc'])
    bob = Decoder(256).eval()
    bob.load_state_dict(checkpoint['dec'])
    archived = torch.load(args.attackers, map_location='cpu', weights_only=True)
    eves = {'eve_no_nonce_transfer': EveReconNoNonce(256).eval(), 'eve_bob_style_transfer': EveBobStyleDecoder(256).eval()}
    eves['eve_no_nonce_transfer'].load_state_dict(archived['eve_models']['no_nonce'])
    eves['eve_bob_style_transfer'].load_state_dict(archived['eve_models']['bob_style'])
    prior = training_mean_image(EXPERIMENTS / 'data/cifar-10-python.tar.gz')
    rng = np.random.default_rng(args.noise_seed)
    manifest = {'dataset': 'Kodak Lossless True Color Image Suite', 'dataset_distribution_url': 'https://r0k.us/graphics/kodak/', 'native_resolution': True, 'resize': False, 'crop': False, 'tile_size': [32, 32], 'tile_order': 'row major; independent nonoverlapping packets; direct reassembly', 'protocol': PROTOCOL, 'key_fingerprint_sha256': hashlib.sha256(key).hexdigest(), 'key_bits': 256, 'nonce_bits': 96, 'nonce_counter_start': hex(NONCE_START), 'nonce_policy': 'unique full96bit counter; disjoint from CIFAR packet ranges', 'latent_real_symbols_per_tile': 256, 'target_power_per_real_symbol': 1.0, 'bob_snr_db': 12.0, 'eve_snr_db': 20.0, 'channel': 'independent real isotropic AWGN', 'noise_seed': args.noise_seed, 'encoder_frozen': True, 'decoder_frozen': True, 'kodak_used_for_training_or_selection': False, 'eve_attacks': 'two archived pretrained reconstruction networks transferred without adaptation; no-nonce and Bob-style', 'prior_baseline': 'pixelwise mean of all50000 official CIFAR-10 training images, repeated per tile', 'full_image_metrics': 'float32 reconstructed full image, before PNG rounding; MSE over all RGB pixels; datasetPSNR from mean imageMSE', 'ssim': 'supplied11x11 Gaussian sigma1.5; zero padding; clamped variances; eps1e-8; finalclamp[0,1]; averaged across allRGBpixels', 'confidence_interval_unit': '24 source images, not9216 independent tiles', 'qualitative_selection': 'indices1,5,9,13,17,21 specified before evaluation', 'checkpoint_sha256': sha256(args.checkpoint), 'attacker_checkpoint_sha256': sha256(args.attackers), 'source_sha256': {rel: sha256(SOURCE / rel) for rel in ['models/autoencoder.py', 'models/attackers.py', 'ssim_ms.py']}, 'cpu_threads': args.threads, 'batch_size': args.batch_size}
    rows = []
    tile_records = {'image_index': [], 'tile_row': [], 'tile_column': []}
    cached = {'images': [], 'normalized_latents': [], 'eve_observations': [], 'nonce_features': [], 'nonce_u32': []}
    recon = args.output / 'reconstructions'
    for name in ['bob', *eves, 'mean_training_image']:
        (recon / name).mkdir(parents=True, exist_ok=True)
    for image_index, path in enumerate(images):
        image_start = time.monotonic()
        image = np.asarray(Image.open(path).convert('RGB'))
        height, width, _ = image.shape
        patches = tile_image(image)
        assert np.array_equal(assemble_tiles(patches, height, width), image)
        latent_parts = []
        with torch.inference_mode():
            for batch_start in range(0, len(patches), args.batch_size):
                tensor = torch.from_numpy(patches[batch_start:batch_start + args.batch_size].copy()).float() / 255.0
                latent = encoder(tensor).numpy()
                latent /= np.sqrt(np.maximum(np.mean(latent * latent, axis=1, keepdims=True), 1e-06))
                latent_parts.append(latent)
        latents = np.concatenate(latent_parts)
        nonces = counter.take(len(patches))
        params = scrambler.parameters_batch(nonces)
        cipher = params.forward(latents)
        bob_received = cipher + rng.normal(0, 10 ** (-12 / 20), cipher.shape).astype(np.float32)
        eve_received = cipher + rng.normal(0, 0.1, cipher.shape).astype(np.float32)
        bob_inputs = params.inverse(bob_received)
        predictions = {'bob': predict_patches(bob, bob_inputs, args.batch_size, patches)}
        for name, model in eves.items():
            predictions[name] = predict_patches(model, eve_received, args.batch_size, patches)
        prior_patches = np.broadcast_to(prior, patches.shape).copy()
        prior_target = torch.from_numpy(patches.copy()).float() / 255.0
        prior_tensor = torch.from_numpy(prior_patches)
        predictions['mean_training_image'] = (prior_patches, (prior_tensor - prior_target).square().mean((1, 2, 3)).numpy(), ssim_per_image(prior_tensor, prior_target).numpy())
        row = {'image_index': image_index, 'name': path.name, 'sha256': sha256(path), 'height': height, 'width': width, 'tiles': len(patches), 'payload_real_symbols': len(patches) * 256, 'raw_nonce_header_bits': len(patches) * 96, 'models': {}}
        for name, (predicted, patch_mse, patch_ssim) in predictions.items():
            full = assemble_tiles(predicted, height, width)
            row['models'][name] = full_image_metrics(full, image)
            row['models'][name].update({'mean_tile_mse': float(patch_mse.mean()), 'mean_tile_ssim': float(patch_ssim.mean()), 'tile_ssim_p95': float(np.percentile(patch_ssim, 95))})
            Image.fromarray(np.rint(np.clip(full, 0, 1) * 255).astype(np.uint8)).save(recon / name / path.name)
            tile_records.setdefault(name + '_mse', []).append(patch_mse)
            tile_records.setdefault(name + '_ssim', []).append(patch_ssim)
        rows.append(row)
        tile_records['image_index'].append(np.full(len(patches), image_index, np.int16))
        tile_records['tile_row'].append(np.repeat(np.arange(height // PATCH), width // PATCH))
        tile_records['tile_column'].append(np.tile(np.arange(width // PATCH), height // PATCH))
        cached['images'].append(patches)
        cached['normalized_latents'].append(latents)
        cached['eve_observations'].append(eve_received)
        cached['nonce_features'].append(nonce_features(nonces))
        cached['nonce_u32'].append(np.array([[int.from_bytes(n[0:4], 'big'), int.from_bytes(n[4:8], 'big'), int.from_bytes(n[8:12], 'big')] for n in nonces], np.uint32))
        write_json(args.output / 'kodak_per_image.json', {'metadata': manifest, 'images': rows})
        print(f"Kodak{image_index + 1}/{len(images)} {path.name}: Bob{row['models']['bob']['psnr']:.2f}dB/SSIM{row['models']['bob']['ssim']:.3f}; Eve-no-nonceSSIM{row['models']['eve_no_nonce_transfer']['ssim']:.3f}; {time.monotonic() - image_start:.1f}s/image; total{time.monotonic() - started:.1f}s", flush=True)
    manifest['nonce_counter_end_exclusive'] = hex(counter.next_value)
    manifest['total_tiles'] = sum((row['tiles'] for row in rows))
    manifest['raw_nonce_bits_per_image'] = rows[0]['raw_nonce_header_bits']
    manifest['payload_real_symbols_per_image'] = rows[0]['payload_real_symbols']
    manifest['elapsed_seconds'] = time.monotonic() - started
    write_json(args.output / 'kodak_per_image.json', {'metadata': manifest, 'images': rows})
    aggregate = aggregate_image_metrics(rows, 20261020)
    write_json(args.output / 'kodak_metrics.json', {'metadata': manifest, **aggregate})
    np.savez(args.output / 'kodak_per_tile.npz', **{name: np.concatenate(values) for name, values in tile_records.items()})
    np.savez(EXPERIMENTS / 'data/kodak_packet_cache.npz', **{name: np.concatenate(values) for name, values in cached.items()}, image_index=np.concatenate(tile_records['image_index']), metadata_json=np.array(json.dumps(manifest)))
    scientific_grid(rows, images, recon, args.output / 'kodak_deterministic_examples')
    print(json.dumps({'completed_images': len(rows), 'total_tiles': manifest['total_tiles'], 'elapsed_seconds': time.monotonic() - started, 'metrics': aggregate['metrics']}), flush=True)
if __name__ == '__main__':
    main()
