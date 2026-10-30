from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from . import common


def defaults():
    return {
        "enabled": True,
        "train_dir": None,
        "val_dir": None,
        "download": True,
        "train_crops": 50000,
        "validation_crops": 5000,
        "epochs": 60,
        "attack_seeds": [801, 802],
        "batch_size": 64,
        "learning_rate": 0.0002,
        "width": 64,
        "eve_snr_db": 20.0,
        "bob_snr_db": 12.0,
        "bootstrap_draws": 10000,
        "include_nonce": True,
        "representative_frame_indices": [0, 6, 12],
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(directory):
    directory = Path(directory)
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not paths:
        raise ValueError(f"No natural-image frames found in {directory}")
    rows = []
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
            pixel_sha256 = hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()
        if min(width, height) < 64:
            raise ValueError(f"Natural-image frame is smaller than 64 pixels: {path}")
        rows.append({"path": str(path.resolve()), "frame_id": path.relative_to(directory).as_posix(), "sha256": sha256_file(path), "pixel_sha256": pixel_sha256, "width": width, "height": height})
    return rows


def assert_independent_frames(train, validation):
    train_paths = {Path(row["path"]).resolve() for row in train}
    val_paths = {Path(row["path"]).resolve() for row in validation}
    if train_paths & val_paths:
        raise ValueError("Natural-image training and validation contain the same frame paths")
    overlap = {row["sha256"] for row in train} & {row["sha256"] for row in validation}
    if overlap:
        raise ValueError("Natural-image training and validation contain duplicate frame contents")
    overlap = {row.get("pixel_sha256", row["sha256"]) for row in train} & {row.get("pixel_sha256", row["sha256"]) for row in validation}
    if overlap:
        raise ValueError("Natural-image training and validation contain identical decoded frames")


def _safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as zf:
        for entry in zf.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Dataset ZIP contains a path outside its destination")
        zf.extractall(destination)


def prepare_corpus(cfg):
    natural = {**defaults(), **cfg.get("natural", {})}
    root = Path(cfg.get("dataset_dir", "data")) / "DIV2K"
    folders = []
    for split, field, expected in (("train", "train_dir", 800), ("valid", "val_dir", 100)):
        explicit = natural.get(field)
        directory = Path(explicit).expanduser() if explicit else root / f"DIV2K_{split}_HR"
        rows = None
        if directory.is_dir():
            try:
                rows = inventory(directory)
            except (OSError, ValueError):
                if explicit:
                    raise
        if rows is None or (not explicit and len(rows) < expected):
            if explicit:
                raise FileNotFoundError(f"Supply an existing natural-image {split} directory: {directory}")
            root.mkdir(parents=True, exist_ok=True)
            name = f"DIV2K_{split}_HR.zip"
            archive = root / name
            url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/" + name
            provenance_path = root / (name + ".provenance.json")
            expected_sha256 = json.loads(provenance_path.read_text())["sha256"] if provenance_path.exists() else None
            if not archive.exists() and not natural["download"]:
                raise FileNotFoundError(f"Official natural-image {split} frames are incomplete and no cached archive is available: {directory}")
            common.download_file(url, archive, expected_sha256=expected_sha256)
            print(f"Natural-image stage: extracting {name}", flush=True)
            _safe_extract(archive, root)
            common.write_json(provenance_path, {"url": url, "sha256": sha256_file(archive), "source": "official DIV2K distribution", "expected_frame_count": expected})
            rows = inventory(directory)
        if not explicit and len(rows) != expected:
            raise ValueError(f"Official DIV2K {split} corpus must contain {expected} frames; found {len(rows)}")
        folders.append(rows)
    assert_independent_frames(*folders)
    return folders


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_crop_bank(rows, count, directory, split, seed):
    if count < 1:
        raise ValueError("Natural-image crop count must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    signature = _signature({"frames": [{k: r[k] for k in ("frame_id", "sha256", "width", "height")} for r in rows], "count": count, "split": split, "seed": seed, "crop_size": 64})
    path = directory / f"{split}_crops.npy"
    coordinates = directory / f"{split}_crops.csv"
    metadata = directory / f"{split}_crops.json"
    if metadata.exists():
        stored = json.loads(metadata.read_text())
        if stored["signature"] != signature:
            raise ValueError(f"Crop cache configuration/content changed: {metadata}. Use a new output directory.")
        if path.exists() and coordinates.exists() and sha256_file(coordinates) == stored["coordinate_manifest_sha256"] and sha256_file(path) == stored["crop_array_sha256"]:
            return np.load(path, mmap_mode="r"), signature
        raise ValueError(f"Crop cache is incomplete or its checksum differs: {directory}")
    rng = np.random.default_rng(seed)
    frame_indices = np.arange(count, dtype=np.int64) % len(rows)
    rng.shuffle(frame_indices)
    ys = np.asarray([rng.integers(0, rows[i]["height"] - 63) for i in frame_indices], dtype=np.int64)
    xs = np.asarray([rng.integers(0, rows[i]["width"] - 63) for i in frame_indices], dtype=np.int64)
    temporary = path.with_suffix(".partial.npy")
    bank = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=(count, 3, 64, 64))
    for frame_index in np.unique(frame_indices):
        with Image.open(rows[frame_index]["path"]) as image:
            image = image.convert("RGB")
            for index in np.flatnonzero(frame_indices == frame_index):
                crop = image.crop((int(xs[index]), int(ys[index]), int(xs[index]) + 64, int(ys[index]) + 64))
                bank[index] = np.asarray(crop).transpose(2, 0, 1)
    bank.flush()
    del bank
    os.replace(temporary, path)
    temporary_coords = coordinates.with_suffix(".tmp")
    with temporary_coords.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["crop_index", "frame_id", "frame_sha256", "x", "y", "width", "height"])
        writer.writeheader()
        for i, frame in enumerate(frame_indices):
            writer.writerow({"crop_index": i, "frame_id": rows[frame]["frame_id"], "frame_sha256": rows[frame]["sha256"], "x": int(xs[i]), "y": int(ys[i]), "width": 64, "height": 64})
    os.replace(temporary_coords, coordinates)
    common.write_json(metadata, {"signature": signature, "count": count, "frame_count": len(rows), "seed": seed, "split": split, "crop_array_sha256": sha256_file(path), "coordinate_manifest_sha256": sha256_file(coordinates), "frame_inventory": rows, "native_resolution_crops": True})
    return np.load(path, mmap_mode="r"), signature


def split_context(images):
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[1:] != (3, 64, 64):
        raise ValueError("Context crops must have shape N x 3 x 64 x 64")
    return np.stack((images[:, :, :32, :32], images[:, :, :32, 32:], images[:, :, 32:, :32], images[:, :, 32:, 32:]), axis=1)


def assemble_context(tiles):
    if torch.is_tensor(tiles):
        return torch.cat((torch.cat((tiles[:, 0], tiles[:, 1]), dim=-1), torch.cat((tiles[:, 2], tiles[:, 3]), dim=-1)), dim=-2)
    tiles = np.asarray(tiles)
    return np.concatenate((np.concatenate((tiles[:, 0], tiles[:, 1]), axis=-1), np.concatenate((tiles[:, 2], tiles[:, 3]), axis=-1)), axis=-2)


def encode_bank(case, bank, directory, split, crop_signature, device):
    directory = Path(directory)
    path = directory / f"{split}_latents.npy"
    meta = directory / f"{split}_latents.json"
    fingerprint = _signature({"crop_signature": crop_signature, "case": case["name"], "codec": _model_hash(case["enc"])})
    if meta.exists():
        stored = json.loads(meta.read_text())
        if stored["signature"] != fingerprint or sha256_file(path) != stored["sha256"]:
            raise ValueError(f"Encoded natural-image cache differs: {meta}")
        return np.load(path, mmap_mode="r")
    temporary = path.with_suffix(".partial.npy")
    result = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(bank), 4, 256))
    for start in range(0, len(bank), 512):
        tiles = split_context(bank[start:start + 512]).reshape(-1, 3, 32, 32)
        result[start:start + len(tiles) // 4] = common.encode_images(case["enc"], tiles, device).reshape(-1, 4, 256)
    result.flush()
    del result
    os.replace(temporary, path)
    common.write_json(meta, {"signature": fingerprint, "sha256": sha256_file(path), "count": len(bank)})
    return np.load(path, mmap_mode="r")


def _model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class NaturalReconstructor(nn.Module):
    def __init__(self, family="context", width=64, include_nonce=True):
        super().__init__()
        if family not in {"tile", "context", "identity_context"}:
            raise ValueError(family)
        if width < 8 or width % 8:
            raise ValueError("Reconstruction width must be a positive multiple of eight")
        self.family = family
        self.include_nonce = include_nonce
        dimension = 3 * 256 + 2 + (6 if include_nonce else 0)
        self.register_buffer("input_mean", torch.zeros(dimension))
        self.register_buffer("input_scale", torch.ones(dimension))
        self.project = nn.Sequential(nn.Linear(dimension, width * 16), nn.SiLU())
        self.net = nn.Sequential(Residual(width), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Residual(width), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Residual(width), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), Residual(width), nn.Conv2d(width, 3, 3, padding=1), nn.Sigmoid())

    def features(self, obs, nonce):
        n = len(obs)
        positions = obs.new_tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]]).expand(n, -1, -1)
        parts = [obs, torch.sort(obs, dim=-1).values, torch.sort(obs.abs(), dim=-1).values, positions]
        if self.include_nonce:
            parts.append(nonce)
        return torch.cat(parts, dim=-1)

    def forward(self, obs, nonce):
        x = (self.features(obs, nonce) - self.input_mean) / self.input_scale
        hidden = self.project(x).reshape(-1, 4, self.project[0].out_features // 16, 4, 4)
        if self.family == "tile":
            output = self.net(hidden.flatten(0, 1)).reshape(-1, 4, 3, 32, 32)
            return assemble_context(output)
        return self.net(assemble_context(hidden))


def frame_bootstrap(rows, draws=10000, seed=0, prior_rows=None):
    n = len(rows["mse"])
    if n < 2:
        raise ValueError("Whole-frame bootstrap requires at least two test frames")
    rng = np.random.default_rng(seed)
    samples = {"dataset_psnr_db": [], "mean_ssim_valid": [], "mean_ssim_legacy": []}
    if prior_rows is not None:
        samples.update({"ssim_valid_minus_prior": [], "mse_reduction_from_prior": []})
    for start in range(0, draws, 512):
        indices = rng.integers(0, n, size=(min(512, draws - start), n))
        samples["dataset_psnr_db"].extend((-10 * np.log10(np.maximum(np.asarray(rows["mse"])[indices].mean(1), 1e-15))).tolist())
        samples["mean_ssim_valid"].extend(np.asarray(rows["ssim_valid"])[indices].mean(1).tolist())
        samples["mean_ssim_legacy"].extend(np.asarray(rows["ssim_legacy"])[indices].mean(1).tolist())
        if prior_rows is not None:
            samples["ssim_valid_minus_prior"].extend((np.asarray(rows["ssim_valid"]) - np.asarray(prior_rows["ssim_valid"]))[indices].mean(1).tolist())
            samples["mse_reduction_from_prior"].extend((np.asarray(prior_rows["mse"]) - np.asarray(rows["mse"]))[indices].mean(1).tolist())
    return {"unit": "whole Kodak frame", "frames": n, "draws": draws, "seed": seed, "intervals_95": {key: [float(v) for v in np.quantile(values, [.025, .975])] for key, values in samples.items()}}


def _atomic_torch(path, value):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_torch(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def _predict(model, obs, nonce, device, batch):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(obs), batch):
            x = torch.as_tensor(np.array(obs[start:start + batch]), device=device)
            n = torch.as_tensor(np.array(nonce[start:start + batch]), device=device)
            outputs.append(model(x, n).cpu().numpy())
    return np.concatenate(outputs)


def _fit(case, natural, train, val, train_u, val_u, val_obs, val_nonce, outdir, family, seed, device, smoke):
    path = outdir / f"{family}_seed{seed}"
    path.mkdir(parents=True, exist_ok=True)
    signature = _signature({"natural": natural, "case": case["name"], "codec": _model_hash(case["enc"]), "family": family, "seed": seed, "train_shape": train.shape, "val_shape": val.shape, "smoke": smoke})
    selection = path / "selection.json"
    if selection.exists():
        result = json.loads(selection.read_text())
        if result["signature"] != signature:
            raise ValueError(f"Natural-image fitting configuration changed: {path}")
        if sha256_file(path / "best_ssim.pt") != result["checkpoint_sha256"]:
            raise ValueError(f"Selected natural-image checkpoint checksum differs: {path}")
        model = NaturalReconstructor(family, natural["width"], natural["include_nonce"]).to(device)
        model.load_state_dict(_load_torch(path / "best_ssim.pt", device)["model"])
        return model, result
    common.torch_seed(seed)
    model = NaturalReconstructor(family, natural["width"], natural["include_nonce"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=natural["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=natural["epochs"])
    start_epoch, history, best_ssim, best_mse = 0, [], -math.inf, math.inf
    normalization_rng = np.random.default_rng(common.stable_seed(seed, "natural_normalization"))
    sample = np.asarray(train_u[:min(2048, len(train_u))]).reshape(-1, 256)
    sampled = common.public_observations(sample, normalization_rng, natural["eve_snr_db"], repetitions=1)[:, 0].reshape(-1, 4, 256)
    if family == "identity_context":
        sampled = np.asarray(train_u[:min(2048, len(train_u))]) + normalization_rng.normal(0, 10 ** (-natural["eve_snr_db"] / 20), (len(sample) // 4, 4, 256)).astype(np.float32)
    with torch.no_grad():
        features = model.features(torch.as_tensor(sampled, device=device), torch.as_tensor(normalization_rng.random((len(sampled), 4, 6)).astype(np.float32), device=device)).flatten(0, 1)
        model.input_mean.copy_(features.mean(0))
        model.input_scale.copy_(features.std(0, unbiased=False).clamp_min(1e-4))
    last = path / "last.pt"
    if last.exists():
        saved = _load_torch(last, device)
        if saved["signature"] != signature:
            raise ValueError(f"Natural-image checkpoint configuration changed: {path}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch, history = saved["epoch"], saved["history"]
        best_ssim, best_mse = saved["best_ssim"], saved["best_mse"]
    for epoch in range(start_epoch, natural["epochs"]):
        started = time.monotonic()
        rng = np.random.default_rng(common.stable_seed(seed, "natural_epoch", epoch))
        order = rng.permutation(len(train))
        model.train()
        total_loss = 0.
        for start in range(0, len(order), natural["batch_size"]):
            indices = order[start:start + natural["batch_size"]]
            latent = np.asarray(train_u[indices])
            if family == "identity_context":
                observations = latent + rng.normal(0, 10 ** (-natural["eve_snr_db"] / 20), latent.shape).astype(np.float32)
            else:
                observations = common.public_observations(latent.reshape(-1, 256), rng, natural["eve_snr_db"], repetitions=1)[:, 0].reshape(-1, 4, 256)
            nonce = rng.integers(0, 65536, (len(indices), 4, 6)).astype(np.float32) / 65535.
            target = torch.as_tensor(np.asarray(train[indices], dtype=np.float32) / 255., device=device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(torch.as_tensor(observations, device=device), torch.as_tensor(nonce, device=device))
            loss = F.mse_loss(predicted, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        scheduler.step()
        predictions = _predict(model, val_obs, val_nonce, device, natural["batch_size"])
        metrics = common.image_rows(predictions, val, device=device)
        mse, ssim = float(np.mean(metrics["mse"])), float(np.mean(metrics["ssim_valid"]))
        row = {"epoch": epoch + 1, "train_mse": total_loss / len(train), "validation_mse": mse, "validation_ssim_valid": ssim, "seconds": time.monotonic() - started, "cumulative_context_presentations": (epoch + 1) * len(train), "cumulative_fresh_packet_presentations": (epoch + 1) * len(train) * 4, "optimizer_updates": (epoch + 1) * math.ceil(len(train) / natural["batch_size"])}
        history.append(row)
        checkpoint = {"model": model.state_dict(), "epoch": epoch + 1, "signature": signature, "validation_mse": mse, "validation_ssim_valid": ssim}
        if ssim > best_ssim:
            best_ssim = ssim
            _atomic_torch(path / "best_ssim.pt", checkpoint)
        if mse < best_mse:
            best_mse = mse
            _atomic_torch(path / "best_mse.pt", checkpoint)
        _atomic_torch(last, {**checkpoint, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history, "best_ssim": best_ssim, "best_mse": best_mse})
        common.write_json(path / "history.json", history)
        print(f"natural {case['name']} {family} seed{seed} epoch{epoch + 1}/{natural['epochs']}: valSSIM={ssim:.6f}, valMSE={mse:.6f} ({row['seconds']:.1f}s)", flush=True)
    best = _load_torch(path / "best_ssim.pt", device)
    model.load_state_dict(best["model"])
    selected_predictions = _predict(model, val_obs, val_nonce, device, natural["batch_size"])
    validation_rows = common.image_rows(selected_predictions, val, device=device)
    common.save_npz(path / "selected_validation_metrics.npz", **validation_rows, crop_index=np.arange(len(val), dtype=np.int64))
    result = {"signature": signature, "family": family, "seed": seed, "selected_epoch": best["epoch"], "validation_ssim_valid": best["validation_ssim_valid"], "validation_mse": best["validation_mse"], "selection_metric": "validation_ssim_valid", "test_used_for_selection": False, "training_loss": "MSE", "normalization_source": "training images and public-family surrogate observations only", "fresh_independent_transforms_per_training_presentation": family != "identity_context", "checkpoint": str((path / "best_ssim.pt").relative_to(outdir)), "checkpoint_sha256": sha256_file(path / "best_ssim.pt"), "model_parameters": sum(p.numel() for p in model.parameters()), "status": "smoke_not_scientific" if smoke else "scientific"}
    common.write_json(selection, result)
    return model, result


def _contexts_for_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError("Kodak frames must have C x H x W RGB shape")
    height, width = frame.shape[1:]
    if height % 64 or width % 64:
        raise ValueError("Native Kodak dimensions must be divisible by 64; no test image is resized or truncated")
    contexts = frame.reshape(3, height // 64, 64, width // 64, 64).transpose(1, 3, 0, 2, 4).reshape(-1, 3, 64, 64)
    return contexts


def _frame_from_contexts(contexts, height, width):
    return contexts.reshape(height // 64, width // 64, 3, 64, 64).transpose(2, 0, 3, 1, 4).reshape(3, height, width)


def _smoke_corpus(outdir):
    rng = np.random.default_rng(2791)
    folders = []
    for split, count in (("train", 3), ("validation", 2)):
        path = outdir / "smoke_corpus" / split
        path.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            image = rng.integers(0, 256, (80, 96, 3), dtype=np.uint8)
            Image.fromarray(image).save(path / f"synthetic_{i:03d}.png")
        folders.append(inventory(path))
    assert_independent_frames(*folders)
    return folders


def run(case, cfg, outdir, device, smoke=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    natural = {**defaults(), **cfg.get("natural", {})}
    if smoke:
        natural.update(train_crops=8, validation_crops=4, epochs=1, attack_seeds=[801], batch_size=4, width=8, bootstrap_draws=32)
    train_inventory, val_inventory = _smoke_corpus(outdir) if smoke else prepare_corpus(cfg)
    stage_signature = _signature({"stage": "natural_v1", "natural": natural, "train_frames": [{"sha256": row["sha256"], "frame_id": row["frame_id"]} for row in train_inventory], "validation_frames": [{"sha256": row["sha256"], "frame_id": row["frame_id"]} for row in val_inventory], "codec": _model_hash(case["enc"]), "case": case["name"], "smoke": smoke})
    result_path = outdir / "summary.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("signature") != stage_signature:
            raise ValueError("Natural-image stage settings changed; choose a fresh output directory")
        return result
    common.write_json(outdir / "frame_split_manifest.json", {"training": train_inventory, "validation": val_inventory, "independence": "disjoint frame paths and SHA-256 contents", "Kodak_used_for_training_or_selection": False})
    bank_dir = outdir / "crop_cache"
    train, train_sig = build_crop_bank(train_inventory, natural["train_crops"], bank_dir, "train", common.stable_seed("natural", "train_crops"))
    validation, val_sig = build_crop_bank(val_inventory, natural["validation_crops"], bank_dir, "validation", common.stable_seed("natural", "val_crops"))
    train_u = encode_bank(case, train, bank_dir, "train", train_sig, device)
    val_u = encode_bank(case, validation, bank_dir, "validation", val_sig, device)
    val_observations, val_nonce = common.actual_observations(val_u.reshape(-1, 256), case, "natural_validation", natural["eve_snr_db"])
    val_observations = val_observations[:, 0].reshape(-1, 4, 256)
    val_nonce = val_nonce[:, 0].reshape(-1, 4, 6)
    identity_rng = np.random.default_rng(common.stable_seed(case["name"], "natural_identity_val"))
    identity_val = np.asarray(val_u) + identity_rng.normal(0, 10 ** (-natural["eve_snr_db"] / 20), val_u.shape).astype(np.float32)
    models, selections = {}, {}
    for family in ("tile", "context", "identity_context"):
        seeds = natural["attack_seeds"] if family != "identity_context" else natural["attack_seeds"][:1]
        for seed in seeds:
            model, selection = _fit(case, natural, train, validation, train_u, val_u, identity_val if family == "identity_context" else val_observations, val_nonce, outdir, family, int(seed), device, smoke)
            key = f"{family}_seed{seed}"
            models[key], selections[key] = model, selection
    strongest = {family: max((key for key in selections if selections[key]["family"] == family), key=lambda key: selections[key]["validation_ssim_valid"]) for family in ("tile", "context", "identity_context")}
    common.write_json(outdir / "selections.json", {"candidates": selections, "family_selected_by_validation": strongest, "test_used_for_selection": False})
    prior_sum = np.zeros((3, 64, 64), dtype=np.float64)
    for start in range(0, len(train), 512):
        prior_sum += np.asarray(train[start:start + 512], dtype=np.float64).sum(axis=0)
    prior = (prior_sum / (len(train) * 255.)).astype(np.float32)
    common.save_npz(outdir / "domain_prior.npz", mean_context=prior)
    if smoke:
        rng = np.random.default_rng(173)
        frames = [rng.integers(0, 256, (3, 64, 128), dtype=np.uint8), rng.integers(0, 256, (3, 128, 64), dtype=np.uint8)]
    else:
        frames = common.load_kodak_frames(cfg)
        if len(frames) != 24:
            raise ValueError(f"Expected all 24 Kodak test frames; got {len(frames)}")
        frames = [np.asarray(frame).transpose(2, 0, 1) for frame in frames]
    kodak_hashes = {hashlib.sha256(np.ascontiguousarray(frame.transpose(1, 2, 0)).tobytes()).hexdigest() for frame in frames}
    if kodak_hashes & {row["pixel_sha256"] for row in train_inventory + val_inventory}:
        raise ValueError("A Kodak test frame is present in the natural-image fitting/validation corpus")
    metric_lists = {name: {metric: [] for metric in ("mse", "ssim_valid", "ssim_legacy")} for name in ["mean_prior", "bob", "identity_decoder", *models]}
    rows = []
    representative = outdir / "representative_frames"
    representative.mkdir(exist_ok=True)
    for frame_index, frame in enumerate(frames):
        contexts = _contexts_for_frame(frame)
        tiles = split_context(contexts).reshape(-1, 3, 32, 32)
        u = common.encode_images(case["enc"], tiles, device).reshape(-1, 4, 256)
        obs, nonce = common.actual_observations(u.reshape(-1, 256), case, f"natural_kodak_frame{frame_index:02d}", natural["eve_snr_db"])
        obs, nonce = obs[:, 0].reshape(-1, 4, 256), nonce[:, 0].reshape(-1, 4, 6)
        rng = np.random.default_rng(common.stable_seed(case["name"], "natural_identity_kodak", frame_index))
        identity_obs = u + rng.normal(0, 10 ** (-natural["eve_snr_db"] / 20), u.shape).astype(np.float32)
        bob_u = common.bob_observations(u.reshape(-1, 256), case, f"natural_kodak_frame{frame_index:02d}", natural["bob_snr_db"]).reshape(-1, 4, 256)
        predictions = {"mean_prior": np.broadcast_to(prior, contexts.shape)}
        with torch.inference_mode():
            for name, symbols in (("bob", bob_u), ("identity_decoder", identity_obs)):
                reconstructed = []
                flat = symbols.reshape(-1, 256)
                for start in range(0, len(flat), natural["batch_size"] * 4):
                    reconstructed.append(case["dec"](torch.as_tensor(flat[start:start + natural["batch_size"] * 4], device=device)).cpu().numpy())
                predictions[name] = assemble_context(np.concatenate(reconstructed).reshape(-1, 4, 3, 32, 32))
        for name, model in models.items():
            predictions[name] = _predict(model, identity_obs if selections[name]["family"] == "identity_context" else obs, nonce, device, natural["batch_size"])
        height, width = frame.shape[1:]
        frame_predictions = {}
        for name, predicted in predictions.items():
            complete = _frame_from_contexts(predicted, height, width)
            frame_predictions[name] = complete
            measured = common.image_rows(complete[None], frame[None], device=device)
            row = {"frame_index": frame_index, "height": height, "width": width, "model": name}
            for metric in metric_lists[name]:
                value = float(measured[metric][0])
                metric_lists[name][metric].append(value)
                row[metric] = value
            rows.append(row)
        if frame_index in natural["representative_frame_indices"]:
            names = ["Original", "Bob", "Mean prior", "Tile attacker", "Context attacker", "Identity control"]
            chosen = [frame.astype(np.float32) / 255., frame_predictions["bob"], frame_predictions["mean_prior"], frame_predictions[strongest["tile"]], frame_predictions[strongest["context"]], frame_predictions[strongest["identity_context"]]]
            panels = []
            from PIL import ImageDraw
            for label, pixels in zip(names, chosen):
                thumb = Image.fromarray(np.uint8(np.clip(pixels.transpose(1, 2, 0), 0, 1) * 255))
                thumb.thumbnail((384, 288), Image.Resampling.LANCZOS)
                panel = Image.new("RGB", (384, 312), "white")
                panel.paste(thumb, ((384 - thumb.width) // 2, 24 + (288 - thumb.height) // 2))
                ImageDraw.Draw(panel).text((8, 5), label, fill="black")
                panels.append(panel)
            canvas = Image.new("RGB", (384 * 3, 312 * 2), "white")
            for j, panel in enumerate(panels):
                canvas.paste(panel, ((j % 3) * 384, (j // 3) * 312))
            canvas.save(representative / f"frame_{frame_index:02d}.png")
        print(f"natural {case['name']}: evaluated Kodak frame {frame_index + 1}/{len(frames)}", flush=True)
    with (outdir / "kodak_frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numerical = {name: {key: np.asarray(values) for key, values in metrics.items()} for name, metrics in metric_lists.items()}
    common.save_npz(outdir / "kodak_frame_metrics.npz", **{f"{name}__{metric}": value for name, metrics in numerical.items() for metric, value in metrics.items()})
    summaries = {}
    for name, metrics in numerical.items():
        summaries[name] = {"metrics": common.image_summary(metrics), "bootstrap": frame_bootstrap(metrics, natural["bootstrap_draws"], common.stable_seed(case["name"], "natural_bootstrap"), numerical["mean_prior"])}
    result = {"signature": stage_signature, "case": case["name"], "status": "smoke_not_scientific" if smoke else "scientific", "frames": len(frames), "natural_training_frames": len(train_inventory), "natural_validation_frames": len(val_inventory), "train_contexts": len(train), "validation_contexts": len(validation), "selections": selections, "family_selected_by_validation": strongest, "summaries": summaries, "test_used_for_selection": False, "test_domain": "Kodak native resolution", "training_domain": "synthetic smoke frames" if smoke else "independent natural-image frames", "context": "nonoverlapping 2 x 2 adjacent 32-pixel tiles, independently transformed per tile", "bob_evaluation": "actual cryptographic transform, isotropic AWGN and exact inverse", "train_crop_signature": train_sig, "validation_crop_signature": val_sig}
    common.write_json(result_path, result)
    return result
