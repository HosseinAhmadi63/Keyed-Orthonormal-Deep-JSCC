import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from evaluation import natural


def _image(path, seed, size=(81, 97)):
    pixels = np.random.default_rng(seed).integers(0, 256, (*size, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return pixels


def test_native_context_roundtrip_preserves_orientation():
    frame = np.arange(3 * 128 * 192, dtype=np.uint32).reshape(3, 128, 192)
    contexts = natural._contexts_for_frame(frame)
    assert contexts.shape == (6, 3, 64, 64)
    assert np.array_equal(natural._frame_from_contexts(contexts, 128, 192), frame)
    assert np.array_equal(natural.assemble_context(natural.split_context(contexts)), contexts)
    with pytest.raises(ValueError, match="resized or truncated"):
        natural._contexts_for_frame(frame[:, :127])


def test_independent_split_rejects_same_frame_after_reencoding(tmp_path):
    fit, val = tmp_path / "train", tmp_path / "validation"
    fit.mkdir()
    val.mkdir()
    pixels = _image(fit / "a.png", 1)
    Image.fromarray(pixels).save(val / "different_name.bmp")
    train_inventory, val_inventory = natural.inventory(fit), natural.inventory(val)
    assert train_inventory[0]["sha256"] != val_inventory[0]["sha256"]
    with pytest.raises(ValueError, match="identical decoded frames"):
        natural.assert_independent_frames(train_inventory, val_inventory)


def test_crop_cache_coordinates_reproduce_native_pixels_and_detect_changes(tmp_path):
    source = tmp_path / "frames"
    source.mkdir()
    originals = {"a.png": _image(source / "a.png", 2), "b.png": _image(source / "b.png", 3)}
    rows = natural.inventory(source)
    bank, signature = natural.build_crop_bank(rows, 7, tmp_path / "cache", "train", 123)
    import csv
    with (tmp_path / "cache/train_crops.csv").open() as stream:
        crops = list(csv.DictReader(stream))
    assert len(crops) == 7
    for index, crop in enumerate(crops):
        x, y = int(crop["x"]), int(crop["y"])
        assert np.array_equal(bank[index], originals[crop["frame_id"]][y:y + 64, x:x + 64].transpose(2, 0, 1))
    repeated, repeated_signature = natural.build_crop_bank(rows, 7, tmp_path / "cache", "train", 123)
    assert repeated_signature == signature
    assert np.array_equal(bank, repeated)
    with pytest.raises(ValueError, match="configuration/content changed"):
        natural.build_crop_bank(rows, 8, tmp_path / "cache", "train", 123)
    with (tmp_path / "cache/train_crops.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="checksum differs"):
        natural.build_crop_bank(rows, 7, tmp_path / "cache", "train", 123)


def test_frame_bootstrap_uses_paired_frame_units():
    rows = {"mse": np.array([0.01, 0.03, 0.02]), "ssim_valid": np.array([0.5, 0.6, 0.7]), "ssim_legacy": np.array([0.6, 0.7, 0.8])}
    prior = {"mse": rows["mse"] + .01, "ssim_valid": rows["ssim_valid"] - .1, "ssim_legacy": rows["ssim_legacy"] - .1}
    result = natural.frame_bootstrap(rows, draws=250, seed=9, prior_rows=prior)
    assert result["frames"] == 3
    assert result["unit"] == "whole Kodak frame"
    assert result["intervals_95"]["ssim_valid_minus_prior"] == pytest.approx([.1, .1])
    assert result["intervals_95"]["mse_reduction_from_prior"] == pytest.approx([.01, .01])


@pytest.mark.parametrize("family", ["tile", "context", "identity_context"])
def test_reconstructors_preserve_context_geometry_and_backpropagate(family):
    model = natural.NaturalReconstructor(family, width=8, include_nonce=True)
    obs = torch.randn(2, 4, 256)
    nonce = torch.rand(2, 4, 6)
    predicted = model(obs, nonce)
    assert predicted.shape == (2, 3, 64, 64)
    assert torch.isfinite(predicted).all()
    predicted.square().mean().backward()
    assert model.project[0].weight.grad.abs().sum() > 0


def test_zip_path_traversal_rejected(tmp_path):
    import zipfile
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="outside its destination"):
        natural._safe_extract(archive, tmp_path / "extract")
    assert not (tmp_path / "escape.txt").exists()


def test_full_natural_smoke_is_marked_non_scientific_and_resumes(tmp_path):
    torch.set_num_threads(1)
    encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 256)).eval()
    decoder = torch.nn.Sequential(torch.nn.Linear(256, 3 * 32 * 32), torch.nn.Sigmoid(), torch.nn.Unflatten(1, (3, 32, 32))).eval()
    for module in (encoder, decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    case = {"name": "natural_unit_smoke", "enc": encoder, "dec": decoder, "key_hex": bytes(range(32)).hex()}
    result = natural.run(case, {"natural": {}}, tmp_path, "cpu", smoke=True)
    assert result["status"] == "smoke_not_scientific"
    assert result["frames"] == 2
    assert result["test_used_for_selection"] is False
    assert len(result["selections"]) == 3
    assert result["family_selected_by_validation"].keys() == {"tile", "context", "identity_context"}
    assert all(record["selection_metric"] == "validation_ssim_valid" for record in result["selections"].values())
    assert all(record["bootstrap"]["frames"] == 2 for record in result["summaries"].values())
    resumed = natural.run(case, {"natural": {}}, tmp_path, "cpu", smoke=True)
    assert resumed == result
    assert (tmp_path / "kodak_frame_metrics.csv").exists()
    assert (tmp_path / "representative_frames/frame_00.png").exists()


def test_official_partial_extraction_is_recovered_from_cached_archive(tmp_path, monkeypatch):
    import zipfile
    root = tmp_path / "DIV2K"
    root.mkdir()
    for split, count in (("train", 800), ("valid", 100)):
        folder = root / f"DIV2K_{split}_HR"
        folder.mkdir()
        _image(folder / "first.png", 10 if split == "train" else 20)
        with zipfile.ZipFile(root / f"DIV2K_{split}_HR.zip", "w") as stream:
            stream.writestr(f"DIV2K_{split}_HR/first.png", (folder / "first.png").read_bytes())
    extracted = []
    monkeypatch.setattr(natural.common, "download_file", lambda *args, **kwargs: args[1])
    def fake_extract(archive, destination):
        extracted.append(Path(archive).name)
    monkeypatch.setattr(natural, "_safe_extract", fake_extract)
    def fake_inventory(directory):
        split = "train" if "train" in str(directory) else "valid"
        count = (800 if split == "train" else 100) if f"DIV2K_{split}_HR.zip" in extracted else 1
        return [{"path": str(Path(directory) / f"{i}.png"), "frame_id": f"{i}.png", "sha256": f"{split}-{i}", "pixel_sha256": f"{split}-{i}", "width": 80, "height": 96} for i in range(count)]
    monkeypatch.setattr(natural, "inventory", fake_inventory)
    train, val = natural.prepare_corpus({"dataset_dir": str(tmp_path), "natural": {"download": False}})
    assert len(train) == 800 and len(val) == 100
    assert extracted == ["DIV2K_train_HR.zip", "DIV2K_valid_HR.zip"]
