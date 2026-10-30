from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from torch import nn
from torch.nn import functional as F
from keyed_jscc.crypto_scrambler import CryptographicScrambler

from . import common
from . import protocol


def defaults():
    return {"enabled": True, "snr_db": [5., 10., 15., 20., 30., 40.], "eve_snr_db": 20., "total_complex_uses": [640, 1024], "groups": [1, 32], "keyed_header_bits": [128, 256], "keyed_header_repetition": 2, "aes_quantizer_bits": [4, 8], "aes_modulation": 16, "bfv_pt_mod": 16, "bfv_ct_mod": 4093, "bfv_n1": 192, "bfv_n2": 192, "bfv_error_std": 1., "bfv_modulation": 4096, "quantizer_percentiles": [.1, 99.9], "attacker_epochs": 40, "attacker_width": 512, "attacker_batch_size": 256, "attacker_seed": 991, "resource_matched_attack_fitting": True, "timing_repeats": 10, "case_names": ["seed1001_lambda0.2"]}


class LatentQuantizer:
    def __init__(self, lower, upper, bits):
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.bits = int(bits)
        if self.bits not in (4, 8) or self.lower.shape != (256,) or self.upper.shape != (256,) or not np.all(self.upper > self.lower):
            raise ValueError("Invalid 256-dimensional 4/8-bit quantizer")

    @classmethod
    def fit(cls, u, bits, percentiles):
        low, high = np.percentile(np.asarray(u, dtype=np.float64), percentiles, axis=0)
        high = np.maximum(high, low + 1e-6)
        return cls(low, high, bits)

    def encode(self, u):
        scaled = (np.asarray(u, dtype=np.float64) - self.lower) / (self.upper - self.lower)
        return np.rint(np.clip(scaled, 0, 1) * (2 ** self.bits - 1)).astype(np.uint8)

    def decode(self, symbols):
        return (self.lower + np.asarray(symbols, dtype=np.float64) * (self.upper - self.lower) / (2 ** self.bits - 1)).astype(np.float32)


def pack_quantized(symbols, bits):
    symbols = np.asarray(symbols, dtype=np.uint8)
    if symbols.ndim != 2 or np.any(symbols >= 2 ** bits):
        raise ValueError("Invalid quantized symbol matrix")
    if bits == 8:
        return symbols.copy()
    if bits != 4 or symbols.shape[1] % 2:
        raise ValueError("Only even-length 4-bit or 8-bit packing is supported")
    return (symbols[:, ::2] << 4) | symbols[:, 1::2]


def unpack_quantized(packed, bits):
    packed = np.asarray(packed, dtype=np.uint8)
    if bits == 8:
        return packed.copy()
    if bits != 4:
        raise ValueError(bits)
    return np.stack((packed >> 4, packed & 15), axis=-1).reshape(len(packed), -1)


def _gray_to_binary(gray):
    gray = np.asarray(gray, dtype=np.int64)
    binary, shift = gray.copy(), 1
    while shift <= int(gray.max(initial=0)).bit_length():
        binary ^= binary >> shift
        shift *= 2
    return binary


def qam_modulate(bits, order):
    bits = np.asarray(bits, dtype=np.uint8)
    log_order = int(round(math.log2(order)))
    if 2 ** log_order != order or log_order % 2:
        raise ValueError("Square power-of-four QAM is required")
    if bits.ndim != 2 or bits.shape[1] % log_order:
        raise ValueError("QAM bit count must be an exact multiple of bits per complex symbol")
    side, axis_bits = int(math.sqrt(order)), log_order // 2
    grouped = bits.reshape(len(bits), -1, 2, axis_bits)
    labels = (grouped * (1 << np.arange(axis_bits - 1, -1, -1))).sum(-1)
    index = _gray_to_binary(labels)
    levels = 2 * index - (side - 1)
    scale = math.sqrt(2 * (side * side - 1) / 3)
    return (levels[:, :, 0] + 1j * levels[:, :, 1]) / scale


def qam_demodulate(symbols, order):
    symbols = np.asarray(symbols)
    side = int(math.sqrt(order))
    axis_bits = int(round(math.log2(side)))
    scale = math.sqrt(2 * (side * side - 1) / 3)
    axes = np.stack((symbols.real, symbols.imag), -1) * scale
    index = np.clip(np.rint((axes + side - 1) / 2), 0, side - 1).astype(np.int64)
    gray = index ^ (index >> 1)
    return ((gray[..., None] >> np.arange(axis_bits - 1, -1, -1)) & 1).astype(np.uint8).reshape(len(symbols), -1)


def qam_channel(bits, order, snr_db, rng, power=1.0):
    transmitted = qam_modulate(bits, order)
    variance = 0. if snr_db is None or np.isinf(snr_db) else 10 ** (-float(snr_db) / 10) / power
    noise = (rng.normal(size=transmitted.shape) + 1j * rng.normal(size=transmitted.shape)) * math.sqrt(variance / 2)
    return qam_demodulate(transmitted + noise, order)


class LinearBFV:
    def __init__(self, seed, n=256, n1=192, n2=192, pt_mod=16, ct_mod=4093, error_std=1.):
        self.n, self.n1, self.n2 = n, n1, n2
        self.t, self.q = int(pt_mod), int(ct_mod)
        self.delta = self.q // self.t
        self.error_std = float(error_std)
        rng = np.random.default_rng(seed)
        self.s = np.rint(rng.normal(0, error_std, (n, n2))).astype(np.int64)
        self.a = rng.integers(0, self.q, (n2, n1), dtype=np.int64)
        error = np.rint(rng.normal(0, error_std, (n, n1))).astype(np.int64)
        self.b = (-self.s @ self.a - error) % self.q

    def encrypt(self, plaintext, rng):
        plaintext = np.asarray(plaintext, dtype=np.int64)
        if plaintext.ndim != 2 or plaintext.shape[1] != self.n or np.any(plaintext < 0) or np.any(plaintext >= self.t):
            raise ValueError("Plaintext is outside the configured BFV message space")
        r = np.rint(rng.normal(0, self.error_std, (len(plaintext), self.n1))).astype(np.int64)
        e0 = np.rint(rng.normal(0, self.error_std, (len(plaintext), self.n))).astype(np.int64)
        e1 = np.rint(rng.normal(0, self.error_std, (len(plaintext), self.n2))).astype(np.int64)
        c0 = (r @ self.b.T + e0 + self.delta * plaintext) % self.q
        c1 = (r @ self.a.T + e1) % self.q
        return np.concatenate((c0, c1), axis=1)

    def decrypt(self, ciphertext):
        ciphertext = np.asarray(ciphertext, dtype=np.int64)
        if ciphertext.ndim != 2 or ciphertext.shape[1] != self.n + self.n2:
            raise ValueError("BFV ciphertext must include both c0 and the transmitted c1 side information")
        recovered = (ciphertext[:, :self.n] + ciphertext[:, self.n:] @ self.s.T) % self.q
        return (np.rint(recovered / self.delta).astype(np.int64) % self.t).astype(np.uint8)


def _bfv_bits(ciphertext):
    return ((np.asarray(ciphertext, dtype=np.int64)[..., None] >> np.arange(11, -1, -1)) & 1).astype(np.uint8).reshape(len(ciphertext), -1)


def _bits_bfv(bits, q):
    return (np.asarray(bits).reshape(len(bits), -1, 12) * (1 << np.arange(11, -1, -1))).sum(-1).astype(np.int64) % q


def aes_encrypt(symbols, quant_bits, key, namespace):
    packed = pack_quantized(symbols, quant_bits)
    aes = AESGCM(key)
    prefix = hashlib.sha256(str(namespace).encode()).digest()[:8]
    result = []
    for i, payload in enumerate(packed):
        nonce = prefix + i.to_bytes(4, "big")
        result.append(np.frombuffer(nonce + aes.encrypt(nonce, payload.tobytes(), None), dtype=np.uint8))
    return np.stack(result)


def aes_decrypt(packets, bits, key):
    aes = AESGCM(key)
    plaintext = np.zeros((len(packets), 256), dtype=np.uint8)
    accepted = np.zeros(len(packets), dtype=bool)
    for i, packet in enumerate(np.asarray(packets, dtype=np.uint8)):
        nonce, ciphertext = packet[:12].tobytes(), packet[12:].tobytes()
        try:
            data = aes.decrypt(nonce, ciphertext, None)
            plaintext[i] = unpack_quantized(np.frombuffer(data, dtype=np.uint8)[None], bits)[0]
            accepted[i] = True
        except InvalidTag:
            pass
    return plaintext, accepted


class BaselineAttacker(nn.Module):
    def __init__(self, dimension, width, classification):
        super().__init__()
        self.classification = classification
        self.net = nn.Sequential(nn.Linear(dimension, width), nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 10 if classification else 3072))

    def forward(self, x):
        output = self.net(x)
        return output if self.classification else torch.sigmoid(output).reshape(-1, 3, 32, 32)


def _atomic_torch(path, value):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _fit_attacker(train_x, val_x, images, labels, fit, val, settings, path, classification, device):
    path.mkdir(parents=True, exist_ok=True)
    common.torch_seed(settings["attacker_seed"] + int(classification))
    model = BaselineAttacker(train_x.shape[1], settings["attacker_width"], classification).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    signature = common.fit_signature(settings, {"shape": train_x.shape, "classification": classification, "fit_features": hashlib.sha256(np.asarray(train_x).tobytes()).hexdigest(), "validation_features": hashlib.sha256(np.asarray(val_x).tobytes()).hexdigest(), "fit_indices": hashlib.sha256(np.asarray(fit).tobytes()).hexdigest(), "val_indices": hashlib.sha256(np.asarray(val).tobytes()).hexdigest()})
    history, start_epoch, best_score = [], 0, -math.inf
    last = path / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location=device, weights_only=False)
        if saved["signature"] != signature:
            raise ValueError(f"Baseline attacker configuration differs: {path}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, start_epoch, best_score = saved["history"], saved["epoch"], saved["best_score"]
    for epoch in range(start_epoch, settings["attacker_epochs"]):
        model.train()
        rng = np.random.default_rng(common.stable_seed(settings["attacker_seed"], classification, epoch))
        order = rng.permutation(len(fit))
        summed = 0.
        for start in range(0, len(order), settings["attacker_batch_size"]):
            indices = order[start:start + settings["attacker_batch_size"]]
            x = torch.as_tensor(np.array(train_x[indices]), device=device)
            target = torch.as_tensor(labels[fit[indices]], dtype=torch.long, device=device) if classification else torch.as_tensor(images[fit[indices]].astype(np.float32) / 255., device=device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(x)
            loss = F.cross_entropy(predicted, target) if classification else F.mse_loss(predicted, target)
            loss.backward()
            optimizer.step()
            summed += float(loss.detach()) * len(indices)
        predictions = _prediction(model, val_x, settings, device)
        if classification:
            score = float(np.mean(predictions.argmax(1) == labels[val]))
            validation_loss = float(-np.log(np.maximum(predictions[np.arange(len(val)), labels[val]], 1e-12)).mean())
        else:
            rows = common.image_rows(predictions, images[val], device=device)
            score, validation_loss = float(np.mean(rows["ssim_valid"])), float(np.mean(rows["mse"]))
        history.append({"epoch": epoch + 1, "fitting_loss": summed / len(fit), "validation_selection_metric": score, "validation_loss": validation_loss, "training_presentations": (epoch + 1) * len(fit)})
        if score > best_score:
            best_score = score
            _atomic_torch(path / "best.pt", {"model": model.state_dict(), "epoch": epoch + 1, "score": score, "signature": signature})
        _atomic_torch(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "best_score": best_score, "history": history, "signature": signature})
        common.write_json(path / "history.json", history)
        print(f"baseline attack {path.name} epoch{epoch + 1}/{settings['attacker_epochs']}: validation={score:.6f}", flush=True)
    best = torch.load(path / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    return model, {"selected_epoch": best["epoch"], "validation_selection_metric": best["score"], "selection_metric": "accuracy" if classification else "ssim_valid", "test_used_for_selection": False, "training_packets": len(fit), "training_presentations": len(fit) * settings["attacker_epochs"], "fresh_supervised_fit": True, "checkpoint_sha256": common.sha256_file(path / "best.pt")}


def _prediction(model, features, settings, device):
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), settings["attacker_batch_size"]):
            value = model(torch.as_tensor(np.array(features[start:start + settings["attacker_batch_size"]]), device=device))
            output.append((value.softmax(1) if model.classification else value).cpu().numpy())
    return np.concatenate(output)


def _decode(decoder, symbols, device, batch=256):
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(symbols), batch):
            predictions.append(decoder(torch.as_tensor(np.asarray(symbols[start:start + batch]), device=device)).cpu().numpy())
    return np.concatenate(predictions)


def _features(bits):
    return (np.asarray(bits, dtype=np.float32) - .5) * 2


def _headers(count, group, bits, snr, power, rng, key=None, sender=7):
    if bits not in (128, 256):
        raise ValueError("Header lengths must match the 128-bit passive or 256-bit authenticated framing")
    if not 1 <= group <= 128:
        raise ValueError("The concrete authenticated group framing supports group sizes 1 through 128")
    key = protocol.auth_key(bytes(range(32))) if key is None else key
    sigma = 0. if snr is None or np.isinf(snr) else math.sqrt(10 ** (-float(snr) / 10) / power)
    transmitted, decoded, accepted, wrong = [], [], [], []
    for offset in range(0, count, group):
        size, base = min(group, count - offset), 1000 + offset
        raw = protocol.original_header((sender << 64) + base, size) if bits == 128 else protocol.seal_header(protocol.encode_header(sender, base, size), key)
        encoded = protocol.ecc_encode(raw)
        observed = 1 - 2 * encoded.astype(np.float64) + rng.normal(0, sigma, len(encoded))
        restored = protocol.ecc_soft_decode(observed)
        actual_nonces = [(sender << 64) + base + slot for slot in range(size)]
        try:
            if bits == 128:
                recovered_base, recovered_size = protocol.parse_original_header(restored, size)
                recovered = [recovered_base + slot for slot in range(size)]
            else:
                header = protocol.open_header(restored, key)
                if header["count"] != size:
                    raise ValueError("Group-count mismatch")
                recovered = [int.from_bytes(protocol.slot_nonce(header, slot), "big") for slot in range(size)]
            is_accepted = True
        except ValueError:
            is_accepted, recovered = False, actual_nonces
        transmitted.extend(actual_nonces)
        decoded.extend(recovered)
        accepted.extend([is_accepted] * size)
        wrong.extend([is_accepted and restored != raw] * size)
    return transmitted, decoded, np.asarray(accepted, dtype=bool), np.asarray(wrong, dtype=bool)


def _header_success(count, group, bits, repetition, snr, power, rng):
    if repetition != 2:
        raise ValueError("Hamming(8,4) has exactly two coded bits per information bit")
    _, _, accepted, wrong = _headers(count, group, bits, snr, power, rng)
    return accepted & ~wrong


def _keyed_channel(u, case, group, header_bits, snr, power, rng):
    sender = common.stable_seed(case["name"], "baseline_group_sender", group)
    if sender == int.from_bytes(b"RV26", "big"):
        sender ^= 1
    master = bytes.fromhex(case["key_hex"])
    nonces, recovered_nonces, accepted, wrong = _headers(len(u), group, header_bits, snr, power, rng, protocol.auth_key(master), sender)
    scrambler = CryptographicScrambler(master, 256)
    parameters = scrambler.parameters_batch(nonces)
    sigma = 0. if snr is None or np.isinf(snr) else math.sqrt(10 ** (-float(snr) / 10) / power)
    received = parameters.forward(u) + rng.normal(0, sigma, u.shape).astype(np.float32)
    inverse_parameters = scrambler.parameters_batch(recovered_nonces) if wrong.any() else parameters
    return inverse_parameters.inverse(received).astype(np.float32), accepted, wrong


def run(case, cfg, outdir, device, smoke=False):
    settings = {**defaults(), **cfg.get("baselines", {})}
    if smoke:
        settings.update(snr_db=[10., 30.], total_complex_uses=[640], groups=[1, 32], attacker_epochs=1, attacker_width=32, attacker_batch_size=16, timing_repeats=2)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    signature = common.fit_signature(settings, {"case": case["name"], "smoke": smoke, "train_u_sha256": hashlib.sha256(np.asarray(case["train_u"]).tobytes()).hexdigest(), "test_u_sha256": hashlib.sha256(np.asarray(case["test_u"]).tobytes()).hexdigest()})
    summary_path = outdir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary["signature"] != signature:
            raise ValueError("Baseline configuration differs; use a new output directory")
        return summary
    fit = np.asarray(case["fit"], dtype=np.int64)
    val = np.asarray(case["val"], dtype=np.int64)
    if smoke:
        fit, val = fit[:32], val[:16]
    if set(fit) & set(val):
        raise ValueError("Baseline fitting/validation indices overlap")
    train_u, test_u = np.asarray(case["train_u"]), np.asarray(case["test_u"])
    images, labels, truth = case["train_images"], case["train_labels"], case["test_images"]
    test_labels = np.asarray(case["test_labels"])
    if smoke:
        test_u, truth, test_labels = test_u[:16], truth[:16], test_labels[:16]
    prior_image = images[fit].astype(np.float64).mean(0).astype(np.float32) / 255.
    prior = np.broadcast_to(prior_image, truth.shape).copy()
    prior_rows = common.image_rows(prior, truth, device=device)
    common.save_npz(outdir / "prior_rows.npz", **prior_rows)
    key = hashlib.sha256(bytes.fromhex(case["key_hex"]) + b"AES-GCM-baseline-v1").digest()
    bfv = LinearBFV(common.stable_seed(case["name"], "bfv_key"), n1=settings["bfv_n1"], n2=settings["bfv_n2"], pt_mod=settings["bfv_pt_mod"], ct_mod=settings["bfv_ct_mod"], error_std=settings["bfv_error_std"])
    if settings["bfv_pt_mod"] != 16 or settings["bfv_ct_mod"] > 4096 or settings["bfv_modulation"] != 4096:
        raise ValueError("This transparent BFV operating point uses t=16, q<=4096 and 4096-QAM")
    quantizers = {bits: LatentQuantizer.fit(train_u[fit], bits, settings["quantizer_percentiles"]) for bits in sorted(set([4] + list(settings["aes_quantizer_bits"])))}
    common.save_npz(outdir / "quantizers.npz", **{f"bits{bits}_{name}": getattr(quantizer, name) for bits, quantizer in quantizers.items() for name in ("lower", "upper")}, fitting_indices=fit, validation_indices=val)
    quantization_oracles = {}
    for bits, quantizer in quantizers.items():
        oracle_prediction = _decode(case["dec"], quantizer.decode(quantizer.encode(test_u)), device)
        oracle_rows = common.image_rows(oracle_prediction, truth, device=device)
        common.save_npz(outdir / f"quantization_oracle_{bits}bit.npz", **oracle_rows)
        quantization_oracles[f"bits{bits}"] = common.image_summary(oracle_rows)
    packets = {}
    for split, u in (("fit", train_u[fit]), ("validation", train_u[val]), ("test", test_u)):
        rng = np.random.default_rng(common.stable_seed(case["name"], "bfv_packets", split))
        packets[("bfv", split)] = _bfv_bits(bfv.encrypt(quantizers[4].encode(u), rng))
        for bits in settings["aes_quantizer_bits"]:
            encrypted = aes_encrypt(quantizers[bits].encode(u), bits, key, f"{case['name']}:{split}:{bits}")
            packets[(f"aes{bits}", split)] = np.unpackbits(encrypted, axis=1)
    common.write_json(outdir / "packet_manifest.json", {f"{scheme}_{split}": {"shape": value.shape, "sha256": hashlib.sha256(value.tobytes()).hexdigest()} for (scheme, split), value in packets.items()})
    attackers, attack_records = {}, {}
    schemes = ["keyed", "identity", "bfv", *[f"aes{bits}" for bits in settings["aes_quantizer_bits"]]]
    split_latents = {"fit": train_u[fit], "validation": train_u[val], "test": test_u}
    for scheme in schemes:
        feature_sets = {}
        for split in ("fit", "validation", "test"):
            rng = np.random.default_rng(common.stable_seed(case["name"], scheme, split, "eve_packets"))
            if scheme == "keyed":
                obs, nonce = common.actual_observations(split_latents[split], case, f"baseline_attacker_{split}", settings["eve_snr_db"])
                feature_sets[split] = np.concatenate((obs[:, 0], nonce[:, 0]), axis=1)
            elif scheme == "identity":
                u = split_latents[split]
                feature_sets[split] = u + rng.normal(0, 10 ** (-settings["eve_snr_db"] / 20), u.shape).astype(np.float32)
            else:
                order = settings["bfv_modulation"] if scheme == "bfv" else settings["aes_modulation"]
                received = qam_channel(packets[(scheme, split)], order, settings["eve_snr_db"], rng)
                feature_sets[split] = _features(received)
        for classification in (False, True):
            name = f"{scheme}_{'classification' if classification else 'reconstruction'}"
            model, record = _fit_attacker(feature_sets["fit"], feature_sets["validation"], images, labels, fit, val, settings, outdir / name, classification, device)
            prediction = _prediction(model, feature_sets["test"], settings, device)
            measured = common.classification_rows(test_labels, prediction) if classification else common.image_rows(prediction, truth, device=device)
            common.save_npz(outdir / f"{name}_test.npz", prediction=prediction if classification else np.empty(0, dtype=np.float32), **{k: v for k, v in measured.items() if isinstance(v, np.ndarray)})
            attack_records[name] = {**record, "metrics": {k: v for k, v in measured.items() if not isinstance(v, np.ndarray)} if classification else common.image_summary(measured), "eve_snr_db": settings["eve_snr_db"], "observation": "keyed: payload and public nonce; identity: unprotected payload; digital schemes: all transmitted ciphertext and public side information, including nonce/tag for AES and c1 for BFV"}
            attackers[name] = model
    operating = []
    all_metric_arrays = {}
    eve_cache = {}
    resource_selections = {}

    def resource_eve(scheme, power):
        tag = f"{scheme}_power{power:.12g}"
        if tag in eve_cache:
            return eve_cache[tag]
        effective = settings["eve_snr_db"] + 10 * math.log10(power)
        def observed(split):
            rng = np.random.default_rng(common.stable_seed(case["name"], tag, split, "resource_eve"))
            u = split_latents[split]
            if scheme == "keyed":
                obs, nonce = common.actual_observations(u, case, f"baseline_resource_eve_{split}", effective)
                return np.concatenate((obs[:, 0], nonce[:, 0]), axis=1)
            if scheme == "identity":
                return u + rng.normal(0, 10 ** (-effective / 20), u.shape).astype(np.float32)
            order = settings["bfv_modulation"] if scheme == "bfv" else settings["aes_modulation"]
            return _features(qam_channel(packets[(scheme, split)], order, settings["eve_snr_db"], rng, power))
        features = observed("test")
        checkpoints = {}
        models = {}
        if settings["resource_matched_attack_fitting"]:
            train_features, val_features = observed("fit"), observed("validation")
            for classification in (False, True):
                task = "classification" if classification else "reconstruction"
                checkpoint_dir = outdir / "resource_attacks" / tag / task
                models[task], selection = _fit_attacker(train_features, val_features, images, labels, fit, val, settings, checkpoint_dir, classification, device)
                checkpoints[task] = str((checkpoint_dir / "best.pt").relative_to(outdir))
                resource_selections[f"{tag}_{task}"] = {**selection, "checkpoint": checkpoints[task], "effective_snr_db": effective, "scheme": scheme, "power_per_active_symbol": power}
            del train_features, val_features
        else:
            for task in ("reconstruction", "classification"):
                models[task] = attackers[f"{scheme}_{task}"]
                checkpoints[task] = f"{scheme}_{task}/best.pt"
        reconstructed = _prediction(models["reconstruction"], features, settings, device)
        probabilities = _prediction(models["classification"], features, settings, device)
        measured = common.image_rows(reconstructed, truth, device=device)
        classification = common.classification_rows(test_labels, probabilities)
        common.save_npz(outdir / f"eve_{tag}.npz", **measured, probabilities=probabilities, correct=classification["correct"], negative_log_probability=classification["negative_log_probability"])
        result = {"eve_nominal_snr_db": settings["eve_snr_db"], "eve_effective_payload_snr_db": effective, "eve_mse": float(np.mean(measured["mse"])), "eve_ssim_valid": float(np.mean(measured["ssim_valid"])), "eve_class_accuracy": classification["accuracy"], "eve_class_cross_entropy": classification["cross_entropy"], "resource_attack_fitting": "fresh_at_effective_snr" if settings["resource_matched_attack_fitting"] else "transfer_from_nominal_snr", "eve_reconstruction_checkpoint": checkpoints["reconstruction"], "eve_classification_checkpoint": checkpoints["classification"]}
        eve_cache[tag] = result
        return result
    for budget in settings["total_complex_uses"]:
        for snr in settings["snr_db"]:
            for group in settings["groups"]:
                for header_bits in settings["keyed_header_bits"]:
                    uses = 128 + header_bits * settings["keyed_header_repetition"] / 2 * math.ceil(len(test_u) / group) / len(test_u)
                    if uses > budget:
                        raise ValueError(f"Equal-resource budget {budget} cannot accommodate keyed framing using {uses} symbols")
                    power = budget / uses
                    domain = f"baseline_keyed_budget{budget}_group{group}_header{header_bits}"
                    rng = np.random.default_rng(common.stable_seed(case["name"], domain, snr, "header"))
                    if settings["keyed_header_repetition"] != 2:
                        raise ValueError("The implemented Hamming(8,4) header expands each information bit by two")
                    recovered, accepted, wrong = _keyed_channel(test_u, case, group, header_bits, snr, power, rng)
                    decoded = _decode(case["dec"], recovered, device)
                    decoded[~accepted] = prior_image
                    name = f"keyed_h{header_bits}_g{group}"
                    metrics = common.image_rows(decoded, truth, device=device)
                    measured_energy = power * (float(np.mean(np.sum(test_u.astype(np.float64) ** 2, axis=1))) / 2 + uses - 128)
                    row = {"scheme": name, "budget_complex_uses": budget, "snr_db": snr, "active_complex_uses_per_image": uses, "total_transmit_energy_per_image": budget, "measured_mean_transmit_energy_per_image": measured_energy, "power_per_active_symbol": power, "protected_metadata_bits": header_bits, "metadata_authentication": header_bits == 256, "payload_authentication": False, "packet_success_fraction": float((accepted & ~wrong).mean()), "undetected_wrong_metadata_images": int(wrong.sum()), **common.image_summary(metrics), **resource_eve("keyed", power)}
                    operating.append(row)
                    all_metric_arrays[f"{name}_C{budget}_S{snr}"] = metrics
            for scheme in ["bfv", *[f"aes{bits}" for bits in settings["aes_quantizer_bits"]]]:
                order = settings["bfv_modulation"] if scheme == "bfv" else settings["aes_modulation"]
                bits = packets[(scheme, "test")]
                uses = bits.shape[1] / math.log2(order)
                if uses > budget:
                    raise ValueError(f"Equal-resource budget {budget} cannot accommodate {scheme} with {uses} symbols")
                power = budget / uses
                rng = np.random.default_rng(common.stable_seed(case["name"], scheme, snr, budget, "bob_packets"))
                received = qam_channel(bits, order, snr, rng, power)
                if scheme == "bfv":
                    decrypted = bfv.decrypt(_bits_bfv(received, bfv.q))
                    decoded = _decode(case["dec"], quantizers[4].decode(decrypted), device)
                    success = np.all(decrypted == quantizers[4].encode(test_u), axis=1)
                else:
                    quant_bits = int(scheme[3:])
                    decrypted, success = aes_decrypt(np.packbits(received, axis=1), quant_bits, key)
                    decoded = _decode(case["dec"], quantizers[quant_bits].decode(decrypted), device)
                    decoded[~success] = prior_image
                metrics = common.image_rows(decoded, truth, device=device)
                measured_energy = budget * float(np.mean(np.abs(qam_modulate(bits, order)) ** 2))
                row = {"scheme": scheme, "budget_complex_uses": budget, "snr_db": snr, "active_complex_uses_per_image": uses, "total_transmit_energy_per_image": budget, "measured_mean_transmit_energy_per_image": measured_energy, "power_per_active_symbol": power, "protected_metadata_bits": 0 if scheme == "bfv" else 224, "metadata_authentication": scheme.startswith("aes"), "payload_authentication": scheme.startswith("aes"), "packet_success_fraction": float(success.mean()), "undetected_wrong_metadata_images": 0, **common.image_summary(metrics), **resource_eve(scheme, power)}
                operating.append(row)
                all_metric_arrays[f"{scheme}_C{budget}_S{snr}"] = metrics
            uses = 128.
            effective = snr + 10 * math.log10(budget / uses)
            rng = np.random.default_rng(common.stable_seed(case["name"], "identity_baseline", snr, budget))
            noisy = test_u + rng.normal(0, 10 ** (-effective / 20), test_u.shape).astype(np.float32)
            decoded = _decode(case["dec"], noisy, device)
            metrics = common.image_rows(decoded, truth, device=device)
            measured_energy = budget / uses * float(np.mean(np.sum(test_u.astype(np.float64) ** 2, axis=1))) / 2
            operating.append({"scheme": "unprotected_identity", "budget_complex_uses": budget, "snr_db": snr, "active_complex_uses_per_image": uses, "total_transmit_energy_per_image": budget, "measured_mean_transmit_energy_per_image": measured_energy, "power_per_active_symbol": budget / uses, "protected_metadata_bits": 0, "metadata_authentication": False, "payload_authentication": False, "packet_success_fraction": 1., "undetected_wrong_metadata_images": 0, **common.image_summary(metrics), **resource_eve("identity", budget / uses)})
            all_metric_arrays[f"identity_C{budget}_S{snr}"] = metrics
    with (outdir / "equal_resource_operating_points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(operating[0]))
        writer.writeheader()
        writer.writerows(operating)
    common.save_npz(outdir / "per_image_operating_points.npz", **{f"{scheme}__{metric}": values for scheme, metrics in all_metric_arrays.items() for metric, values in metrics.items()})
    timing_rng = np.random.default_rng(571)
    timing_u = test_u[:min(128, len(test_u))]
    timing_quantized = quantizers[4].encode(timing_u)
    timing_values = {name: [] for name in ("bfv_encrypt", "bfv_decrypt", "aes4_encrypt", "aes4_decrypt")}
    for repeat in range(settings["timing_repeats"] + 2):
        started = time.perf_counter()
        ciphertext = bfv.encrypt(timing_quantized, timing_rng)
        durations = {"bfv_encrypt": 1000 * (time.perf_counter() - started) / len(ciphertext)}
        started = time.perf_counter()
        bfv.decrypt(ciphertext)
        durations["bfv_decrypt"] = 1000 * (time.perf_counter() - started) / len(ciphertext)
        started = time.perf_counter()
        ciphertext = aes_encrypt(timing_quantized, 4, key, f"{case['name']}:timing:{repeat}")
        durations["aes4_encrypt"] = 1000 * (time.perf_counter() - started) / len(ciphertext)
        started = time.perf_counter()
        aes_decrypt(ciphertext, 4, key)
        durations["aes4_decrypt"] = 1000 * (time.perf_counter() - started) / len(ciphertext)
        if repeat >= 2:
            for name, duration in durations.items():
                timing_values[name].append(duration)
    timing_summary = {"batch": len(timing_u), "warmup_repeats": 2, "measured_repeats": settings["timing_repeats"], "scope": "cryptographic operations only; codec, quantization, modem and channel simulation excluded", "values": {name: {"median": float(np.median(values)), "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9)), "samples": values} for name, values in timing_values.items()}}
    summary = {"signature": signature, "case": case["name"], "status": "smoke_not_scientific" if smoke else "scientific", "comparison_scope": "frozen common learned codec; independently implemented conventional AES-GCM and linear-BFV mechanism inspired by DeepJSCEC, not an end-to-end DeepJSCEC reproduction", "published_mechanism_reference": "https://arxiv.org/abs/2208.09245", "bfv_parameters": {"n": 256, "n1": bfv.n1, "n2": bfv.n2, "t": bfv.t, "q": bfv.q, "error_std": bfv.error_std, "fresh_ephemeral_noise_per_packet": True, "transmitted_c1_symbols": bfv.n2, "public_key_coefficients": bfv.a.size + bfv.b.size, "public_key_provisioning": "amortized session setup, excluded from per-image channel use count", "integer_arithmetic": "int64 matrix products, reduction modulo q"}, "aes_parameters": {"algorithm": "AES-256-GCM", "nonce_bits": 96, "tag_bits": 128, "nonce_policy": "deterministic disjoint experimental namespace and per-packet counter; reproduction-only key", "modulation": settings["aes_modulation"]}, "keyed_header_interpretation": {"128": "passive framing, error detection only", "256": "metadata authentication tag included, payload not authenticated"}, "channel_model": "complex AWGN; unit average constellation energy before equal-resource power allocation", "energy_accounting": "Each row has identical reserved complex channel uses and total energy; unused uses are idle and active symbols receive total_energy/active_uses power. No common peak-power constraint is imposed.", "attacks": attack_records, "operating_points": operating, "timings_ms_per_image_cpu_batch_amortized": timing_summary, "quantization_only_oracles": quantization_oracles, "prior": common.image_summary(prior_rows), "test_used_for_selection": False, "fixed_codec_quantization": "per-coordinate clipping/scale learned on fitting images only; no baseline-specific codec retraining", "receiver_failure_policy": "AES authentication/header rejection uses fitting-set mean image; BFV always outputs its decrypted quantized latent"}
    summary["energy_accounting"] = "Equal reserved complex channel uses and nominal total energy using unit-average-energy QAM constellations. Unused uses are idle; active symbol power equals nominal_budget/active_uses. Measured finite-packet mean energies are separately recorded. No common peak-power constraint is imposed."
    summary["header_channel_design"] = {"ecc": "extended Hamming(8,4), soft maximum-likelihood codeword decoding, QPSK", "passive": "96-bit nonce, 16-bit group count, CRC16; 128 information bits, 256 coded bits, 128 complex uses per group", "authenticated": "128-bit structured metadata and 128-bit HMAC-SHA256 tag; 256 information bits, 512 coded bits, 256 complex uses per group", "accepted_wrong_metadata": "decoded actual nonce is applied to the payload inverse and affected images are counted", "partial_group": "actual last group count and header cost are included"}
    summary["resource_eve_metadata"] = "Eve is supplied correct public metadata, conservatively independent of any channel errors in her metadata reception."
    summary["attack_training_packets"] = "One independently encrypted/noisy packet per fitting image; neural fitting revisits these packets across epochs. All public nonce, tag and BFV c1 side-information bits are included."
    summary["security_interpretation"] = "Measured reconstruction and class inference concern this finite matched neural suite. AES-GCM's conventional cryptographic properties are distinct from these empirical measurements; poor attack scores do not establish comparable cryptographic guarantees for the other schemes."
    summary["resource_attack_selections"] = resource_selections
    common.write_json(summary_path, summary)
    return summary
