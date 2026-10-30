from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from keyed_jscc.crypto_scrambler import CryptographicScrambler


def defaults():
    return {
        "snr_db": [0, 5, 10, 15, 20, 30],
        "draws": [0, 1, 2],
        "max_test_images": 3600,
        "sample_rate_hz": 1000000,
        "carrier_hz": 3500000000,
        "speed_m_s": 15.0,
        "rician_k_db": 6.0,
                "tap_delays_samples": [0, 1, 3],
        "tap_power_db": [0.0, -3.0, -6.0],
        "propagation_delay_samples": 3,
        "max_timing_search_samples": 8,
        "preamble_symbols": 64,
        "pilot_symbols": 128,
        "channel_batch": 128,
        "kpa_counts": [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 512, 4096],
        "kpa_subset_seeds": [0, 1, 2],
        "kpa_snr_db": 20,
        "kpa_rcond": 1e-6,
        "repeat_plaintext_counts": [1, 2, 4, 8, 16, 32, 64],
        "run_kpa": True,
        "scenarios": ["flat_rayleigh_perfect", "flat_rayleigh_estimated", "flat_rician_perfect", "flat_rician_estimated", "multipath_cfo_perfect", "multipath_cfo_estimated", "multipath_cfo_unsynchronized"],
    }


def pack_complex(u):
    u = np.asarray(u)
    if u.shape[-1] != 256:
        raise ValueError("The physical mapping requires exactly 256 real symbols")
    return (u[..., :128] + 1j * u[..., 128:]) / np.sqrt(2)


def unpack_complex(x):
    x = np.asarray(x)
    if x.shape[-1] != 128:
        raise ValueError("The physical mapping requires exactly 128 complex symbols")
    return np.concatenate([x.real, x.imag], axis=-1) * np.sqrt(2)


def complex_matrix_to_real(matrix):
    matrix = np.asarray(matrix)
    return np.block([[matrix.real, -matrix.imag], [matrix.imag, matrix.real]])


def flat_conditional_noise_covariance(h, estimate, noise_variance, transform=None):
    """Covariance after scalar equalization, expressed in decoder real coordinates.

    Given h and its estimate, receiver noise is circular white. A noisy channel
    estimate additionally changes the signal map by h/estimate; that distortion
    is not silently relabeled as independent AWGN.
    """
    variance = float(noise_variance) / abs(estimate) ** 2
    covariance = variance * np.eye(256)
    if transform is not None:
        covariance = transform @ covariance @ transform.T
    signal = complex_matrix_to_real((h / estimate) * np.eye(128))
    if transform is not None:
        signal = transform @ signal @ transform.T
    return signal, covariance


def _draw_taps(rng, count, scenario, cfg):
    if scenario.startswith("flat_rayleigh"):
        return (rng.normal(size=(count, 1)) + 1j * rng.normal(size=(count, 1))) / np.sqrt(2)
    if scenario.startswith("flat_rician"):
        k = 10 ** (float(cfg["rician_k_db"]) / 10)
        scattered = (rng.normal(size=(count, 1)) + 1j * rng.normal(size=(count, 1))) / np.sqrt(2)
        phase = np.exp(1j * rng.uniform(-np.pi, np.pi, (count, 1)))
        return np.sqrt(k / (k + 1)) * phase + scattered / np.sqrt(k + 1)
    delays = np.asarray(cfg["tap_delays_samples"], dtype=int)
    power = 10 ** (np.asarray(cfg["tap_power_db"], dtype=float) / 10)
    power /= power.sum()
    taps = np.zeros((count, int(delays.max()) + 1), dtype=np.complex128)
    taps[:, delays] = (rng.normal(size=(count, len(delays))) + 1j * rng.normal(size=(count, len(delays)))) * np.sqrt(power / 2)
    return taps


def _channel_convolve(frame, taps):
    output = np.zeros((len(frame), frame.shape[1] + taps.shape[1] - 1), dtype=np.complex128)
    for lag in range(taps.shape[1]):
        output[:, lag:lag + frame.shape[1]] += taps[:, lag, None] * frame
    return output


def transmit_pair(identity, keyed, rng, snr_db, scenario, cfg):
    """Same physical channel, CSI-estimation realization and AWGN for both arms."""
    n = len(identity)
    if int(cfg["pilot_symbols"]) != 128:
        raise ValueError("This declared pilot design uses exactly 128 known complex symbols")
    if float(cfg["sample_rate_hz"]) <= 0 or float(cfg["carrier_hz"]) <= 0:
        raise ValueError("Sample rate and carrier must be positive")
    delays = np.asarray(cfg["tap_delays_samples"], dtype=int)
    if len(delays) != len(cfg["tap_power_db"]) or len(np.unique(delays)) != len(delays) or np.any(delays < 0) or 0 not in delays:
        raise ValueError("Tap delays must be unique nonnegative integers including zero")
    multipath = scenario.startswith("multipath")
    taps = _draw_taps(rng, n, scenario, cfg)
    cp = taps.shape[1] - 1
    delay = int(cfg["propagation_delay_samples"]) if multipath else 0
    search = int(cfg["max_timing_search_samples"])
    if delay < 0 or search < delay:
        raise ValueError("Declared arrival uncertainty does not cover propagation delay")
    preamble_length = int(cfg["preamble_symbols"])
    if preamble_length < 16 or preamble_length % 2:
        raise ValueError("Preamble length must be even and at least 16")
    half = preamble_length // 2
    pre_rng = np.random.default_rng(32007)
    half_pre = np.exp(1j * (np.pi / 4 + np.pi / 2 * pre_rng.integers(0, 4, half)))
    preamble = np.concatenate([half_pre, half_pre])
    pilot = np.exp(1j * (np.pi / 4 + np.pi / 2 * pre_rng.integers(0, 4, 128)))
    block_length = 128 + cp
    pilot_block = np.concatenate([np.broadcast_to(pilot, (n, 128)), np.zeros((n, cp))], axis=1)
    prefix = np.concatenate([np.broadcast_to(preamble, (n, preamble_length)), np.zeros((n, cp)), pilot_block, np.zeros((n, search))], axis=1)
    pilot_start = preamble_length + cp
    payload_start = prefix.shape[1]
    frame_length = prefix.shape[1] + block_length
    frequency_pilot = np.fft.fft(np.concatenate([pilot, np.zeros(cp)]))
    guard = search + taps.shape[1] - 1
    noise_variance = 10 ** (-float(snr_db) / 10)
    noise = (rng.normal(size=(n, frame_length + guard)) + 1j * rng.normal(size=(n, frame_length + guard))) * np.sqrt(noise_variance / 2)
    cfo = float(cfg["speed_m_s"]) * float(cfg["carrier_hz"]) / 299792458.0 if multipath else 0.0
    angular_cfo = 2 * np.pi * cfo / float(cfg["sample_rate_hz"])
    length = frame_length + guard
    time_index = np.arange(length)
    all_rx = []
    for transmitted in (identity, keyed):
        payload = np.concatenate([pack_complex(transmitted), np.zeros((n, cp))], axis=1)
        frame = np.concatenate([prefix, payload], axis=1)
        convolved = _channel_convolve(frame, taps)
        rx = np.zeros((n, length), dtype=np.complex128)
        available = min(convolved.shape[1], length - delay)
        rx[:, delay:delay + available] = convolved[:, :available]
        rx *= np.exp(1j * angular_cfo * time_index)[None, :]
        all_rx.append(rx + noise)
    rx_prefix = all_rx[0]
    if scenario.endswith("perfect"):
        estimated_delay = np.full(n, delay, dtype=int)
        omega = np.full(n, angular_cfo)
    elif scenario.endswith("unsynchronized"):
        estimated_delay = np.zeros(n, dtype=int)
        omega = np.zeros(n)
    else:
        scores = np.stack([np.abs(np.sum(rx_prefix[:, offset:offset + half] * half_pre.conj(), axis=1)) for offset in range(search + 1)], axis=1)
        estimated_delay = scores.argmax(axis=1)
        first = np.take_along_axis(rx_prefix, estimated_delay[:, None] + np.arange(half)[None, :], axis=1)
        second = np.take_along_axis(rx_prefix, estimated_delay[:, None] + half + np.arange(half)[None, :], axis=1)
        omega = np.angle(np.sum(first.conj() * second, axis=1)) / half
    corrected = [rx * np.exp(-1j * omega[:, None] * time_index[None, :]) for rx in all_rx]
    h_true = np.fft.fft(taps, n=block_length, axis=1)
    if scenario.endswith("perfect"):
        h_estimate = h_true
    else:
        pilot_rx = np.take_along_axis(corrected[0], estimated_delay[:, None] + pilot_start + np.arange(block_length)[None, :], axis=1)
        if not multipath:
            h_scalar = np.mean(pilot_rx[:, :128] * pilot.conj()[None, :], axis=1)
            h_estimate = np.broadcast_to(h_scalar[:, None], (n, block_length))
        else:
            estimate_frequency = np.fft.fft(pilot_rx, axis=1) / frequency_pilot[None, :]
            estimated_taps = np.fft.ifft(estimate_frequency, axis=1)
            keep = np.zeros(block_length, dtype=bool)
            keep[np.asarray(cfg["tap_delays_samples"], dtype=int)] = True
            estimated_taps[:, ~keep] = 0
            h_estimate = np.fft.fft(estimated_taps, axis=1)
    if not multipath:
        inverse = 1 / np.where(np.abs(h_estimate) < 1e-12, 1e-12 + 0j, h_estimate)
        equalizer = "scalar ZF after synchronization; deep-fade noise enhancement retained"
    else:
        inverse = h_estimate.conj() / (np.abs(h_estimate) ** 2 + noise_variance)
        equalizer = "zero-padded single-carrier frequency-domain MMSE; residual signal distortion and colored noise retained"
    estimates = []
    for rx in corrected:
        observed = np.take_along_axis(rx, estimated_delay[:, None] + payload_start + np.arange(block_length)[None, :], axis=1)
        restored = np.fft.ifft(np.fft.fft(observed, axis=1) * inverse, axis=1)
        estimates.append(unpack_complex(restored[:, :128]).astype(np.float32))
    diagnostics = {
        "timing_error_samples": estimated_delay - delay,
        "cfo_error_hz": (omega - angular_cfo) * float(cfg["sample_rate_hz"]) / (2 * np.pi),
        "mean_channel_gain": np.mean(abs(h_true) ** 2, axis=1),
        "csi_frequency_mse": np.mean(abs(h_estimate - h_true) ** 2, axis=1),
        "equalized_white_noise_power": noise_variance * np.mean(abs(inverse) ** 2, axis=1),
        "equalized_noise_frequency_power_cv": np.std(abs(inverse) ** 2, axis=1) / np.maximum(np.mean(abs(inverse) ** 2, axis=1), 1e-12),
    }
    accounting = {"payload_complex_uses": 128, "preamble_complex_uses": preamble_length,
                  "pilot_complex_uses": 128, "zero_padding_guard_complex_uses": 3 * cp,
                  "pilot_to_payload_sync_guard_uses": search,
                  "arrival_uncertainty_guard_uses": guard,
                  "total_complex_uses": length, "equalizer": equalizer,
                  "propagation_delay_samples": delay, "sample_rate_hz": cfg["sample_rate_hz"],
                  "cfo_hz": cfo, "cfo_interpretation": "constant frequency-offset proxy for maximum kinematic Doppler; taps fixed per packet, not a time-varying scattering process",
                  "payload_mean_symbol_energy_identity": float(np.mean(abs(pack_complex(identity)) ** 2)),
                  "payload_mean_symbol_energy_keyed": float(np.mean(abs(pack_complex(keyed)) ** 2)),
                  "total_energy_note": "unit-energy pilots/preamble and zero guards; identity/keyed payload energies match exactly by orthogonality",
                  "noise_variance_per_complex_sample": noise_variance,
                  "csi_note": "perfect CSI is an oracle control; estimated CSI and timing/CFO are computed from shared noisy preamble/pilot; no test-image fitting"}
    energy = []
    for source in (identity, keyed):
        frame = np.concatenate([prefix, pack_complex(source), np.zeros((n, cp))], axis=1)
        energy.append(np.sum(abs(frame) ** 2, axis=1))
    diagnostics["transmitted_energy_identity"] = energy[0]
    diagnostics["transmitted_energy_keyed"] = energy[1]
    return estimates[0], estimates[1], diagnostics, accounting


def _decode(dec, latent, device, batch=128):
    import torch
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(latent), batch):
            value = torch.as_tensor(latent[start:start + batch], dtype=torch.float32, device=device)
            predictions.append(dec(value).float().cpu().numpy())
    return np.concatenate(predictions)


def _metric_record(dec, latent, truth, device):
    from .common import image_rows, image_summary
    prediction = _decode(dec, latent, device)
    rows = image_rows(prediction, truth, device)
    return prediction, rows, image_summary(rows)


def _kpa(case, cfg, outdir, device, smoke):
    from .common import stable_seed, save_npz, write_json
    counts = [1, 4, 16] if smoke else cfg["kpa_counts"]
    seeds = [0] if smoke else cfg["kpa_subset_seeds"]
    fitting = np.asarray(case["fit"])
    counts = [int(value) for value in counts if int(value) <= len(fitting)]
    rng = np.random.default_rng(stable_seed(case["name"], "reused-transform-test"))
    key = bytes.fromhex(case["key_hex"])
    params = CryptographicScrambler(key).parameters(stable_seed(case["name"], "reused-transform-nonce"))
    full_train = np.asarray(case["train_u"])
    full_test = np.asarray(case["test_u"][:cfg["max_test_images"]])
    truth = np.asarray(case["test_images"][:len(full_test)])
    train_y = params.forward(full_train)
    test_y = params.forward(full_test)
    std = 10 ** (-float(cfg["kpa_snr_db"]) / 20)
    train_y = train_y + rng.normal(0, std, train_y.shape).astype(np.float32)
    test_y = test_y + rng.normal(0, std, test_y.shape).astype(np.float32)
    result = []
    correct = params.inverse(test_y)
    prediction, rows, summary = _metric_record(case["dec"], correct, truth, device)
    save_npz(outdir / "kpa_correct_inverse.npz", predictions=prediction, **rows)
    result.append({"kind": "correct_inverse", "pairs": 0, "subset_seed": None, **summary})
    for subset_seed in seeds:
        order_rng = np.random.default_rng(stable_seed(case["name"], "kpa-subset", subset_seed))
        selected = order_rng.permutation(fitting)
        for count in counts:
            destination = outdir / f"kpa_pairs{count}_subset{subset_seed}.json"
            if destination.exists():
                result.append(json.loads(destination.read_text()))
                continue
            indices = selected[:count]
            design = train_y[indices].astype(np.float64)
            target = full_train[indices].astype(np.float64)
            left, singular, right = np.linalg.svd(design, full_matrices=False)
            cutoff = float(cfg["kpa_rcond"]) * singular.max()
            keep = singular > cutoff
            reciprocal = np.zeros_like(singular)
            reciprocal[keep] = 1 / singular[keep]
            inverse = (right.T * reciprocal) @ left.T @ target
            latent = (test_y @ inverse).astype(np.float32)
            prediction, rows, summary = _metric_record(case["dec"], latent, truth, device)
            record = {"kind": "known_plaintext_linear_inverse", "pairs": count, "subset_seed": subset_seed,
                      "rank": int(keep.sum()), "design_condition_nonzero": float(singular[keep].max() / singular[keep].min()),
                      "rcond": cfg["kpa_rcond"], "training_indices": indices.tolist(), "test_used_for_selection": False,
                      "interpretation": "sufficient pair counts in this codec/channel sample, not a universal minimum", **summary}
            save_npz(destination.with_suffix(".npz"), inverse_estimate=inverse, singular_values=singular,
                     selected_training_indices=indices, predictions=prediction, **rows)
            write_json(destination, record)
            result.append(record)
            print(f"KPA {case['name']}: {count} diverse pairs, subset {subset_seed}", flush=True)
    repeat = []
    source_index = int(fitting[0])
    one = full_train[source_index:source_index + 1]
    scrambled_one = params.forward(one)
    for count in ([1, 4] if smoke else cfg["repeat_plaintext_counts"]):
        average = scrambled_one + rng.normal(0, std / np.sqrt(count), scrambled_one.shape)
        rel_error = float(np.mean((average - scrambled_one) ** 2))
        repeat.append({"identical_plaintext_receptions": int(count), "ciphertext_average_mse": rel_error,
                       "expected_noise_mse": std**2 / count, "design_rank_maximum": 1,
                       "interpretation": "noise averaging of one ciphertext reveals one transformed vector; cannot identify a general 256-dimensional map"})
    return {"diverse_known_plaintexts": result, "identical_plaintext_averaging": repeat,
            "snr_db": cfg["kpa_snr_db"], "nonce_policy": "intentional transform-reuse failure control only"}


def run(case, cfg, outdir, device, smoke=False):
    from .common import stable_seed, fit_signature, write_json, save_npz
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ccfg = {**defaults(), **cfg.get("channels", {})}
    if smoke:
        ccfg.update({"snr_db": [10], "draws": [0], "max_test_images": min(8, len(case["test_u"])),
                     "scenarios": ["flat_rayleigh_perfect", "multipath_cfo_estimated"]})
    signature = fit_signature(cfg, {"stage": "channels", "version": 3, "case": case["name"], "smoke": smoke})
    final = outdir / "summary.json"
    if final.exists():
        result = json.loads(final.read_text())
        if result.get("signature") == signature:
            return result
        raise RuntimeError("Channel output already belongs to a different configuration")
    u = np.asarray(case["test_u"][:ccfg["max_test_images"]])
    truth = np.asarray(case["test_images"][:len(u)])
    nonce_start = stable_seed(case["name"], "channel-protocol") << 48
    nonces = [nonce_start + i for i in range(len(u))]
    params = CryptographicScrambler(bytes.fromhex(case["key_hex"])).parameters_batch(nonces)
    scrambled = params.forward(u)
    results = []
    for scenario in ccfg["scenarios"]:
        for snr in ccfg["snr_db"]:
            for draw in ccfg["draws"]:
                label = f"{scenario}_snr{snr:g}_draw{draw}"
                record_path = outdir / f"{label}.json"
                if record_path.exists():
                    record = json.loads(record_path.read_text())
                    if record.get("signature") != signature:
                        raise RuntimeError("Partial channel result has incompatible configuration")
                    results.append(record)
                    continue
                observation_identity, observation_keyed, diagnostics, accounting = [], [], {}, None
                rng = np.random.default_rng(stable_seed(case["name"], label))
                for start in range(0, len(u), int(ccfg["channel_batch"])):
                    end = min(len(u), start + int(ccfg["channel_batch"]))
                    identity, keyed, batch_diag, accounting = transmit_pair(u[start:end], scrambled[start:end], rng, snr, scenario, ccfg)
                    observation_identity.append(identity)
                    observation_keyed.append(keyed)
                    for name, value in batch_diag.items():
                        diagnostics.setdefault(name, []).append(value)
                identity = np.concatenate(observation_identity)
                keyed = params.inverse(np.concatenate(observation_keyed)).astype(np.float32)
                diagnostics = {name: np.concatenate(value) for name, value in diagnostics.items()}
                if not np.allclose(diagnostics["transmitted_energy_identity"], diagnostics["transmitted_energy_keyed"], rtol=1e-6, atol=1e-5):
                    raise RuntimeError("The paired physical links do not use matched transmitted energy")
                latent_error_identity = identity - u
                latent_error_keyed = keyed - u
                pred_identity, rows_identity, summary_identity = _metric_record(case["dec"], identity, truth, device)
                pred_keyed, rows_keyed, summary_keyed = _metric_record(case["dec"], keyed, truth, device)
                save_npz(outdir / f"{label}.npz", example_indices=np.arange(min(16, len(u))),
                         example_predictions_identity=pred_identity[:16], example_predictions_keyed=pred_keyed[:16],
                         latent_mse_per_image_identity=np.mean(latent_error_identity**2, axis=1),
                         latent_mse_per_image_keyed=np.mean(latent_error_keyed**2, axis=1),
                         latent_error_covariance_identity=np.cov(latent_error_identity, rowvar=False).astype(np.float32),
                         latent_error_covariance_keyed=np.cov(latent_error_keyed, rowvar=False).astype(np.float32),
                         **{f"identity_{name}": value for name, value in rows_identity.items()},
                         **{f"keyed_{name}": value for name, value in rows_keyed.items()}, **diagnostics)
                diagnostic_summary = {name: {"mean": float(np.mean(value)), "p95": float(np.quantile(value, .95))} for name, value in diagnostics.items()}
                record = {"signature": signature, "scenario": scenario, "snr_db": snr, "draw": draw,
                          "identity": summary_identity, "keyed": summary_keyed,
                          "latent_mse_identity": float(np.mean(latent_error_identity**2)),
                          "latent_mse_keyed": float(np.mean(latent_error_keyed**2)),
                          "diagnostics": diagnostic_summary, "physical_link": accounting,
                          "codec_policy": "frozen AWGN-trained codec; no channel retraining or test-driven selection",
                          "paired_channel_policy": "same channel, preamble/pilot noise, payload noise and equalizer for identity/keyed",
                          "nonce_policy": "unique per image; SNR/draw arms are counterfactual channel evaluations of the same recorded packet",
                          "test_used_for_selection": False}
                write_json(record_path, record)
                results.append(record)
                print(f"Channel {case['name']}: {label}", flush=True)
    kpa = _kpa(case, ccfg, outdir, device, smoke) if ccfg["run_kpa"] else {"status": "disabled_by_configuration"}
    summary = {"signature": signature, "case": case["name"], "channels": results, "known_plaintext": kpa,
               "configuration": ccfg, "smoke": smoke, "test_used_for_selection": False,
               "flat_fading_theory": "conditional on a perfectly known scalar complex gain, inversion restores the same fixed-codec decoder-input law as the corresponding identity link; averaged fading noise is a scale mixture, not fixed-SNR AWGN",
               "nonflat_theory": "frequency-selective equalization generally leaves colored noise and signal distortion, so no general AWGN utility-equivalence claim is made"}
    write_json(final, summary)
    return summary
