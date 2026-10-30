import numpy as np
import pytest
import torch

from evaluation import baselines


@pytest.mark.parametrize("order", [4, 16, 64, 4096])
def test_gray_qam_exact_noiseless_roundtrip(order):
    rng = np.random.default_rng(order)
    width = int(np.log2(order))
    bits = rng.integers(0, 2, (7, 200 * width), dtype=np.uint8)
    modulated = baselines.qam_modulate(bits, order)
    demodulated = baselines.qam_demodulate(modulated, order)
    assert np.array_equal(demodulated, bits)
    assert np.array_equal(baselines.qam_channel(bits, order, float("inf"), rng), bits)


@pytest.mark.parametrize("bits", [4, 8])
def test_quantizer_and_packing(bits):
    rng = np.random.default_rng(9)
    train = rng.normal(size=(100, 256))
    quantizer = baselines.LatentQuantizer.fit(train, bits, [.1, 99.9])
    encoded = quantizer.encode(train)
    packed = baselines.pack_quantized(encoded, bits)
    assert packed.shape[1] == 256 * bits // 8
    assert np.array_equal(baselines.unpack_quantized(packed, bits), encoded)
    decoded = quantizer.decode(encoded)
    assert decoded.shape == train.shape
    assert np.mean((decoded - train) ** 2) < (0.03 if bits == 4 else .005)


def test_aes_requires_valid_nonce_payload_and_tag():
    rng = np.random.default_rng(41)
    symbols = rng.integers(0, 16, (5, 256), dtype=np.uint8)
    key = bytes(range(32))
    packets = baselines.aes_encrypt(symbols, 4, key, "unit_fit")
    assert packets.shape == (5, 12 + 128 + 16)
    decoded, accepted = baselines.aes_decrypt(packets, 4, key)
    assert accepted.all()
    assert np.array_equal(symbols, decoded)
    assert len({packet[:12].tobytes() for packet in packets}) == len(packets)
    alternate = baselines.aes_encrypt(symbols, 4, key, "unit_validation")
    assert not ({packet[:12].tobytes() for packet in packets} & {packet[:12].tobytes() for packet in alternate})
    for column in (0, 12, packets.shape[1] - 1):
        corrupted = packets.copy()
        corrupted[0, column] ^= 1
        _, accepted = baselines.aes_decrypt(corrupted, 4, key)
        assert not accepted[0]
        assert accepted[1:].all()


def test_linear_bfv_fresh_per_packet_and_full_public_side_information():
    bfv = baselines.LinearBFV(7)
    rng = np.random.default_rng(12)
    plaintext = rng.integers(0, 16, (100, 256), dtype=np.uint8)
    ciphertext = bfv.encrypt(plaintext, rng)
    assert ciphertext.shape == (100, 448)
    assert ciphertext.dtype == np.int64
    assert np.array_equal(bfv.decrypt(ciphertext), plaintext)
    bits = baselines._bfv_bits(ciphertext)
    assert bits.shape == (100, 448 * 12)
    assert np.array_equal(baselines._bits_bfv(bits, bfv.q), ciphertext)
    repeated = np.repeat(plaintext[:1], 3, axis=0)
    encrypted = bfv.encrypt(repeated, rng)
    assert len({row.tobytes() for row in encrypted}) == 3
    with pytest.raises(ValueError, match="transmitted c1"):
        bfv.decrypt(ciphertext[:, :256])


def test_noiseless_headers_and_group_failure_expansion():
    assert baselines._header_success(17, 8, 128, 2, float("inf"), 1., np.random.default_rng(2)).all()
    flags = baselines._header_success(17, 8, 256, 2, -30, 1., np.random.default_rng(2))
    assert not flags.any()
    assert flags.shape == (17,)


def test_baseline_smoke_runs_metric_aligned_fits_and_equal_energy(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(29)
    train_u, test_u = rng.normal(size=(60, 256)).astype(np.float32), rng.normal(size=(18, 256)).astype(np.float32)
    decoder = torch.nn.Sequential(torch.nn.Linear(256, 3072), torch.nn.Sigmoid(), torch.nn.Unflatten(1, (3, 32, 32))).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    case = {"name": "baseline_unit", "train_u": train_u, "test_u": test_u, "train_images": rng.integers(0, 256, (60, 3, 32, 32), dtype=np.uint8), "test_images": rng.integers(0, 256, (18, 3, 32, 32), dtype=np.uint8), "train_labels": np.arange(60) % 10, "test_labels": np.arange(18) % 10, "fit": np.arange(40), "val": np.arange(40, 60), "dec": decoder, "key_hex": bytes(range(32)).hex()}
    result = baselines.run(case, {}, tmp_path, "cpu", smoke=True)
    assert result["status"] == "smoke_not_scientific"
    assert result["test_used_for_selection"] is False
    assert len(result["attacks"]) == 10
    assert all(row["total_transmit_energy_per_image"] == 640 for row in result["operating_points"])
    assert all(row["active_complex_uses_per_image"] <= 640 for row in result["operating_points"])
    assert all("eve_class_accuracy" in row and "eve_ssim_valid" in row for row in result["operating_points"])
    assert all(row["resource_attack_fitting"] == "fresh_at_effective_snr" for row in result["operating_points"])
    assert result["resource_attack_selections"]
    assert all(record["test_used_for_selection"] is False and record["checkpoint_sha256"] for record in result["resource_attack_selections"].values())
    assert all(record["selection_metric"] == ("accuracy" if "classification" in name else "ssim_valid") for name, record in result["attacks"].items())
    assert baselines.run(case, {}, tmp_path, "cpu", smoke=True) == result
