import concurrent.futures
import multiprocessing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from keyed_jscc.crypto_scrambler import CryptographicScrambler
from evaluation.channels import (complex_matrix_to_real, defaults, flat_conditional_noise_covariance,
                               pack_complex, transmit_pair, unpack_complex)
from evaluation.protocol import (DurableNonceAllocator, DurableReplayWindow, auth_key,
                               decode_header, ecc_decode, ecc_encode, ecc_soft_decode, encode_header,
                               original_header, parse_original_header,
                               open_header, seal_header, slot_nonce)


def _reserve_worker(arguments):
    path, sender, count = arguments
    return [value.hex() for value in DurableNonceAllocator(path, "test-generation", sender).take(count)]


class ChannelTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(73)
        self.u = self.rng.normal(size=(64, 256)).astype(np.float32)
        self.u /= np.sqrt(np.mean(self.u**2, axis=1, keepdims=True))
        self.params = CryptographicScrambler(bytes(range(32))).parameters_batch(list(range(64)))

    def test_complex_mapping_preserves_physical_energy(self):
        np.testing.assert_allclose(unpack_complex(pack_complex(self.u)), self.u, atol=2e-7)
        np.testing.assert_allclose(np.sum(abs(pack_complex(self.u))**2, axis=1), 128, atol=2e-5)

    def test_flat_conditional_covariance_and_signal_map(self):
                                                                          
        params = CryptographicScrambler(bytes(range(32))).parameters(19)
        inverse = params.inverse(np.eye(256)).T
        signal, covariance = flat_conditional_noise_covariance(.7 + .4j, .7 + .4j, .1, inverse)
        np.testing.assert_allclose(signal, np.eye(256), atol=2e-8)
        np.testing.assert_allclose(covariance, np.eye(256) * .1 / .65, atol=2e-8)
        _, wrong_covariance = flat_conditional_noise_covariance(.7 + .4j, .65 + .3j, .1, inverse)
        self.assertFalse(np.allclose(wrong_covariance, covariance))
                                                                             
        noise = (self.rng.normal(size=(20000, 128)) + 1j * self.rng.normal(size=(20000, 128))) * np.sqrt(.1 / 2)
        samples = params.inverse(unpack_complex(noise / (.7 + .4j)))
        diagonal = np.var(samples, axis=0)
        np.testing.assert_allclose(diagonal, np.diag(covariance), rtol=.04)
        cross = np.cov(samples[:, :8], rowvar=False)
        self.assertLess(np.max(abs(cross - np.diag(np.diag(cross)))), .006)

    def test_complex_real_signal_matrix_mapping(self):
        scalar = .4 + .3j
        real_matrix = complex_matrix_to_real(scalar * np.eye(128))
        np.testing.assert_allclose(self.u @ real_matrix.T, unpack_complex(scalar * pack_complex(self.u)), atol=2e-7)

    def test_noiseless_perfect_flat_recovers_both_arms(self):
        identity, keyed, diag, accounting = transmit_pair(self.u, self.params.forward(self.u), self.rng, 160, "flat_rician_perfect", defaults())
        np.testing.assert_allclose(identity, self.u, atol=1e-6)
        np.testing.assert_allclose(self.params.inverse(keyed), self.u, atol=1e-6)
        np.testing.assert_allclose(diag["transmitted_energy_identity"], diag["transmitted_energy_keyed"], atol=5e-5)
        self.assertEqual(accounting["payload_complex_uses"], 128)

    def test_multipath_equal_resource_and_noise_color_recorded(self):
        identity, keyed, diag, accounting = transmit_pair(self.u, self.params.forward(self.u), self.rng, 20, "multipath_cfo_perfect", defaults())
        self.assertEqual(identity.shape, self.u.shape)
        self.assertTrue(np.isfinite(keyed).all())
        np.testing.assert_allclose(diag["transmitted_energy_identity"], diag["transmitted_energy_keyed"], atol=5e-5)
        self.assertTrue(np.any(diag["equalized_noise_frequency_power_cv"] > .1))
        self.assertGreater(accounting["total_complex_uses"], 128)
        self.assertGreater(accounting["cfo_hz"], 0)
        self.assertTrue(np.all(diag["timing_error_samples"] == 0))

    def test_actual_synchronization_beats_omitted_sync_at_high_snr(self):
        cfg = defaults()
        seed = 1234
        synchronized, _, diag_sync, _ = transmit_pair(self.u, self.params.forward(self.u), np.random.default_rng(seed), 40, "multipath_cfo_estimated", cfg)
        omitted, _, diag_bad, _ = transmit_pair(self.u, self.params.forward(self.u), np.random.default_rng(seed), 40, "multipath_cfo_unsynchronized", cfg)
        self.assertLess(np.mean((synchronized - self.u)**2), np.mean((omitted - self.u)**2))
        self.assertTrue(np.all(diag_bad["timing_error_samples"] == -3))


class ProtocolTests(unittest.TestCase):
    def test_restart_and_burned_reservations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nonce.sqlite"
            first = DurableNonceAllocator(path, "generation", 7).take(3)
            second = DurableNonceAllocator(path, "generation", 7).take(2)
            self.assertEqual(int.from_bytes(second[0][4:], "big"), 3)
            self.assertFalse(set(first) & set(second))
            self.assertNotEqual(DurableNonceAllocator(path, "generation", 8).take()[0], first[0])

    def test_concurrent_process_reservations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "nonce.sqlite")
            DurableNonceAllocator(path, "test-generation", 3)
            context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=3, mp_context=context) as pool:
                reserved = list(pool.map(_reserve_worker, [(path, 3, 9)] * 6))
            values = [value for group in reserved for value in group]
            self.assertEqual(len(values), len(set(values)))
            self.assertEqual(sorted(int(value[8:], 16) for value in values), list(range(54)))

    def test_failure_and_exhaustion_do_not_return_reused_nonce(self):
        with tempfile.TemporaryDirectory() as folder:
            allocator = DurableNonceAllocator(Path(folder) / "nonce.sqlite", "generation", 7, 2**64 - 2)
            with self.assertRaises(OverflowError):
                allocator.take(3)
            values = allocator.take(2)
            self.assertEqual(int.from_bytes(values[0][4:], "big"), 2**64 - 2)
            with self.assertRaises(OverflowError):
                allocator.take()
            with self.assertRaises(ValueError):
                allocator.take(0)

    def test_authenticated_fields_tampering_replay_restart_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            key = auth_key(bytes(range(32)))
            raw = encode_header(7, 100, 8, direction=1)
            sealed = seal_header(raw, key)
            header = open_header(sealed, key)
            self.assertEqual(slot_nonce(header, 3), (7).to_bytes(4, "big") + (103).to_bytes(8, "big"))
            for bit in range(len(sealed) * 8):
                changed = bytearray(sealed)
                changed[bit // 8] ^= 1 << (bit % 8)
                with self.assertRaises(ValueError):
                    open_header(bytes(changed), key)
            path = Path(folder) / "replay.sqlite"
            receiver = DurableReplayWindow(path)
            receiver.accept(sealed, key)
            with self.assertRaises(ValueError):
                DurableReplayWindow(path).accept(sealed, key)
            later = seal_header(encode_header(7, 120, 8, direction=1), key)
            self.assertEqual(DurableReplayWindow(path).accept(later, key)["base"], 120)
            reordered = seal_header(encode_header(7, 112, 8, direction=1), key)
            receiver.accept(reordered, key)
            with self.assertRaises(ValueError):
                receiver.accept(seal_header(encode_header(7, 106, 8, direction=1), key), key)

    def test_auth_failure_does_not_poison_replay_state(self):
        with tempfile.TemporaryDirectory() as folder:
            receiver = DurableReplayWindow(Path(folder) / "replay.sqlite")
            key = auth_key(bytes(range(32)))
            wrong = auth_key(bytes(range(1, 33)))
            packet = seal_header(encode_header(1, 0, 1), key)
            with self.assertRaises(ValueError):
                receiver.accept(packet, wrong)
            receiver.accept(packet, key)

    def test_large_legitimate_counter_jump_uses_bounded_replay_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            key = auth_key(bytes(range(32)))
            receiver = DurableReplayWindow(Path(folder) / "replay.sqlite")
            receiver.accept(seal_header(encode_header(7, 2**63, 8), key), key)
            with self.assertRaises(ValueError):
                receiver.accept(seal_header(encode_header(7, 0, 1), key), key)

    def test_original_header_crc_count_and_soft_decoding(self):
        import binascii
        self.assertEqual(binascii.crc_hqx(b"123456789", 0xffff), 0x29b1)
        raw = original_header((7 << 64) + 1000, 128)
        self.assertEqual(parse_original_header(raw, 128), ((7 << 64) + 1000, 128))
        wave = 1 - 2 * ecc_encode(raw).astype(float)
        for bit in range(len(wave)):
            changed = wave.copy()
            changed[bit] *= -1
            self.assertEqual(ecc_soft_decode(changed), raw)
        damaged = bytearray(raw)
        damaged[0] ^= 1
        with self.assertRaises(ValueError):
            parse_original_header(bytes(damaged), 128)
        with self.assertRaises(ValueError):
            parse_original_header(raw, 8)

    def test_hamming_single_errors_and_double_error_detection(self):
        raw = bytes(range(32))
        bits = ecc_encode(raw)
        self.assertEqual(ecc_decode(bits), raw)
        for index in range(8):
            changed = bits.copy()
            changed[index] ^= 1
            self.assertEqual(ecc_decode(changed), raw)
        changed = bits.copy()
        changed[:2] ^= 1
        with self.assertRaises(ValueError):
            ecc_decode(changed)


if __name__ == "__main__":
    unittest.main()
