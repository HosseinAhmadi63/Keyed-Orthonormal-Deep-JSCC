from __future__ import annotations

import hashlib
import binascii
from contextlib import closing
import hmac
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import struct
import tempfile
import time

import numpy as np
import cryptography
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from keyed_jscc.crypto_scrambler import CryptographicScrambler


AUTH_DOMAIN = b"JSCC-ORTHO-METADATA-AUTH-v1\x00"
HEADER = struct.Struct(">BBBBIQ")
TAG_BYTES = 16


def protocol_defaults():
    return {
        "groups": [1, 8, 32, 128],
        "header_snr_db": [0, 2, 4, 6, 8, 10, 12, 16, 20],
        "header_trials": 20000,
        "benchmark_repeats": 40,
        "benchmark_batches": [1, 128],
        "replay_window": 4096,
    }


class DurableNonceAllocator:
    """Reserve nonce ranges in a durable, process-serialized SQLite transaction.

    The database must reside on persistent local storage and must never be rolled
    back to an older backup while its associated key remains active. Sender IDs
    must be provisioned uniquely among devices sharing a key generation. Each
    distinct key generation requires an explicit namespace. Unused reservations
    are deliberately burned after crashes. Network filesystem semantics are not
    assumed. Secret keys are never stored in this database.
    """

    def __init__(self, path, namespace: str, sender: int, start=0):
        if not namespace or not isinstance(namespace, str):
            raise ValueError("An explicit key-generation namespace is required")
        if not 0 <= int(sender) < 2**32 or not 0 <= int(start) < 2**64:
            raise ValueError("Invalid sender or counter")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace, self.sender = namespace, int(sender)
        with closing(self._connect()) as db:
            db.execute("CREATE TABLE IF NOT EXISTS counters (namespace TEXT, sender INTEGER, next_value TEXT NOT NULL, PRIMARY KEY(namespace,sender))")
            db.execute("INSERT OR IGNORE INTO counters VALUES (?,?,?)", (namespace, self.sender, str(int(start))))

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=60, isolation_level=None)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA journal_mode=DELETE")
        except BaseException:
            db.close()
            raise
        return db

    def take(self, count=1):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("count must be a positive integer")
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            start = int(db.execute("SELECT next_value FROM counters WHERE namespace=? AND sender=?", (self.namespace, self.sender)).fetchone()[0])
            end = start + int(count)
            if end > 2**64:
                raise OverflowError("Sender counter exhausted; provision a fresh key generation")
            db.execute("UPDATE counters SET next_value=? WHERE namespace=? AND sender=?", (str(end), self.namespace, self.sender))
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()
        return [self.sender.to_bytes(4, "big") + value.to_bytes(8, "big") for value in range(start, end)]


def auth_key(master_key: bytes, namespace="experiment-v1"):
    if len(master_key) != 32:
        raise ValueError("The master key must contain 32 bytes")
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=AUTH_DOMAIN + namespace.encode("utf-8")).derive(master_key)


def encode_header(sender, base, count, direction=0, mapping=0, version=1):
    if not 1 <= int(count) <= 128 or int(base) + int(count) > 2**64:
        raise ValueError("Invalid group counter range")
    if not 0 <= int(sender) < 2**32 or not 0 <= int(base) < 2**64:
        raise ValueError("Invalid sender or counter")
    if version != 1 or direction not in (0, 1) or mapping != 0:
        raise ValueError("Unsupported version, direction or slot mapping")
    return HEADER.pack(version, direction, int(count), mapping, int(sender), int(base))


def decode_header(raw):
    if len(raw) != HEADER.size:
        raise ValueError("Invalid header length")
    version, direction, count, mapping, sender, base = HEADER.unpack(raw)
    encode_header(sender, base, count, direction, mapping, version)
    return {"version": version, "direction": direction, "count": count,
            "mapping": mapping, "sender": sender, "base": base}


def seal_header(raw, key):
    decode_header(raw)
    return raw + hmac.new(key, AUTH_DOMAIN + raw, hashlib.sha256).digest()[:TAG_BYTES]


def open_header(packet, key):
    if len(packet) != HEADER.size + TAG_BYTES:
        raise ValueError("Invalid authenticated header length")
    raw, tag = packet[:HEADER.size], packet[HEADER.size:]
    expected = hmac.new(key, AUTH_DOMAIN + raw, hashlib.sha256).digest()[:TAG_BYTES]
    if not hmac.compare_digest(tag, expected):
        raise ValueError("Metadata authentication failed")
    return decode_header(raw)


class DurableReplayWindow:
    """Durable receiver replay state; authentication must precede state updates."""

    def __init__(self, path, namespace="experiment-v1", width=4096):
        if int(width) < 128:
            raise ValueError("Replay window must cover at least one full group")
        self.path, self.namespace, self.width = Path(path), namespace, int(width)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.execute("CREATE TABLE IF NOT EXISTS replay (namespace TEXT, sender INTEGER, direction INTEGER, highest TEXT, bitmap TEXT, PRIMARY KEY(namespace,sender,direction))")

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=60, isolation_level=None)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA journal_mode=DELETE")
        except BaseException:
            db.close()
            raise
        return db

    def accept(self, packet, key):
        header = open_header(packet, key)
        identity = (self.namespace, header["sender"], header["direction"])
        values = range(header["base"], header["base"] + header["count"])
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT highest,bitmap FROM replay WHERE namespace=? AND sender=? AND direction=?", identity).fetchone()
            highest, bitmap = (-1, 0) if row is None else (int(row[0]), int(row[1]))
            new_highest = max(highest, values.stop - 1)
            advance = new_highest - highest
            shifted = 0 if advance >= self.width else (bitmap << advance) & ((1 << self.width) - 1)
            for value in values:
                distance = new_highest - value
                if distance >= self.width or shifted & (1 << distance):
                    raise ValueError("Replayed or expired nonce range")
                shifted |= 1 << distance
            db.execute("INSERT OR REPLACE INTO replay VALUES (?,?,?,?,?)", (*identity, str(new_highest), str(shifted)))
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()
        return header


def slot_nonce(header, slot):
    if not 0 <= int(slot) < header["count"]:
        raise ValueError("Slot falls outside its authenticated group")
    return int(header["sender"]).to_bytes(4, "big") + (int(header["base"]) + int(slot)).to_bytes(8, "big")



def original_header(base_nonce, count):
    if not 0 <= int(base_nonce) < 2**96 or not 1 <= int(count) <= 65535 or int(base_nonce) + int(count) > 2**96:
        raise ValueError("Invalid original-header nonce range")
    body = int(base_nonce).to_bytes(12, "big") + int(count).to_bytes(2, "big")
    return body + binascii.crc_hqx(body, 0xffff).to_bytes(2, "big")


def parse_original_header(raw, expected_count=None):
    if len(raw) != 16:
        raise ValueError("Invalid original-header length")
    nonce, count = int.from_bytes(raw[:12], "big"), int.from_bytes(raw[12:14], "big")
    if binascii.crc_hqx(raw[:14], 0xffff).to_bytes(2, "big") != raw[14:]:
        raise ValueError("CRC16 mismatch")
    original_header(nonce, count)
    if expected_count is not None and count != expected_count:
        raise ValueError("Group-count mismatch")
    return nonce, count


def _hamming_codebook():
    words = []
    for value in range(16):
        d = np.array([(value >> k) & 1 for k in range(3, -1, -1)], dtype=np.uint8)
        c = np.zeros(8, dtype=np.uint8)
        c[[2, 4, 5, 6]] = d
        c[0] = c[2] ^ c[4] ^ c[6]
        c[1] = c[2] ^ c[5] ^ c[6]
        c[3] = c[4] ^ c[5] ^ c[6]
        c[7] = np.bitwise_xor.reduce(c[:7])
        words.append(c)
    return np.stack(words)


CODEBOOK = _hamming_codebook()


def ecc_encode(raw):
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(-1, 4)
    index = bits @ np.array([8, 4, 2, 1], dtype=np.uint8)
    return CODEBOOK[index].reshape(-1)


def ecc_decode(bits):
    words = np.asarray(bits, dtype=np.uint8).reshape(-1, 8)
    distance = (words[:, None, :] != CODEBOOK[None, :, :]).sum(2)
    best = distance.argmin(1)
    if np.any(distance[np.arange(len(words)), best] > 1):
        raise ValueError("Uncorrectable SECDED word")
    decoded = ((best[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.uint8)
    return np.packbits(decoded.reshape(-1)).tobytes()



def ecc_soft_decode(received):
    values = np.asarray(received, dtype=np.float64)
    single = values.ndim == 1
    values = values.reshape(-1, values.shape[-1] // 8, 8)
    sign = 1 - 2 * CODEBOOK.astype(np.float64)
    indices = (values @ sign.T).argmax(-1)
    nibble_bits = ((indices[..., None] >> np.arange(3, -1, -1)) & 1).astype(np.uint8)
    decoded = np.packbits(nibble_bits.reshape(len(values), -1), axis=1)
    result = [row.tobytes() for row in decoded]
    return result[0] if single else result


def _timings(function, repeats):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"median_ms_per_batch": float(np.median(samples)), "p95_ms_per_batch": float(np.quantile(samples, .95)), "repeats": repeats}


def _header_mc(rng, snr_db, trials, batch=256):
    key = auth_key(bytes(range(32)))
    protected = seal_header(encode_header(7, 1000, 128), key)
    plain = original_header((7 << 64) + 1000, 128)
    out = []
    for label, raw in (("hamming_nonce96_count16_crc16", plain), ("hamming_authenticated_256bit_information", protected)):
        encoded = ecc_encode(raw)
        errors = corrected = accepted_wrong = 0
        sigma = math.sqrt(10 ** (-snr_db / 10))
        for start in range(0, trials, batch):
            count = min(batch, trials - start)
            observed = (1 - 2 * encoded.astype(float))[None, :] + rng.normal(0, sigma, (count, len(encoded)))
            received = ecc_soft_decode(observed)
            for restored in received:
                try:
                    if label.startswith("hamming_authenticated"):
                        open_header(restored, key)
                    else:
                        parse_original_header(restored, 128)
                    if restored != raw:
                        accepted_wrong += 1
                        errors += 1
                    else:
                        corrected += 1
                except ValueError:
                    errors += 1
        rate = errors / trials
        z = 1.959963984540054
        denominator = 1 + z*z/trials
        center = (rate + z*z/(2*trials))/denominator
        radius = z*math.sqrt(rate*(1-rate)/trials + z*z/(4*trials*trials))/denominator
        out.append({"scheme": label, "snr_db_complex_symbol": float(snr_db), "trials": trials, "actual_group_count": 128,
                    "failure_wilson95": [max(0.,center-radius),min(1.,center+radius)],
                    "zero_error_one_sided95_upper": 1 - .05**(1/trials) if errors == 0 else None,
                    "failed_or_wrong_headers": errors, "failure_rate": errors / trials,
                    "accepted_wrong_headers": accepted_wrong, "correct_headers": corrected,
                    "coded_bits": len(encoded), "qpsk_complex_uses": len(encoded) // 2,
                    "decoder": "exhaustive soft maximum-likelihood extended-Hamming word decoding followed by CRC16 or MAC verification",
                    "noise_model": "independent real components of unit-energy QPSK; Eb not held fixed"})
    return out


def run_protocol(cfg, outdir, smoke=False):
    from .common import fit_signature, write_json
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pcfg = {**protocol_defaults(), **cfg.get("protocol", {})}
    signature = fit_signature(cfg, {"stage": "protocol", "version": 3, "smoke": smoke})
    final = outdir / "summary.json"
    if final.exists():
        cached = json.loads(final.read_text())
        if cached.get("signature") == signature:
            return cached
        raise RuntimeError("Protocol output already belongs to a different configuration")
    tests = {}
    key = auth_key(bytes(range(32)))
    with tempfile.TemporaryDirectory(prefix="jscc_protocol_") as temporary:
        folder = Path(temporary)
        allocator = DurableNonceAllocator(folder / "nonces.sqlite", "public-reproduction-key-v1", 7)
        first = allocator.take(8)
        restart = DurableNonceAllocator(folder / "nonces.sqlite", "public-reproduction-key-v1", 7)
        later = restart.take(8)
        tests["restart_reservations_disjoint"] = not set(first) & set(later)
        tests["unused_reservation_burned"] = int.from_bytes(later[0][4:], "big") == 8
        replay = DurableReplayWindow(folder / "replay.sqlite", width=pcfg["replay_window"])
        packet = seal_header(encode_header(7, 16, 8), key)
        header = replay.accept(packet, key)
        tests["per_slot_nonce_matches_allocator_policy"] = slot_nonce(header, 3) == (7).to_bytes(4, "big") + (19).to_bytes(8, "big")
        try:
            replay.accept(packet, key)
            tests["replay_rejected"] = False
        except ValueError:
            tests["replay_rejected"] = True
        again = DurableReplayWindow(folder / "replay.sqlite", width=pcfg["replay_window"])
        try:
            again.accept(packet, key)
            tests["receiver_restart_replay_rejected"] = False
        except ValueError:
            tests["receiver_restart_replay_rejected"] = True
        tamper_rejected = 0
        for bit in range(len(packet) * 8):
            changed = bytearray(packet)
            changed[bit // 8] ^= 1 << (bit % 8)
            try:
                open_header(bytes(changed), key)
            except ValueError:
                tamper_rejected += 1
        tests["single_bit_tampering_rejected"] = tamper_rejected == len(packet) * 8
        tests["tamper_trials"] = len(packet) * 8
        packet_later = seal_header(encode_header(7, 32, 8), key)
        tests["dropped_group_does_not_desynchronize_later_group"] = again.accept(packet_later, key)["base"] == 32
        tests["sender_namespaces_disjoint"] = DurableNonceAllocator(folder / "nonces.sqlite", "public-reproduction-key-v1", 8).take()[0] != first[0]
        limit = DurableNonceAllocator(folder / "nonces.sqlite", "exhaustion-test", 9, 2**64 - 1)
        limit.take()
        try:
            limit.take()
            tests["counter_exhaustion_rejected"] = False
        except OverflowError:
            tests["counter_exhaustion_rejected"] = True
        forged_crc_header = original_header((7 << 64) + 9999, 128)
        tests["crc_is_forgeable_error_detection_only"] = parse_original_header(forged_crc_header, 128)[0] == (7 << 64) + 9999
        rng = np.random.default_rng(2026102601)
        benchmark = []
        repeats = 3 if smoke else int(pcfg["benchmark_repeats"])
        for batch in ([1, 4] if smoke else pcfg["benchmark_batches"]):
            u = rng.normal(size=(batch, 256)).astype(np.float32)
            scrambler = CryptographicScrambler(bytes(range(32)))
            nonces = allocator.take(batch)
            params = scrambler.parameters_batch(nonces)
            transformed = params.forward(u)
            raw = encode_header(7, 10000, min(batch, 128))
            jobs = {
                "parameter_generation_sender": lambda: scrambler.parameters_batch(nonces),
                "parameter_regeneration_receiver": lambda: scrambler.parameters_batch(nonces),
                "forward_transform": lambda: params.forward(u),
                "inverse_transform": lambda: params.inverse(transformed),
                "mac_generate": lambda: [seal_header(raw, key) for _ in range(batch)],
                "mac_verify": lambda: [open_header(seal_header(raw, key), key) for _ in range(batch)],
                "durable_nonce_reservation": lambda: allocator.take(batch),
            }
            protected_raw = seal_header(raw, key)
            jobs["mac_verify"] = lambda: [open_header(protected_raw, key) for _ in range(batch)]
            receiver = DurableReplayWindow(folder / f"benchmark-replay-{batch}.sqlite")
            replay_packets = [seal_header(encode_header(7, 1000000 + i, 1), key) for i in range((repeats + 1) * batch)]
            replay_position = [0]
            def receive_batch():
                begin = replay_position[0]
                replay_position[0] += batch
                return [receiver.accept(packet, key) for packet in replay_packets[begin:begin + batch]]
            coded_header = ecc_encode(protected_raw)
            header_wave = 1 - 2 * coded_header.astype(float)
            jobs["receiver_auth_and_durable_replay"] = receive_batch
            jobs["hamming_encode"] = lambda: [ecc_encode(protected_raw) for _ in range(batch)]
            jobs["hamming_soft_decode"] = lambda: ecc_soft_decode(np.broadcast_to(header_wave, (batch, len(header_wave))))
            for component, job in jobs.items():
                measured = _timings(job, repeats)
                measured.update({"batch": batch, "component": component,
                                 "median_ms_per_image": measured["median_ms_per_batch"] / batch,
                                 "p95_ms_per_image": measured["p95_ms_per_batch"] / batch,
                                 "unit_note": "MAC jobs perform one header per loop item; group-header per-image cost divides this by group size; nonce reservation performs one durable transaction per batch"})
                benchmark.append(measured)
    header_mc = []
    trials = 32 if smoke else int(pcfg["header_trials"])
    for snr in ([0, 10] if smoke else pcfg["header_snr_db"]):
        header_mc.extend(_header_mc(np.random.default_rng(2026102602 + int(10 * snr)), snr, trials))
    accounting = []
    for group in pcfg["groups"]:
        for auth in [False, True]:
            information = 128 + (128 if auth else 0)
            uses = information
            accounting.append({"group_size": group, "authenticated_metadata": auth,
                               "information_bits_per_header": information, "hamming_coded_bits": 2 * information,
                               "qpsk_header_uses": uses, "payload_uses_per_image": 128,
                               "total_complex_uses_per_image": 128 + uses / group,
                               "header_overhead_fraction": uses / (128 * group),
                               "frame_384_tiles_complex_uses": 384 * (128 + uses / group),
                               "preamble_pilots_and_payload_integrity_included": False,
                               "header_layout": "version8,direction8,count8,mapping8,sender32,base64,tag128" if auth else "nonce96,count16,CRC16 (original error-detection reference)",
                               "interpretation": "derived accounting; authenticated metadata does not authenticate continuous payload"})
    derived_latency = []
    for batch in ([1, 4] if smoke else pcfg["benchmark_batches"]):
        rows = {r["component"]: r for r in benchmark if r["batch"] == batch}
        for group in pcfg["groups"]:
            per_image = sum(rows[name]["median_ms_per_image"] for name in ("parameter_generation_sender", "parameter_regeneration_receiver", "forward_transform", "inverse_transform", "durable_nonce_reservation"))
            per_image += sum(rows[name]["median_ms_per_image"] for name in ("mac_generate", "receiver_auth_and_durable_replay", "hamming_encode", "hamming_soft_decode")) / group
            derived_latency.append({"measurement_batch": batch, "group_size": group, "derived_extra_processing_ms_per_tile": per_image, "derived_extra_processing_ms_for_384_tiles": per_image * 384, "scope": "sum of component medians, not a measured end-to-end latency; includes authenticated-header FEC and durable replay handling; excludes codec, radio, queueing and retransmissions"})
    write_json(outdir / "metadata_validation_vector.json", {"public_reproduction_master_key_hex": bytes(range(32)).hex(), "namespace": "experiment-v1", "raw_header_hex": encode_header(7, 16, 8).hex(), "authenticated_header_hex": seal_header(encode_header(7, 16, 8), key).hex(), "operational_use": False})
    result = {
        "signature": signature, "test_used_for_selection": False, "protocol_tests": tests,
        "header_monte_carlo": header_mc, "header_accounting": accounting, "cpu_benchmarks": benchmark, "derived_processing_accounting": derived_latency,
        "hardware": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
                     "python": platform.python_version(), "numpy": np.__version__, "sqlite": sqlite3.sqlite_version, "cryptography": cryptography.__version__, "logical_cpu_count": os.cpu_count()},
        "protocol_definition": {"nonce": "sender32 || counter64", "header_128_bits": "version8,direction8,count8,mapping8,sender32,base64",
                                "slot_mapping": "mapping=0: nonce(sender,base+slot), slot in [0,count)",
                                "authentication": "domain-separated HKDF-SHA256 key; HMAC-SHA256 truncated to 128 bits",
                                "layout_relation_to_original": "the original 128-bit nonce96+count16+CRC16 error-detection header is compared with a 128-bit structured metadata header plus 128-bit MAC; CRC16 is replaced by keyed authentication, not counted twice",
                                "replay": "authenticate before durable sliding-window update; explicit group/slot counters",
                                "crash_rule": "reserve all group counters durably before transmitting any slot; burn unused reservations",
                                "assumptions": ["persistent local SQLite storage honors fsync", "no rollback of nonce/replay state under an active key", "unique sender prefix per shared-key device", "fresh key-generation namespace on key rotation", "packet framing identifies authenticated header and slot index"],
                                "scope": "metadata origin/integrity and group replay rejection; no continuous-payload authentication or protection against deletion/jamming"},
        "timing_scope": "CPU microbenchmarks, warm process; includes durable reserve per batch and authenticated receiver replay updates; excludes codec, radio, queuing and link-level retransmissions",
        "smoke": smoke,
    }
    boolean_tests = [value for value in tests.values() if isinstance(value, bool)]
    if not all(boolean_tests):
        raise RuntimeError("A protocol control test failed")
    write_json(final, result)
    return result
