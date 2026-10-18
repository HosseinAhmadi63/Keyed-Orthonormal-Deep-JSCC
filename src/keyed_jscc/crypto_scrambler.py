from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from numbers import Integral
from pathlib import Path
import secrets
import struct
from typing import Iterable, Sequence
import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
PROTOCOL = 'JSCC-ORTHO-HKDF-AESCTR-v1'
DOMAIN = b'JSCC-ORTHO-HKDF-AESCTR-v1\x00'
LABELS = (b'sign1', b'sign2', b'permutation')
KEY_BYTES = 32
NONCE_BYTES = 12

def encode_nonce(nonce: bytes | bytearray | memoryview | int) -> bytes:
    if isinstance(nonce, Integral) and (not isinstance(nonce, bool)):
        value = int(nonce)
        if not 0 <= value < 1 << 96:
            raise ValueError('nonce integer must be in [0, 2**96)')
        return value.to_bytes(NONCE_BYTES, 'big')
    if isinstance(nonce, (bytes, bytearray, memoryview)):
        value = bytes(nonce)
        if len(value) != NONCE_BYTES:
            raise ValueError('nonce must contain exactly 12 bytes')
        return value
    raise TypeError('nonce must be a 96-bit nonnegative integer or 12 bytes')

def generate_key() -> bytes:
    return secrets.token_bytes(KEY_BYTES)

class NonceCounter:

    def __init__(self, start: int=0):
        encode_nonce(start)
        self.next_value = int(start)

    def take(self, count: int=1) -> list[bytes]:
        if not isinstance(count, Integral) or isinstance(count, bool) or count < 0:
            raise ValueError('count must be a nonnegative integer')
        count = int(count)
        end = self.next_value + count
        if end > 1 << 96:
            raise OverflowError('96-bit nonce counter exhausted; rotate the key')
        result = [value.to_bytes(12, 'big') for value in range(self.next_value, end)]
        self.next_value = end
        return result

def nonce_features(nonces: Iterable[bytes | int]) -> np.ndarray:
    raw = b''.join((encode_nonce(value) for value in nonces))
    return np.frombuffer(raw, dtype='>u2').reshape(-1, 6).astype(np.float32) / np.float32(65535.0)

class _AESStream:

    def __init__(self, material: bytes, buffer_bytes: int=4096):
        if len(material) != 48:
            raise ValueError('AES stream material must be 48 bytes')
        self._encryptor = Cipher(algorithms.AES(material[:32]), modes.CTR(material[32:])).encryptor()
        self._buffer = b''
        self._offset = 0
        self._buffer_bytes = buffer_bytes
        self._blocks_generated = 0
        self._initial_counter = int.from_bytes(material[32:], 'big')

    def read(self, size: int) -> bytes:
        if size < 0:
            raise ValueError('stream read size cannot be negative')
        if size == 0:
            return b''
        if len(self._buffer) - self._offset < size:
            remainder = self._buffer[self._offset:]
            needed = max(self._buffer_bytes, size - len(remainder))
            needed = (needed + 15) // 16 * 16
            new_blocks = needed // 16
            if self._initial_counter + self._blocks_generated + new_blocks > 1 << 128:
                raise OverflowError('AES-CTR counter would wrap')
            self._buffer = remainder + self._encryptor.update(bytes(needed))
            self._blocks_generated += new_blocks
            self._offset = 0
        result = self._buffer[self._offset:self._offset + size]
        self._offset += size
        return result

def _uniform_below(stream: _AESStream, bound: int) -> int:
    if not 1 <= bound <= 1 << 32:
        raise ValueError('bound must be between 1 and 2**32')
    limit = (1 << 32) - (1 << 32) % bound
    while True:
        candidate = int.from_bytes(stream.read(4), 'big')
        if candidate < limit:
            return candidate % bound

def fwht_normalized(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 1:
        raise ValueError('FWHT input must have at least one dimension')
    dimension = array.shape[-1]
    if dimension < 1 or dimension & dimension - 1:
        raise ValueError('last dimension must be a positive power of two')
    if np.iscomplexobj(array):
        raise ValueError('this real JSCC protocol does not accept complex symbols')
    dtype = array.dtype if array.dtype in (np.float32, np.float64) else np.float64
    result = np.array(array, dtype=dtype, copy=True)
    original_shape = result.shape
    flat = result.reshape(-1, dimension)
    stride = 1
    while stride < dimension:
        pairs = flat.reshape(-1, dimension // (2 * stride), 2, stride)
        left = pairs[:, :, 0, :].copy()
        right = pairs[:, :, 1, :].copy()
        pairs[:, :, 0, :] = left + right
        pairs[:, :, 1, :] = left - right
        stride *= 2
    result /= np.asarray(math.sqrt(dimension), dtype=dtype)
    return result.reshape(original_shape)

@dataclass(frozen=True)
class TransformParameters:
    sign1: np.ndarray
    sign2: np.ndarray
    perm: np.ndarray
    inv_perm: np.ndarray

    def forward(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim < 1 or array.shape[-1] != self.sign1.shape[-1]:
            raise ValueError('input last dimension does not match transform')
        result = fwht_normalized(array * self.sign1) * self.sign2
        if self.perm.ndim == 1:
            return result[..., self.perm]
        if result.shape != self.perm.shape:
            raise ValueError('batched parameters require a matching BxN input')
        return np.take_along_axis(result, self.perm, axis=1)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim < 1 or array.shape[-1] != self.sign1.shape[-1]:
            raise ValueError('input last dimension does not match transform')
        if self.inv_perm.ndim == 1:
            aligned = array[..., self.inv_perm]
        else:
            if array.shape != self.inv_perm.shape:
                raise ValueError('batched parameters require a matching BxN input')
            aligned = np.take_along_axis(array, self.inv_perm, axis=1)
        result = fwht_normalized(aligned * self.sign2)
        return result * self.sign1

class CryptographicScrambler:

    def __init__(self, master_key: bytes, dimension: int=256):
        if not isinstance(master_key, bytes) or len(master_key) != KEY_BYTES:
            raise ValueError('master_key must contain exactly 32 bytes')
        if not isinstance(dimension, Integral) or isinstance(dimension, bool) or dimension < 1 or int(dimension) & int(dimension) - 1 or (dimension >= 1 << 32):
            raise ValueError('dimension must be a positive power of two below 2**32')
        self._master_key = master_key
        self.dimension = int(dimension)

    def _derive(self, nonce: bytes | int, label: bytes) -> bytes:
        if label not in LABELS:
            raise ValueError('unknown protocol stream label')
        info = DOMAIN + struct.pack('>I', self.dimension) + label
        return HKDF(algorithm=hashes.SHA256(), length=48, salt=encode_nonce(nonce), info=info).derive(self._master_key)

    def parameters(self, nonce: bytes | int) -> TransformParameters:
        nonce_bytes = encode_nonce(nonce)
        dimension = self.dimension
        signs = []
        for label in LABELS[:2]:
            stream = _AESStream(self._derive(nonce_bytes, label), buffer_bytes=32)
            raw = np.frombuffer(stream.read((dimension + 7) // 8), dtype=np.uint8)
            bits = np.unpackbits(raw, bitorder='little')[:dimension]
            signs.append(1.0 - 2.0 * bits.astype(np.float32))
        stream = _AESStream(self._derive(nonce_bytes, LABELS[2]))
        permutation = np.arange(dimension, dtype=np.int64)
        for high in range(dimension - 1, 0, -1):
            low = _uniform_below(stream, high + 1)
            permutation[high], permutation[low] = (permutation[low], permutation[high])
        inverse = np.empty(dimension, dtype=np.int64)
        inverse[permutation] = np.arange(dimension, dtype=np.int64)
        return TransformParameters(signs[0], signs[1], permutation, inverse)

    def parameters_batch(self, nonces: Sequence[bytes | int]) -> TransformParameters:
        params = [self.parameters(nonce) for nonce in nonces]
        if not params:
            return TransformParameters(np.empty((0, self.dimension), dtype=np.float32), np.empty((0, self.dimension), dtype=np.float32), np.empty((0, self.dimension), dtype=np.int64), np.empty((0, self.dimension), dtype=np.int64))
        return TransformParameters(np.stack([value.sign1 for value in params]), np.stack([value.sign2 for value in params]), np.stack([value.perm for value in params]), np.stack([value.inv_perm for value in params]))

    def scramble(self, values: np.ndarray, nonce: bytes | int) -> np.ndarray:
        return self.parameters(nonce).forward(values)

    def descramble(self, values: np.ndarray, nonce: bytes | int) -> np.ndarray:
        return self.parameters(nonce).inverse(values)

    def scramble_batch(self, values: np.ndarray, nonces: Sequence[bytes | int]) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 2 or array.shape != (len(nonces), self.dimension):
            raise ValueError('batch input must have shape (number of nonces, N)')
        params = self.parameters_batch(nonces)
        result = fwht_normalized(array * params.sign1) * params.sign2
        return np.take_along_axis(result, params.perm, axis=1)

    def descramble_batch(self, values: np.ndarray, nonces: Sequence[bytes | int]) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 2 or array.shape != (len(nonces), self.dimension):
            raise ValueError('batch input must have shape (number of nonces, N)')
        params = self.parameters_batch(nonces)
        result = np.take_along_axis(array, params.inv_perm, axis=1)
        return fwht_normalized(result * params.sign2) * params.sign1
    forward = scramble
    inverse = descramble
CryptoScrambler = CryptographicScrambler

def _stdlib_hkdf(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    output, previous = (b'', b'')
    for counter in range(1, (length + 31) // 32 + 1):
        previous = hmac.new(prk, previous + info + bytes([counter]), hashlib.sha256).digest()
        output += previous
    return output[:length]

def run_self_tests() -> dict:
    actual = HKDF(algorithm=hashes.SHA256(), length=42, salt=bytes.fromhex('000102030405060708090a0b0c'), info=bytes.fromhex('f0f1f2f3f4f5f6f7f8f9')).derive(b'\x0b' * 22)
    expected = bytes.fromhex('3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865')
    assert actual == expected, 'RFC 5869 HKDF known-answer test failed'
    aes_key = bytes.fromhex('603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4')
    ctr = bytes.fromhex('f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff')
    plaintext = bytes.fromhex('6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e5130c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710')
    encryptor = Cipher(algorithms.AES(aes_key), modes.CTR(ctr)).encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    assert ciphertext.hex() == '601ec313775789a5b7a7f504bbf3d228f443e3ca4d62b59aca84e990cacaf5c52b0930daa23de94ce87017ba2d84988ddfc9c58db67aada613c2dd08457941a6'
    key = bytes(range(32))
    scrambler = CryptographicScrambler(key, 256)
    nonce = bytes(range(12))
    for label in LABELS:
        expected_material = _stdlib_hkdf(key, nonce, DOMAIN + struct.pack('>I', 256) + label, 48)
        assert scrambler._derive(nonce, label) == expected_material
    materials = [scrambler._derive(nonce, label) for label in LABELS]
    assert len(set(materials)) == 3
    rng = np.random.default_rng(1729)
    maximum_errors = {}
    for dtype in (np.float32, np.float64):
        inputs = rng.normal(size=(12, 256)).astype(dtype)
        original = inputs.copy()
        nonces = NonceCounter(2 ** 70).take(inputs.shape[0])
        scrambled = scrambler.scramble_batch(inputs, nonces)
        restored = scrambler.descramble_batch(scrambled, nonces)
        tolerance = 2e-06 if dtype == np.float32 else 2e-14
        assert np.array_equal(inputs, original), 'input was modified'
        assert np.allclose(restored, inputs, rtol=tolerance, atol=tolerance)
        assert np.allclose(np.linalg.norm(scrambled, axis=1), np.linalg.norm(inputs, axis=1), rtol=tolerance, atol=tolerance)
        maximum_errors[np.dtype(dtype).name] = float(np.max(np.abs(restored - inputs)))
    params = scrambler.parameters(nonce)
    assert np.array_equal(np.sort(params.perm), np.arange(256))
    assert np.array_equal(params.inv_perm[params.perm], np.arange(256))
    assert np.array_equal(params.sign1, scrambler.parameters(nonce).sign1)
    assert np.array_equal(params.perm, scrambler.parameters(nonce).perm)
    different = scrambler.parameters((int.from_bytes(nonce, 'big') + 1).to_bytes(12, 'big'))
    assert not np.array_equal(params.perm, different.perm)
    matrix_rows = params.forward(np.eye(256))
    assert np.allclose(matrix_rows @ matrix_rows.T, np.eye(256), atol=1e-14)

    class ScriptedStream:

        def __init__(self):
            self.words = iter([4294967295, 7])
            self.calls = 0

        def read(self, size):
            assert size == 4
            self.calls += 1
            return next(self.words).to_bytes(4, 'big')
    scripted = ScriptedStream()
    assert _uniform_below(scripted, 3) == 1 and scripted.calls == 2
    counter = NonceCounter((1 << 96) - 1)
    assert counter.take(1) == [b'\xff' * 12]
    try:
        counter.take(1)
        raise AssertionError('counter exhaustion was not detected')
    except OverflowError:
        pass
    first = np.zeros(256)
    first[0] = 16.0
    second = np.zeros(256)
    second[:2] = math.sqrt(128.0)
    fourth_a = float(np.sum(params.forward(first) ** 4))
    fourth_b = float(np.sum(params.forward(second) ** 4))
    assert np.isclose(fourth_a, 256.0)
    assert np.isclose(fourth_b, 512.0)
    return {'protocol': PROTOCOL, 'passed': True, 'primitive_tests': ['RFC5869-A.1', 'NIST-SP800-38A-F.5.5-four-blocks'], 'inversion_max_absolute_error': maximum_errors, 'fourth_moment_limitation': {'basis_latent': fourth_a, 'two_basis_latent': fourth_b}}

def build_test_vectors() -> dict:
    key = bytes(range(32))
    nonce = bytes(range(12))
    vectors = []
    for dimension in (16, 256):
        scrambler = CryptographicScrambler(key, dimension)
        params = scrambler.parameters(nonce)
        inputs = np.arange(dimension, dtype=np.float64) / dimension
        output = params.forward(inputs)
        signature = params.sign1.astype(np.int8).tobytes() + params.sign2.astype(np.int8).tobytes() + params.perm.astype('>u4').tobytes()
        entry = {'dimension': dimension, 'materials_hex': {label.decode(): scrambler._derive(nonce, label).hex() for label in LABELS}, 'stream_first32_hex': {label.decode(): _AESStream(scrambler._derive(nonce, label)).read(32).hex() for label in LABELS}, 'parameters_sha256': hashlib.sha256(signature).hexdigest(), 'output_float64_be_sha256': hashlib.sha256(output.astype('>f8').tobytes()).hexdigest()}
        if dimension == 16:
            entry.update({'sign1': params.sign1.astype(int).tolist(), 'sign2': params.sign2.astype(int).tolist(), 'permutation': params.perm.tolist(), 'inverse_permutation': params.inv_perm.tolist(), 'input': inputs.tolist(), 'output': output.tolist()})
        vectors.append(entry)
    return {'protocol': PROTOCOL, 'example_key_hex': key.hex(), 'example_nonce_hex': nonce.hex(), 'example_key_is_public_and_for_tests_only': True, 'domain_hex': DOMAIN.hex(), 'labels': [label.decode() for label in LABELS], 'vectors': vectors, 'validation': run_self_tests()}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-vectors', type=Path, help='write deterministic protocol vectors to JSON')
    args = parser.parse_args()
    if args.write_vectors is not None:
        args.write_vectors.parent.mkdir(parents=True, exist_ok=True)
        args.write_vectors.write_text(json.dumps(build_test_vectors(), indent=2) + '\n')
        print(json.dumps({'vectors_written': str(args.write_vectors), **run_self_tests()}))
    else:
        print(json.dumps(run_self_tests(), indent=2))
