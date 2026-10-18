from __future__ import annotations
import argparse
import ast
import json
import math
from numbers import Integral
from pathlib import Path
import time
from typing import Optional, Tuple
import numpy as np
from crypto_scrambler import TransformParameters, fwht_normalized
MASK32 = np.uint64(4294967295)
GAMMA = np.uint64(2654435769)
SIGN1_DOMAIN = np.uint64(324508639)
SIGN2_DOMAIN = np.uint64(610839776)
PERM_DOMAIN = np.uint64(826366246)

def _nonce_array(nonces) -> np.ndarray:
    array = np.asarray(nonces)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError('legacy nonces must be a one-dimensional integer array')
    return array.astype(np.uint64)

def legacy_nonce_features(nonces) -> np.ndarray:
    values = _nonce_array(nonces)
    chunks = np.stack([values >> np.uint64(shift) & np.uint64(65535) for shift in (0, 16, 32, 48)], axis=1).astype(np.float32)
    return chunks / np.float32(65535.0) * np.float32(2.0) - np.float32(1.0)

def seed32_from_key_nonce(key_u64: int, nonces) -> np.ndarray:
    values = _nonce_array(nonces)
    key = np.uint64(int(key_u64) % (1 << 64))
    klo, khi = (key & MASK32, key >> np.uint64(32) & MASK32)
    nlo, nhi = (values & MASK32, values >> np.uint64(32) & MASK32)
    return (nlo ^ klo) + (GAMMA * (nhi ^ khi) & MASK32) & MASK32

def splitmix32(seeds: np.ndarray, count: int) -> np.ndarray:
    seeds = np.asarray(seeds, dtype=np.uint64).reshape(-1, 1)
    indices = np.arange(1, count + 1, dtype=np.uint64).reshape(1, -1)
    values = seeds + GAMMA * indices & MASK32
    values = values ^ values >> np.uint64(16)
    values = values * np.uint64(2246822507) & MASK32
    values = values ^ values >> np.uint64(13)
    values = values * np.uint64(3266489909) & MASK32
    return values ^ values >> np.uint64(16)

class LegacyScramblerNumpy:

    def __init__(self, key_u64: int, dimension: int=256, use_perm: bool=True):
        if not isinstance(dimension, Integral) or isinstance(dimension, bool) or dimension < 1 or int(dimension) & int(dimension) - 1:
            raise ValueError('dimension must be a positive power of two')
        self.key_u64 = int(key_u64) % (1 << 64)
        self.dimension = int(dimension)
        self.use_perm = bool(use_perm)

    def parameters_batch(self, nonces) -> TransformParameters:
        values = _nonce_array(nonces)
        seeds = seed32_from_key_nonce(self.key_u64, values)
        sign1 = 1.0 - 2.0 * (splitmix32(seeds ^ SIGN1_DOMAIN, self.dimension) & 1).astype(np.float32)
        sign2 = 1.0 - 2.0 * (splitmix32(seeds ^ SIGN2_DOMAIN, self.dimension) & 1).astype(np.float32)
        if self.use_perm:
            scores = splitmix32(seeds ^ PERM_DOMAIN, self.dimension)
            permutation = np.argsort(scores, axis=1).astype(np.int64)
        else:
            permutation = np.broadcast_to(np.arange(self.dimension, dtype=np.int64), (values.size, self.dimension)).copy()
        inverse = np.empty_like(permutation)
        np.put_along_axis(inverse, permutation, np.broadcast_to(np.arange(self.dimension), permutation.shape), axis=1)
        return TransformParameters(sign1, sign2, permutation, inverse)

    def parameters(self, nonce: int) -> TransformParameters:
        params = self.parameters_batch(np.asarray([int(nonce) % (1 << 64)], dtype=np.uint64))
        return TransformParameters(params.sign1[0], params.sign2[0], params.perm[0], params.inv_perm[0])

    def scramble_batch(self, values: np.ndarray, nonces) -> np.ndarray:
        array = np.asarray(values)
        params = self.parameters_batch(nonces)
        if array.ndim != 2 or array.shape != params.sign1.shape:
            raise ValueError('batch input must have shape (number of nonces, N)')
        return params.forward(array)

    def descramble_batch(self, values: np.ndarray, nonces) -> np.ndarray:
        array = np.asarray(values)
        params = self.parameters_batch(nonces)
        if array.ndim != 2 or array.shape != params.sign1.shape:
            raise ValueError('batch input must have shape (number of nonces, N)')
        return params.inverse(array)

    def scramble(self, values: np.ndarray, nonce: int) -> np.ndarray:
        return self.parameters(nonce).forward(values)

    def descramble(self, values: np.ndarray, nonce: int) -> np.ndarray:
        return self.parameters(nonce).inverse(values)
LegacyScrambler = LegacyScramblerNumpy

def verify_original_source(source_path: Path) -> dict:
    import torch
    from torch import nn
    names = {'is_power_of_two', 'wrap_to_int64', '_mask32', 'add32', 'xor32', 'mul32', '_to_u32_halves_from_i64', 'seed32_from_key_nonce', 'splitmix32', 'fwht', 'KeyedScrambler'}
    source = source_path.read_text()
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    missing = names - {node.name for node in nodes}
    auxiliary_source = None
    if missing and missing <= {'is_power_of_two', 'wrap_to_int64'}:
        utility_path = source_path.parent / 'utils.py'
        auxiliary_source = utility_path.read_text()
        utility_tree = ast.parse(auxiliary_source)
        nodes = [node for node in utility_tree.body if isinstance(node, ast.FunctionDef) and node.name in missing] + nodes
    if {node.name for node in nodes} != names:
        raise ValueError('source does not contain all expected original definitions')
    original = {'torch': torch, 'nn': nn, 'math': math, 'Tuple': Tuple, 'Optional': Optional, 'MASK32': 4294967295, 'MASK16': 65535}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), 'exec'), original)
    rng = np.random.default_rng(20261010)
    nonces = rng.integers(0, 1 << 63, size=16, dtype=np.int64)
    test_keys = [0, 81985529216486895, 18364758544493064720, (1 << 64) - 1]
    comparisons = []
    for dimension in (16, 256):
        values = rng.normal(size=(16, dimension)).astype(np.float32)
        for key in test_keys:
            for use_perm in (False, True):
                actual = LegacyScramblerNumpy(key, dimension, use_perm)
                params = actual.parameters_batch(nonces)
                reference = original['KeyedScrambler'](dimension, key, use_perm)
                tnonce = torch.from_numpy(nonces)
                perm, inv, sign1, sign2 = reference._params(tnonce, reference.key_i64)
                assert np.array_equal(params.sign1, sign1.numpy())
                assert np.array_equal(params.sign2, sign2.numpy())
                assert np.array_equal(params.perm, perm.numpy())
                assert np.array_equal(params.inv_perm, inv.numpy())
                output = actual.scramble_batch(values, nonces)
                expected = reference.scramble(torch.from_numpy(values), tnonce).numpy()
                assert np.array_equal(output, expected), 'historical forward output differs'
                restored = actual.descramble_batch(output, nonces)
                assert np.allclose(restored, values, rtol=2e-06, atol=2e-06)
                comparisons.append({'dimension': dimension, 'key_hex': f'{key:016x}', 'permutation': use_perm, 'forward_exact_match': True, 'max_inverse_error': float(np.max(np.abs(restored - values)))})
    return {'passed': True, 'original_source': str(source_path), 'source_sha256': __import__('hashlib').sha256(source.encode()).hexdigest(), 'auxiliary_utils_sha256': None if auxiliary_source is None else __import__('hashlib').sha256(auxiliary_source.encode()).hexdigest(), 'nonce_count_per_case': len(nonces), 'cases': comparisons}
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-source', type=Path)
    parser.add_argument('--write-report', type=Path)
    args = parser.parse_args()
    if args.verify_source:
        report = verify_original_source(args.verify_source)
        if args.write_report:
            args.write_report.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'passed': report['passed'], 'cases': len(report['cases']), 'nonce_count_per_case': report['nonce_count_per_case']}))
    else:
        test = LegacyScramblerNumpy(81985529216486895)
        nonces = np.arange(1000, dtype=np.int64)
        values = np.ones((1000, 256), dtype=np.float32)
        started = time.perf_counter()
        output = test.scramble_batch(values, nonces)
        print(json.dumps({'packets': len(nonces), 'seconds': time.perf_counter() - started, 'output_shape': list(output.shape)}))
