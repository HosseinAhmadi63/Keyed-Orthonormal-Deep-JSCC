import json
from pathlib import Path

import numpy as np
import pytest

from keyed_jscc.crypto_scrambler import CryptographicScrambler, NonceCounter, build_test_vectors, run_self_tests


def test_primitive_known_answers():
    assert run_self_tests()['passed']


def test_protocol_vectors():
    path = Path(__file__).resolve().parents[1] / 'configs' / 'protocol_vectors.json'
    expected = json.loads(path.read_text())
    assert build_test_vectors()['vectors'] == expected['vectors']


@pytest.mark.parametrize('dimension', [16, 256])
def test_round_trip_and_energy(dimension):
    rng = np.random.default_rng(20261018)
    values = rng.normal(size=(5, dimension))
    scrambler = CryptographicScrambler(bytes(range(32)), dimension)
    nonces = list(range(5))
    packets = scrambler.scramble_batch(values, nonces)
    decoded = scrambler.descramble_batch(packets, nonces)
    np.testing.assert_allclose(decoded, values, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(np.sum(packets**2, axis=1), np.sum(values**2, axis=1), rtol=2e-14)


def test_nonce_changes_transform():
    values = np.random.default_rng(7).normal(size=256)
    scrambler = CryptographicScrambler(bytes(range(32)), 256)
    assert not np.allclose(scrambler.scramble(values, 0), scrambler.scramble(values, 1))


def test_nonce_counter_exhaustion():
    counter = NonceCounter((1 << 96) - 1)
    assert counter.take() == [((1 << 96) - 1).to_bytes(12, 'big')]
    with pytest.raises(OverflowError):
        counter.take()
