import numpy as np

from keyed_jscc.public_sampling import invariant_features, sample_packets, theoretical_moments


def test_coordinate_order_invariance():
    rng = np.random.default_rng(81)
    values = rng.normal(size=(20, 256)).astype(np.float32)
    permutations = rng.permuted(np.broadcast_to(np.arange(256), values.shape), axis=1)
    permuted = np.take_along_axis(values, permutations, axis=1)
    for view in ['fusion', 'magnitude', 'signed']:
        np.testing.assert_allclose(invariant_features(values, view), invariant_features(permuted, view), atol=1e-6)


def test_noiseless_packet_power():
    rng = np.random.default_rng(82)
    values = rng.normal(size=(50, 256)).astype(np.float32)
    for family in ['full', 'permutation_only', 'identity']:
        packets = sample_packets(values, rng, np.inf, family)
        np.testing.assert_allclose(np.sum(packets**2, axis=1), np.sum(values**2, axis=1), rtol=5e-7)


def test_distribution_moments_detect_different_latents():
    values = np.zeros((2, 256))
    values[0, 0] = 16
    values[1, :2] = np.sqrt(128)
    moments = theoretical_moments(values)
    np.testing.assert_allclose(moments['second'], [1, 1])
    assert moments['fourth'][0] < moments['fourth'][1]
