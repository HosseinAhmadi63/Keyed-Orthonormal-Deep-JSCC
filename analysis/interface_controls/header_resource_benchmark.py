from __future__ import annotations
import argparse, binascii, json, math, os, sys, time
from pathlib import Path
import numpy as np
from scipy.fft import dct, idct
BASE = Path(__file__).resolve().parent
H = 256
NONCE_START = 25438558367014162481750736896

def hamming_tables():
    data = (np.arange(16)[:, None] >> np.arange(3, -1, -1) & 1).astype(np.uint8)
    a, b, c, d = data.T
    code = np.column_stack((a ^ b ^ d, a ^ c ^ d, a, b ^ c ^ d, b, c, d, np.zeros(16, np.uint8)))
    code[:, -1] = code[:, :7].sum(1) % 2
    return (code, 1 - 2 * code.astype(np.float32))
CODE, SIGN = hamming_tables()

def header_bytes(base, count):
    if not (0 <= base < 1 << 96 and 1 <= count <= 65535 and (base + count <= 1 << 96)):
        raise ValueError('Nonce/count outside the protocol range')
    body = int(base).to_bytes(12, 'big') + int(count).to_bytes(2, 'big')
    return body + binascii.crc_hqx(body, 65535).to_bytes(2, 'big')

def header_waveforms(rows):
    a = np.asarray(rows, dtype=np.uint8)
    nib = np.column_stack((a >> 4, a & 15)).reshape(len(a), 2, 16).transpose(0, 2, 1).reshape(len(a), 32)
    return SIGN[nib].reshape(len(a), 256)

def decode_waveforms(received):
    nib = (np.asarray(received).reshape(-1, 32, 8) @ SIGN.T).argmax(-1).astype(np.uint8)
    return (nib[:, ::2] * 16 + nib[:, 1::2]).astype(np.uint8)

def parse_header(raw, expected_count=None):
    raw = bytes(raw)
    body = raw[:14]
    base = int.from_bytes(body[:12], 'big')
    count = int.from_bytes(body[12:], 'big')
    ok = len(raw) == 16 and int.from_bytes(raw[14:], 'big') == binascii.crc_hqx(body, 65535) and (1 <= count <= 65535) and (base + count <= 1 << 96)
    if expected_count is not None:
        ok = ok and count == expected_count
    return (ok, base, count)

def wilson(errors, n):
    z = 1.959963984540054
    p = errors / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]

def selftests():
    distance = np.count_nonzero(CODE[:, None, :] != CODE[None, :, :], axis=-1) + np.eye(16, dtype=int) * 99
    assert distance.min() == 4 and np.unique(CODE, axis=0).shape[0] == 16
    assert binascii.crc_hqx(b'123456789', 65535) == 10673
    raw = header_bytes(NONCE_START, 32)
    array = np.frombuffer(raw, np.uint8)[None]
    sent = header_waveforms(array)
    assert np.array_equal(decode_waveforms(sent), array)
    for bit in range(256):
        noisy = sent.copy()
        noisy[0, bit] *= -1
        assert np.array_equal(decode_waveforms(noisy), array)
    ok, base, count = parse_header(raw, 32)
    assert ok and base == NONCE_START and (count == 32)
    flipped = bytearray(raw)
    flipped[0] ^= 1
    assert not parse_header(flipped, 32)[0]
    rng = np.random.default_rng(20261015)
    waveform = []
    for count in [1, 8, 32, 128]:
        length = count * 256
        u = rng.normal(size=length)
        z = rng.normal(size=length)
        extra = rng.normal(size=H)
        sigma = 10 ** (-12 / 20)
        transmit = idct(np.r_[u, np.zeros(H)], norm='ortho', workers=2)
        channel_noise = idct(sigma * np.r_[z, extra], norm='ortho', workers=2)
        recovered = dct(transmit + channel_noise, norm='ortho', workers=2)[:length]
        energy_error = abs(float(transmit @ transmit - u @ u)) / float(u @ u)
        reconstruction_error = float(np.max(np.abs(recovered - (u + sigma * z))))
        assert energy_error < 1e-12 and reconstruction_error < 1e-12
        waveform.append({'packets': count, 'real_uses': length + H, 'relative_energy_error': energy_error, 'max_sufficient_statistic_error': reconstruction_error})
    return {'hamming_minimum_distance': int(distance.min()), 'single_bit_tests': 256, 'crc123456789': '29b1', 'header_example_hex': raw.hex(), 'coded_example_hex': np.packbits((sent[0] < 0).astype(np.uint8), bitorder='big').tobytes().hex(), 'dct_physical_waveform_tests': waveform}

def header_monte_carlo(output, n=10000, seed=20261015):
    from evidence_common import write_json
    truth = np.array([np.frombuffer(header_bytes(NONCE_START + i * 128, 128), np.uint8) for i in range(n)])
    sent = header_waveforms(truth)
    noise = np.random.default_rng(seed).normal(size=sent.shape).astype(np.float32)
    results = {}
    for snr in [0, 3, 6, 9, 12]:
        decoded = decode_waveforms(sent + noise * 10 ** (-snr / 20))
        changed = np.any(decoded != truth, axis=1)
        accept = np.array([parse_header(x, 128)[0] for x in decoded])
        wrong = changed & accept
        detected = ~accept
        errors = int(changed.sum())
        results[str(snr)] = {'header_real_symbol_snr_db': snr, 'frames': n, 'wrong_decoded_headers': errors, 'decoded_header_error_rate': errors / n, 'decoded_header_error_wilson95': wilson(errors, n), 'crc_detected_failures': int(detected.sum()), 'undetected_wrong_metadata': int(wrong.sum()), 'undetected_metadata_wilson95': wilson(int(wrong.sum()), n), 'zero_error_one_sided95_upper': float(1 - 0.05 ** (1 / n)) if errors == 0 else None}
        print('HEADER', snr, results[str(snr)], flush=True)
    report = {'protocol': protocol(), 'selftests': selftests(), 'seed': seed, 'snr_convention': 'per-real-dimension signal power divided by real Gaussian variance', 'shared_noise_across_snr': True, 'monte_carlo': results}
    write_json(output / 'header_reliability.json', report)
    return report

def protocol():
    return {'information_bits': {'group_base_nonce': 96, 'actual_packet_count': 16, 'CRC16': 16}, 'header_information_bits': 128, 'coding': '32 extended-Hamming(8,4) words; minimum distance4; exhaustive soft maximum-likelihood word decoding', 'coded_header_bits': 256, 'modulation': 'bits map0->+sqrt(P),1->-sqrt(P) on real coordinates; adjacent pairs form QPSK with complex energy2P', 'header_real_uses': 256, 'header_complex_uses': 128, 'crc': 'polynomial0x1021, initial0xffff, no xor-out; big-endian16bit checksum of nonce12bytes||count2bytes', 'bit_order': 'MSB first; high nibble then low nibble; code positions[p1,p2,d1,p4,d2,d3,d4,p8]', 'nonce_policy': 'reserve count consecutive96bit nonces; packet slotj usesbase+j; never reuse a counter under a key; persistent allocation across restart; independent disjoint ranges across senders', 'integrity_assumption': 'passiveAWGN only; CRC is error detection, not authentication; active tampering/replay require an authenticated design with extra accounted overhead', 'framing_assumption': 'group boundaries, expected physical packet count and slot order known; synchronization/pilots/framing common to both compared schemes and excluded', 'detected_failure': 'abstain on entire group and use fitting-only mean image for utility; no silent packet removal', 'undetected_failure': 'decode with the wrong received nonce; record separately; no oracle erasure', 'rate_note': 'larger groups trade nonce/header amortization for buffering, joint outage and latency; waiting time is not measured'}

def grouped_bootstrap(proposed, reference, group_ids, draws=10000, seed=20261016):
    groups = np.unique(group_ids)
    sizes = np.bincount(group_ids).astype(np.float64)
    metrics = {'mse': 'mse', 'legacy_ssim': 'legacy_zero_clipped', 'valid_ssim': 'reference_valid_unclipped'}
    sums = {label: {key: np.bincount(group_ids, weights=arrays[value]) for key, value in metrics.items()} for label, arrays in [('proposed', proposed), ('reference', reference)]}
    rng = np.random.default_rng(seed)
    samples = {label: {key: [] for key in metrics} for label in sums}
    for start in range(0, draws, 128):
        ii = rng.integers(0, len(groups), size=(min(128, draws - start), len(groups)))
        denominator = sizes[ii].sum(1)
        for label in sums:
            for metric in metrics:
                samples[label][metric].extend((sums[label][metric][ii].sum(1) / denominator).tolist())
    samples = {label: {k: np.asarray(v) for k, v in m.items()} for label, m in samples.items()}
    ci = lambda x: np.quantile(x, [0.025, 0.975]).tolist()
    out = {}
    for label, arrays in [('proposed', proposed), ('reference', reference)]:
        out[label] = {'mse': float(arrays['mse'].mean(dtype=np.float64)), 'dataset_psnr_db': float(-10 * np.log10(arrays['mse'].mean(dtype=np.float64))), 'dataset_psnr_ci95': ci(-10 * np.log10(samples[label]['mse']))}
        for k, v in metrics.items():
            if k != 'mse':
                out[label][k] = {'mean': float(arrays[v].mean(dtype=np.float64)), 'ci95': ci(samples[label][k])}
    out['proposed_minus_reference'] = {k: {'mean': float(proposed[v].mean(dtype=np.float64) - reference[v].mean(dtype=np.float64)), 'ci95': ci(samples['proposed'][k] - samples['reference'][k])} for k, v in metrics.items() if k != 'mse'}
    out['bootstrap'] = {'unit': 'whole transmitted group', 'groups': len(groups), 'draws': draws, 'seed': seed, 'last_partial_group_size': int(sizes[-1]), 'weighting': 'sumgroupmetric/sumgroupimages in each paired resample'}
    return out

def utility(output, limit=None, source_snr=12.0):
    import torch
    from evidence_common import ASSETS, INPUTS, load_decoder, predict, metric_arrays, sha256, write_json
    sys.path.insert(0, str(ASSETS / 'evaluation'))
    from crypto_scrambler import CryptoScrambler
    with np.load(INPUTS / 'cifar_holdout.npz') as a:
        u = a['test_latents']
        images = a['test_images']
        indices = a['test_indices']
    n = limit or len(u)
    u, images, indices = (u[:n], images[:n], indices[:n])
    u = u.astype(np.float64)
    u *= np.sqrt(256 / np.square(u).sum(axis=1, keepdims=True))
    shuffle = np.random.default_rng(202610151).permutation(n)
    u, images, indices = (u[shuffle], images[shuffle], indices[shuffle])
    key = bytes.fromhex({'key_hex': 'aa47c63ebedee13713bbd1e4ed31f1849fed3be178b9fa04ef79da31fdc4a212', 'wrong_key_hex': 'eb47ecb539e8f5f3e365eafa8213f19aa7979fb1fea47f2f63abccff2f17193e'}['key_hex'])
    crypto = CryptoScrambler(key, 256)
    decoder, checkpoint = load_decoder()
    with np.load(ASSETS / 'evaluation/results/reconstruction_crypto_final/reconstruction_per_image.npz') as a:
        prior = a['fitting_mean_image_first16_reconstructions'][0]
    sigma = 10 ** (-source_snr / 20)
    z = np.random.default_rng(202610152).normal(size=u.shape)
    reference_pred = predict(decoder, u + sigma * z)
    reference = metric_arrays(reference_pred, images)
    report = {'protocol': protocol(), 'publication_run': n == 3600, 'n': n, 'reference_source_snr_db': source_snr, 'noise_variance_per_real_dimension': sigma * sigma, 'target_image_energy_real_units': 256, 'energy_note': 'normalized transmitted signal energy, not joules/device energy; both use original frozen latent group energy; keyed header/payload share uniform per-real power', 'energy_arithmetic': 'float64 radial normalization and channel/transform arithmetic; decoder weights and inference remain float32; scalar allocation count/(count+1) is known from header count', 'reference': 'unitaryDCT embedding of[u_group,zeros256] uses same total dimensions/energy; physicalnoise generated through the inverseDCT; sufficient statistic u+sigma*z', 'noise_coupling': 'same independent per-image latent-coordinate Gaussian draw across schemes; cipher-domain noise rotated orthogonally; no labels used', 'nonce_allocations': 'disjoint offsets for each group profile,sourceSNR and pilot/publication status; persistent next-exclusive values recorded in group ledger', 'source_checkpoint_sha256': sha256(checkpoint), 'numpy_version': np.__version__, 'torch_version': str(torch.__version__), 'selftests': selftests(), 'profiles': {}}
    all_arrays = {'test_indices': indices, **{'reference__' + k: v for k, v in reference.items()}}
    started = time.monotonic()
    for profile in [1, 8, 32, 128]:
        rng = np.random.default_rng(202610153 + profile)
        group_ids = np.arange(n) // profile
        allocation = int(round(source_snr * 100)) + (1000000 if limit is not None else 0)
        next_nonce = NONCE_START + allocation * (1 << 48) + profile * (1 << 32)
        predicted = np.empty((n, 3, 32, 32), np.float32)
        records = []
        for group, start in enumerate(range(0, n, profile)):
            stop = min(start + profile, n)
            count = stop - start
            uu = u[start:stop]
            zz = z[start:stop]
            base = next_nonce
            next_nonce += count
            nonces = [base + j for j in range(count)]
            p = crypto.parameters_batch(nonces)
            cipher = p.forward(uu)
            energy = float(np.square(uu.astype(np.float64)).sum())
            cipher_energy = float(np.square(cipher.astype(np.float64)).sum())
            alpha2 = count / (count + 1)
            alpha = np.sqrt(alpha2)
            payload = alpha * cipher + sigma * p.forward(zz)
            sent_header = header_waveforms(np.frombuffer(header_bytes(base, count), np.uint8)[None])[0]
            receive_header = alpha * sent_header + sigma * rng.normal(size=H)
            decoded = decode_waveforms(receive_header[None])[0]
            accepted, received_base, received_count = parse_header(decoded, count)
            undetected = accepted and received_base != base
            if not accepted:
                predicted[start:stop] = prior
            else:
                inverse = p if received_base == base else crypto.parameters_batch([received_base + j for j in range(count)])
                predicted[start:stop] = predict(decoder, inverse.inverse(payload) / alpha)
            flat = uu.astype(np.float64).ravel()
            noise = zz.astype(np.float64).ravel()
            tx = idct(np.r_[flat, np.zeros(H)], norm='ortho', workers=2)
            w = idct(sigma * np.r_[noise, rng.normal(size=H)], norm='ortho', workers=2)
            sufficient = dct(tx + w, norm='ortho', workers=2)[:len(flat)]
            assert np.max(np.abs(sufficient - (flat + sigma * noise))) < 1e-10
            reference_energy = float(tx @ tx)
            proposed_energy = alpha2 * (cipher_energy + H)
            assert abs(reference_energy - proposed_energy) / energy < 1e-10
            records.append({'group': group, 'count': count, 'base_nonce_hex': hex(base), 'next_nonce_exclusive_hex': hex(next_nonce), 'header_accepted': bool(accepted), 'undetected_wrong_nonce': bool(undetected), 'alpha2_real_power': alpha2, 'keyed_payload_snr_db': float(source_snr + 10 * np.log10(alpha2)), 'complex_uses': (count * 256 + H) // 2, 'reference_energy': reference_energy, 'proposed_energy': proposed_energy, 'header_energy': alpha2 * H, 'payload_energy': alpha2 * cipher_energy})
        metrics = metric_arrays(predicted, images)
        summary = grouped_bootstrap(metrics, reference, group_ids)
        failures = sum((not row['header_accepted'] for row in records))
        summary.update({'group_records': records, 'group_profile': profile, 'groups': len(records), 'detected_failed_groups': failures, 'failed_group_wilson95': wilson(failures, len(records)), 'undetected_wrong_nonce_groups': sum((row['undetected_wrong_nonce'] for row in records)), 'header_complex_uses': len(records) * 128, 'payload_complex_uses': n * 128, 'total_complex_uses': n * 128 + len(records) * 128, 'actual_header_to_payload_fraction': len(records) / n, 'total_transmit_energy_real_units': float(sum((row['reference_energy'] for row in records))), 'keyed_payload_snr_db_range': [min((row['keyed_payload_snr_db'] for row in records)), max((row['keyed_payload_snr_db'] for row in records))]})
        report['profiles'][str(profile)] = summary
        for name, value in metrics.items():
            all_arrays[f'B{profile}__{name}'] = value
        all_arrays[f'B{profile}__group_id'] = group_ids
        write_json(output / 'equal_resource_utility.json', report)
        np.savez(output / 'equal_resource_per_image.npz', **all_arrays)
        print('RESOURCE', profile, 'legacySSIM', summary['proposed']['legacy_ssim'], 'reference', summary['reference']['legacy_ssim'], 'failures', failures, 'elapsed', time.monotonic() - started, flush=True)
    return report
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['selftest', 'header', 'utility', 'all'], default='all')
    p.add_argument('--output', type=Path, default=BASE / 'results/header_resource')
    p.add_argument('--limit', type=int)
    p.add_argument('--source-snr', type=float, default=12.0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.mode == 'selftest':
        print(json.dumps(selftests(), indent=2))
    if a.mode in ['header', 'all']:
        header_monte_carlo(a.output)
    if a.mode in ['utility', 'all']:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        utility(a.output, a.limit, a.source_snr)
