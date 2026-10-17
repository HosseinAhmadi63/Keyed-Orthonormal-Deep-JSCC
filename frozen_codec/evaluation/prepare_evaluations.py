from pathlib import Path
import hashlib, json, pickle, tarfile, time, sys, secrets
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'archived_implementation'
sys.path.insert(0, str(SOURCE))
from models.autoencoder import Encoder, Decoder
from models.attackers import EveRecon, EveReconStrong, EveReconNoNonce, EveBobStyleDecoder, InterceptClf
from ssim_ms import ssim_per_image
from crypto_scrambler import CryptoScrambler, nonce_features, NonceCounter
from legacy_scrambler_numpy import LegacyScramblerNumpy, legacy_nonce_features
BASE = ROOT / 'models'
OUT = ROOT / 'evaluation'
DATA = OUT / 'data'
RESULTS = OUT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, a):
    Path(p).write_text(json.dumps(a, indent=2) + '\n')

def summary(mse, ssim):
    return {'n': len(mse), 'mse': float(np.mean(mse, dtype=np.float64)), 'psnr': float(-10 * np.log10(np.mean(mse, dtype=np.float64))), 'ssim': float(np.mean(ssim, dtype=np.float64)), 'ssim_p95': float(np.percentile(ssim, 95)), 'ssim_max': float(np.max(ssim))}

def predict(model, obs, images, nonces=None):
    mm = []
    ss = []
    examples = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(obs), 128):
            ob = torch.from_numpy(obs[start:start + 128])
            x = torch.from_numpy(images[start:start + 128].copy()).float() / 255
            pred = model(ob) if nonces is None else model(ob, torch.from_numpy(nonces[start:start + 128]))
            mm.extend((pred - x).square().mean((1, 2, 3)).numpy())
            ss.extend(ssim_per_image(pred, x).numpy())
            if start == 0:
                examples = pred[:16].numpy()
    return (np.array(mm), np.array(ss), np.array(examples))

def main():
    archive = DATA / 'cifar-10-python.tar.gz'
    assert hashlib.md5(archive.read_bytes()).hexdigest() == 'c58f30108f718f92721af3b95e74349a'
    with tarfile.open(archive, 'r:gz') as t:
        trainparts = []
        labels = []
        for i in range(1, 6):
            d = pickle.load(t.extractfile(f'cifar-10-batches-py/data_batch_{i}'), encoding='bytes')
            trainparts.append(d[b'data'])
            labels.extend(d[b'labels'])
        d = pickle.load(t.extractfile('cifar-10-batches-py/test_batch'), encoding='bytes')
    train_images = np.concatenate(trainparts).reshape(-1, 3, 32, 32)
    y_train = np.array(labels, dtype=np.int64)
    all_test_images = d[b'data'].reshape(-1, 3, 32, 32)
    all_test_labels = np.array(d[b'labels'], dtype=np.int64)
    archived = json.loads((BASE / 'json/final_eval.json').read_text())
    truth = np.array(archived['distributions']['int_true'])
    assert np.array_equal(truth, all_test_labels[:6400]), 'Dataset ordering differs from archived evaluation.'
    test_images = all_test_images[6400:].copy()
    y_test = all_test_labels[6400:].copy()
    ck = torch.load(BASE / 'pt/best_main.pt', map_location='cpu', weights_only=True)
    enc = Encoder(256).eval()
    enc.load_state_dict(ck['enc'])
    dec = Decoder(256).eval()
    dec.load_state_dict(ck['dec'])
    latentpath = DATA / 'frozen_latents.npz'
    if latentpath.exists():
        with np.load(latentpath) as a:
            train_z = a['train_latents']
            test_z = a['test_latents']
    else:
        arrays = []
        for label, images in [('training', train_images), ('holdout', test_images)]:
            zparts = []
            beg = time.monotonic()
            with torch.no_grad():
                for start in range(0, len(images), 128):
                    x = torch.from_numpy(images[start:start + 128].copy()).float() / 255
                    z = enc(x).numpy()
                    zparts.append(z)
                    if start % 6400 == 0:
                        print(f'Encoding {label}: {start}/{len(images)}; {time.monotonic() - beg:.1f}s', flush=True)
            arrays.append(np.concatenate(zparts))
            print(f'Encoded {label}: {len(images)}', flush=True)
        train_z, test_z = arrays
        np.savez(latentpath, train_latents=train_z, test_latents=test_z)

    def norm(z):
        return z / np.sqrt(np.maximum(np.mean(z * z, axis=1, keepdims=True), 1e-06))
    train_u, test_u = (norm(train_z), norm(test_z))
    keypath = OUT / 'experiment_keys.json'
    if keypath.exists():
        keyrecord = json.loads(keypath.read_text())
    else:
        keyrecord = {'key_hex': 'aa47c63ebedee13713bbd1e4ed31f1849fed3be178b9fa04ef79da31fdc4a212', 'wrong_key_hex': 'eb47ecb539e8f5f3e365eafa8213f19aa7979fb1fea47f2f63abccff2f17193e'}
        dump(keypath, keyrecord)
    publickey = bytes.fromhex(keyrecord['key_hex'])
    wrongkey = bytes.fromhex(keyrecord['wrong_key_hex'])
    crypto = CryptoScrambler(publickey, 256)
    wrong = CryptoScrambler(wrongkey, 256)
    n_rng = np.random.default_rng(20261010)
    order = n_rng.permutation(len(train_u) + len(test_u))
    offset = 23002549206610311231137906688
    all_n = [int(offset) + int(i) for i in order]
    train_n, test_n = (all_n[:50000], all_n[50000:])
    legacy = LegacyScramblerNumpy(1311768467463790320, 256)
    gen = torch.Generator().manual_seed(42 + 12345)
    train_old = torch.randint(0, 2 ** 63 - 1, (50000,), generator=gen, dtype=torch.int64).numpy()
    gen = torch.Generator().manual_seed(43 + 12345)
    test_old = torch.randint(0, 2 ** 63 - 1, (10000,), generator=gen, dtype=torch.int64).numpy()[6400:]
    legacy_test_params = legacy.parameters_batch(test_old)
    crypto_test_params = crypto.parameters_batch(test_n)
    noise_rng = np.random.default_rng(20261012)
    train_w = noise_rng.normal(0, 0.1, train_u.shape).astype(np.float32)
    test_w = noise_rng.normal(0, 0.1, test_u.shape).astype(np.float32)
    bob_w = noise_rng.normal(0, 10 ** (-12 / 20), test_u.shape).astype(np.float32)
    base_meta = {'dataset': 'CIFAR-10', 'official_archive_md5': 'c58f30108f718f92721af3b95e74349a', 'source_checkpoint_sha256': sha(BASE / 'pt/best_main.pt'), 'selected_main_epoch': ck['best']['epoch'], 'original_selection_images': list(range(6400)), 'holdout_indices': list(range(6400, 10000)), 'test_used_for_model_selection': False, 'encoder_retrained': False, 'power_normalized': True, 'target_power': 1.0, 'snr_bob_db': 12.0, 'snr_eve_db': 20.0, 'noise_seed': 20261012, 'nonce_assignment_seed': 20261010, 'one_intercepted_packet_per_image': True, 'crypto_example_master_key_hex': publickey.hex(), 'example_keys_public_for_reproducibility_only': True, 'nonce_counter_offset': hex(offset), 'data_order_verified_against_archived_first6400_labels': True}
    np.savez(DATA / 'basic_holdout.npz', train_latents=train_z, test_latents=test_z, test_images=test_images, y_test=y_test, crypto_nonces_u32=np.array([[n >> 64, n >> 32 & 4294967295, n & 4294967295] for n in test_n], dtype=np.uint32), legacy_nonces=test_old, test_indices=np.arange(6400, 10000))
    for protocol in ['crypto', 'legacy', 'identity', 'fixed_crypto', 'permutation_only']:
        beg = time.monotonic()
        path = DATA / f'cache_{protocol}.npz'
        if path.exists():
            print('Cache already exists: ' + protocol, flush=True)
            continue
        if protocol == 'crypto':
            train_cipher = crypto.scramble_batch(train_u, train_n)
            test_cipher = crypto_test_params.forward(test_u)
            tn = nonce_features(train_n)
            vn = nonce_features(test_n)
        elif protocol == 'legacy':
            train_cipher = legacy.scramble_batch(train_u, train_old)
            test_cipher = legacy_test_params.forward(test_u)
            tn = np.pad(legacy_nonce_features(train_old), ((0, 0), (0, 2)))
            vn = np.pad(legacy_nonce_features(test_old), ((0, 0), (0, 2)))
        elif protocol == 'identity':
            train_cipher = train_u
            test_cipher = test_u
            tn = nonce_features(train_n)
            vn = nonce_features(test_n)
        elif protocol == 'fixed_crypto':
            p = crypto.parameters(offset)
            train_cipher = p.forward(train_u)
            test_cipher = p.forward(test_u)
            tn = np.zeros((50000, 6), np.float32)
            vn = np.zeros((3600, 6), np.float32)
        elif protocol == 'permutation_only':
            p = crypto.parameters_batch(train_n)
            train_cipher = np.take_along_axis(train_u, p.perm, axis=1)
            test_cipher = np.take_along_axis(test_u, crypto_test_params.perm, axis=1)
            tn = nonce_features(train_n)
            vn = nonce_features(test_n)
        meta = {**base_meta, 'protocol': protocol, 'secret_key_given_to_attackers': False}
        np.savez(path, train_obs=train_cipher + train_w, train_nonce_features=tn, y_train=y_train, test_obs=test_cipher + test_w, test_nonce_features=vn, y_test=y_test, train_images=train_images, test_images=test_images, train_latents=train_u, test_latents=test_u, test_indices=np.arange(6400, 10000), metadata_json=np.array(json.dumps(meta)))
        print(f'Prepared {protocol} cache in {time.monotonic() - beg:.1f}s', flush=True)
        dump(RESULTS / f'cache_{protocol}_metadata.json', meta)
    variants = {'identity': test_u + bob_w, 'legacy': legacy_test_params.inverse(legacy_test_params.forward(test_u) + bob_w), 'crypto': crypto_test_params.inverse(crypto_test_params.forward(test_u) + bob_w), 'crypto_wrong_key': wrong.parameters_batch(test_n).inverse(crypto_test_params.forward(test_u) + bob_w), 'crypto_nonce_bitflip': crypto.parameters_batch([n ^ 1 for n in test_n]).inverse(crypto_test_params.forward(test_u) + bob_w)}
    metrics = {}
    perimage = {}
    for name, z in variants.items():
        beg = time.monotonic()
        m, s, ex = predict(dec, z, test_images)
        metrics[name] = summary(m, s)
        perimage[name + '_mse'] = m
        perimage[name + '_ssim'] = s
        perimage[name + '_first16'] = ex
        dump(RESULTS / 'bob_holdout_metrics.json', {'metadata': base_meta, 'models': metrics})
        np.savez(RESULTS / 'bob_holdout_per_image.npz', **perimage)
        print(f'Bob holdout {name}: {metrics[name]} ({time.monotonic() - beg:.1f}s)', flush=True)
    atk = torch.load(BASE / 'pt/best_response_attackers.pt', map_location='cpu', weights_only=True)
    models = {'base': EveRecon(256, 32), 'strong': EveReconStrong(256, 64), 'no_nonce': EveReconNoNonce(256), 'bob_style': EveBobStyleDecoder(256)}
    for k, m in models.items():
        m.load_state_dict(atk['eve_models'][k])
        m.eval()
    for protocol in ['legacy', 'crypto']:
        with np.load(DATA / f'cache_{protocol}.npz') as a:
            obs = a['test_obs']
        metrics = {}
        perimage = {}
        for name, model in models.items():
            nonce_input = test_old if protocol == 'legacy' else np.array([n & 18446744073709551615 for n in test_n], dtype=np.int64)
            m, s, ex = predict(model, obs, test_images, nonce_input if name in ['base', 'strong'] else None)
            metrics[name] = summary(m, s)
            perimage[name + '_mse'] = m
            perimage[name + '_ssim'] = s
            perimage[name + '_first16'] = ex
            print(f'Archived transfer attack {protocol}/{name}: {metrics[name]}', flush=True)
        dump(RESULTS / f'archived_transfer_{protocol}.json', {'metadata': base_meta, 'models': metrics})
        np.savez(RESULTS / f'archived_transfer_{protocol}_per_image.npz', **perimage)
    dump(RESULTS / 'preparation_manifest.json', base_meta)
    print('Preparation and holdout inference complete.', flush=True)
if __name__ == '__main__':
    main()
