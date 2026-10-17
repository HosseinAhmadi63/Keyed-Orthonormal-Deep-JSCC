from pathlib import Path
import hashlib, json, math, pickle, tarfile, urllib.request, zipfile
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from crypto_scrambler import CryptoScrambler, nonce_features
from legacy_ssim import ssim_per_image

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()

def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)

def normalize(z):
    return z.float() / z.float().square().mean(1, keepdim=True).clamp_min(1e-06).sqrt()

def fwht(x):
    n = x.shape[-1]
    h = 1
    z = x
    while h < n:
        a = z.reshape(-1, n // (2 * h), 2, h)
        left, right = (a[:, :, 0, :], a[:, :, 1, :])
        z = torch.stack((left + right, left - right), dim=2).reshape(-1, n)
        h *= 2
    return z / math.sqrt(n)

def cipher_torch(u, params):
    d = u.device
    a = torch.as_tensor(params.sign1, device=d, dtype=u.dtype)
    b = torch.as_tensor(params.sign2, device=d, dtype=u.dtype)
    p = torch.as_tensor(params.perm.astype(np.int64), device=d)
    return torch.gather(fwht(u * a) * b, 1, p)

class OnlineClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.nonce = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(288, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 10))

    def forward(self, obs, nonce):
        return self.head(torch.cat((obs, self.nonce(2 * nonce - 1)), dim=1))

class ImageMLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.SiLU(), nn.Linear(512, 1024), nn.SiLU(), nn.Linear(1024, 3072), nn.Sigmoid())

    def forward(self, x, nonce=None):
        return self.net(x).reshape(-1, 3, 32, 32)

def load_data(folder, cfg, smoke=False):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    if smoke:
        r = np.random.default_rng(991)
        images = r.integers(0, 256, (80, 3, 32, 32), dtype=np.uint8)
        return (images[:64], np.arange(64, dtype=np.int64) % 10, images[64:], np.arange(16, dtype=np.int64) % 10)
    archive = folder / 'cifar-10-python.tar.gz'
    expected = 'c58f30108f718f92721af3b95e74349a'
    if archive.exists() and hashlib.md5(archive.read_bytes()).hexdigest() != expected:
        bad = archive.with_name('cifar-10-python.invalid-' + sha(archive)[:12] + '.tar.gz')
        archive.replace(bad)
        print('Dataset checksum mismatch; downloading replacement', flush=True)
    if not archive.exists():
        print('Downloading official CIFAR-10 (~170 MB).', flush=True)
        temporary = archive.with_suffix('.download')
        errors = []
        for host in ['www.cs.toronto.edu/~kriz', 'cave.cs.toronto.edu/kriz']:
            try:
                urllib.request.urlretrieve('https://' + host + '/cifar-10-python.tar.gz', temporary)
                if hashlib.md5(temporary.read_bytes()).hexdigest() != expected:
                    raise RuntimeError('Downloaded archive checksum mismatch')
                temporary.replace(archive)
                break
            except Exception as exc:
                errors.append(str(exc))
        if not archive.exists():
            raise RuntimeError('Official dataset download failed: ' + '; '.join(errors))
    xs = []
    ys = []
    with tarfile.open(archive, 'r:gz') as tar:
        for i in range(1, 6):
            d = pickle.load(tar.extractfile(f'cifar-10-batches-py/data_batch_{i}'), encoding='bytes')
            xs.append(d[b'data'])
            ys.extend(d[b'labels'])
        d = pickle.load(tar.extractfile('cifar-10-batches-py/test_batch'), encoding='bytes')
    return (np.concatenate(xs).reshape(-1, 3, 32, 32), np.array(ys, dtype=np.int64), d[b'data'].reshape(-1, 3, 32, 32)[cfg['holdout_start']:].copy(), np.array(d[b'labels'], dtype=np.int64)[cfg['holdout_start']:].copy())

def split_indices(y, cfg, smoke=False):
    if smoke:
        return (np.arange(32), np.arange(32, 64))
    r = np.random.default_rng(cfg['split_seed'])
    val = []
    for cls in range(10):
        val.extend(r.permutation(np.flatnonzero(y == cls))[:cfg['validation_per_class']])
    val = np.array(sorted(val), dtype=np.int64)
    fit = np.setdiff1d(np.arange(len(y)), val)
    assert len(fit) == 45000 and len(val) == 5000 and (not np.intersect1d(fit, val).size)
    return (fit, val)

@torch.no_grad()
def encode(enc, images, device, batch=256):
    enc.eval()
    out = []
    for i in range(0, len(images), batch):
        x = torch.as_tensor(images[i:i + batch].copy(), device=device).float() / 255
        out.append(normalize(enc(x)).cpu().numpy())
    return np.concatenate(out)

def actual_packets(u, key_hex, offset, seed, snr=20):
    r = np.random.default_rng(seed)
    nonces = [int(offset) + int(i) for i in r.permutation(len(u))]
    p = CryptoScrambler(bytes.fromhex(key_hex), u.shape[1]).parameters_batch(nonces)
    obs = p.forward(u) + r.normal(0, 10 ** (-snr / 20), u.shape).astype(np.float32)
    return (obs.astype(np.float32), nonce_features(nonces), nonces, p)

@torch.no_grad()
def valid_ssim(a, b):
    a = a.float()
    b = b.float()
    t = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    w = torch.exp(-t.square() / (2 * 1.5 ** 2))
    w = w / w.sum()
    w = (w[:, None] * w[None, :])[None, None].repeat(3, 1, 1, 1)
    conv = lambda v: F.conv2d(v, w, groups=3)
    ma, mb = (conv(a), conv(b))
    va = conv(a * a) - ma * ma
    vb = conv(b * b) - mb * mb
    cov = conv(a * b) - ma * mb
    return ((2 * ma * mb + 0.01 ** 2) * (2 * cov + 0.03 ** 2) / ((ma * ma + mb * mb + 0.01 ** 2) * (va + vb + 0.03 ** 2))).mean((1, 2, 3))

@torch.no_grad()
def image_metrics(predict, images, features, device, batch=128):
    rows = {k: [] for k in ['mse', 'ssim_legacy', 'ssim_valid']}
    examples = []
    for i in range(0, len(images), batch):
        y = torch.as_tensor(images[i:i + batch].copy(), device=device).float() / 255
        p = predict(None if features is None else features[i:i + batch], len(y)).float()
        if p.shape != y.shape or not torch.isfinite(p).all():
            raise RuntimeError('Invalid image prediction')
        rows['mse'].append((p - y).square().mean((1, 2, 3)).cpu().numpy())
        rows['ssim_legacy'].append(ssim_per_image(p, y).cpu().numpy())
        rows['ssim_valid'].append(valid_ssim(p, y).cpu().numpy())
        if not examples:
            examples.append(p[:16].cpu().numpy())
    rows = {k: np.concatenate(v) for k, v in rows.items()}
    rows['first_examples'] = examples[0]
    return rows

def summarize_images(rows):
    return {'n': len(rows['mse']), 'mse': float(np.mean(rows['mse'], dtype=np.float64)), 'dataset_psnr': float(-10 * np.log10(np.mean(rows['mse'], dtype=np.float64))), 'mean_ssim_legacy': float(np.mean(rows['ssim_legacy'], dtype=np.float64)), 'mean_ssim_valid': float(np.mean(rows['ssim_valid'], dtype=np.float64))}

def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]

def class_metrics(y, probs):
    assert np.isfinite(probs).all() and np.max(np.abs(probs.sum(1) - 1)) < 0.0001
    pred = probs.argmax(1)
    cm = np.bincount(10 * y + pred, minlength=100).reshape(10, 10)
    counts = cm.sum(1)
    k = int(np.trace(cm))
    recall = np.diag(cm) / np.maximum(counts, 1)
    return {'n': len(y), 'correct': k, 'accuracy': k / len(y), 'wilson95': wilson(k, len(y)), 'balanced_accuracy': float(recall[counts > 0].mean()), 'confusion': cm.tolist(), 'recall': recall.tolist()}

def return_archives(results, destination, max_bytes):
    results = Path(results)
    destination = Path(destination)
    destination.mkdir(exist_ok=True)
    for p in destination.glob('RETURN_MODELS_[0-9][0-9].zip'):
        p.unlink()
    paths = [p for p in sorted(results.rglob('*')) if p.is_file() and p.name not in ['last.pt', 'attack_last.pt'] and ('.tmp' not in p.name)]
    inventory = [{'file': p.relative_to(results).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths]
    write_json(destination / 'RETURN_INVENTORY.json', inventory)
    reports = [p for p in paths if p.suffix not in ['.pt'] and p.name != 'latents.npz']
    models = [p for p in paths if p not in reports]
    with zipfile.ZipFile(destination / 'RESULTS_REPORTS.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for p in reports:
            z.write(p, p.relative_to(results).as_posix())
        z.write(destination / 'RETURN_INVENTORY.json', 'RETURN_INVENTORY.json')
    groups = []
    current = []
    size = 0
    for p in models:
        if current and size + p.stat().st_size > max_bytes:
            groups.append(current)
            current = []
            size = 0
        current.append(p)
        size += p.stat().st_size
    if current:
        groups.append(current)
    for i, group in enumerate(groups, 1):
        dest = destination / f'RETURN_MODELS_{i:02}.zip'
        with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as z:
            for p in group:
                z.write(p, p.relative_to(results).as_posix())
        if dest.stat().st_size > 120000000:
            raise RuntimeError('Model archive exceeds upload budget')
    write_json(destination / 'ARCHIVE_SHA256.json', {p.name: sha(p) for p in sorted(destination.glob('*.zip'))})
    name = 'SMOKE_RETURN_RESULTS.zip' if destination.name == 'SMOKE_RETURN' else 'RETURN_RESULTS.zip'
    outer = destination.parent / name
    temporary = outer.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in sorted(destination.iterdir()):
            if p.is_file():
                z.write(p, p.name)
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise RuntimeError('Return archive integrity check failed')
    temporary.replace(outer)
    outer.with_suffix('.sha256').write_text(sha(outer) + '  ' + outer.name + '\n')
