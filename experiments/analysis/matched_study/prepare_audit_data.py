import argparse, hashlib, pickle, tarfile
from pathlib import Path
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    h = hashlib.md5()
    with a.archive.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    assert h.hexdigest() == 'c58f30108f718f92721af3b95e74349a', 'Official archive checksum mismatch; no pickle is loaded.'
    xs = []
    ys = []
    with tarfile.open(a.archive, 'r:gz') as t:
        for i in range(1, 6):
            d = pickle.load(t.extractfile(f'cifar-10-batches-py/data_batch_{i}'), encoding='bytes')
            xs.append(d[b'data'])
            ys.extend(d[b'labels'])
        d = pickle.load(t.extractfile('cifar-10-batches-py/test_batch'), encoding='bytes')
    y = np.array(ys, np.int64)
    images = np.concatenate(xs).reshape(-1, 3, 32, 32)
    x = d[b'data'].reshape(-1, 3, 32, 32)[6400:].copy()
    yt = np.array(d[b'labels'], np.int64)[6400:].copy()
    with np.load(a.split) as d:
        fit = d['fitting']
    prior = (images[fit].mean(0, dtype=np.float64) / 255).astype(np.float32)
    a.output.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(a.output, y_train=y, y_test=yt, test_images=x, fitting_mean_image=prior)
    print({'official_archive_md5': h.hexdigest(), 'train_n': len(y), 'holdout_n': len(yt)}, flush=True)
if __name__ == '__main__':
    main()
