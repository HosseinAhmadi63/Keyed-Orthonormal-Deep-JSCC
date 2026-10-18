from pathlib import Path
import os
import concurrent.futures, subprocess, hashlib, time
folder = Path(os.environ.get('JSCC_ASSETS', str(Path(__file__).resolve().parents[1]))) / 'evaluation/data'
folder.mkdir(parents=True, exist_ok=True)
size = 170498071
parts = 16
url = 'https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'

def get(i):
    lo = size * i // parts
    hi = size * (i + 1) // parts - 1
    p = folder / f'part{i:02d}'
    if p.exists() and p.stat().st_size == hi - lo + 1:
        return p
    r = subprocess.run(['curl', '-sS', '--fail', '--max-time', '210', '--range', f'{lo}-{hi}', url, '-o', str(p)], capture_output=True, text=True)
    if r.returncode or not p.exists() or p.stat().st_size != hi - lo + 1:
        raise RuntimeError(f'part{i}: {r.stderr} size {(p.stat().st_size if p.exists() else 0)}')
    print(f'Completed dataset part {i + 1}/16', flush=True)
    return p
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    ps = list(pool.map(get, range(parts)))
with (folder / 'cifar-10-python.tar.gz').open('wb') as out:
    for p in ps:
        out.write(p.read_bytes())
actual = hashlib.md5((folder / 'cifar-10-python.tar.gz').read_bytes()).hexdigest()
assert actual == 'c58f30108f718f92721af3b95e74349a', actual
print('Official CIFAR-10 archive verified: ' + actual, flush=True)
for p in ps:
    p.unlink()
