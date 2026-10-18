from pathlib import Path
import hashlib


def main():
    root = Path(__file__).resolve().parents[1] / 'data'
    entries = (root / 'SHA256SUMS.txt').read_text().splitlines()
    for entry in entries:
        expected, filename = entry.split('  ', 1)
        actual = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Numerical input checksum mismatch: {filename}')
    print(f'Verified {len(entries)} numerical input files')


if __name__ == '__main__':
    main()
