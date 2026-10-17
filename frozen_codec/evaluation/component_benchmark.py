from pathlib import Path
import json, time, sys, platform, subprocess
import numpy as np
import torch
from crypto_scrambler import CryptoScrambler
from prepare_evaluations import ROOT, BASE, DATA, RESULTS
from models.autoencoder import Encoder, Decoder

def main():
    torch.set_num_threads(2)
    a = np.load(DATA / 'basic_holdout.npz', allow_pickle=False)
    images = torch.from_numpy(a['test_images'][:128].copy()).float() / 255
    z = a['test_latents'][:128]
    u = z / np.sqrt(np.maximum(np.mean(z * z, axis=1, keepdims=True), 1e-06))
    keys = {'key_hex': 'aa47c63ebedee13713bbd1e4ed31f1849fed3be178b9fa04ef79da31fdc4a212', 'wrong_key_hex': 'eb47ecb539e8f5f3e365eafa8213f19aa7979fb1fea47f2f63abccff2f17193e'}
    c = CryptoScrambler(bytes.fromhex(keys['key_hex']), 256)
    nonces = [20509796110279700719756902400 + i for i in range(128)]
    p = c.parameters_batch(nonces)
    s = p.forward(u)
    ck = torch.load(BASE / 'pt/best_main.pt', map_location='cpu', weights_only=True)
    enc = Encoder(256).eval()
    enc.load_state_dict(ck['enc'])
    dec = Decoder(256).eval()
    dec.load_state_dict(ck['dec'])
    obs = torch.from_numpy(u)
    tests = {'parameter_generation': lambda: c.parameters_batch(nonces), 'scramble_FWHT_signs_permutation': lambda: p.forward(u), 'inverse_FWHT_signs_permutation': lambda: p.inverse(s), 'encoder': lambda: enc(images), 'decoder': lambda: dec(obs)}
    output = {}
    with torch.inference_mode():
        for name, func in tests.items():
            for _ in range(2):
                func()
            measured = []
            for _ in range(5):
                begin = time.perf_counter()
                func()
                measured.append(time.perf_counter() - begin)
            med = float(np.median(measured))
            output[name] = {'seconds_all_repeats': measured, 'median_batch_ms': 1000 * med, 'median_per_image_ms': 1000 * med / 128}
            print(name, output[name]['median_batch_ms'], flush=True)
    try:
        brand = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    except Exception:
        brand = platform.machine()
    result = {'batch_size': 128, 'latent_n': 256, 'cpu_threads_torch': 2, 'warmup_calls': 2, 'timed_calls': 5, 'hardware': brand, 'platform': platform.platform(), 'numpy': np.__version__, 'torch': torch.__version__, 'noise_used_for_timing': False, 'codec_retrained': False, 'components': output}
    (RESULTS / 'component_benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
if __name__ == '__main__':
    main()
