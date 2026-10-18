# Experiment commands

## Saved numerical analysis

```bash
python scripts/reproduce_analysis.py
python scripts/verify_inputs.py
python -m pytest
```

Use `--output-dir outputs/analysis` to write a separate analysis run. `--no-plots` recomputes the numerical tables without rendering figures.

## Paired codec and attacker training

```bash
python scripts/run_experiments.py --device cuda --config configs/experiment.json
```

`configs/experiment.json` specifies the three paired codec seeds, two defense weights, 30-epoch codec and attacker budgets, validation stopping, latent dimension 256, and the 12 dB intended-receiver and 20 dB eavesdropper channels. Of the 50,000 CIFAR-10 training images, 45,000 are used for fitting and 5000 for validation. The reported holdout contains 3600 CIFAR-10 test images.

The training runner supports `--stage train`, `--stage attacks`, `--output` and `--data`. Keep the same output directory when continuing an experiment. A synthetic execution is available with `--device cpu --smoke --output outputs/smoke`.

## Protocol and header tests

```bash
python -m keyed_jscc.crypto_scrambler
python experiments/analysis/interface_controls/header_resource_benchmark.py --mode selftest --output outputs/header_selftest
python experiments/analysis/interface_controls/header_resource_benchmark.py --mode header --output outputs/header_reliability
```

The protocol tests cover key derivation, Advanced Encryption Standard (AES) counter-mode known answers, transform inversion, power preservation and nonce exhaustion. `configs/protocol_vectors.json` specifies deterministic transform examples and byte-level signatures. Header tests cover the Hamming code, cyclic redundancy check (CRC) and noisy-header decoding. Header experiments require SciPy, included in the training extra.

## Complementary model evaluations

The implementations under `experiments/` cover fixed-codec reconstruction and classification attacks, signed and invariant-aware predictors, transform-reuse and wrong-key controls, SSIM definitions, equal-resource transmission and Kodak transfer. Their model-based evaluations use saved checkpoints and observation caches. Install the training extra before running them.

`JSCC_ASSETS` selects the fixed-codec asset directory, with `models/pt/`, `models/json/` and `evaluation/data/` or `evaluation/results/`. `JSCC_INPUTS` selects image and latent replay inputs, including `cifar_holdout.npz` and `all50k_mean_image.npy`. Obtain the recorded asset bundle from the corresponding author when replaying the fixed-codec study.

In PowerShell, set an asset path with `$env:JSCC_ASSETS = 'C:\path\assets'`. On macOS or Linux, use `export JSCC_ASSETS=/path/to/assets`.

Examples after supplying the assets:

```bash
python experiments/frozen_codec/evaluation/prepare_evaluations.py
python experiments/frozen_codec/evaluation/run_classifier_protocols.py --base /path/to/assets/evaluation --protocols crypto,identity,fixed_crypto
python experiments/frozen_codec/evaluation/reconstruction_attacks.py /path/to/cache_crypto.npz outputs/reconstruction --epochs 30 --extended --device cpu
python experiments/analysis/interface_controls/ssim_sensitivity.py --scope cifar --output outputs/ssim_cifar
python experiments/analysis/interface_controls/header_resource_benchmark.py --mode utility --source-snr 12 --output outputs/resource_12dB
```

The invariant-aware suite accepts explicit observation-cache and split files:

```bash
python experiments/analysis/invariant_attacks/train_attacks.py --cache /path/to/cache_crypto.npz --split /path/to/split_indices.npz --output outputs/invariant --ssim-module experiments/frozen_codec/archived_implementation/ssim_ms.py --suite classifiers --phase all --protocol crypto
```

Kodak frame evaluation accepts the dataset and trained checkpoints explicitly:

```bash
python experiments/frozen_codec/evaluation/kodak_evaluation.py --data /path/to/kodak --checkpoint /path/to/best_main.pt --attackers /path/to/best_response_attackers.pt --output outputs/kodak
```

## Implementation map

| Component | Location |
| --- | --- |
| Key derivation, nonce handling and orthonormal transform | `src/keyed_jscc/crypto_scrambler.py` |
| Image encoder and decoder | `src/keyed_jscc/codec_models.py` |
| Reconstruction and classification attackers | `src/keyed_jscc/attack_models.py`, `fit_attackers.py` |
| Public-family simulation and invariant features | `src/keyed_jscc/public_sampling.py` |
| Paired codec training and repeated evaluation | `src/keyed_jscc/runner.py` |
| Saved numerical analysis | `src/keyed_jscc/analysis.py` |
| Fixed-codec evaluations | `experiments/frozen_codec/evaluation/` |
| Invariant, SSIM and transmission-resource controls | `experiments/analysis/` |
