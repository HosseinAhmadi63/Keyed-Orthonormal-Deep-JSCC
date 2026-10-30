# Keyed Orthonormal Deep JSCC

Python implementation for **Keyed Orthonormal Latent Scrambling for Wireless Image Transmission with Deep JSCC**, by Hossein Ahmadi, Fatemeh Hashemi, Amirhossein Ahmadi and Ali Kuhestani.

A shared secret key and a public packet nonce select an energy-preserving transformation of an image codec's latent representation. The intended receiver regenerates and inverts that transformation before decoding. The software implements paired codec training, reconstruction and inference attacks, fading and multipath receivers, protection-mechanism comparisons, and persistent authenticated metadata handling. Privacy measurements concern the specified attacker families, fitting budgets and observation conditions.

## Workflows

| Workflow | Entry point | Required inputs |
| --- | --- | --- |
| CPU analysis of complementary numerical records | `scripts/reproduce_analysis.py` | Files in `data/` |
| CPU figure replay for the comprehensive evaluation | `scripts/plot_evaluation.py` | Compact tables in `data/evaluation/` |
| Paired codec and attacker training | `scripts/run_experiments.py` | Official CIFAR-10 download |
| Comprehensive fixed-codec evaluation | `scripts/run_evaluation.py` | Saved experiment assets plus CIFAR-10 and DIV2K downloads |

The saved codecs and latent archives required by the comprehensive evaluation are distributed separately. A code-only checkout cannot run that evaluation, including its synthetic smoke mode, until the complete asset bundle is placed in `assets/`. See [asset requirements](assets/README.md). The bundle contains approximately 602 MiB of model, latent and Kodak packet files and is available from the corresponding author upon reasonable request.

## CPU numerical analysis

With Python 3.12 in a virtual environment:

```bash
python -m pip install -e ".[dev]"
python scripts/reproduce_analysis.py
python scripts/verify_inputs.py
python scripts/plot_evaluation.py
```

This workflow recomputes tables and grid figures from the recorded numerical inputs under `data/` and writes outputs to `results/generated/`. The complementary inputs have their own [field and averaging definitions](docs/RESULT_SCHEMA.md). The comprehensive evaluation tables in [data/evaluation](data/evaluation/README.md) supply five additional grid figures under `results/evaluation/`; replaying them requires neither datasets nor model weights. Their exporter, `scripts/export_evaluation_tables.py`, regenerates the compact tables from complete numerical records.

## GPU evaluation

The tested scientific environment uses Python 3.12, PyTorch 2.8.0 and CUDA 12.8 on an NVIDIA RTX 5080 with approximately 16 GiB of GPU memory. Install the CUDA-enabled PyTorch build and the pinned evaluation dependencies in a separate environment:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-evaluation.txt
python -m pip install -e . --no-deps
python scripts/hardware_check.py
```

After obtaining the required saved assets:

```bash
python scripts/run_evaluation.py --device cuda --config configs/evaluation.json --output outputs/evaluation
```

The configuration covers metric-aligned selection, 30/60/120-epoch attack budgets, higher-SNR and noiseless interception, joint observations of the same image, class/attribute inference, natural-image context attacks, fading/multipath, transform reuse, AES-GCM and linear-BFV mechanisms, and nonce/header processing. See [evaluation definitions and commands](docs/EVALUATION.md).

For codec training and complementary controls, see [experiment commands](docs/EXPERIMENTS.md). [PyCharm instructions](docs/PYCHARM.md) describe selecting the project environment and running its entry points.

## Tests

After installing the evaluation environment:

```bash
python -m pytest
```

Unit tests use synthetic fixtures. One saved-codec integration test is skipped when the separate assets are absent; set `KEYED_JSCC_ASSETS` to their directory to enable it. The evaluation runner's smoke mode is a separate integration check and requires the complete asset bundle:

```bash
python scripts/run_evaluation.py --device cpu --smoke --config configs/evaluation.json --output outputs/smoke
```

Synthetic outputs are software checks and are explicitly distinguished from scientific experiment results.

Published module paths and source fingerprints differ from the recorded execution package. The numerical implementation, seeds and generator domain identifiers are preserved. Use a fresh output directory for this software version; hash-locked outputs from a different source snapshot are not resumable.

## Project structure

```text
assets/                  Asset requirements, hashes and lightweight fixtures
configs/                 Training and evaluation settings
data/                    Recorded numerical analysis inputs
docs/                    Setup, experiment definitions and result fields
evaluation/              Fixed-codec attack, channel and protocol evaluations
experiments/             Complementary codec and interface controls
scripts/                 Analysis, training, evaluation and hardware entry points
src/keyed_jscc/           Keyed transform, codec models, fitting and analysis
tests/                   Numerical, model and protocol tests
```

## Data, models and citation

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Kodak](https://r0k.us/graphics/kodak/) remain available from their original providers under their respective terms. CIFAR-10 and DIV2K are downloaded by the evaluation code when absent. The Kodak packet cache is part of the separately supplied experiment assets. Complete numerical records, trained weights and latent archives are available from [the corresponding author](mailto:hossein.ahmadi@polito.it) upon reasonable request.

Published key values are public reproduction fixtures. Operational transmissions require independently generated secret keys and durable, unique nonce allocation.

Software citation metadata are in [CITATION.cff](CITATION.cff). The implementation uses the [MIT License](LICENSE).
