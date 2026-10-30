# Comprehensive fixed-codec evaluation

## Scope and inputs

`configs/evaluation.json` defines a fixed-codec study. Six saved codecs pair training seeds 1001, 1002 and 1003 with defense weights 0 and 0.2. A separate seed-42 codec supplies additional natural-image measurements. Codec weights stay fixed throughout this evaluation.

The CIFAR-10 partition contains 45,000 fitting images, 5000 training-derived validation images and 3600 reporting images at official test indices 6400–9999. Feature normalization and checkpoint/model selection use fitting and validation partitions. Reporting images recur across experimental conditions, so repeated observations are not independent source images.

The complete saved asset bundle must be present in `assets/`, or in the directory supplied through `--assets-dir`. The runner verifies every file against `assets/ASSET_MANIFEST.json` before executing any selected stage or smoke check. The source-only checkout therefore does not provide an immediately runnable full or smoke evaluation. See [the asset inventory](../assets/README.md) for the separately supplied files.

## Experiment stages

| Stage | Configured evaluation |
| --- | --- |
| `attacks` | Seventeen primary predictors and four stationary controls per codec; metric/loss selections; paired larger-budget, high-SNR, packet-set and attribute models |
| `diagnostics` | Latent/observable statistics and retained codec-training loss records |
| `channels` | Rayleigh/Rician and multipath receivers, synchronization, transform reuse and repeated-plaintext controls |
| `baselines` | Common-codec keyed, AES-GCM and linear-BFV mechanisms under equal reserved channel uses and nominal transmit energy |
| `natural` | DIV2K-fitted tile and adjacent four-tile context reconstructors, with native-resolution Kodak reporting |
| `protocol` | Durable nonce reservation, authenticated metadata, replay handling, noisy headers and component timings |

The six stages schedule 21 case-level jobs of unequal duration. The default scientific run requires CUDA. A representative execution on the stated RTX 5080 host completed in approximately 8.2 hours, after startup and initial CIFAR-10 preparation; this is a measured execution, not a runtime guarantee for other systems.

## Run and resume

Use the environment described in [the repository README](../README.md):

```bash
python scripts/run_evaluation.py --device cuda --config configs/evaluation.json --output outputs/evaluation
```

Running the same command resumes a compatible output directory. Completed stages are checked against their stored inventories; interrupted neural fitting resumes from a completed epoch. Configuration, source and input hashes bind an output directory to its execution. Use a different output directory when changing them, and run only one process per output directory.

Stages may be selected explicitly:

```bash
python scripts/run_evaluation.py --device cuda --config configs/evaluation.json --output outputs/channels --stages channels
```

The full asset check still applies. A CPU synthetic integration check is:

```bash
python scripts/run_evaluation.py --device cpu --smoke --config configs/evaluation.json --output outputs/smoke
```

It checks software execution using synthetic images and short fitting schedules; its outputs are marked as non-scientific. Unit tests can exercise individual components without the large saved asset bundle.

## Selection and interpretation

Primary reconstruction checkpoints and model families maximize validation valid-window SSIM. Class-label inference maximizes validation accuracy; the binary task maximizes balanced accuracy; attribute regression minimizes validation normalized MSE. MSE- and cross-entropy-selected alternatives remain available for the criterion comparison. Optimization losses and selection metrics are recorded independently.

Public-family fitting generates fresh signs, permutations and channel noise from known fitting images without using the evaluated secret key in the prediction model. Evaluation uses the actual keyed generator. Budget snapshots select the best available validation checkpoint along one 120-epoch trajectory, rather than independent training at each budget.

The high-SNR study uses fresh fits at 20, 30 and 40 dB and without noise. Packet-set models receive up to 16 fresh-nonce transmissions of the same image. Natural-image models train on 800 DIV2K frames, select on 100 separate frames and report on 24 Kodak frames; joint context spans four adjacent tiles. These access and model bounds delimit the empirical privacy measurements.

The channel model uses packet-constant taps and a Doppler-related carrier-frequency-offset proxy. The common-codec digital references use the specified quantizers and uncoded hard-decision modulation. The linear-BFV implementation is an adapted mechanism comparison, not a complete trained DeepJSCEC reproduction or a security-matched cryptographic parameter set. Metadata authentication and replay processing concern the header; continuous payload integrity is a separate service.

## Outputs and collection

The output directory contains `RUN_MANIFEST.json`, `STATUS.json`, the environment record, per-stage summaries, validation selections, fitting histories and per-image numerical records. Stage completion records preserve file hashes. Status is updated at stage boundaries rather than after every training epoch.

The runner produces a numerical-results ZIP and checksum when execution completes. Large trained model files and full image-prediction tensors remain in the output directory; `RESULTS_INVENTORY.json` identifies those retained files and their hashes. Numerical metric vectors are exported in full, while large prediction arrays are represented by their first 16 images. Preserve the complete output directory when retaining the scientific record.

Additional collection commands use the same configuration/output path:

```bash
python scripts/run_evaluation.py --config configs/evaluation.json --output outputs/evaluation --archive-only
python scripts/run_evaluation.py --config configs/evaluation.json --output outputs/evaluation --collect-models
python scripts/run_evaluation.py --config configs/evaluation.json --output outputs/evaluation --collect-full-results
```

The first command exports numerical records, the second exports selected model checkpoints in independently readable ZIP parts, and the third collects complete scientific outputs. Downloaded datasets and reproducible crop caches are excluded from these archives.

## Compact numerical replay

The aggregate tables in `data/evaluation/` retain the measured scientific results independently of the large assets. Run `python scripts/plot_evaluation.py` to generate five PNG/SVG grid figures and interpretation notes in `results/evaluation/`. No fitting occurs. The [table schema](../data/evaluation/README.md) describes units, averaging and selection. `provenance.json` identifies the recorded execution modules and source records by hash; these identify the historical execution rather than the publication-adapted module paths.
