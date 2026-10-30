# Recorded evaluation tables

These compact tables contain aggregate numerical records for the keyed latent interface. They can be inspected without image datasets, trained weights or GPU access. The accompanying source code performs the corresponding experiments when their required assets are available.

## Files and units

| Files | Contents |
|---|---|
| `attack_candidates.csv`, `attack_selected.csv` | Every primary predictor and the validation-selected suite results for all six codecs. |
| `attack_controls.csv`, `image_priors.csv` | Stationary identity/reused-transform controls and image/class priors. |
| `attack_fitting.csv` | All 174 primary, control and extended fits, with epochs, source-image counts, optimizer updates and simulated packet draws. |
| `attack_budget.csv` | Best available checkpoints within 30, 60 and 120 epochs along each optimization trajectory. |
| `attack_snr.csv` | Freshly fitted high-SNR and noiseless predictors, with key and noise-draw identifiers. |
| `attack_packet_sets.csv` | Joint predictions from 1, 2, 4, 8 or 16 fresh packets of the same source image. |
| `attack_binary.csv`, `attack_attributes.csv` | Animal/vehicle inference and three direct image-attribute estimates. |
| `channel_outcomes.csv` | Matched identity/keyed receiver measurements by codec, channel configuration, SNR and draw. |
| `reuse_recovery.csv`, `reuse_averaging.csv` | Diverse known-image inverse fits and repeated reception of one identical plaintext. |
| `natural_frames.csv`, `natural_summary.csv`, `natural_selections.csv` | Native-frame Kodak outcomes, frame-bootstrap intervals and DIV2K-based model selections. |
| `protection_operating_points.csv`, `protection_timings.csv` | Common-codec protection mechanisms, reserved/active resources, utility, compact inference and CPU cryptographic-operation timings. |
| `header_reliability.csv`, `header_resources.csv` | Header decoding trials and complete-group communication accounting. |
| `protocol_timings.csv`, `protocol_processing.csv` | Measured component timings and separately derived processing sums. |
| `codec_loss_diagnostics.csv` | Online-head cross-entropy and uniform-target loss over the last five codec-training epochs. |
| `experiment_config.json`, `protocol_checks.json`, `provenance.json` | Fixed experiment settings, protocol-check outcomes, input-record hashes and table hashes. |

Accuracy and AUROC values are fractions, not percentages. Reconstruction SSIM uses valid windows unless the column explicitly says `legacy`; that column denotes the zero-padded secondary score. Dataset PSNR is `-10 log10(mean MSE)`, rather than the average of image-wise PSNR values. Empty cells indicate quantities that do not apply to that task.

`criterion=metric` selects the reporting-task validation metric; `criterion=loss` selects MSE or cross-entropy. The `task_metric` column identifies the task's primary reporting metric, independently of that criterion. Binary inference uses balanced accuracy, and attribute fitting uses normalized MSE. `suite_selected` indicates validation-based model-family selection.

The CIFAR reporting set contains 3600 source images reused across conditions. Keys, channel draws, candidate predictors and packet counts do not create additional independent source images. Natural-image reporting contains 24 complete Kodak frames, with zero-based frame indices; its intervals resample whole frames. Budget snapshots share optimization trajectories. Fit source-image counts and simulated packet presentations are distinct quantities.

Protocol timing loop items are tiles, headers or counters reserved in a batch, as labeled. A durable reservation performs one transaction for the batch, so its per-counter cost is amortized. Authentication, Hamming processing and replay entries measure a header, with a per-tile group cost obtained by dividing by group size. The processing table sums component medians; it does not measure end-to-end radio latency.

Protection operating points reserve equal nominal energy and total complex uses, with actual active uses and measured finite-waveform energy reported separately. Digital references use the specified uncoded modulation and compact hard-decision predictors. These are common fixed-codec mechanism measurements. Known-image recovery counts describe the measured codec/channel/estimator conditions, rather than a universal safe reuse threshold.

To export these tables from a complete numerical-results directory:

```bash
python scripts/export_evaluation_tables.py --results PATH_TO_RESULTS --output data/evaluation
```

The export uses Python's standard library. `--module-directory` optionally records hashes of the executed evaluation modules. Original datasets and full checkpoint/prediction assets are distributed separately under the availability conditions stated in the project documentation.
