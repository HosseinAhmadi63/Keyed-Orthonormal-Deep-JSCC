# Numerical inputs and results

The files in `data/` contain recorded numerical outcomes. Running `python scripts/reproduce_analysis.py` reads these inputs and writes recomputed tables and figures to `results/generated/`.

## Matched codec study

`matched_per_image.csv.gz` contains 21,600 rows: 3600 held-out images for each of six codecs. `case` identifies the training seed and defense weight; `test_index` identifies the CIFAR-10 test image, from 6400 through 9999. Each outcome has already been averaged across the nine combinations of three keys and three independent noise draws for that image.

The `bob_*`, `eve_*` and `prior_*` columns describe intended-receiver reconstruction, selected-attacker reconstruction and the fitting-image mean predictor. `mse` is mean squared error on images scaled to [0,1]. `ssim_valid` is structural similarity measured on valid image windows; `ssim_legacy` uses the zero-padded implementation. `accuracy` is the fraction of the nine repeated selected-classifier predictions that matched the image's label.

Dataset PSNR is `-10 log10(mean(mse))` for unit image dynamic range. Averaging per-image PSNR gives a different quantity and is not used for the principal comparison.

`key_noise_cells.csv` contains the 54 separate codec/key/noise evaluations. `codec_selections.csv` identifies the validation-selected codec epochs, classifiers and reconstructors. `attacker_candidates.csv` contains 102 candidate fits; `positive_controls.csv` contains 18 identity, permutation or correct-access controls. Epoch-level measurements appear in `codec_training_curves.csv` and `attacker_training_curves.csv`.

## Uncertainty and pairing

Matched-study intervals resample source images with 2000 bootstrap draws while retaining their averaged key/noise outcomes. They condition on the fitted models and recorded channel evaluations. Paired defense-weight comparisons match training seeds 1001, 1002 and 1003; their individual differences remain visible in `paired_seed_contrasts.csv`.

Kodak resampling uses the 24 original images as units. Equal-resource resampling uses transmitted groups, keeping the partial final group and the paired schemes together. These analyses and SSIM sensitivity use 10,000 bootstrap draws. Seeds and sample counts are written into the relevant numerical outputs or specified in the analysis functions.

## Complementary measurements

| Input | Measurements |
| --- | --- |
| `equal_resource_per_image.csv.gz` | Per-image reconstruction for keyed and reference schemes, indexed by source SNR, group size and group index |
| `resource_profiles.csv` | Actual channel uses, coded-header cost, signal-energy accounting, payload SNR and failed groups |
| `header_monte_carlo.csv` | Header errors, CRC-detected failures and undetected metadata errors by SNR |
| `ssim_per_image.csv.gz` | Four SSIM definitions applied to the same CIFAR-10 or Kodak reconstruction |
| `kodak_per_image.csv` | MSE and SSIM for 24 Kodak frames and five evaluated models |
| `reference_*_values.csv` | Numerical targets used to check the recomputed matched, resource, Kodak and SSIM results |

`numerical_checks.csv` reports the number of reference values checked and the largest absolute numerical difference. Empty CSV fields denote metrics that do not apply to a particular candidate, rather than zero-valued observations.
