# Saved experiment assets

The comprehensive fixed-codec evaluation requires the files listed in `ASSET_MANIFEST.json`. File size and SHA-256 checks identify the recorded inputs. The source repository supplies lightweight manifests, split indices, public test keys, validation vectors and codec-training curves. Large files are distributed separately.

## Separately supplied files

| Files | Count | Purpose |
| --- | ---: | --- |
| `seed1001_lambda0*/best_codec.pt`, `seed1002_lambda0*/best_codec.pt`, `seed1003_lambda0*/best_codec.pt` | 6 | The paired saved codecs |
| The corresponding `latents.npz` files in those six case directories | 6 | CIFAR training/reporting latents from the saved encoders |
| `frozen_seed42/best_codec.pt` | 1 | Fixed reference codec |
| `kodak_packets.npz` | 1 | Native-resolution Kodak tile inputs and their recorded metadata |

These 14 files total 630,764,185 bytes, approximately 602 MiB. Obtain the saved experiment assets from [the corresponding author](mailto:hossein.ahmadi@polito.it) upon reasonable request. Preserve the paths from the manifest when placing them inside `assets/`; do not change their contents or replace them with newly trained weights when replaying the recorded fixed-codec study.

Alternatively, pass `--assets-dir PATH_TO_ASSETS` to use an external asset directory with the same layout.

The evaluation runner verifies the entire manifest before executing selected stages, including its smoke mode. A source-only checkout therefore cannot run the full evaluation or smoke command without this bundle. Unit tests use small synthetic fixtures; the saved-codec integration test is skipped when its model and latent files are absent. It can be enabled by setting `KEYED_JSCC_ASSETS` to the saved-asset directory.

## Lightweight files

- `ASSET_MANIFEST.json`: complete input file inventory and hashes.
- `DATASETS.json`: public dataset locations and archive checksums.
- `split_manifest.npz`: recorded fitting, validation and reporting indices.
- `experiment_keys.json` and `legacy_experiment_keys.json`: public reproduction fixtures.
- `protocol_vectors.json`: deterministic protocol validation examples.
- `codec_training_curves.csv`: retained codec-fitting records.

The public reproduction keys are unsuitable for protecting operational traffic. Use independently generated secret keys with persistent, unique nonce allocation in a deployment.

CIFAR-10 and DIV2K are obtained from their original providers by the evaluation code. Full image downloads, crop caches, model checkpoints and latent archives are excluded from Git tracking. Dataset terms remain those of the original providers.
