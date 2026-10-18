# Numerical experiment inputs

These CSV and compressed CSV files contain reconstruction, classification, convergence, channel-resource and metric-sensitivity measurements from the image-transmission experiments. The default analysis reads them directly:

```bash
python scripts/reproduce_analysis.py
```

The file and column definitions, averaging units and bootstrap procedures are described in [the result schema](../docs/RESULT_SCHEMA.md). File hashes are listed in `SHA256SUMS.txt` and can be checked with `python scripts/verify_inputs.py`.

Obtain the image datasets from the original [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [Kodak](https://r0k.us/graphics/kodak/) providers. The training entry point downloads and verifies CIFAR-10 automatically. Model checkpoints and latent archives are available from the corresponding author upon reasonable request.
