# GPU training setup

The recorded six-codec experiment used Python 3.12.10, PyTorch 2.8.0 with CUDA 12.8, NumPy 2.5.3, cryptography 46.0.2 and Matplotlib 3.10.7 on an NVIDIA GeForce RTX 5080 with approximately 16 GB GPU memory.

In the project's Python 3.12 environment on Windows or Linux, install the CUDA 12.8 PyTorch wheel, then the training dependencies:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-training.txt
python -m pip install -e ".[train,dev]"
python scripts/hardware_check.py
```

The PyTorch command follows the [official PyTorch 2.8.0 installation instructions](https://pytorch.org/get-started/previous-versions/#v280). Other devices can use the appropriate official CPU or CUDA wheel.

## Complete experiment

```bash
python scripts/run_experiments.py --device cuda --config configs/experiment.json
```

The configuration trains three matched pairs of codecs for 30 epochs and fits the specified reconstruction and classification attackers. Bob's signal-to-noise ratio is 12 dB and Eve's is 20 dB. Evaluation includes three keys and three channel draws per codec.

The runner prints epoch progress and saves checkpoints throughout the run. Repeating an unchanged command resumes saved compatible work. Use separate output folders for different settings:

```bash
python scripts/run_experiments.py --device cuda --config configs/experiment.json --output outputs/experiment
```

Fresh training produces its own keys, channel draws and checkpoints. The bundled numerical analysis reads the recorded experiment outcomes; it is the direct route to recomputing the reported tables and comparison plots.
