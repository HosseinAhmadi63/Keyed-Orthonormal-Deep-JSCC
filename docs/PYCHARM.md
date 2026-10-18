# PyCharm setup

Use Python 3.12 and open the repository folder as a PyCharm project.

1. Select **Settings → Project → Python Interpreter → Add Interpreter → Local Interpreter**.
2. Create a virtual environment named `.venv` using Python 3.12.
3. Open the project terminal and run `python -m pip install -e ".[dev]"`.
4. Run `python scripts/reproduce_analysis.py`, followed by `python -m pytest`.

The analysis uses the recorded numerical inputs in `data/`. Its generated tables and figures are written to `results/generated/`.

## Run configurations

For a saved-analysis configuration, select `scripts/reproduce_analysis.py` as the script and the repository root as the working directory. Use the project's `.venv` interpreter.

For training, select `scripts/run_experiments.py`, set parameters to `--device cuda --config configs/experiment.json`, and use the repository root as the working directory. First complete [GPU setup](GPU_SETUP.md).

## Terminal installation

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts/reproduce_analysis.py
python -m pytest
```

macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/reproduce_analysis.py
python -m pytest
```

The optional `environment.yml` provides a Python 3.12 Conda environment. `requirements.txt` records the dependency versions used for the portable numerical-analysis environment.
