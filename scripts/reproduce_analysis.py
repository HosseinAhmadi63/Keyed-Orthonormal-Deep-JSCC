from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from keyed_jscc.analysis import reproduce_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "generated")
    parser.add_argument("--no-plots", action="store_true")
    arguments = parser.parse_args()
    checks = reproduce_analysis(arguments.data_dir, arguments.output_dir, plots=not arguments.no_plots)
    for row in checks:
        print(f"{row['analysis']}: {row['reference_values_checked']} values, maximum absolute error {row['maximum_absolute_error']:.3g}")
    print(f"Results saved to {arguments.output_dir}")


if __name__ == "__main__":
    main()
