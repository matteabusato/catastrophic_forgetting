"""Plot both attention CE and output MSE for selected saved runs."""

import argparse
from pathlib import Path

from src.plotting import plot_run_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_run_metrics(args.run_dirs, args.output)
    print(f"Saved CE and MSE plots to {args.output}")


if __name__ == "__main__":
    main()
