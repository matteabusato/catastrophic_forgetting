#!/usr/bin/env python3
"""Plot mean and across-seed spread for long-run CIFAR-10 metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


METHODS = ("warm_start", "cold_start", "lr_rewarm")
COLORS = {
    "warm_start": "#1f5f99",
    "cold_start": "#e76f00",
    "lr_rewarm": "#2a9d8f",
}
METRICS = (
    ("train_accuracy", "Training accuracy", "Accuracy"),
    ("heldout_accuracy", "Held-out accuracy", "Accuracy"),
    ("train_loss", "Training cross-entropy", "Cross-entropy"),
    ("heldout_loss", "Held-out cross-entropy", "Cross-entropy"),
)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=script_dir / "outputs" / "long_run"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary_path = args.input_dir / "trajectory_summary.csv"
    if not summary_path.is_file():
        raise SystemExit(
            f"Missing {summary_path}; check that the long-run job completed "
            "and wrote trajectory_summary.csv."
        )
    rows = read_rows(summary_path)
    if not rows:
        raise SystemExit(f"No trajectory rows found in {summary_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required; install it in the active environment") from exc

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, constrained_layout=True)
    for axis, (metric, title, ylabel) in zip(axes.flat, METRICS):
        for method in METHODS:
            method_rows = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: int(row["full_data_epoch"]),
            )
            if not method_rows:
                continue
            epochs = [int(row["full_data_epoch"]) for row in method_rows]
            means = [float(row[f"{metric}_mean"]) for row in method_rows]
            stds = [float(row[f"{metric}_std"]) for row in method_rows]
            color = COLORS[method]
            axis.plot(epochs, means, color=color, linewidth=1.8, label=method)
            axis.fill_between(
                epochs,
                [mean - std for mean, std in zip(means, stds)],
                [mean + std for mean, std in zip(means, stds)],
                color=color,
                alpha=0.18,
                linewidth=0,
            )
        axis.set_title(title)
        axis.set_xlabel("Full-data training epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend()

    fig.suptitle("Long-run warm, cold, and LR-rewarmed CIFAR-10 runs")
    output_path = args.output or (args.input_dir / "long_run_metrics.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"plot={output_path.resolve()}")


if __name__ == "__main__":
    main()
