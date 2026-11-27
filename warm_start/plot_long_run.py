#!/usr/bin/env python3
"""Plot mean and across-seed spread for long-run CIFAR-10 metrics."""

from __future__ import annotations

import argparse
import csv
import math
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


def aggregate_seed_curves(input_dir: Path) -> List[Dict[str, str]]:
    """Build trajectory mean/std rows directly from each seed's raw curves."""
    values: Dict[tuple, Dict[str, List[float]]] = {}
    metric_names = (
        "train_accuracy", "heldout_accuracy", "train_loss", "heldout_loss"
    )
    for curve_path in sorted(input_dir.glob("seed_*/*/curves.csv")):
        for row in read_rows(curve_path):
            method = row.get("method", curve_path.parent.name)
            epoch = int(row["full_data_epoch"])
            bucket = values.setdefault((method, epoch), {m: [] for m in metric_names})
            for metric in metric_names:
                bucket[metric].append(float(row[metric]))

    aggregate = []
    for (method, epoch), metrics in sorted(values.items()):
        row = {"method": method, "full_data_epoch": str(epoch)}
        for metric, observations in metrics.items():
            row_mean = sum(observations) / len(observations)
            row[f"{metric}_mean"] = str(row_mean)
            std = (
                math.sqrt(sum((x - row_mean) ** 2 for x in observations) / (len(observations) - 1))
                if len(observations) > 1
                else 0.0
            )
            row[f"{metric}_std"] = str(std)
            row[f"{metric}_num_seeds"] = str(len(observations))
        aggregate.append(row)
    return aggregate


def read_endpoint_summary(path: Path) -> List[Dict[str, str]]:
    """Adapt across-seed endpoint summaries to the plotter's metric schema."""
    endpoint_rows = read_rows(path)
    metrics = {metric for metric, _, _ in METRICS}
    result = []
    for method in METHODS:
        summary = {"method": method, "full_data_epoch": "0"}
        for row in endpoint_rows:
            if row["method"] == method and row["metric"] in metrics:
                metric = row["metric"]
                summary[f"{metric}_mean"] = row["mean"]
                summary[f"{metric}_std"] = row["std"]
                summary[f"{metric}_num_seeds"] = row["num_seeds"]
        if any(key.endswith("_mean") for key in summary):
            result.append(summary)
    return result


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=script_dir / "outputs" / "long_run"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary_path = args.input_dir / "trajectory_summary.csv"
    endpoint_only = False
    if summary_path.is_file():
        rows = read_rows(summary_path)
    else:
        rows = aggregate_seed_curves(args.input_dir)
        if not rows:
            endpoint_path = args.input_dir / "summary_across_seeds.csv"
            if endpoint_path.is_file():
                rows = read_endpoint_summary(endpoint_path)
                endpoint_only = True
    if not rows:
        raise SystemExit(
            f"No metric rows or summaries found under {args.input_dir}."
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required; install it in the active environment") from exc

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, constrained_layout=True)
    for axis, (metric, title, ylabel) in zip(axes.flat, METRICS):
        if endpoint_only:
            method_rows = [row for row in rows if f"{metric}_mean" in row]
            labels = [row["method"] for row in method_rows]
            means = [float(row[f"{metric}_mean"]) for row in method_rows]
            stds = [float(row[f"{metric}_std"]) for row in method_rows]
            axis.errorbar(range(len(labels)), means, yerr=stds, fmt="o", capsize=4)
            axis.set_xticks(range(len(labels)))
            axis.set_xticklabels(labels, rotation=15)
            axis.set_xlabel("Training arm at saved endpoint")
            axis.set_title(f"{title} (endpoint mean ± std)")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.25)
            continue
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

    title = (
        "Long-run endpoint metrics across seeds"
        if endpoint_only
        else "Long-run warm, cold, and LR-rewarmed CIFAR-10 runs"
    )
    fig.suptitle(title)
    output_path = args.output or (args.input_dir / "long_run_metrics.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"plot={output_path.resolve()}")


if __name__ == "__main__":
    main()
