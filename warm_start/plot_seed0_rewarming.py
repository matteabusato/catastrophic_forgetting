#!/usr/bin/env python3
"""Plot seed-0 baseline and ELR re-warming comparison curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def read_rows(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metrics file not found: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def epoch_map(rows: List[dict], method: str) -> Dict[int, dict]:
    return {
        int(row["epoch"]): row
        for row in rows
        if row.get("method") == method
    }


def continuation_map(rows: List[dict]) -> Dict[int, dict]:
    return {int(row["epoch"]): row for row in rows}


def stitched_curve(
    warm_rows: Dict[int, dict], continuation_rows: Dict[int, dict], metric: str
) -> tuple[List[int], List[float]]:
    # All intervention runs share this exact warm-start half-data trajectory.
    epochs = sorted(set(range(1, 351)) | set(continuation_rows))
    values = []
    for epoch in epochs:
        row = warm_rows[epoch] if epoch <= 350 else continuation_rows[epoch]
        values.append(float(row[metric]))
    return epochs, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parent / "outputs" / "seed_00"
    parser.add_argument("--seed-dir", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=default_root)
    parser.add_argument(
        "--exclude-projection",
        action="store_true",
        help="Omit the noisy LR re-warming plus projection run from both plots.",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to make these plots") from exc

    baseline_rows = read_rows(args.seed_dir / "curves.csv")
    lr_rows = continuation_map(read_rows(args.seed_dir / "lr_rewarm" / "curves.csv"))
    warm = epoch_map(baseline_rows, "warm_start")
    cold = epoch_map(baseline_rows, "cold_start")
    curves = {
        "Cold start": (cold, "#555555"),
        "Warm baseline": (warm, "#2a9d8f"),
        "Warm + LR re-warming": (lr_rows, "#e76f51"),
    }
    if not args.exclude_projection:
        project_rows = continuation_map(
            read_rows(args.seed_dir / "lr_rewarm_project" / "curves.csv")
        )
        curves["Warm + LR re-warming + projection"] = (project_rows, "#457b9d")
    if not warm or not cold:
        raise ValueError("Baseline curves.csv must contain warm_start and cold_start rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_no_projection" if args.exclude_projection else ""

    # Make one independent single-axis figure per accuracy metric.
    for metric, ylabel, filename in (
        ("train_accuracy", "Training accuracy", f"seed0_training_accuracy{suffix}.png"),
        ("heldout_accuracy", "Test accuracy", f"seed0_test_accuracy{suffix}.png"),
    ):
        fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
        for label, (source, color) in curves.items():
            if label.startswith("Warm +"):
                epochs, values = stitched_curve(warm, source, metric)
            else:
                epochs = sorted(source)
                values = [float(source[e][metric]) for e in epochs]
            ax.plot(epochs, values, color=color, linewidth=1.8, label=label)

        ax.axvline(350, color="black", linestyle=":", linewidth=1)
        ax.axvline(700, color="black", linestyle="-.", linewidth=1, alpha=0.65)
        ax.text(355, 0.03, "full data added", rotation=90, va="bottom", fontsize=9)
        ax.text(705, 0.03, "original 700-epoch horizon", rotation=90, va="bottom", fontsize=9)
        ax.set(xlim=(1, 1050), ylim=(0, 1), xlabel="Epoch", ylabel=ylabel)
        ax.set_title("Warm vs cold starting")
        ax.text(0.01, 0.98, "seed 0", transform=ax.transAxes, va="top", fontsize=9)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", frameon=True)
        plot_path = args.output_dir / filename
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
