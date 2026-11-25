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
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to make these plots") from exc

    baseline_rows = read_rows(args.seed_dir / "curves.csv")
    lr_rows = continuation_map(read_rows(args.seed_dir / "lr_rewarm" / "curves.csv"))
    project_rows = continuation_map(
        read_rows(args.seed_dir / "lr_rewarm_project" / "curves.csv")
    )
    warm = epoch_map(baseline_rows, "warm_start")
    cold = epoch_map(baseline_rows, "cold_start")
    curves = {
        "Cold start": (cold, "#555555"),
        "Warm baseline": (warm, "#2a9d8f"),
        "Warm + LR re-warming": (lr_rows, "#e76f51"),
        "Warm + LR re-warming + projection": (project_rows, "#457b9d"),
    }
    if not warm or not cold:
        raise ValueError("Baseline curves.csv must contain warm_start and cold_start rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy view: solid lines are training accuracy (left axis); dashed
    # lines are held-out accuracy (right axis). Intervention lines reuse the
    # shared warm-start history through epoch 350 before their own continuation.
    fig, train_ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    heldout_ax = train_ax.twinx()
    for label, (source, color) in curves.items():
        if label.startswith("Warm +"):
            train_epochs, train_values = stitched_curve(warm, source, "train_accuracy")
            test_epochs, test_values = stitched_curve(warm, source, "heldout_accuracy")
        else:
            train_epochs = sorted(source)
            train_values = [float(source[e]["train_accuracy"]) for e in train_epochs]
            test_epochs = sorted(source)
            test_values = [float(source[e]["heldout_accuracy"]) for e in test_epochs]
        train_ax.plot(train_epochs, train_values, color=color, linewidth=1.8, label=label)
        heldout_ax.plot(
            test_epochs, test_values, color=color, linewidth=1.8, linestyle="--", alpha=0.9
        )

    train_ax.axvline(350, color="black", linestyle=":", linewidth=1)
    train_ax.axvline(700, color="black", linestyle="-.", linewidth=1, alpha=0.65)
    train_ax.text(355, 0.03, "full data added", rotation=90, va="bottom", fontsize=9)
    train_ax.text(705, 0.03, "original 700-epoch horizon", rotation=90, va="bottom", fontsize=9)
    train_ax.set(xlim=(1, 1050), ylim=(0, 1), xlabel="Epoch", ylabel="Training accuracy")
    heldout_ax.set(ylim=(0, 1), ylabel="Held-out accuracy")
    train_ax.set_title("Warm vs cold starting")
    train_ax.text(
        0.01, 0.98, "seed 0 · solid: train · dashed: held-out",
        transform=train_ax.transAxes, va="top", fontsize=9,
    )
    train_ax.grid(alpha=0.2)
    train_ax.legend(loc="lower right", frameon=True)
    fig.savefig(args.output_dir / "seed0_accuracy_comparison.png", dpi=180)
    plt.close(fig)

    # Accuracy generalization gap, train minus held-out, puts the target effect
    # directly on one scale and makes the extra 350 epochs visible.
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for label, (source, color) in curves.items():
        if label.startswith("Warm +"):
            epochs, train_values = stitched_curve(warm, source, "train_accuracy")
            _, test_values = stitched_curve(warm, source, "heldout_accuracy")
        else:
            epochs = sorted(source)
            train_values = [float(source[e]["train_accuracy"]) for e in epochs]
            test_values = [float(source[e]["heldout_accuracy"]) for e in epochs]
        gaps = [train - test for train, test in zip(train_values, test_values)]
        ax.plot(epochs, gaps, color=color, linewidth=1.8, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(350, color="black", linestyle=":", linewidth=1, label="full data added")
    ax.axvline(700, color="black", linestyle="-.", linewidth=1, alpha=0.65)
    ax.set(xlim=(1, 1050), xlabel="Epoch", ylabel="Train accuracy − held-out accuracy")
    ax.set_title("Seed-0 generalization gap")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", frameon=True)
    fig.savefig(args.output_dir / "seed0_generalization_gap.png", dpi=180)
    plt.close(fig)

    print(f"Saved {args.output_dir / 'seed0_accuracy_comparison.png'}")
    print(f"Saved {args.output_dir / 'seed0_generalization_gap.png'}")


if __name__ == "__main__":
    main()
