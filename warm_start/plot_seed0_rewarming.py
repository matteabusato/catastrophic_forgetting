#!/usr/bin/env python3
"""Make the seed-0 warm, LR-scheduled warm, and cold-start comparison plot."""

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


def rows_by_epoch(rows: List[dict], method: str = "") -> Dict[int, dict]:
    return {
        int(row["epoch"]): row
        for row in rows
        if not method or row.get("method") == method
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    seed_dir = Path(__file__).resolve().parent / "outputs" / "seed_00"
    parser.add_argument("--seed-dir", type=Path, default=seed_dir)
    parser.add_argument("--output", type=Path, default=seed_dir / "seed0_warm_vs_warm_lr_vs_cold.png")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to make the plot") from exc

    baseline = read_rows(args.seed_dir / "curves.csv")
    lr_schedule = read_rows(args.seed_dir / "lr_rewarm" / "curves.csv")
    warm_half = rows_by_epoch(baseline, "warm_start")
    warm_full = rows_by_epoch(baseline, "warm_start")
    cold = rows_by_epoch(baseline, "cold_start")
    lr_full = rows_by_epoch(lr_schedule)

    # Plot the shared half-data stage once. After epoch 350, compare the
    # baseline continuation, LR-scheduled continuation, and cold-start arm.
    pretrain_epochs = sorted(e for e, row in warm_half.items() if row["stage"] == "half_data")
    warm_epochs = sorted(e for e, row in warm_full.items() if row["stage"] == "full_data")
    cold_epochs = sorted(cold)
    lr_epochs = sorted(e for e in lr_full if 350 < e <= 700)
    if not pretrain_epochs or not warm_epochs or not cold_epochs or not lr_epochs:
        raise ValueError("Seed-0 baseline and LR-schedule CSVs are incomplete")

    panels = (
        ("train_accuracy", "Train accuracy", (0.4, 1.02)),
        ("heldout_accuracy", "Test accuracy", (0.42, 0.81)),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, constrained_layout=True)
    colors = {
        "pretrain": "#79a9d8",
        "warm": "#1f5f99",
        "lr": "#2a9d8f",
        "cold": "#e76f00",
    }

    for ax, (metric, ylabel, y_limits) in zip(axes, panels):
        ax.plot(
            pretrain_epochs,
            [float(warm_half[e][metric]) for e in pretrain_epochs],
            linestyle="--",
            linewidth=2,
            color=colors["pretrain"],
            label="50% data (pretraining)",
        )
        ax.plot(
            warm_epochs,
            [float(warm_full[e][metric]) for e in warm_epochs],
            linewidth=2,
            color=colors["warm"],
            label="Warm start, then full data",
        )
        ax.plot(
            lr_epochs,
            [float(lr_full[e][metric]) for e in lr_epochs],
            linewidth=2,
            color=colors["lr"],
            label="Warm + LR schedule",
        )
        ax.plot(
            cold_epochs,
            [float(cold[e][metric]) for e in cold_epochs],
            linewidth=2,
            color=colors["cold"],
            label="Cold start",
        )
        ax.axvline(350, color="gray", linestyle=":", linewidth=1.5)
        ax.set(xlim=(0, 700), ylim=y_limits, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", frameon=True, fontsize=9)

    fig.suptitle("Warm vs warm + LR schedule vs cold starting", fontsize=16)
    axes[0].text(356, 0.42, "full data added", rotation=90, va="bottom", fontsize=9, color="gray")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
