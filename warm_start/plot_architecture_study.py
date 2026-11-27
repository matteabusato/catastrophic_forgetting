#!/usr/bin/env python3
"""Plot train/held-out accuracy and loss for one architecture run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


MODELS = (
    "three_layer_tanh_mlp",
    "wide_relu_mlp",
    "plain_cnn_no_skips",
    "tiny_vit_patch4",
)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    seed_dir = (
        script_dir / "outputs" / "architecture_study" / args.model
        / f"seed_{args.seed:02d}"
    )
    curves_path = seed_dir / "curves.csv"
    rows = read_rows(curves_path)
    if not rows:
        raise SystemExit(f"No metric rows found in {curves_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required; install it in the active environment") from exc

    warm_half = [row for row in rows if row["method"] == "warm_start" and row["phase"] == "half_data"]
    warm_full = [row for row in rows if row["method"] == "warm_start" and row["phase"] == "full_data"]
    cold = [row for row in rows if row["method"] == "cold_start"]

    def plot_series(axis, records, x_values, metric, **kwargs):
        axis.plot(x_values, [float(row[metric]) for row in records], **kwargs)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, constrained_layout=True)
    panels = (
        (axes[0, 0], "train_accuracy", "Training accuracy"),
        (axes[0, 1], "heldout_accuracy", "Held-out accuracy"),
        (axes[1, 0], "train_loss", "Training cross-entropy"),
        (axes[1, 1], "heldout_loss", "Held-out cross-entropy"),
    )

    for axis, metric, title in panels:
        if warm_half:
            plot_series(
                axis, warm_half,
                [int(row["total_epoch"]) for row in warm_half], metric,
                color="#77a9d1", linestyle="--", linewidth=1.5,
                label="Warm: half-fit phase",
            )
        if warm_full:
            plot_series(
                axis, warm_full,
                [int(row["total_epoch"]) for row in warm_full], metric,
                color="#1769aa", linewidth=1.7,
                label="Warm: full-fit phase",
            )
        if cold:
            plot_series(
                axis, cold,
                [int(row["total_epoch"]) + 350 for row in cold], metric,
                color="#e87500", linewidth=1.7,
                label="Cold: full-fit phase (aligned at 350)",
            )
        axis.axvline(350, color="gray", linestyle=":", linewidth=1.2)
        axis.set_title(title)
        axis.set_xlabel("Phase-aligned epoch (cold phase starts at 350)")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 1].set_ylabel("Accuracy")
    axes[1, 0].set_ylabel("Cross-entropy")
    axes[1, 1].set_ylabel("Cross-entropy")
    fig.suptitle(f"{args.model}, seed {args.seed}: warm versus cold")

    output_path = args.output or (seed_dir / "train_heldout_metrics.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"plot={output_path.resolve()}")


if __name__ == "__main__":
    main()
