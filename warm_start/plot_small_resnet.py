#!/usr/bin/env python3
"""Plot train and held-out accuracy from the small ResNet experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "outputs" / "small_resnet" / "seed_00" / "curves.csv",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Curve file not found: {args.input}")
    output_path = args.output or args.input.with_name("accuracy_curves.png")

    import matplotlib.pyplot as plt

    rows = read_rows(args.input)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharex=True, sharey=True)
    series = (
        ("warm_start", "half_data", "Warm: half-data phase", "#165A91", "--"),
        ("warm_start", "full_data", "Warm: full-data phase", "#165A91", "-"),
        (
            "cold_start", "full_data", "Cold: full-data training (aligned at 350)",
            "#E87500", "-",
        ),
    )
    for axis, metric, title in (
        (axes[0], "train_accuracy", "Training accuracy"),
        (axes[1], "heldout_accuracy", "Held-out accuracy"),
    ):
        for method, phase, label, color, linestyle in series:
            points = [
                row for row in rows
                if row["method"] == method and row["phase"] == phase
            ]
            if not points:
                continue
            points.sort(key=lambda row: int(row["total_epoch"]))
            x_offset = 350 if method == "cold_start" else 0
            x = [int(row["total_epoch"]) + x_offset for row in points]
            y = [float(row[metric]) for row in points]
            axis.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=1.7)

        axis.axvline(350, color="0.45", linestyle=":", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Phase-aligned epoch (cold full-data phase starts at 350)")
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend()

    fig.suptitle(
        "CIFAR-10 ResNet [1,1,1,1]: warm vs. cold\n"
        "Warm: 350 half-data + 350 full-data epochs; cold: 350 full-data epochs"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"plot={output_path.resolve()}")


if __name__ == "__main__":
    main()
