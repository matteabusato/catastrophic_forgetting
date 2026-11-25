#!/usr/bin/env python3
"""Analyze layer-wise weight norms in the seed-0 LR re-warming run."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch

import replicate_figure1 as baseline


def checkpoint_weight_norms(path: Path) -> Dict[str, float]:
    checkpoint = torch.load(path, map_location="cpu")
    model = baseline.ResNet(num_classes=10)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {
        name: parameter.detach().double().norm().item()
        for name, parameter in model.named_parameters()
        if name.endswith(".weight") and parameter.ndim >= 2
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_seed_dir = Path(__file__).resolve().parent / "outputs" / "seed_00"
    parser.add_argument("--seed-dir", type=Path, default=default_seed_dir)
    parser.add_argument("--output-dir", type=Path, default=default_seed_dir)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit("matplotlib and numpy are required to make the plot") from exc

    start_checkpoint = args.seed_dir / "warm_after_half.pt"
    run_dir = args.seed_dir / "lr_rewarm"
    curves_path = run_dir / "curves.csv"
    if not curves_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {curves_path}")
    baseline_curves_path = args.seed_dir / "curves.csv"
    if not baseline_curves_path.is_file():
        raise FileNotFoundError(f"Baseline metrics file not found: {baseline_curves_path}")

    with curves_path.open(newline="") as handle:
        curve_rows = list(csv.DictReader(handle))
    curve_by_epoch = {int(row["epoch"]): row for row in curve_rows}
    if not curve_rows:
        raise ValueError(f"No continuation metrics found in {curves_path}")
    with baseline_curves_path.open(newline="") as handle:
        baseline_rows = list(csv.DictReader(handle))
    transition_rows = [
        row for row in baseline_rows
        if row.get("method") == "warm_start" and int(row["epoch"]) == 350
    ]
    if not transition_rows:
        raise ValueError("Could not find the seed-0 warm checkpoint metrics at epoch 350")
    transition_heldout_accuracy = float(transition_rows[0]["heldout_accuracy"])
    transition_lr = float(curve_rows[0]["learning_rate"])

    start_norms = checkpoint_weight_norms(start_checkpoint)
    checkpoints: List[Tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint_epoch_*.pt"):
        match = re.fullmatch(r"checkpoint_epoch_(\d+)\.pt", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    if not checkpoints:
        raise FileNotFoundError(f"No 50-epoch checkpoints found under {run_dir}")

    epochs = [350]
    layer_names = sorted(start_norms)
    norm_history = {name: [1.0] for name in layer_names}
    effective_lr_ratio_history = {
        name: [transition_lr / baseline.LEARNING_RATE] for name in layer_names
    }
    output_rows = []
    for name in layer_names:
        raw_proxy = transition_lr / start_norms[name]
        output_rows.append(
            {
                "global_epoch": 350,
                "full_data_epoch": 0,
                "layer": name,
                "learning_rate": transition_lr,
                "heldout_accuracy": transition_heldout_accuracy,
                "weight_frobenius_norm": start_norms[name],
                "lr_over_norm": raw_proxy,
                "lr_over_norm_ratio_to_baseline_at_epoch350": transition_lr / baseline.LEARNING_RATE,
            }
        )
    for full_data_epoch, path in checkpoints:
        norms = checkpoint_weight_norms(path)
        global_epoch = 350 + full_data_epoch
        if global_epoch not in curve_by_epoch:
            raise ValueError(f"No accuracy/LR row for checkpoint at epoch {global_epoch}")
        metrics = curve_by_epoch[global_epoch]
        learning_rate = float(metrics["learning_rate"])
        heldout = float(metrics["heldout_accuracy"])
        epochs.append(global_epoch)
        for name in layer_names:
            value = norms[name]
            norm_history[name].append(value / start_norms[name])
            raw_proxy = learning_rate / value
            proxy_ratio = raw_proxy / (baseline.LEARNING_RATE / start_norms[name])
            effective_lr_ratio_history[name].append(proxy_ratio)
            output_rows.append(
                {
                    "global_epoch": global_epoch,
                    "full_data_epoch": full_data_epoch,
                    "layer": name,
                    "learning_rate": learning_rate,
                    "heldout_accuracy": heldout,
                    "weight_frobenius_norm": value,
                    "lr_over_norm": raw_proxy,
                    "lr_over_norm_ratio_to_baseline_at_epoch350": proxy_ratio,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = args.output_dir / "lr_rewarm_layer_norms.csv"
    with csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    # Heatmap uses each layer's norm relative to its own transition norm, so
    # differences in tensor size do not hide the growth pattern.
    matrix = np.array([norm_history[name] for name in layer_names])
    fig, (norm_ax, metric_ax) = plt.subplots(
        2, 1, figsize=(13, 10), sharex=False, constrained_layout=True,
        gridspec_kw={"height_ratios": [1.4, 1]},
    )
    image = norm_ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    norm_ax.set_yticks(range(len(layer_names)))
    norm_ax.set_yticklabels(layer_names)
    norm_ax.set_xticks(range(len(epochs)))
    norm_ax.set_xticklabels(epochs)
    norm_ax.set_xlabel("Overall epoch")
    norm_ax.set_ylabel("Conv/linear weight tensor")
    norm_ax.set_title("Layer weight norms relative to the epoch-350 checkpoint")
    colorbar = fig.colorbar(image, ax=norm_ax, pad=0.01)
    colorbar.set_label("Norm ratio")

    curve_epochs = [int(row["epoch"]) for row in curve_rows]
    learning_rates = [float(row["learning_rate"]) for row in curve_rows]
    heldout_accuracy = [float(row["heldout_accuracy"]) for row in curve_rows]
    lr_ax = metric_ax
    accuracy_ax = lr_ax.twinx()
    lr_line, = lr_ax.plot(curve_epochs, learning_rates, color="#e76f00", label="Learning rate")
    accuracy_line, = accuracy_ax.plot(
        curve_epochs, heldout_accuracy, color="#1f5f99", label="Held-out accuracy"
    )
    lr_ax.set(xlim=(350, 1050), xlabel="Overall epoch", ylabel="Learning rate")
    accuracy_ax.set_ylabel("Held-out accuracy")
    lr_ax.axvline(350, color="gray", linestyle=":", linewidth=1.2, label="_nolegend_")
    lr_ax.grid(alpha=0.25)
    lr_ax.set_title("LR re-warming schedule and held-out accuracy")
    lr_ax.legend(
        [lr_line, accuracy_line], ["Learning rate", "Held-out accuracy"], loc="best"
    )

    plot_out = args.output_dir / "lr_rewarm_norm_trajectory.png"
    fig.savefig(plot_out, dpi=180)
    plt.close(fig)

    # Direct per-layer proxy: eta / ||W_l||, shown relative to the baseline
    # proxy eta_baseline / ||W_l|| at the transition checkpoint. A value of 1
    # means the same estimated effective LR as that reference point.
    proxy_matrix = np.array(
        [effective_lr_ratio_history[name] for name in layer_names]
    )
    fig, (proxy_ax, accuracy_ax) = plt.subplots(
        2, 1, figsize=(13, 10), sharex=False, constrained_layout=True,
        gridspec_kw={"height_ratios": [1.4, 1]},
    )
    proxy_image = proxy_ax.imshow(
        proxy_matrix, aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=10
    )
    proxy_ax.set_yticks(range(len(layer_names)))
    proxy_ax.set_yticklabels(layer_names)
    proxy_ax.set_xticks(range(len(epochs)))
    proxy_ax.set_xticklabels(epochs)
    proxy_ax.set_xlabel("Overall epoch")
    proxy_ax.set_ylabel("Conv/linear weight tensor")
    proxy_ax.set_title("Per-layer LR / norm proxy relative to epoch-350 baseline")
    proxy_colorbar = fig.colorbar(proxy_image, ax=proxy_ax, pad=0.01)
    proxy_colorbar.set_label("Relative LR / norm (baseline = 1; color scale capped at 10)")

    accuracy_epochs = [350] + [int(row["epoch"]) for row in curve_rows]
    accuracy_values = [transition_heldout_accuracy] + [
        float(row["heldout_accuracy"]) for row in curve_rows
    ]
    accuracy_ax.plot(accuracy_epochs, accuracy_values, color="#1f5f99", linewidth=1.5)
    accuracy_ax.set(
        xlim=(350, 1050), xlabel="Overall epoch", ylabel="Held-out accuracy"
    )
    accuracy_ax.axvline(350, color="gray", linestyle=":", linewidth=1.2)
    accuracy_ax.grid(alpha=0.25)
    accuracy_ax.set_title("Held-out accuracy")

    proxy_plot_out = args.output_dir / "lr_over_norm_vs_accuracy.png"
    fig.savefig(proxy_plot_out, dpi=180)
    plt.close(fig)
    print(f"Saved {csv_out}")
    print(f"Saved {plot_out}")
    print(f"Saved {proxy_plot_out}")


if __name__ == "__main__":
    main()
