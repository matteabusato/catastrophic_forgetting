"""Portable metrics plus local PyTorch snapshots for auditing each paired run."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .experiment import ExperimentResult


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_result(result: ExperimentResult, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", asdict(result.config))
    write_json(directory / "summary.json", result.summary)
    write_json(directory / "metadata.json", {
        "seeds": result.seeds, "torch_version": str(torch.__version__),
        "loss": "output_mse_mean_over_examples_tokens_features",
        "input_distribution": "iid standard Gaussian tokens",
        "centering": "subtract trace(W.T@W/sqrt(P))/d * I_T before row softmax",
        "optimizer_reset_at_transition": True,
        "timing": "optimization loop only, excludes metric evaluation",
    })
    for name, run in result.runs.items():
        with (directory / f"{name}_history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run.history[0]))
            writer.writeheader()
            writer.writerows(run.history)
    torch.save({
        "teacher_W": result.teacher.W.detach().cpu(),
        "initial_W": result.initial_W.cpu(),
        "pretrained_W": result.pretrained_W.cpu(),
        "warm_final_W": result.warm_model.W.detach().cpu(),
        "fresh_final_W": result.fresh_model.W.detach().cpu(),
    }, directory / "weights.pt")
    torch.save({
        "train_X": result.data.full.X.cpu(), "train_y": result.data.full.y.cpu(),
        "train_attention": result.data.full.attention.cpu(),
        "test_X": result.data.test.X.cpu(), "test_y": result.data.test.y.cpu(),
        "test_attention": result.data.test.attention.cpu(),
        "indices_A": result.data.indices_A.cpu(), "indices_B": result.data.indices_B.cpu(),
    }, directory / "datasets.pt")
