import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from src.data_model import Teachers
from src.training import (
    PretrainingCheckpoint,
    PretrainingMetric,
)

def ensure_run_directories(
    run_dir: Path,
) -> None:

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        run_dir
        / "checkpoints"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

def save_config(
    run_dir: Path,
    config: dict,
) -> None:

    ensure_run_directories(
        run_dir
    )

    path = run_dir / "config.json"

    with open(path, "w") as f:
        json.dump(
            config,
            f,
            indent=2,
            sort_keys=True,
        )


def load_config(
    run_dir: Path,
) -> dict:

    path = run_dir / "config.json"

    with open(path, "r") as f:
        return json.load(f)
    
def save_teachers(
    run_dir: Path,
    teachers: Teachers,
) -> None:

    path = run_dir / "teachers.pt"

    state = {
        "W_star": (
            teachers.W_star
            .detach()
            .cpu()
        ),
        "w1_star": (
            teachers.w1_star
            .detach()
            .cpu()
        ),
        "w2_star": (
            teachers.w2_star
            .detach()
            .cpu()
        ),
    }

    torch.save(
        state,
        path,
    )


def load_teachers(
    run_dir: Path,
    device: str,
    dtype: torch.dtype,
) -> Teachers:

    state = torch.load(
        run_dir / "teachers.pt",
        map_location="cpu",
        weights_only=True,
    )

    return Teachers(
        W_star=state["W_star"].to(
            device=device,
            dtype=dtype,
        ),
        w1_star=state["w1_star"].to(
            device=device,
            dtype=dtype,
        ),
        w2_star=state["w2_star"].to(
            device=device,
            dtype=dtype,
        ),
    )


PRETRAINING_METRIC_FIELDS = [
    "step",
    "step_over_d",
    "step_over_d2",
    "representation_error",
    "generalization_error",
]


def load_metric_steps(
    run_dir: Path,
) -> set[int]:

    path = run_dir / "metrics.csv"

    if not path.exists():
        return set()

    steps = set()

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            steps.add(
                int(row["step"])
            )

    return steps


def append_pretraining_metric(
    run_dir: Path,
    metric: PretrainingMetric,
) -> None:

    path = run_dir / "metrics.csv"

    write_header = (
        not path.exists()
        or path.stat().st_size == 0
    )

    with open(path, "a", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=(
                PRETRAINING_METRIC_FIELDS
            ),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "step": metric.step,
                "step_over_d": (
                    metric.step_over_d
                ),
                "step_over_d2": (
                    metric.step_over_d2
                ),
                "representation_error": (
                    metric.representation_error
                ),
                "generalization_error": (
                    metric.generalization_error
                ),
            }
        )

def checkpoint_dir(
    run_dir: Path,
    step: int,
) -> Path:

    return (
        run_dir
        / "checkpoints"
        / f"step_{step:09d}"
    )

def save_pretraining_checkpoint(
    run_dir: Path,
    checkpoint: PretrainingCheckpoint,
) -> None:

    ckpt_dir = checkpoint_dir(
        run_dir,
        checkpoint.step,
    )

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------
    # W
    # ----------------------------------------------

    torch.save(
        checkpoint.W,
        ckpt_dir / "W.pt",
    )

    # ----------------------------------------------
    # Resume state
    # ----------------------------------------------

    training_state = {
        "step": checkpoint.step,
        "optimizer_state": (
            checkpoint.optimizer_state
        ),
        "generator_state": (
            checkpoint.generator_state
        ),
    }

    torch.save(
        training_state,
        ckpt_dir / "training_state.pt",
    )

    # ----------------------------------------------
    # Human-readable checkpoint metrics
    # ----------------------------------------------

    metrics = {
        "step": checkpoint.step,
        "representation_error": (
            checkpoint.representation_error
        ),
        "generalization_error": (
            checkpoint.generalization_error
        ),
    }

    with open(
        ckpt_dir / "metrics.json",
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    # ----------------------------------------------
    # Point "latest" to this checkpoint
    # ----------------------------------------------

    latest = {
        "step": checkpoint.step,
        "checkpoint_dir": (
            ckpt_dir.name
        ),
    }

    latest_path = (
        run_dir / "latest.json"
    )

    tmp_path = (
        run_dir / "latest.tmp.json"
    )

    with open(tmp_path, "w") as f:
        json.dump(
            latest,
            f,
            indent=2,
        )

    os.replace(
        tmp_path,
        latest_path,
    )

def get_latest_checkpoint_step(
    run_dir: Path,
) -> int:

    path = run_dir / "latest.json"

    if not path.exists():
        raise FileNotFoundError(
            f"No latest.json found in {run_dir}"
        )

    with open(path, "r") as f:
        latest = json.load(f)

    return int(
        latest["step"]
    )


def load_pretraining_checkpoint(
    run_dir: Path,
    step: Optional[int] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict:

    if step is None:
        step = get_latest_checkpoint_step(
            run_dir
        )

    ckpt_dir = checkpoint_dir(
        run_dir,
        step,
    )

    W = torch.load(
        ckpt_dir / "W.pt",
        map_location="cpu",
        weights_only=True,
    )

    state = torch.load(
        ckpt_dir / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )

    with open(
        ckpt_dir / "metrics.json",
        "r",
    ) as f:

        metrics = json.load(f)

    return {
        "step": step,

        "W": W.to(
            device=device,
            dtype=dtype,
        ),

        "optimizer_state": (
            state["optimizer_state"]
        ),

        "generator_state": (
            state["generator_state"]
        ),

        "metrics": metrics,

    }

def append_run_history(
    run_dir: Path,
    start_step: int,
    target_step: int,
) -> None:

    path = (
        run_dir
        / "run_history.jsonl"
    )

    record = {
        "timestamp_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "start_step": start_step,
        "target_step": target_step,
    }

    with open(path, "a") as f:
        f.write(
            json.dumps(record)
            + "\n"
        )