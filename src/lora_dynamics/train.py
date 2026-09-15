#!/usr/bin/env python3
"""Example:
    python train.py \
        --experiment matched_two_stage \
        --d 200 \
        --T 5 \
        --kappa-star 1.0 \
        --kappa 1.0 \
        --alpha 2.0 \
        --alpha-prime 2.0 \
        --lr-W 1e-3 \
        --lr-w 1e-3 \
        --log-every 500 \
        --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from model import (
    ModelConfig,
    StudentModel,
    TeacherModel,
    sample_gaussian_input,
    set_seed,
    squared_sequence_loss,
)
from observables import (
    ForgettingTracker,
    compute_all_observables,
    make_fixed_test_set,
)
from protocols import (
    ExperimentName,
    Protocol,
    ProtocolConfig,
    StepInstruction,
    protocol_summary,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

EXPERIMENT_CHOICES = (
    "joint",
    "W_then_w",
    "w_then_W",
    "matched_two_stage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the solvable attention + rank-one LoRA model with "
            "single-example SGD."
        )
    )

    # Protocol.
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=EXPERIMENT_CHOICES,
        help="Training protocol.",
    )
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--alpha-prime", type=float, default=2.0)
    parser.add_argument(
        "--p-finetune",
        type=float,
        default=0.5,
        help="P(draw S') at each mixed-protocol SGD step.",
    )

    # Model.
    parser.add_argument("--d", type=int, default=200)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--kappa-star", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument(
        "--activation",
        type=str,
        default="softmax",
        choices=("softmax", "identity"),
    )
    parser.add_argument("--init-std-W", type=float, default=1.0)
    parser.add_argument("--init-std-w", type=float, default=1.0)

    # Plain SGD: no momentum, no weight decay, no mini-batching.
    parser.add_argument(
        "--lr-W",
        type=float,
        default=1e-3,
        help="SGD learning rate for the extensive-rank parameter W.",
    )
    parser.add_argument(
        "--lr-w",
        type=float,
        default=1e-3,
        help="SGD learning rate for the rank-one LoRA vector w.",
    )

    # Logging/evaluation.
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help=(
            "Record observables every this many SGD updates. "
            "Step 0, phase boundaries, and the final step are always recorded."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help=(
            "Number of fixed held-out samples per task. "
            "Default: 10*d, as in the finite-D simulations of the paper."
        ),
    )
    parser.add_argument(
        "--gradient-probe-size",
        type=int,
        default=1,
        help=(
            "Number of fixed held-out inputs per task used for gradient norms "
            "and gradient-cosine diagnostics. Set to 0 to disable gradient "
            "diagnostics. This does not affect single-sample SGD training."
        ),
    )

    # Reproducibility.
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Main seed for teacher/student initialization and training data.",
    )
    parser.add_argument(
        "--schedule-seed",
        type=int,
        default=None,
        help=(
            "Seed for S/S' choices in mixed protocols. "
            "Default: same value as --seed."
        ),
    )

    # Device / precision.
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=("float32", "float64"),
    )

    # Output.
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/training_dynamics"),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit run-directory name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing run directory with the same name.",
    )

    args = parser.parse_args()

    if args.alpha <= 0:
        parser.error("--alpha must be > 0.")
    if args.alpha_prime <= 0:
        parser.error("--alpha-prime must be > 0.")
    if not 0.0 <= args.p_finetune <= 1.0:
        parser.error("--p-finetune must lie in [0,1].")
    if args.lr_W <= 0 or args.lr_w <= 0:
        parser.error("Both learning rates must be > 0.")
    if args.log_every <= 0:
        parser.error("--log-every must be > 0.")
    if args.test_size is not None and args.test_size <= 0:
        parser.error("--test-size must be > 0 when specified.")
    if args.gradient_probe_size < 0:
        parser.error("--gradient-probe-size must be >= 0.")

    return args


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device_arg!r} requested, but CUDA is not available."
        )
    return device


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def _slug_number(x: float) -> str:
    s = f"{x:.10g}"
    return s.replace("-", "m").replace(".", "p").replace("+", "")


def default_run_name(args: argparse.Namespace) -> str:
    return (
        f"d{args.d}_T{args.T}"
        f"_ks{_slug_number(args.kappa_star)}"
        f"_k{_slug_number(args.kappa)}"
        f"_a{_slug_number(args.alpha)}"
        f"_ap{_slug_number(args.alpha_prime)}"
        f"_lrW{_slug_number(args.lr_W)}"
        f"_lrw{_slug_number(args.lr_w)}"
        f"_seed{args.seed}"
    )


def prepare_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name or default_run_name(args)
    run_dir = args.output_root / args.experiment / run_name

    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Run directory already exists:\n  {run_dir}\n"
                "Use --overwrite or choose --run-name."
            )
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def json_dump(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=True)


def phase_boundary_steps(protocol_config: ProtocolConfig) -> set[int]:
    if protocol_config.experiment == "joint":
        return set()

    if protocol_config.experiment == "w_then_W":
        return {protocol_config.n_small}

    # W_then_w and matched_two_stage both switch after n_large.
    return {protocol_config.n_large}


def should_log(
    step: int,
    *,
    total_steps: int,
    log_every: int,
    boundary_steps: set[int],
) -> bool:
    return (
        step == 0
        or step == total_steps
        or step in boundary_steps
        or step % log_every == 0
    )


class MetricRecorder:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.rows: list[Dict[str, Any]] = []

    def append(self, row: Dict[str, Any]) -> None:
        self.rows.append(dict(row))

    @staticmethod
    def _all_keys(rows: Iterable[Dict[str, Any]]) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        return keys

    def save(self) -> None:
        if not self.rows:
            return

        keys = self._all_keys(self.rows)

        # CSV is convenient for quick inspection.
        csv_path = self.run_dir / "metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({key: row.get(key, "") for key in keys})

        # NPZ is convenient for the plotting notebook.
        arrays: Dict[str, np.ndarray] = {}
        for key in keys:
            values = [row.get(key, np.nan) for row in self.rows]

            # Preserve text fields as unicode arrays; numerical fields as
            # float64 to avoid object arrays / pickle requirements.
            if any(isinstance(v, str) for v in values):
                arrays[key] = np.asarray(
                    ["" if not isinstance(v, str) else v for v in values],
                    dtype=str,
                )
            else:
                converted = []
                for v in values:
                    if v is None:
                        converted.append(np.nan)
                    elif isinstance(v, (bool, np.bool_)):
                        converted.append(float(v))
                    else:
                        converted.append(float(v))
                arrays[key] = np.asarray(converted, dtype=np.float64)

        np.savez_compressed(self.run_dir / "metrics.npz", **arrays)


# ---------------------------------------------------------------------------
# Metric checkpointing
# ---------------------------------------------------------------------------


def make_initial_window_stats() -> Dict[str, float]:
    return {
        "loss_sum": 0.0,
        "count": 0.0,
        "loss_S_sum": 0.0,
        "count_S": 0.0,
        "loss_Sprime_sum": 0.0,
        "count_Sprime": 0.0,
    }


def update_window_stats(
    stats: Dict[str, float],
    *,
    task: str,
    loss: float,
) -> None:
    stats["loss_sum"] += loss
    stats["count"] += 1.0

    if task == "S":
        stats["loss_S_sum"] += loss
        stats["count_S"] += 1.0
    elif task == "Sprime":
        stats["loss_Sprime_sum"] += loss
        stats["count_Sprime"] += 1.0
    else:
        raise ValueError(f"Unknown task: {task}")


def safe_mean(total: float, count: float) -> float:
    return total / count if count > 0 else float("nan")


def log_checkpoint(
    *,
    step: int,
    instruction: Optional[StepInstruction],
    teacher: TeacherModel,
    student: StudentModel,
    old_test_set,
    new_test_set,
    X_old_probe: Optional[torch.Tensor],
    X_new_probe: Optional[torch.Tensor],
    forgetting_tracker: ForgettingTracker,
    recorder: MetricRecorder,
    window_stats: Dict[str, float],
    last_train_loss: float,
    n_S_seen: int,
    n_Sprime_seen: int,
    d: int,
    lr_W: float,
    lr_w: float,
    elapsed_sec: float,
) -> Dict[str, Any]:
    metrics = compute_all_observables(
        teacher,
        student,
        old_test_set=old_test_set,
        new_test_set=new_test_set,
        X_old_probe=X_old_probe,
        X_new_probe=X_new_probe,
        forgetting_tracker=forgetting_tracker,
        delta=0.0,
    )

    if instruction is None:
        phase = "initial"
        phase_step = 0
        last_task = "none"
        train_W = False
        train_w = False
        include_lora = True
    else:
        phase = instruction.phase
        phase_step = instruction.phase_step
        last_task = instruction.task
        train_W = instruction.train_W
        train_w = instruction.train_w
        include_lora = instruction.include_lora

    row: Dict[str, Any] = {
        # Progress coordinates.
        "step": step,
        "samples_seen": step,  # true SGD: exactly one sample per update
        "step_over_d": step / float(d),
        "step_over_d2": step / float(d * d),
        "phase": phase,
        "phase_step": phase_step,
        "last_task": last_task,
        "train_W": train_W,
        "train_w": train_w,
        "include_lora": include_lora,

        # Actual task counts are useful for mixed protocols.
        "n_S_seen": n_S_seen,
        "n_Sprime_seen": n_Sprime_seen,
        "n_S_over_d2": n_S_seen / float(d * d),
        "n_Sprime_over_d": n_Sprime_seen / float(d),

        # SGD loss diagnostics.
        "train_loss_last": last_train_loss,
        "train_loss_mean_since_log": safe_mean(
            window_stats["loss_sum"], window_stats["count"]
        ),
        "train_loss_S_mean_since_log": safe_mean(
            window_stats["loss_S_sum"], window_stats["count_S"]
        ),
        "train_loss_Sprime_mean_since_log": safe_mean(
            window_stats["loss_Sprime_sum"], window_stats["count_Sprime"]
        ),

        "lr_W": lr_W,
        "lr_w": lr_w,
        "elapsed_sec": elapsed_sec,
    }
    row.update(metrics)

    recorder.append(row)
    recorder.save()

    return row


def print_checkpoint(row: Dict[str, Any], total_steps: int) -> None:
    step = int(row["step"])
    phase = row["phase"]

    msg = (
        f"[{step:>8d}/{total_steps}] "
        f"phase={phase:<22s} "
        f"E_S={row['E_S']:.4e} "
        f"E_S'={row['E_Sprime']:.4e} "
        f"E_S(W)={row['E_S_W_only']:.4e} "
        f"rho_W={row['rho_W']:.4f} "
        f"rho_w={row['rho_w']:.4f} "
        f"F_S={row['F_S']:.4e}"
    )
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    if device.type == "cuda" and dtype == torch.float32:
        # Allows TF32-backed matmuls on supported NVIDIA GPUs while retaining
        # float32 tensors. This is a speed choice, not a training batch.
        torch.set_float32_matmul_precision("high")

    schedule_seed = args.seed if args.schedule_seed is None else args.schedule_seed
    test_size = 10 * args.d if args.test_size is None else args.test_size

    if args.gradient_probe_size > test_size:
        raise ValueError(
            "--gradient-probe-size cannot exceed --test-size because the "
            "gradient probes are taken from the fixed test sets."
        )

    run_dir = prepare_run_dir(args)

    model_config = ModelConfig(
        d=args.d,
        T=args.T,
        kappa_star=args.kappa_star,
        kappa=args.kappa,
        activation=args.activation,
        init_std_W=args.init_std_W,
        init_std_w=args.init_std_w,
        dtype=dtype,
    )
    model_config.validate()

    protocol_config = ProtocolConfig(
        experiment=args.experiment,
        d=args.d,
        alpha=args.alpha,
        alpha_prime=args.alpha_prime,
        p_finetune=args.p_finetune,
        schedule_seed=schedule_seed,
    )
    protocol_config.validate()
    protocol = Protocol(protocol_config)

    # Main RNG: teacher and student initialization.
    set_seed(args.seed)

    teacher = TeacherModel(model_config, device=device)
    student = StudentModel(model_config, device=device)

    # Private generator for *training* sequences. This keeps the training
    # stream independent of fixed-test construction and logging frequency.
    train_generator = torch.Generator(device=device)
    train_data_seed = args.seed + 1_000_003
    train_generator.manual_seed(train_data_seed)

    # Fixed evaluation sets use private RNGs internally, so making them here
    # does not alter the training sequence.
    old_test_seed = args.seed + 2_000_003
    new_test_seed = args.seed + 3_000_003

    print(
        f"Creating fixed test sets: {test_size} samples/task on {device}...",
        flush=True,
    )
    old_test_set = make_fixed_test_set(
        teacher,
        task="S",
        n_samples=test_size,
        seed=old_test_seed,
        device=device,
    )
    new_test_set = make_fixed_test_set(
        teacher,
        task="Sprime",
        n_samples=test_size,
        seed=new_test_seed,
        device=device,
    )

    if args.gradient_probe_size > 0:
        X_old_probe = old_test_set.X[: args.gradient_probe_size]
        X_new_probe = new_test_set.X[: args.gradient_probe_size]
    else:
        X_old_probe = None
        X_new_probe = None

    # Two parameter groups make the W and w learning rates independently
    # configurable while still using a single plain-SGD optimizer.
    optimizer = torch.optim.SGD(
        [
            {"params": [student.W], "lr": args.lr_W},
            {"params": [student.w], "lr": args.lr_w},
        ],
        momentum=0.0,
        weight_decay=0.0,
    )

    config_payload: Dict[str, Any] = {
        "model": {
            "d": args.d,
            "T": args.T,
            "kappa_star": args.kappa_star,
            "kappa": args.kappa,
            "P_star": model_config.P_star,
            "P": model_config.P,
            "activation": args.activation,
            "init_std_W": args.init_std_W,
            "init_std_w": args.init_std_w,
            "dtype": args.dtype,
        },
        "protocol": protocol_summary(protocol_config),
        "training": {
            "optimizer": "SGD",
            "single_sample_sgd": True,
            "batch_size": 1,
            "lr_W": args.lr_W,
            "lr_w": args.lr_w,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "log_every": args.log_every,
            "test_size_per_task": test_size,
            "gradient_probe_size": args.gradient_probe_size,
            "teacher_noise_Delta": 0.0,
        },
        "seeds": {
            "main_seed": args.seed,
            "schedule_seed": schedule_seed,
            "train_data_seed": train_data_seed,
            "old_test_seed": old_test_seed,
            "new_test_seed": new_test_seed,
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
    }
    json_dump(run_dir / "config.json", config_payload)

    recorder = MetricRecorder(run_dir)
    forgetting_tracker = ForgettingTracker()

    boundaries = phase_boundary_steps(protocol_config)
    total_steps = protocol_config.total_steps

    n_S_seen = 0
    n_Sprime_seen = 0
    last_train_loss = float("nan")
    window_stats = make_initial_window_stats()

    start_time = time.perf_counter()

    # Step 0: essential for seeing the complete training dynamics.
    row = log_checkpoint(
        step=0,
        instruction=None,
        teacher=teacher,
        student=student,
        old_test_set=old_test_set,
        new_test_set=new_test_set,
        X_old_probe=X_old_probe,
        X_new_probe=X_new_probe,
        forgetting_tracker=forgetting_tracker,
        recorder=recorder,
        window_stats=window_stats,
        last_train_loss=last_train_loss,
        n_S_seen=n_S_seen,
        n_Sprime_seen=n_Sprime_seen,
        d=args.d,
        lr_W=args.lr_W,
        lr_w=args.lr_w,
        elapsed_sec=time.perf_counter() - start_time,
    )
    print_checkpoint(row, total_steps)

    # The "since last log" window starts after the initial checkpoint.
    window_stats = make_initial_window_stats()

    previous_trainability: tuple[Optional[bool], Optional[bool]] = (None, None)

    for instruction in protocol:
        # Freeze/unfreeze only when the protocol state actually changes.
        desired = (instruction.train_W, instruction.train_w)
        if desired != previous_trainability:
            # Clear stale gradients before changing requires_grad flags.
            optimizer.zero_grad(set_to_none=True)
            student.set_trainable(
                train_W=instruction.train_W,
                train_w=instruction.train_w,
            )
            previous_trainability = desired

        optimizer.zero_grad(set_to_none=True)

        # Exactly ONE Gaussian sequence per SGD update.
        X = sample_gaussian_input(
            args.T,
            args.d,
            device=device,
            dtype=dtype,
            generator=train_generator,
        )

        if instruction.task == "S":
            y = teacher.pretrain_target(X)
            n_S_seen += 1
        elif instruction.task == "Sprime":
            y = teacher.finetune_target(X)
            n_Sprime_seen += 1
        else:
            raise RuntimeError(f"Unknown task {instruction.task!r}.")

        prediction = student(
            X,
            include_lora=instruction.include_lora,
        )
        loss = squared_sequence_loss(prediction, y)

        # A scalar check avoids silently propagating NaN/Inf runs.
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                f"Non-finite loss at step {instruction.step}: {loss.item()}"
            )

        loss.backward()
        optimizer.step()

        last_train_loss = float(loss.detach().item())
        update_window_stats(
            window_stats,
            task=instruction.task,
            loss=last_train_loss,
        )

        if should_log(
            instruction.step,
            total_steps=total_steps,
            log_every=args.log_every,
            boundary_steps=boundaries,
        ):
            row = log_checkpoint(
                step=instruction.step,
                instruction=instruction,
                teacher=teacher,
                student=student,
                old_test_set=old_test_set,
                new_test_set=new_test_set,
                X_old_probe=X_old_probe,
                X_new_probe=X_new_probe,
                forgetting_tracker=forgetting_tracker,
                recorder=recorder,
                window_stats=window_stats,
                last_train_loss=last_train_loss,
                n_S_seen=n_S_seen,
                n_Sprime_seen=n_Sprime_seen,
                d=args.d,
                lr_W=args.lr_W,
                lr_w=args.lr_w,
                elapsed_sec=time.perf_counter() - start_time,
            )
            print_checkpoint(row, total_steps)
            window_stats = make_initial_window_stats()

    elapsed_sec = time.perf_counter() - start_time

    # Save the trained model and teacher so any trajectory can later be
    # inspected/re-evaluated without regenerating the disorder realization.
    torch.save(
        {
            "step": total_steps,
            "teacher_state_dict": teacher.state_dict(),
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": config_payload["model"],
            "protocol_config": config_payload["protocol"],
            "training_config": config_payload["training"],
            "seeds": config_payload["seeds"],
        },
        run_dir / "final_state.pt",
    )

    final_row = recorder.rows[-1]

    summary: Dict[str, Any] = {
        "experiment": args.experiment,
        "run_dir": str(run_dir),
        "completed": True,
        "total_steps": total_steps,
        "n_S_seen": n_S_seen,
        "n_Sprime_seen": n_Sprime_seen,
        "elapsed_sec": elapsed_sec,
        "final": {
            key: value
            for key, value in final_row.items()
            if key
            not in {
                "phase",
                "last_task",
            }
        },
        "final_phase": final_row["phase"],
        "final_last_task": final_row["last_task"],
    }

    if device.type == "cuda":
        summary["peak_cuda_memory_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )

    json_dump(run_dir / "summary.json", summary)

    print()
    print("Training complete.", flush=True)
    print(f"Results saved to: {run_dir}", flush=True)
    print(f"Elapsed time: {elapsed_sec:.2f} s", flush=True)


if __name__ == "__main__":
    main()