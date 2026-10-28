"""Paired A -> A union B versus fresh A union B experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import math
import statistics
from typing import Callable

import torch

from .data import DataSplit, make_split
from .model import Student, Teacher
from .training import TrainingConfig, TrainingRun, train


@dataclass(frozen=True)
class ExperimentConfig:
    d: int = 16
    T: int = 4
    P: int = 32
    P_teacher: int = 32
    n_train: int = 128
    n_test: int = 2048
    initial_fraction: float = 0.5
    init_scale: float = 1.0
    teacher_scale: float = 1.0
    seed: int = 0
    dtype: str = "float64"
    device: str = "cpu"
    pretraining: TrainingConfig = field(default_factory=TrainingConfig)
    final_training: TrainingConfig = field(default_factory=TrainingConfig)
    training_match_atol: float = 1e-6
    training_match_rtol: float = 0.05

    def __post_init__(self):
        if min(self.d, self.P, self.P_teacher, self.n_test) < 1 or self.T < 2:
            raise ValueError("Positive dimensions/sample counts and T>=2 required.")
        if self.n_train < 2 or not 0 < self.initial_fraction < 1:
            raise ValueError("Need n_train>=2 and 0<initial_fraction<1.")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64.")
        if any(not math.isfinite(x) or x <= 0 for x in (self.init_scale, self.teacher_scale)):
            raise ValueError("Initialization scales must be positive and finite.")
        if any(not math.isfinite(x) or x < 0 for x in
               (self.training_match_atol, self.training_match_rtol)):
            raise ValueError("Training-match tolerances must be nonnegative and finite.")


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    seeds: dict[str, int]
    data: DataSplit
    teacher: Teacher
    initial_W: torch.Tensor
    pretrained_W: torch.Tensor
    warm_model: Student
    fresh_model: Student
    runs: dict[str, TrainingRun]
    summary: dict


def experiment_seeds(seed: int) -> dict[str, int]:
    """Stable role-specific seeds, also used to recover interrupted legacy runs."""
    roles = ("teacher", "initialization", "train_data", "test_data",
             "split", "pretraining_order", "final_order")
    # Role-specific streams avoid overlapping RNG seeds across adjacent replicates.
    return {role: int.from_bytes(hashlib.sha256(
        f"warm_start:{seed}:{role}".encode()).digest()[:8], "big") % (2**63)
        for role in roles}


def run_experiment(config: ExperimentConfig,
                   metric_callback: Callable[[dict], None] | None = None) -> ExperimentResult:
    """One independent paired replicate; all random streams have recorded seeds."""
    seeds = experiment_seeds(config.seed)
    dtype = getattr(torch, config.dtype)
    teacher = Teacher(config.d, config.P_teacher, seed=seeds["teacher"],
                      init_scale=config.teacher_scale, dtype=dtype, device=config.device)
    initial = Student(config.d, config.P, seed=seeds["initialization"],
                      init_scale=config.init_scale, dtype=dtype, device=config.device)
    initial_W = initial.W.detach().clone()
    data = make_split(teacher, config.n_train, config.n_test, config.T,
                      config.initial_fraction, seeds["train_data"],
                      seeds["test_data"], seeds["split"])
    warm, fresh = deepcopy(initial), deepcopy(initial)
    evaluations = {"full": data.full, "A": data.A, "B": data.B, "test": data.test}
    common = dict(evaluation_sets=evaluations, teacher_S=teacher.S, initial_W=initial_W,
                  metric_callback=metric_callback)
    pretrain = train(warm, data.A, config.pretraining, seeds["pretraining_order"],
                     "pretraining_A", **common)
    pretrained_W = warm.W.detach().clone()
    warm_run = train(warm, data.full, config.final_training, seeds["final_order"],
                     "warm_full", **common)
    fresh_run = train(fresh, data.full, config.final_training, seeds["final_order"],
                      "fresh_full", **common)
    warm_final, fresh_final = warm_run.history[-1], fresh_run.history[-1]
    warm_train, fresh_train = warm_final["train_mse"], fresh_final["train_mse"]
    difference = abs(warm_train - fresh_train)
    matched = difference <= (config.training_match_atol
                            + config.training_match_rtol * max(warm_train, fresh_train))
    summary = {
        "seed": config.seed,
        "warm_train_mse": warm_train, "fresh_train_mse": fresh_train,
        "warm_test_mse": warm_final["test_mse"],
        "fresh_test_mse": fresh_final["test_mse"],
        "delta_gen": warm_final["test_mse"] - fresh_final["test_mse"],
        "absolute_train_mse_difference": difference,
        "comparable_training_error": matched,
        "both_reached_loss_tolerance": max(warm_train, fresh_train)
                                       <= config.final_training.loss_tolerance,
        "warm_final_updates": warm_run.updates,
        "warm_total_updates": pretrain.updates + warm_run.updates,
        "fresh_updates": fresh_run.updates,
        "warm_final_train_seconds": warm_run.train_seconds,
        "warm_total_train_seconds": pretrain.train_seconds + warm_run.train_seconds,
        "fresh_train_seconds": fresh_run.train_seconds,
        "stop_reasons": {r.phase: r.stop_reason for r in (pretrain, warm_run, fresh_run)},
    }
    return ExperimentResult(config, seeds, data, teacher, initial_W, pretrained_W,
                            warm, fresh, {"pretraining": pretrain, "warm": warm_run,
                                          "fresh": fresh_run}, summary)


def summarize_pairs(summaries: list[dict]) -> dict:
    """Sample SD and SE across independent paired replicates; no significance claim."""
    if not summaries:
        raise ValueError("Need at least one replicate.")
    if len({s["seed"] for s in summaries}) != len(summaries):
        raise ValueError("Replicate seeds must be distinct.")
    deltas = [s["delta_gen"] for s in summaries]
    if not all(math.isfinite(delta) for delta in deltas):
        raise ValueError("Generalization gaps must be finite.")
    n = len(deltas)
    std = statistics.stdev(deltas) if n > 1 else None
    return {"n_pairs": n, "mean_delta_gen": statistics.mean(deltas),
            "std_delta_gen": std,
            "se_delta_gen": std / math.sqrt(n) if std is not None else None,
            "positive_gap_pairs": sum(delta > 0 for delta in deltas),
            "comparable_training_pairs": sum(s["comparable_training_error"] for s in summaries),
            "both_fitted_pairs": sum(s["both_reached_loss_tolerance"] for s in summaries)}
