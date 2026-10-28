"""Finite-data epochs with a new optimizer per call and train-only stopping."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Callable

import torch

from .data import Dataset
from .losses import output_mse
from .model import Student
from .observables import evaluate, matrix_metrics


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 200
    min_epochs: int = 1
    batch_size: int = 32
    optimizer: str = "adam"
    lr: float = 1e-3
    eval_every: int = 10
    loss_tolerance: float = 1e-8
    # Disabled by default: plateau is not evidence of fitting the training set.
    plateau_patience: int | None = None
    plateau_atol: float = 1e-10
    plateau_rtol: float = 1e-6
    eval_batch_size: int = 1024
    # When set, the update budget replaces max_epochs (including partial epochs).
    max_steps: int | None = None
    log_every_steps: int | None = None
    early_stopping: bool = True

    def __post_init__(self):
        if not 1 <= self.min_epochs <= self.max_epochs:
            raise ValueError("Require 1<=min_epochs<=max_epochs.")
        if min(self.batch_size, self.eval_every, self.eval_batch_size) < 1:
            raise ValueError("Batch sizes and eval_every must be positive.")
        if self.optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer must be 'adam' or 'sgd'.")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be positive and finite.")
        if any(not math.isfinite(x) or x < 0 for x in
               (self.loss_tolerance, self.plateau_atol, self.plateau_rtol)):
            raise ValueError("Tolerances must be nonnegative and finite.")
        if self.plateau_patience is not None and self.plateau_patience < 1:
            raise ValueError("plateau_patience must be positive or None.")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive or None.")
        if self.log_every_steps is not None and self.log_every_steps < 1:
            raise ValueError("log_every_steps must be positive or None.")


@dataclass
class TrainingRun:
    phase: str
    history: list[dict] = field(default_factory=list)
    stop_reason: str = "max_epochs"
    epochs: int = 0
    updates: int = 0
    examples_seen: int = 0
    train_seconds: float = 0.0


def _synchronize(model: Student) -> None:
    if model.W.device.type == "cuda":
        torch.cuda.synchronize(model.W.device)
    elif model.W.device.type == "mps":
        torch.mps.synchronize()


def train(model: Student, train_data: Dataset, config: TrainingConfig,
          seed: int, phase: str, evaluation_sets: dict[str, Dataset] | None = None,
          teacher_S: torch.Tensor | None = None,
          initial_W: torch.Tensor | None = None,
          metric_callback: Callable[[dict], None] | None = None) -> TrainingRun:
    """Train W on fixed examples. Evaluation sets never affect updates or stopping.

    log_every_steps switches evaluation from epochs to optimizer updates.
    Stopping is checked at evaluations. Plateau patience counts evaluations.
    Runtime measures updates, excluding diagnostic evaluation and callbacks.
    """
    evaluation_sets = evaluation_sets or {}
    if "train" in evaluation_sets:
        raise ValueError("'train' is reserved for the actual training dataset.")
    optimizer_type = torch.optim.Adam if config.optimizer == "adam" else torch.optim.SGD
    optimizer = optimizer_type([model.W], lr=config.lr)
    generator = torch.Generator(device=model.W.device).manual_seed(seed)
    run = TrainingRun(phase=phase)
    if config.max_steps is not None:
        run.stop_reason = "max_steps"

    def record() -> float:
        row = dict(phase=phase, epoch=run.epochs, updates=run.updates,
                   examples_seen=run.examples_seen, train_seconds=run.train_seconds)
        for name, data in {"train": train_data, **evaluation_sets}.items():
            for key, value in evaluate(model, data, config.eval_batch_size).items():
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite {name}_{key} in {phase}.")
                row[f"{name}_{key}"] = value
            # The regression loss and prediction error use the same MSE definition.
            row[f"{name}_loss"] = row[f"{name}_mse"]
            row[f"{name}_error"] = row[f"{name}_mse"]
        if teacher_S is not None:
            row.update(matrix_metrics(model, teacher_S, initial_W))
        run.history.append(row)
        if metric_callback is not None:
            metric_callback(dict(row))
        return row["train_mse"]

    model.train()
    best = record()
    stale = 0

    def should_stop(train_mse: float) -> bool:
        nonlocal best, stale
        threshold = config.plateau_atol + config.plateau_rtol * abs(best)
        if train_mse < best - threshold:
            best, stale = train_mse, 0
        else:
            stale += 1
        if not config.early_stopping or run.epochs < config.min_epochs:
            return False
        if train_mse <= config.loss_tolerance:
            run.stop_reason = "loss_tolerance"
            return True
        if config.plateau_patience is not None and stale >= config.plateau_patience:
            run.stop_reason = "plateau"
            return True
        return False

    batches_per_epoch = math.ceil(len(train_data) / config.batch_size)
    epochs = (math.ceil(config.max_steps / batches_per_epoch)
              if config.max_steps is not None else config.max_epochs)
    for epoch in range(1, epochs + 1):
        _synchronize(model)
        started = perf_counter()
        order = torch.randperm(len(train_data), generator=generator, device=model.W.device)
        for batch, indices in enumerate(order.split(config.batch_size), start=1):
            optimizer.zero_grad(set_to_none=True)
            loss = output_mse(model(train_data.X[indices]), train_data.y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {phase}, epoch {epoch}.")
            loss.backward()
            if model.W.grad is None or not torch.isfinite(model.W.grad).all():
                raise FloatingPointError(f"Non-finite or missing gradient in {phase}.")
            optimizer.step()
            run.updates += 1
            run.examples_seen += len(indices)
            # epoch denotes the number of completed passes at this update.
            run.epochs = epoch if batch == batches_per_epoch else epoch - 1
            last_update = (run.updates == config.max_steps if config.max_steps is not None
                           else epoch == epochs and batch == batches_per_epoch)
            step_log = config.log_every_steps is not None and (
                run.updates % config.log_every_steps == 0 or last_update)
            if step_log or (last_update and config.max_steps is not None):
                _synchronize(model)
                run.train_seconds += perf_counter() - started
                if should_stop(record()):
                    return run
                _synchronize(model)
                started = perf_counter()
            if last_update and config.max_steps is not None:
                return run
        _synchronize(model)
        run.train_seconds += perf_counter() - started
        run.epochs = epoch
        if config.log_every_steps is not None:
            continue
        if epoch % config.eval_every and epoch != epochs:
            continue
        if should_stop(record()):
            break
    return run
