"""Fixed-budget online SGD on one teacher, with separate fixed-set evaluation."""

from __future__ import annotations

import math
from typing import Callable
import torch
from .data import GaussianStream
from .losses import output_mse
from .model import Student, Teacher


def train_phase(model: Student, teacher: Teacher, stream: GaussianStream,
                steps: int, batch_size: int, lr: float, eval_every: int,
                record: Callable[[int, float | None], dict],
                on_record: Callable[[dict], None] | None = None) -> list[dict]:
    """A new stateless SGD optimizer per phase; evaluation never controls updates.

    record receives phase-local updates and the latest pre-update minibatch MSE.
    Always evaluate at step zero and the exact final update. No convergence claim
    or early stopping is inferred from a fixed compute budget.
    """
    if min(batch_size, eval_every) < 1 or steps < 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Invalid step budget, batch size, evaluation interval or lr.")
    optimizer = torch.optim.SGD([model.W], lr=lr)
    history = []

    def log(step, loss):
        row = record(step, loss)
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise FloatingPointError(f"Non-finite diagnostic {key} at step {step}.")
        history.append(row)
        if on_record:
            on_record(dict(row))

    model.train()
    log(0, None)
    for step in range(1, steps + 1):
        X = stream.sample(batch_size)
        with torch.no_grad():
            target = teacher(X)
        optimizer.zero_grad(set_to_none=True)
        loss = output_mse(model(X), target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}.")
        loss.backward()
        if model.W.grad is None or not torch.isfinite(model.W.grad).all():
            raise FloatingPointError(f"Missing or non-finite gradient at step {step}.")
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            log(step, loss.item())
    return history
