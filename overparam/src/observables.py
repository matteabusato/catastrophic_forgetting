"""Held-out task errors, attention diagnostics, representation geometry and forgetting."""

from __future__ import annotations

import torch
from torch import Tensor
from .data import Dataset
from .losses import output_mse, attention_kl
from .model import Student


@torch.no_grad()
def evaluate(model: Student, data: Dataset, batch_size: int = 256,
             S: Tensor | None = None) -> dict[str, float]:
    if batch_size < 1 or len(data) < 1:
        raise ValueError("Need nonempty data and positive batch_size.")
    from .model import centered_logits
    S = model.S if S is None else S
    total = dict(mse=0.0, attention_kl=0.0, attention_entropy=0.0)
    for start in range(0, len(data), batch_size):
        X = data.X[start:start + batch_size]
        logits = centered_logits(X, S)
        attention = logits.softmax(-1)
        values = dict(mse=output_mse(attention @ X, data.y[start:start + batch_size]),
                      attention_kl=attention_kl(logits, data.attention[start:start + batch_size]),
                      attention_entropy=-torch.special.xlogy(attention, attention).sum(-1).mean())
        for key, value in values.items():
            total[key] += len(X) * value.item()
    return {key: value / len(data) for key, value in total.items()}


@torch.no_grad()
def matrix_metrics(model: Student, teachers: dict, initial_W: Tensor,
                   task1_S: Tensor | None = None, S: Tensor | None = None) -> dict[str, float]:
    S = model.S if S is None else S
    tiny = torch.finfo(S.dtype).tiny
    out = dict(weight_norm=model.W.norm().item(), trace_S_over_d=(S.trace() / model.d).item(),
               relative_weight_displacement=((model.W - initial_W).norm()
                                             / initial_W.norm().clamp_min(tiny)).item())
    for name, teacher in teachers.items():
        target = teacher.S
        out[f"{name}_matrix_error"] = ((S - target).square().sum()
                                            / target.square().sum().clamp_min(tiny)).item()
        out[f"{name}_matrix_overlap"] = ((S * target).sum()
                                          / (S.norm() * target.norm()).clamp_min(tiny)).item()
    if task1_S is not None:
        out["task1_matrix_displacement"] = ((S - task1_S).norm()
                                              / task1_S.norm().clamp_min(tiny)).item()
    return out


def forgetting(current: float, task1_endpoint: float, initial: float) -> dict:
    """Relative forgetting is undefined if Task 1 acquired no meaningful improvement."""
    acquired = initial - task1_endpoint
    absolute = current - task1_endpoint
    return dict(forgetting_abs=absolute, acquired_task1_improvement=acquired,
                forgetting_relative=absolute / acquired if acquired > 1e-12 else None)
