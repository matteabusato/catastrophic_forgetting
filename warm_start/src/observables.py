"""Prediction, attention and matrix diagnostics, evaluated without gradients."""

from __future__ import annotations

import torch
from torch import Tensor

from .data import Dataset
from .losses import attention_kl, output_mse
from .model import Student


@torch.no_grad()
def evaluate(model: Student, data: Dataset, batch_size: int = 1024) -> dict[str, float]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    totals = dict(mse=0.0, attention_kl=0.0, attention_entropy=0.0)
    for start in range(0, len(data), batch_size):
        X = data.X[start:start + batch_size]
        logits = model.attention_logits(X)
        attention = logits.softmax(dim=-1)
        values = {
            "mse": output_mse(attention @ X, data.y[start:start + batch_size]),
            "attention_kl": attention_kl(logits, data.attention[start:start + batch_size]),
            "attention_entropy": -torch.special.xlogy(attention, attention).sum(-1).mean(),
        }
        for key, value in values.items():
            totals[key] += len(X) * value.item()
    return {key: value / len(data) for key, value in totals.items()}


@torch.no_grad()
def matrix_metrics(model: Student, teacher_S: Tensor,
                   initial_W: Tensor | None = None) -> dict[str, float]:
    S = model.S
    tiny = torch.finfo(S.dtype).tiny
    result = {
        "representation_error": ((S - teacher_S).square().sum()
                                 / teacher_S.square().sum().clamp_min(tiny)).item(),
        "matrix_overlap": ((S * teacher_S).sum()
                           / (S.norm() * teacher_S.norm()).clamp_min(tiny)).item(),
        "weight_norm": model.W.norm().item(),
        "trace_S_over_d": (S.trace() / model.d).item(),
    }
    if initial_W is not None:
        result["relative_weight_displacement"] = (
            (model.W - initial_W).norm() / initial_W.norm().clamp_min(tiny)).item()
    return result
