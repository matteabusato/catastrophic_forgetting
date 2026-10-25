from __future__ import annotations

"""Training objectives. CE is averaged over examples and query positions."""

import torch
from torch import Tensor

from src.data_model import teacher_attention, teacher_forward

LOSS_TYPES = ("output_mse", "attention_ce")
ATTENTION_REDUCTION = "mean_examples_and_query_positions_sum_keys"


def validate_loss_type(loss_type: str) -> None:
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"Unknown loss {loss_type!r}; expected one of {LOSS_TYPES}.")


def resolve_loss_type(requested: str | None, *configs: dict) -> str:
    """Inherit the source objective; legacy configurations use output MSE."""
    sources = [c.get("loss_type", "output_mse") for c in configs]
    chosen = requested or (sources[0] if sources else "output_mse")
    validate_loss_type(chosen)
    if any(source != chosen for source in sources):
        raise ValueError(f"Loss mismatch: requested {chosen!r}, source objectives {sources}.")
    for config in configs:
        reduction = config.get("attention_reduction", ATTENTION_REDUCTION)
        if reduction != ATTENTION_REDUCTION:
            raise ValueError(f"Incompatible attention loss reduction: {reduction!r}")
    return chosen


def mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
    # Preserve the original training normalization (sum positions, divide by d).
    return (prediction - target).square().sum(dim=(-2, -1)).mean() / prediction.shape[-1]


def attention_cross_entropy(student_logits: Tensor, teacher_attention: Tensor) -> Tensor:
    if student_logits.shape != teacher_attention.shape:
        raise ValueError("Student logits and teacher attention must have the same shape.")
    if student_logits.ndim < 2:
        raise ValueError("Expected attention tensors with query and key dimensions.")
    return -(teacher_attention.detach() * student_logits.log_softmax(dim=-1)).sum(dim=-1).mean()


def training_loss(model, X: Tensor, target_output: Tensor,
                  target_attention: Tensor | None, loss_type: str) -> Tensor:
    validate_loss_type(loss_type)
    if loss_type == "output_mse":
        return mse_loss(model(X), target_output)
    if target_attention is None:
        raise ValueError("Attention CE training requires teacher attention targets.")
    return attention_cross_entropy(model.attention_logits(X), target_attention)


def teacher_student_loss(model, X: Tensor, S_teacher: Tensor, loss_type: str) -> Tensor:
    validate_loss_type(loss_type)
    if loss_type == "attention_ce":
        with torch.no_grad():
            target = teacher_attention(X, S_teacher)
        return attention_cross_entropy(model.attention_logits(X), target)
    with torch.no_grad():
        target = teacher_forward(X, S_teacher)
    return mse_loss(model(X), target)
