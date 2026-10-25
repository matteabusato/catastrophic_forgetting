"""Shared centered attention map for the teacher and student."""

import torch
from torch import Tensor


def attention_logits(X: Tensor, S: Tensor) -> Tensor:
    d, T = X.shape[-1], X.shape[-2]
    scores = X @ S @ X.transpose(-1, -2) / d
    center = torch.trace(S) / d * torch.eye(T, dtype=X.dtype, device=X.device)
    return scores - center
