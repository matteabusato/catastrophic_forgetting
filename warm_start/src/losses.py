"""One primary objective, normalized identically in training and evaluation."""

import torch
from torch import Tensor


def output_mse(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean over examples, tokens and features: sum squared error / (N*T*d)."""
    if prediction.shape != target.shape or prediction.numel() == 0:
        raise ValueError("Prediction and target must have identical nonempty shapes.")
    return (prediction - target).square().mean()


def attention_kl(logits: Tensor, target: Tensor) -> Tensor:
    """KL(teacher || student), averaged over examples and query positions."""
    if logits.shape != target.shape:
        raise ValueError("Attention logits and targets must have identical shapes.")
    return (torch.special.xlogy(target, target)
            - target * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
