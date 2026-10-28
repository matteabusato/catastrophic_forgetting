"""Consistently normalized output MSE and diagnostic teacher-to-student KL."""

import torch
from torch import Tensor


def output_mse(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape or prediction.numel() == 0:
        raise ValueError("Prediction and target must have matching nonempty shapes.")
    return (prediction - target).square().mean()


def attention_kl(logits: Tensor, target: Tensor) -> Tensor:
    return (torch.special.xlogy(target, target)
            - target * logits.log_softmax(-1)).sum(-1).mean()
