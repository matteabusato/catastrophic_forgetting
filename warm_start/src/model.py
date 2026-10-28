"""One trainable factor W, with analytical Gaussian logit centering."""

import math

import torch
from torch import Tensor, nn


def centered_logits(X: Tensor, S: Tensor) -> Tensor:
    """X: (..., T, d), S: (d, d); expectation is over iid N(0,I_d) tokens.

    E_X[X S X.T / d | S] = tr(S) I_T / d. The trace stays in autograd.
    """
    if X.ndim < 2 or S.ndim != 2 or S.shape != (X.shape[-1], X.shape[-1]):
        raise ValueError("Expected X (..., T, d) and S (d, d).")
    d, T = X.shape[-1], X.shape[-2]
    return (X @ S @ X.transpose(-1, -2)
            - S.trace() * torch.eye(T, device=X.device, dtype=X.dtype)) / d


class Student(nn.Module):
    """W is (P,d), S=W.T W/sqrt(P), y=softmax(centered logits) X."""

    def __init__(self, d: int, P: int, seed: int = 0, init_scale: float = 1.0,
                 dtype: torch.dtype = torch.float64, device: str = "cpu"):
        super().__init__()
        if d < 1 or P < 1 or not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError("d, P and init_scale must be positive and finite.")
        self.d, self.P = d, P
        # Local RNG: model construction does not change the global RNG stream.
        generator = torch.Generator(device=device).manual_seed(seed)
        self.W = nn.Parameter(init_scale * torch.randn(
            P, d, generator=generator, dtype=dtype, device=device))

    @property
    def S(self) -> Tensor:
        return self.W.T @ self.W / math.sqrt(self.P)

    def attention_logits(self, X: Tensor) -> Tensor:
        return centered_logits(X, self.S)

    def attention(self, X: Tensor) -> Tensor:
        return self.attention_logits(X).softmax(dim=-1)

    def forward(self, X: Tensor) -> Tensor:
        return self.attention(X) @ X


class Teacher(Student):
    """Same function class, independently sampled factor, permanently frozen."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_grad_(False)
        self.eval()
