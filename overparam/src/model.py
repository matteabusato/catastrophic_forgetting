"""Full-factor student and frozen teacher with analytically centered attention."""

import math
import torch
from torch import Tensor, nn


def centered_logits(X: Tensor, S: Tensor) -> Tensor:
    """Subtract E_X[X S X.T]=tr(S) I for iid standard Gaussian tokens."""
    if X.ndim < 2 or S.shape != (X.shape[-1], X.shape[-1]):
        raise ValueError("Expected X (...,T,d) and S (d,d).")
    T, d = X.shape[-2:]
    return (X @ S @ X.transpose(-1, -2)
            - S.trace() * torch.eye(T, dtype=X.dtype, device=X.device)) / d


class Student(nn.Module):
    """Only W of shape (d,P) is trainable; S=WW.T/sqrt(P)."""

    def __init__(self, d: int, P: int, seed: int, std: float = 1.0,
                 dtype: torch.dtype = torch.float64, device: str = "cpu"):
        super().__init__()
        if min(d, P) < 1 or not math.isfinite(std) or std <= 0:
            raise ValueError("d, P and std must be positive and finite.")
        self.d, self.P = d, P
        # Draw on CPU for portable, local RNG streams; transfer after sampling.
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter((std * torch.randn(d, P, generator=g, dtype=dtype)).to(device))

    @property
    def S(self) -> Tensor:
        return self.W @ self.W.T / math.sqrt(self.P)

    def attention_logits(self, X: Tensor) -> Tensor:
        return centered_logits(X, self.S)

    def attention(self, X: Tensor) -> Tensor:
        return self.attention_logits(X).softmax(-1)

    def forward(self, X: Tensor) -> Tensor:
        return self.attention(X) @ X


class Teacher(Student):
    """Same attention map with a permanently frozen independently sampled W."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_grad_(False)
        self.eval()
