"""Independent teachers, reproducible online Gaussian inputs and fixed diagnostics."""

from dataclasses import dataclass
import hashlib
import torch
from torch import Tensor
from .model import Teacher


def role_seed(seed: int, role: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{role}".encode()).digest()[:8], "big") % (2**63 - 1)


class GaussianStream:
    """Local CPU RNG; equal seeds give equal inputs independently of student width."""

    def __init__(self, seed: int, tokens: int, d: int, dtype: torch.dtype, device: str):
        self.generator = torch.Generator().manual_seed(seed)
        self.tokens, self.d, self.dtype, self.device = tokens, d, dtype, device

    def sample(self, n: int) -> Tensor:
        if n < 1:
            raise ValueError("n must be positive.")
        return torch.randn(n, self.tokens, self.d, generator=self.generator,
                           dtype=self.dtype).to(self.device)


@dataclass(frozen=True)
class Dataset:
    X: Tensor
    y: Tensor
    attention: Tensor

    def __len__(self):
        return len(self.X)


def make_teachers(d: int, P_star: int, seed: int, std: float,
                  dtype: torch.dtype, device: str) -> dict[str, Teacher]:
    """rho=0: independent factors, fixed P_star across every student width."""
    return {name: Teacher(d, P_star, role_seed(seed, name), std, dtype, device)
            for name in ("task1", "task2")}


@torch.no_grad()
def label_inputs(X: Tensor, teacher: Teacher, batch_size: int = 256) -> Dataset:
    if X.ndim != 3 or len(X) == 0 or batch_size < 1:
        raise ValueError("Expected nonempty X (N,T,d) and positive batch_size.")
    attention = torch.cat([teacher.attention(x) for x in X.split(batch_size)])
    return Dataset(X, attention @ X, attention)
