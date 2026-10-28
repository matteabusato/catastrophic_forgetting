"""Fixed, noiseless teacher-labelled Gaussian sequences; no online resampling."""

from dataclasses import dataclass

import torch
from torch import Tensor

from .model import Teacher


@dataclass(frozen=True)
class Dataset:
    X: Tensor
    y: Tensor
    attention: Tensor

    def __post_init__(self):
        if self.X.ndim != 3 or self.y.shape != self.X.shape or self.X.shape[0] < 1:
            raise ValueError("Expected nonempty X,y of shape (N,T,d).")
        if self.attention.shape != (*self.X.shape[:2], self.X.shape[1]):
            raise ValueError("Expected attention targets of shape (N,T,T).")

    def __len__(self) -> int:
        return self.X.shape[0]

    def subset(self, indices: Tensor) -> "Dataset":
        return Dataset(self.X[indices], self.y[indices], self.attention[indices])


@dataclass(frozen=True)
class DataSplit:
    full: Dataset
    A: Dataset
    B: Dataset
    test: Dataset
    indices_A: Tensor
    indices_B: Tensor


@torch.no_grad()
def generate_dataset(teacher: Teacher, n: int, T: int, seed: int,
                     batch_size: int = 1024) -> Dataset:
    if n < 1 or T < 2 or batch_size < 1:
        raise ValueError("Require n>=1, T>=2 and batch_size>=1.")
    generator = torch.Generator(device=teacher.W.device).manual_seed(seed)
    X = torch.randn(n, T, teacher.d, generator=generator,
                    device=teacher.W.device, dtype=teacher.W.dtype)
    attention = torch.cat([teacher.attention(chunk) for chunk in X.split(batch_size)])
    return Dataset(X, attention @ X, attention)


def make_split(teacher: Teacher, n_train: int, n_test: int, T: int,
               initial_fraction: float, train_seed: int, test_seed: int,
               split_seed: int) -> DataSplit:
    if n_train < 2 or not 0 < initial_fraction < 1:
        raise ValueError("Need n_train>=2 and 0<initial_fraction<1.")
    if train_seed == test_seed:
        raise ValueError("Train and test RNG seeds must differ.")
    n_A = int(n_train * initial_fraction)
    if not 0 < n_A < n_train:
        raise ValueError("initial_fraction must produce two nonempty subsets.")
    full = generate_dataset(teacher, n_train, T, train_seed)
    test = generate_dataset(teacher, n_test, T, test_seed)
    generator = torch.Generator(device=teacher.W.device).manual_seed(split_seed)
    indices = torch.randperm(n_train, generator=generator, device=teacher.W.device)
    a, b = indices[:n_A], indices[n_A:]
    return DataSplit(full, full.subset(a), full.subset(b), test, a, b)
