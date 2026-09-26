from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Teachers:
    W_star: Tensor
    w1_star: Tensor
    w2_star: Tensor

    @property
    def P_star(self) -> int:
        return self.W_star.shape[1]

    @property
    def d(self) -> int:
        return self.W_star.shape[0]

    @property
    def S_star(self) -> Tensor:
        return (self.W_star @ self.W_star.T / torch.sqrt(torch.tensor(self.P_star, 
                dtype=self.W_star.dtype, device=self.W_star.device,)))


def sample_teachers(d: int, kappa_star: float, device: str = "cpu", dtype: torch.dtype = torch.float64,
    seed: int = 0,) -> Teachers:

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    P_star = int(round(kappa_star * d))

    W_star = torch.randn(d, P_star, generator=g, device=device, dtype=dtype,)

    w1_star = torch.randn(d, generator=g, device=device, dtype=dtype,)

    w2_star = torch.randn(d, generator=g, device=device, dtype=dtype,)

    return Teachers(W_star=W_star, w1_star=w1_star, w2_star=w2_star,)


def teacher_forward(X: Tensor, S: Tensor,) -> Tensor:
    d = X.shape[-1]
    T = X.shape[-2]

    A = X @ S @ X.transpose(-1, -2) / d

    # E_X[A] = Tr(S) / d * I_T
    expectation = (torch.trace(S) / d) * torch.eye(T,dtype=X.dtype,device=X.device,)

    A_centered = A - expectation

    attention = torch.softmax(A_centered, dim=-1)

    return attention @ X


def pretraining_teacher(X: Tensor, teachers: Teachers,) -> Tensor:
    return teacher_forward(X, teachers.S_star,)


def task1_teacher(X: Tensor, teachers: Teachers,) -> Tensor:
    S1 = teachers.S_star + torch.outer(teachers.w1_star, teachers.w1_star,)

    return teacher_forward(X, S1)


def task2_teacher(X: Tensor, teachers: Teachers,) -> Tensor:
    S2 = teachers.S_star + torch.outer(teachers.w2_star, teachers.w2_star,)

    return teacher_forward(X, S2)


@dataclass
class Dataset:
    X: Tensor
    y: Tensor


def sample_inputs(n_samples: int, T: int, d: int, device: str, dtype: torch.dtype,
    generator: torch.Generator,) -> Tensor:

    return torch.randn(n_samples, T, d, generator=generator, device=device, dtype=dtype,)


def generate_dataset(n_samples: int, T: int, d: int, S_teacher: Tensor, seed: int,
    device: str = "cpu", dtype: torch.dtype = torch.float64,) -> Dataset:

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    X = sample_inputs(n_samples=n_samples, T=T, d=d, device=device, dtype=dtype, generator=g,)

    with torch.no_grad():
        y = teacher_forward(X, S_teacher)

    return Dataset(X=X, y=y)
