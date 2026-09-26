#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor, nn


ActivationName = Literal["softmax", "identity"]


@dataclass(frozen=True)
class ModelConfig:
    d: int = 200
    T: int = 5
    kappa_star: float = 1.0
    kappa: float = 1.0
    activation: ActivationName = "softmax"
    init_std_W: float = 1.0
    init_std_w: float = 1.0
    dtype: torch.dtype = torch.float32

    @property
    def P_star(self) -> int:
        p = int(round(self.kappa_star * self.d))
        if p <= 0:
            raise ValueError("P_star must be positive.")
        return p

    @property
    def P(self) -> int:
        p = int(round(self.kappa * self.d))
        if p <= 0:
            raise ValueError("P must be positive.")
        return p

    def validate(self) -> None:
        if self.d <= 0:
            raise ValueError("d must be positive.")
        if self.T <= 0:
            raise ValueError("T must be positive.")
        if self.kappa_star <= 0:
            raise ValueError("kappa_star must be positive.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.activation not in ("softmax", "identity"):
            raise ValueError(
                f"Unknown activation {self.activation!r}. "
                "Use 'softmax' or 'identity'."
            )
        if self.init_std_W <= 0 or self.init_std_w <= 0:
            raise ValueError("Initialization standard deviations must be positive.")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_gaussian_input(
    T: int,
    d: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    return torch.randn(
        T,
        d,
        device=device,
        dtype=dtype,
        generator=generator,
    )


def _check_shapes(X: Tensor, W: Tensor, w: Optional[Tensor]) -> None:
    if X.ndim != 2:
        raise ValueError(f"X must have shape [T, d], got {tuple(X.shape)}.")
    if W.ndim != 2:
        raise ValueError(f"W must have shape [d, P], got {tuple(W.shape)}.")

    d = X.shape[1]
    if W.shape[0] != d:
        raise ValueError(f"Dimension mismatch: X has d={d}, while W has shape {tuple(W.shape)}.")

    if w is not None:
        if w.ndim != 1 or w.shape[0] != d:
            raise ValueError(f"w must have shape [{d}], got {tuple(w.shape)}.")


def centered_pre_activation(
    X: Tensor,
    W: Tensor,
    w: Optional[Tensor] = None,
) -> Tensor:
    _check_shapes(X, W, w)

    T, d = X.shape
    P = W.shape[1]

    # Extensive-rank contribution, factorized to avoid constructing WW^T.
    XW = X @ W                                      # [T, P]
    A = (XW @ XW.transpose(0, 1)) / (d * (P ** 0.5))

    # Exact Gaussian expectation of the extensive-rank diagonal contribution.
    center = W.square().sum() / (d * (P ** 0.5))

    if w is not None:
        Xw = X @ w                                  # [T]
        A = A + torch.outer(Xw, Xw) / d
        center = center + w.square().sum() / d

    eye = torch.eye(T, device=X.device, dtype=X.dtype)
    return A - center * eye


def apply_activation(z: Tensor, activation: ActivationName) -> Tensor:
    if z.ndim != 2 or z.shape[0] != z.shape[1]:
        raise ValueError(
            f"z must be a square [T, T] matrix, got {tuple(z.shape)}."
        )

    if activation == "softmax":
        return torch.softmax(z, dim=-1)

    if activation == "identity":
        T = z.shape[0]
        return z / T

    raise ValueError(
        f"Unknown activation {activation!r}. Use 'softmax' or 'identity'."
    )


def sequence_prediction(
    X: Tensor,
    W: Tensor,
    w: Optional[Tensor],
    *,
    activation: ActivationName,
) -> Tensor:
    
    z = centered_pre_activation(X, W, w)
    return apply_activation(z, activation) @ X


def squared_sequence_loss(y_hat: Tensor, y: Tensor) -> Tensor:

    if y_hat.shape != y.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: "
            f"{tuple(y_hat.shape)} vs {tuple(y.shape)}."
        )

    if y_hat.ndim != 2:
        raise ValueError("Expected y_hat and y to have shape [T, d].")

    d = y_hat.shape[1]
    return (y_hat - y).square().sum() / d


class TeacherModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config

        W_star = torch.randn(
            config.d,
            config.P_star,
            device=device,
            dtype=config.dtype,
        )
        w_star = torch.randn(
            config.d,
            device=device,
            dtype=config.dtype,
        )

        self.register_buffer("W_star", W_star)
        self.register_buffer("w_star", w_star)

    @torch.no_grad()
    def pretrain_target(self, X: Tensor) -> Tensor:
        """Generate y_S(X): extensive-rank teacher only."""
        return sequence_prediction(
            X,
            self.W_star,
            None,
            activation=self.config.activation,
        )

    @torch.no_grad()
    def finetune_target(self, X: Tensor) -> Tensor:
        """Generate y_S'(X): extensive-rank teacher + rank-one teacher."""
        return sequence_prediction(
            X,
            self.W_star,
            self.w_star,
            activation=self.config.activation,
        )


class StudentModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.W = nn.Parameter(
            torch.empty(
                config.d,
                config.P,
                device=device,
                dtype=config.dtype,
            )
        )
        self.w = nn.Parameter(
            torch.empty(
                config.d,
                device=device,
                dtype=config.dtype,
            )
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.W.normal_(mean=0.0, std=self.config.init_std_W)
        self.w.normal_(mean=0.0, std=self.config.init_std_w)

    def forward(
        self,
        X: Tensor,
        *,
        include_lora: bool = True,
    ) -> Tensor:
        w = self.w if include_lora else None
        return sequence_prediction(
            X,
            self.W,
            w,
            activation=self.config.activation,
        )

    def set_trainable(
        self,
        *,
        train_W: bool,
        train_w: bool,
    ) -> None:
        self.W.requires_grad_(train_W)
        self.w.requires_grad_(train_w)


def _sanity_check() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ModelConfig(
        d=200,
        T=5,
        kappa_star=1.0,
        kappa=1.0,
        activation="softmax",
    )

    set_seed(0)
    teacher = TeacherModel(config, device=device)
    student = StudentModel(config, device=device)

    X = sample_gaussian_input(
        config.T,
        config.d,
        device=device,
        dtype=config.dtype,
    )

    y_S = teacher.pretrain_target(X)
    y_Sprime = teacher.finetune_target(X)

    with torch.no_grad():
        student.W.copy_(teacher.W_star)
        student.w.zero_()

    old_loss = squared_sequence_loss(
        student(X, include_lora=True),
        y_S,
    )

    with torch.no_grad():
        student.w.copy_(teacher.w_star)

    ft_loss = squared_sequence_loss(
        student(X, include_lora=True),
        y_Sprime,
    )

    print(f"device           : {device}")
    print(f"old-task loss    : {old_loss.item():.6e}")
    print(f"fine-tuning loss : {ft_loss.item():.6e}")


if __name__ == "__main__":
    _sanity_check()