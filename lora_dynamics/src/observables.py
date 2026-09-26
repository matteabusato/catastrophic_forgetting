#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Literal, Optional

import torch
from torch import Tensor

from model import (
    StudentModel,
    TeacherModel,
    sample_gaussian_input,
    sequence_prediction,
    squared_sequence_loss,
)


TaskName = Literal["S", "Sprime"]


@dataclass
class FixedTestSet:
    X: Tensor       # [n, T, d]
    y: Tensor       # [n, T, d]
    task: TaskName

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])


@dataclass
class ForgettingTracker:
    best_old_error: float = float("inf")

    def update(self, old_error: float) -> float:
        self.best_old_error = min(self.best_old_error, float(old_error))
        return float(old_error) - self.best_old_error


def effective_matrix(W: Tensor) -> Tensor:
    r"""Return S = P^{-1/2} W W^T."""
    if W.ndim != 2:
        raise ValueError(f"W must have shape [d, P], got {tuple(W.shape)}.")
    P = W.shape[1]
    return (W @ W.transpose(0, 1)) / sqrt(P)


def centered_matrix(S: Tensor) -> Tensor:
    r"""Return S - Tr(S) I_d / d."""
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"S must be square, got {tuple(S.shape)}.")
    d = S.shape[0]
    eye = torch.eye(d, device=S.device, dtype=S.dtype)
    return S - (torch.trace(S) / d) * eye


@torch.no_grad()
def compute_order_parameters(
    teacher: TeacherModel,
    student: StudentModel,
    *,
    delta: float = 0.0,
    eps: float = 1e-12,
) -> Dict[str, float]:
    W_star = teacher.W_star
    w_star = teacher.w_star
    W = student.W
    w = student.w

    d = int(W.shape[0])
    if W_star.shape[0] != d or w_star.shape[0] != d or w.shape[0] != d:
        raise ValueError("Teacher and student must have the same ambient dimension d.")

    S_star = effective_matrix(W_star)
    S = effective_matrix(W)

    d_float = float(d)
    d2 = d_float * d_float

    # Extensive-rank order parameters.
    Q0_t = torch.sum(S_star * S_star) / d2
    Q_t = torch.sum(S * S) / d2
    M_t = torch.sum(S * S_star) / d2

    Q0 = float(Q0_t.item())
    Q = float(Q_t.item())
    M = float(M_t.item())

    denom_W = sqrt(max(Q * Q0, eps))
    rho_W = M / denom_W
    eps_W = Q + Q0 - 2.0 * M

    # Rank-one order parameters.
    q0_t = torch.dot(w_star, w_star) / d_float
    q_t = torch.dot(w, w) / d_float
    m_t = torch.dot(w, w_star) / d_float

    q0 = float(q0_t.item())
    q = float(q_t.item())
    m = float(m_t.item())

    denom_w = sqrt(max(q * q0, eps))
    rho_w = abs(m) / denom_w
    eps_w = q * q + q0 * q0 - 2.0 * m * m

    # Centered effective matrices for cross-contamination diagnostics.
    S_centered = centered_matrix(S)
    S_star_centered = centered_matrix(S_star)

    C_W_from_wstar = float(
        (torch.dot(w_star, S_centered @ w_star) / d2).item()
    )
    C_w_from_Wstar = float(
        (torch.dot(w, S_star_centered @ w) / d2).item()
    )
    C_W_w = float(
        (torch.dot(w, S_centered @ w) / d2).item()
    )

    delta_eff = 0.5 * float(delta) + eps_W

    return {
        "Q0": Q0,
        "Q": Q,
        "M": M,
        "rho_W": rho_W,
        "eps_W": eps_W,
        "q0": q0,
        "q": q,
        "m": m,
        "rho_w": rho_w,
        "eps_w": eps_w,
        "C_W_from_wstar": C_W_from_wstar,
        "C_w_from_Wstar": C_w_from_Wstar,
        "C_W_w": C_W_w,
        "Delta_eff": delta_eff,
    }


@torch.no_grad()
def make_fixed_test_set(
    teacher: TeacherModel,
    *,
    task: TaskName,
    n_samples: int,
    seed: int,
    device: Optional[torch.device | str] = None,
) -> FixedTestSet:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if task not in ("S", "Sprime"):
        raise ValueError("task must be 'S' or 'Sprime'.")

    if device is None:
        device = teacher.W_star.device
    device = torch.device(device)

    config = teacher.config
    # A private generator prevents construction of the fixed test set from
    # consuming the random stream used later by SGD.
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    X_all = torch.empty(
        n_samples,
        config.T,
        config.d,
        device=device,
        dtype=config.dtype,
    )
    y_all = torch.empty_like(X_all)

    target_fn = teacher.pretrain_target if task == "S" else teacher.finetune_target

    # We deliberately generate/evaluate one sequence at a time. This keeps the
    # semantics identical to the non-batched model; test-set construction is
    # done only once per run, so this is not performance-critical at d=200.
    for i in range(n_samples):
        X = sample_gaussian_input(
            config.T,
            config.d,
            device=device,
            dtype=config.dtype,
            generator=gen,
        )
        X_all[i].copy_(X)
        y_all[i].copy_(target_fn(X))

    return FixedTestSet(X=X_all, y=y_all, task=task)


@torch.no_grad()
def _mean_test_loss(
    student: StudentModel,
    test_set: FixedTestSet,
    *,
    include_lora: bool,
) -> float:
    total = torch.zeros((), device=student.W.device, dtype=student.W.dtype)

    for i in range(test_set.n_samples):
        pred = student(test_set.X[i], include_lora=include_lora)
        total += squared_sequence_loss(pred, test_set.y[i])

    return float((total / test_set.n_samples).item())


@torch.no_grad()
def evaluate_test_errors(
    student: StudentModel,
    old_test_set: FixedTestSet,
    new_test_set: FixedTestSet,
) -> Dict[str, float]:
    if old_test_set.task != "S":
        raise ValueError("old_test_set must have task='S'.")
    if new_test_set.task != "Sprime":
        raise ValueError("new_test_set must have task='Sprime'.")

    E_S = _mean_test_loss(student, old_test_set, include_lora=True)
    E_Sprime = _mean_test_loss(student, new_test_set, include_lora=True)
    E_S_W_only = _mean_test_loss(student, old_test_set, include_lora=False)

    return {
        "E_S": E_S,
        "E_Sprime": E_Sprime,
        "E_S_W_only": E_S_W_only,
    }


def _task_loss_from_free_parameters(
    teacher: TeacherModel,
    X: Tensor,
    *,
    task: TaskName,
    W: Tensor,
    w: Tensor,
) -> Tensor:
    if task == "S":
        y = teacher.pretrain_target(X)
    elif task == "Sprime":
        y = teacher.finetune_target(X)
    else:
        raise ValueError("task must be 'S' or 'Sprime'.")

    pred = sequence_prediction(
        X,
        W,
        w,
        activation=teacher.config.activation,
    )
    return squared_sequence_loss(pred, y)


def _mean_task_loss_from_free_parameters(
    teacher: TeacherModel,
    X_probe: Tensor,
    *,
    task: TaskName,
    W: Tensor,
    w: Tensor,
) -> Tensor:
    if X_probe.ndim == 2:
        return _task_loss_from_free_parameters(
            teacher, X_probe, task=task, W=W, w=w
        )

    if X_probe.ndim != 3:
        raise ValueError(
            f"X_probe must have shape [T,d] or [n,T,d], got {tuple(X_probe.shape)}."
        )

    losses = []
    for i in range(X_probe.shape[0]):
        losses.append(
            _task_loss_from_free_parameters(
                teacher, X_probe[i], task=task, W=W, w=w
            )
        )
    return torch.stack(losses).mean()


def _cosine(a: Tensor, b: Tensor, eps: float = 1e-20) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = torch.linalg.vector_norm(a_flat) * torch.linalg.vector_norm(b_flat)
    if float(denom.detach().item()) <= eps:
        return float("nan")
    return float((torch.dot(a_flat, b_flat) / denom).detach().item())


def compute_gradient_observables(
    teacher: TeacherModel,
    student: StudentModel,
    *,
    X_old_probe: Tensor,
    X_new_probe: Tensor,
) -> Dict[str, float]:
    W = student.W.detach().clone().requires_grad_(True)
    w = student.w.detach().clone().requires_grad_(True)

    loss_S = _mean_task_loss_from_free_parameters(
        teacher,
        X_old_probe,
        task="S",
        W=W,
        w=w,
    )
    grad_W_S, grad_w_S = torch.autograd.grad(
        loss_S,
        (W, w),
        retain_graph=False,
        create_graph=False,
    )

    # Recreate leaves for the second task so the two diagnostic graphs are
    # completely independent.
    W2 = student.W.detach().clone().requires_grad_(True)
    w2 = student.w.detach().clone().requires_grad_(True)

    loss_Sprime = _mean_task_loss_from_free_parameters(
        teacher,
        X_new_probe,
        task="Sprime",
        W=W2,
        w=w2,
    )
    grad_W_Sprime, grad_w_Sprime = torch.autograd.grad(
        loss_Sprime,
        (W2, w2),
        retain_graph=False,
        create_graph=False,
    )

    grad_W_S_norm = float(torch.linalg.vector_norm(grad_W_S).detach().item())
    grad_W_Sprime_norm = float(
        torch.linalg.vector_norm(grad_W_Sprime).detach().item()
    )
    grad_w_S_norm = float(torch.linalg.vector_norm(grad_w_S).detach().item())
    grad_w_Sprime_norm = float(
        torch.linalg.vector_norm(grad_w_Sprime).detach().item()
    )

    return {
        "grad_W_S_norm": grad_W_S_norm,
        "grad_W_Sprime_norm": grad_W_Sprime_norm,
        "grad_w_S_norm": grad_w_S_norm,
        "grad_w_Sprime_norm": grad_w_Sprime_norm,
        "Gamma_W": _cosine(grad_W_S, grad_W_Sprime),
        "Gamma_w": _cosine(grad_w_S, grad_w_Sprime),
    }


def compute_all_observables(
    teacher: TeacherModel,
    student: StudentModel,
    *,
    old_test_set: FixedTestSet,
    new_test_set: FixedTestSet,
    X_old_probe: Optional[Tensor] = None,
    X_new_probe: Optional[Tensor] = None,
    forgetting_tracker: Optional[ForgettingTracker] = None,
    delta: float = 0.0,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics.update(compute_order_parameters(teacher, student, delta=delta))
    metrics.update(evaluate_test_errors(student, old_test_set, new_test_set))

    if (X_old_probe is None) != (X_new_probe is None):
        raise ValueError(
            "Provide both X_old_probe and X_new_probe, or neither of them."
        )

    if X_old_probe is not None and X_new_probe is not None:
        metrics.update(
            compute_gradient_observables(
                teacher,
                student,
                X_old_probe=X_old_probe,
                X_new_probe=X_new_probe,
            )
        )

    if forgetting_tracker is not None:
        metrics["F_S"] = forgetting_tracker.update(metrics["E_S"])

    return metrics


def _sanity_check() -> None:
    from model import ModelConfig, set_seed

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

    old_test = make_fixed_test_set(
        teacher, task="S", n_samples=8, seed=10, device=device
    )
    new_test = make_fixed_test_set(
        teacher, task="Sprime", n_samples=8, seed=11, device=device
    )

    # Exact teacher matching for both blocks.
    with torch.no_grad():
        student.W.copy_(teacher.W_star)
        student.w.copy_(teacher.w_star)

    op = compute_order_parameters(teacher, student)
    errors = evaluate_test_errors(student, old_test, new_test)

    print(f"device             : {device}")
    print(f"Q0                 : {op['Q0']:.6f}")
    print(f"Q                  : {op['Q']:.6f}")
    print(f"M                  : {op['M']:.6f}")
    print(f"rho_W              : {op['rho_W']:.6f}  (should be 1)")
    print(f"eps_W              : {op['eps_W']:.6e}  (should be 0)")
    print(f"rho_w              : {op['rho_w']:.6f}  (should be 1)")
    print(f"eps_w              : {op['eps_w']:.6e}  (should be 0)")
    print(f"E_Sprime           : {errors['E_Sprime']:.6e}  (should be 0)")
    print(f"E_S_W_only         : {errors['E_S_W_only']:.6e}  (should be 0)")
    print(
        f"E_S (LoRA enabled) : {errors['E_S']:.6e}  "
        "(generally > 0: LoRA interferes with old task)"
    )


if __name__ == "__main__":
    _sanity_check()