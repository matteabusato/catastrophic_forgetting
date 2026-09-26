import torch
from torch import Tensor

from src.model import Student


def representation_error(S_student: Tensor, S_teacher: Tensor,) -> Tensor:
    """
    Relative squared Frobenius reconstruction error:

        ||S_student - S_teacher||_F^2
        --------------------------------
              ||S_teacher||_F^2
    """
    numerator = torch.sum((S_student - S_teacher) ** 2)
    denominator = torch.sum(S_teacher ** 2)

    return numerator / denominator


@torch.no_grad()
def generalization_error(model: Student, X: Tensor, y: Tensor,) -> Tensor:
    """
    Mean squared prediction error on a fixed dataset.
    """

    y_pred = model(X)

    return torch.mean((y_pred - y) ** 2)


def forgetting(task1_error_after_task1: float, task1_error_after_task2: float,) -> float:
    """
    Task-1 forgetting:

        F1 = eps_1(after Task 2) - eps_1(after Task 1)
    """

    return task1_error_after_task2 - task1_error_after_task1


def vector_overlap(w: Tensor, w_star: Tensor, eps: float = 1e-12,) -> Tensor:
    """
    Cosine overlap between a learned adapter and a task-specific teacher direction.
    """

    denominator = torch.linalg.vector_norm(w) * torch.linalg.vector_norm(w_star)

    return torch.dot(w, w_star) / (denominator + eps)


def residual_norm(S_student: Tensor, S_teacher: Tensor,) -> Tensor:
    """
    Absolute Frobenius norm of the residual S* - S.
    """

    return torch.linalg.matrix_norm(S_teacher - S_student, ord="fro",)