import torch

from model import Student
from observables import representation_error, generalization_error, forgetting, vector_overlap


DTYPE = torch.float64



def test_representation_error_zero_for_identical_matrices():
    S = torch.randn(10, 10, dtype=torch.float64)

    error = representation_error(S, S)

    assert torch.allclose(
        error,
        torch.tensor(0.0, dtype=torch.float64),
    )

def test_representation_error_known_value():
    S_teacher = torch.eye(
        5,
        dtype=torch.float64,
    )

    S_student = torch.zeros(
        5,
        5,
        dtype=torch.float64,
    )

    error = representation_error(
        S_student,
        S_teacher,
    )

    # ||0 - I||_F^2 / ||I||_F^2 = 1
    assert torch.allclose(
        error,
        torch.tensor(1.0, dtype=torch.float64),
    )

def test_forgetting():
    error_after_task1 = 0.1
    error_after_task2 = 0.35

    F = forgetting(
        error_after_task1,
        error_after_task2,
    )

    assert abs(F - 0.25) < 1e-12
def test_vector_overlap_identical():
    w = torch.tensor(
        [1.0, 2.0, 3.0],
        dtype=torch.float64,
    )

    overlap = vector_overlap(w, w)

    assert torch.allclose(
        overlap,
        torch.tensor(1.0, dtype=torch.float64),
    )


def test_vector_overlap_orthogonal():
    w1 = torch.tensor(
        [1.0, 0.0],
        dtype=torch.float64,
    )

    w2 = torch.tensor(
        [0.0, 1.0],
        dtype=torch.float64,
    )

    overlap = vector_overlap(w1, w2)

    assert torch.allclose(
        overlap,
        torch.tensor(0.0, dtype=torch.float64),
    )

def test_generalization_error_zero_for_exact_predictions():
    model = Student(
        d=10,
        kappa=1.0,
        dtype=torch.float64,
    )

    X = torch.randn(
        20,
        5,
        10,
        dtype=torch.float64,
    )

    with torch.no_grad():
        y = model(X)

    error = generalization_error(
        model,
        X,
        y,
    )

    assert torch.allclose(
        error,
        torch.tensor(0.0, dtype=torch.float64),
    )