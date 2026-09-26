import torch

from data_model import (
    sample_teachers,
    teacher_forward,
    generate_dataset,
)


def test_teacher_shapes():
    d = 20
    kappa_star = 1.5

    teachers = sample_teachers(
        d=d,
        kappa_star=kappa_star,
        seed=0,
    )

    assert teachers.W_star.shape == (20, 30)
    assert teachers.w1_star.shape == (20,)
    assert teachers.w2_star.shape == (20,)

    assert teachers.S_star.shape == (20, 20)
    assert teachers.S1_star.shape == (20, 20)
    assert teachers.S2_star.shape == (20, 20)


def test_teacher_matrices_are_symmetric():
    teachers = sample_teachers(
        d=20,
        kappa_star=1.0,
        seed=0,
    )

    assert torch.allclose(
        teachers.S_star,
        teachers.S_star.T,
    )

    assert torch.allclose(
        teachers.S1_star,
        teachers.S1_star.T,
    )

    assert torch.allclose(
        teachers.S2_star,
        teachers.S2_star.T,
    )

def test_teacher_reproducibility():
    t1 = sample_teachers(
        d=20,
        kappa_star=1.0,
        seed=42,
    )

    t2 = sample_teachers(
        d=20,
        kappa_star=1.0,
        seed=42,
    )

    assert torch.equal(t1.W_star, t2.W_star)
    assert torch.equal(t1.w1_star, t2.w1_star)
    assert torch.equal(t1.w2_star, t2.w2_star)

def test_dataset_shapes():
    d = 20
    T = 5
    N = 100

    teachers = sample_teachers(
        d=d,
        kappa_star=1.0,
        seed=0,
    )

    dataset = generate_dataset(
        n_samples=N,
        T=T,
        d=d,
        S_teacher=teachers.S_star,
        seed=1,
    )

    assert dataset.X.shape == (N, T, d)
    assert dataset.y.shape == (N, T, d)

def test_expected_pre_activation():
    torch.manual_seed(0)

    d = 20
    T = 5
    N = 50_000

    teachers = sample_teachers(
        d=d,
        kappa_star=1.0,
        seed=0,
    )

    S = teachers.S_star

    X = torch.randn(
        N,
        T,
        d,
        dtype=torch.float64,
    )

    A = X @ S @ X.transpose(-1, -2) / d

    empirical = A.mean(dim=0)

    expected = (
        torch.trace(S) / d
    ) * torch.eye(
        T,
        dtype=torch.float64,
    )

    assert torch.allclose(
        empirical,
        expected,
        atol=5e-2,
        rtol=5e-2,
    )