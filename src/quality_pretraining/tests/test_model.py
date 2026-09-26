import torch

from model import Student


DTYPE = torch.float64


def test_student_shapes():
    d = 20
    kappa = 1.5

    model = Student(
        d=d,
        kappa=kappa,
        dtype=DTYPE,
    )

    P = int(round(kappa * d))

    assert model.W.shape == (d, P)
    assert model.w.shape == (d,)
    assert model.S_large.shape == (d, d)
    assert model.S.shape == (d, d)


def test_large_rank_matrix_is_symmetric():
    model = Student(
        d=20,
        kappa=1.0,
        dtype=DTYPE,
    )

    assert torch.allclose(
        model.S_large,
        model.S_large.T,
    )


def test_full_matrix_is_symmetric():
    model = Student(
        d=20,
        kappa=1.0,
        use_low_rank=True,
        dtype=DTYPE,
    )

    model.reset_low_rank(
        scale=0.1,
        seed=0,
    )

    assert torch.allclose(
        model.S,
        model.S.T,
    )


def test_low_rank_component_is_inactive_when_disabled():
    model = Student(
        d=20,
        kappa=1.0,
        use_low_rank=False,
        dtype=DTYPE,
    )

    # Give w a non-zero value.
    model.reset_low_rank(
        scale=0.1,
        seed=0,
    )

    # Since use_low_rank=False, w should not contribute.
    assert torch.allclose(
        model.S,
        model.S_large,
    )


def test_low_rank_component_is_added_when_enabled():
    model = Student(
        d=20,
        kappa=1.0,
        use_low_rank=False,
        dtype=DTYPE,
    )

    model.reset_low_rank(
        scale=0.1,
        seed=0,
    )

    model.set_finetuning_mode()

    expected = (
        model.S_large
        + torch.outer(model.w, model.w)
    )

    assert torch.allclose(
        model.S,
        expected,
    )


def test_forward_shape():
    d = 20
    T = 5
    N = 10

    model = Student(
        d=d,
        kappa=1.0,
        dtype=DTYPE,
    )

    X = torch.randn(
        N,
        T,
        d,
        dtype=DTYPE,
    )

    y = model(X)

    assert y.shape == (N, T, d)


def test_pretraining_mode():
    model = Student(
        d=20,
        kappa=1.0,
        use_low_rank=True,
        dtype=DTYPE,
    )

    model.reset_low_rank(
        scale=0.1,
        seed=0,
    )

    # Make sure w is initially non-zero.
    assert not torch.allclose(
        model.w,
        torch.zeros_like(model.w),
    )

    model.set_pretraining_mode()

    assert model.use_low_rank is False
    assert model.W.requires_grad is True
    assert model.w.requires_grad is False

    # set_pretraining_mode should also reset w to zero.
    assert torch.allclose(
        model.w,
        torch.zeros_like(model.w),
    )


def test_finetuning_mode():
    model = Student(
        d=20,
        kappa=1.0,
        dtype=DTYPE,
    )

    model.reset_low_rank(
        scale=0.1,
        seed=0,
    )

    w_before = model.w.detach().clone()

    model.set_finetuning_mode()

    assert model.use_low_rank is True
    assert model.W.requires_grad is False
    assert model.w.requires_grad is True

    # Switching mode should not change w.
    assert torch.equal(
        model.w,
        w_before,
    )


def test_reset_low_rank_is_reproducible():
    model = Student(
        d=20,
        kappa=1.0,
        dtype=DTYPE,
    )

    model.reset_low_rank(
        scale=0.1,
        seed=42,
    )

    w1 = model.w.detach().clone()

    model.reset_low_rank(
        scale=0.1,
        seed=42,
    )

    w2 = model.w.detach().clone()

    assert torch.equal(
        w1,
        w2,
    )


def test_reset_low_rank_different_seeds():
    model = Student(
        d=20,
        kappa=1.0,
        dtype=DTYPE,
    )

    model.reset_low_rank(
        scale=0.1,
        seed=1,
    )

    w1 = model.w.detach().clone()

    model.reset_low_rank(
        scale=0.1,
        seed=2,
    )

    w2 = model.w.detach().clone()

    assert not torch.equal(
        w1,
        w2,
    )


def test_reset_low_rank_is_nonzero():
    model = Student(
        d=20,
        kappa=1.0,
        dtype=DTYPE,
    )

    model.reset_low_rank(
        scale=0.1,
        seed=0,
    )

    assert not torch.allclose(
        model.w,
        torch.zeros_like(model.w),
    )


def test_load_large_rank():
    d = 20
    kappa = 1.5
    P = int(round(kappa * d))

    model = Student(
        d=d,
        kappa=kappa,
        dtype=DTYPE,
    )

    W_checkpoint = torch.randn(
        d,
        P,
        dtype=DTYPE,
    )

    model.load_large_rank(
        W_checkpoint,
    )

    assert torch.allclose(
        model.W,
        W_checkpoint,
    )


def test_pretraining_gradient_flow():
    """
    During pre-training:
        W should receive gradients.
        w should not receive gradients.
    """

    torch.manual_seed(0)

    d = 10
    T = 4
    N = 8

    model = Student(
        d=d,
        kappa=1.0,
        dtype=DTYPE,
    )

    model.set_pretraining_mode()

    X = torch.randn(
        N,
        T,
        d,
        dtype=DTYPE,
    )

    target = torch.randn(
        N,
        T,
        d,
        dtype=DTYPE,
    )

    prediction = model(X)

    loss = torch.mean(
        (prediction - target) ** 2
    )

    loss.backward()

    assert model.W.grad is not None
    assert model.W.grad.norm() > 0

    assert model.w.grad is None


def test_finetuning_gradient_flow():
    """
    During fine-tuning:
        W should be frozen.
        w should receive gradients.
    """

    torch.manual_seed(0)

    d = 10
    T = 4
    N = 8

    model = Student(
        d=d,
        kappa=1.0,
        dtype=DTYPE,
    )

    # w must be non-zero before fine-tuning,
    # otherwise d(ww^T)/dw = 0 at w = 0.
    model.reset_low_rank(
        scale=0.1,
        seed=0,
    )

    model.set_finetuning_mode()

    X = torch.randn(
        N,
        T,
        d,
        dtype=DTYPE,
    )

    target = torch.randn(
        N,
        T,
        d,
        dtype=DTYPE,
    )

    prediction = model(X)

    loss = torch.mean(
        (prediction - target) ** 2
    )

    loss.backward()

    assert model.W.grad is None

    assert model.w.grad is not None
    assert model.w.grad.norm() > 0