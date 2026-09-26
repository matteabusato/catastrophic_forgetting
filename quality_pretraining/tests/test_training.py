import math

import torch

from data_model import (
    sample_teachers,
    generate_dataset,
)
from model import Student
from observables import generalization_error
from training import (
    pretrain,
    finetune_sequential,
)


DTYPE = torch.float64


# ============================================================
# Helpers
# ============================================================


def make_tiny_problem():
    """
    Create a very small RQ1 instance for testing.

    The dimensions are deliberately small so the tests run
    quickly on CPU.
    """

    d = 6
    T = 3
    kappa = 1.0
    kappa_star = 1.0

    teachers = sample_teachers(
        d=d,
        kappa_star=kappa_star,
        dtype=DTYPE,
        seed=0,
    )

    S_pre = teachers.S_star

    S_task1 = (
        teachers.S_star
        + torch.outer(
            teachers.w1_star,
            teachers.w1_star,
        )
    )

    S_task2 = (
        teachers.S_star
        + torch.outer(
            teachers.w2_star,
            teachers.w2_star,
        )
    )

    # --------------------------
    # Training datasets
    # --------------------------

    pretrain_train = generate_dataset(
        n_samples=64,
        T=T,
        d=d,
        S_teacher=S_pre,
        seed=1,
    )

    task1_train = generate_dataset(
        n_samples=64,
        T=T,
        d=d,
        S_teacher=S_task1,
        seed=2,
    )

    task2_train = generate_dataset(
        n_samples=64,
        T=T,
        d=d,
        S_teacher=S_task2,
        seed=3,
    )

    # --------------------------
    # Independent test datasets
    # --------------------------

    pretrain_test = generate_dataset(
        n_samples=48,
        T=T,
        d=d,
        S_teacher=S_pre,
        seed=11,
    )

    task1_test = generate_dataset(
        n_samples=48,
        T=T,
        d=d,
        S_teacher=S_task1,
        seed=12,
    )

    task2_test = generate_dataset(
        n_samples=48,
        T=T,
        d=d,
        S_teacher=S_task2,
        seed=13,
    )

    return {
        "d": d,
        "T": T,
        "kappa": kappa,
        "teachers": teachers,
        "pretrain_train": pretrain_train,
        "pretrain_test": pretrain_test,
        "task1_train": task1_train,
        "task1_test": task1_test,
        "task2_train": task2_train,
        "task2_test": task2_test,
    }


def get_pretrained_checkpoint(problem):
    """
    Produce one small pre-training checkpoint that can be
    reused in the fine-tuning tests.
    """

    model = Student(
        d=problem["d"],
        kappa=problem["kappa"],
        dtype=DTYPE,
    )

    checkpoints = pretrain(
        model=model,
        train_data=problem["pretrain_train"],
        test_data=problem["pretrain_test"],
        S_star=problem["teachers"].S_star,
        n_steps=12,
        lr=5e-2,
        checkpoint_steps=[0, 12],
        seed=0,
    )

    return checkpoints[-1].W


# ============================================================
# Pre-training tests
# ============================================================


def test_pretrain_changes_W_and_keeps_w_inactive():
    problem = make_tiny_problem()

    model = Student(
        d=problem["d"],
        kappa=problem["kappa"],
        dtype=DTYPE,
    )

    # Make w non-zero first so that we can verify that
    # pretraining mode really resets/disables it.
    model.reset_low_rank(
        scale=0.1,
        seed=123,
    )

    W_before = (
        model.W
        .detach()
        .clone()
    )

    checkpoints = pretrain(
        model=model,
        train_data=problem["pretrain_train"],
        test_data=problem["pretrain_test"],
        S_star=problem["teachers"].S_star,
        n_steps=12,
        lr=5e-2,
        checkpoint_steps=[0, 12],
        seed=0,
    )

    W_after = (
        model.W
        .detach()
        .clone()
    )

    # W should have been trained.
    assert not torch.equal(
        W_before,
        W_after,
    )

    # w should be inactive and exactly zero.
    assert torch.equal(
        model.w,
        torch.zeros_like(model.w),
    )

    assert model.W.requires_grad
    assert not model.w.requires_grad

    # Both requested checkpoints should exist.
    assert len(checkpoints) == 2
    assert checkpoints[0].step == 0
    assert checkpoints[1].step == 12


def test_pretraining_checkpoints_are_exact():
    problem = make_tiny_problem()

    model = Student(
        d=problem["d"],
        kappa=problem["kappa"],
        dtype=DTYPE,
    )

    W_initial = (
        model.W
        .detach()
        .cpu()
        .clone()
    )

    checkpoints = pretrain(
        model=model,
        train_data=problem["pretrain_train"],
        test_data=problem["pretrain_test"],
        S_star=problem["teachers"].S_star,
        n_steps=10,
        lr=5e-2,
        checkpoint_steps=[0, 10],
        seed=0,
    )

    # Step 0 should correspond exactly to the initialization.
    assert torch.equal(
        checkpoints[0].W,
        W_initial,
    )

    # Final saved checkpoint should correspond exactly
    # to the model state after pre-training.
    assert torch.equal(
        checkpoints[-1].W,
        model.W.detach().cpu(),
    )


def test_pretraining_checkpoint_metrics_are_finite():
    problem = make_tiny_problem()

    model = Student(
        d=problem["d"],
        kappa=problem["kappa"],
        dtype=DTYPE,
    )

    checkpoints = pretrain(
        model=model,
        train_data=problem["pretrain_train"],
        test_data=problem["pretrain_test"],
        S_star=problem["teachers"].S_star,
        n_steps=10,
        lr=5e-2,
        checkpoint_steps=[0, 5, 10],
        seed=0,
    )

    for checkpoint in checkpoints:

        assert math.isfinite(
            checkpoint.representation_error
        )

        assert math.isfinite(
            checkpoint.generalization_error
        )

        assert (
            checkpoint.representation_error >= 0
        )

        assert (
            checkpoint.generalization_error >= 0
        )


# ============================================================
# Fine-tuning tests
# ============================================================


def test_finetuning_keeps_W_exactly_frozen():
    problem = make_tiny_problem()

    W_checkpoint = get_pretrained_checkpoint(
        problem
    )

    W_before = W_checkpoint.clone()

    model = Student(
        d=problem["d"],
        kappa=problem["kappa"],
        dtype=DTYPE,
    )

    finetune_sequential(
        model=model,
        W_checkpoint=W_checkpoint,
        pretraining_test=problem["pretrain_test"],
        task1_train=problem["task1_train"],
        task1_test=problem["task1_test"],
        task2_train=problem["task2_train"],
        task2_test=problem["task2_test"],
        teachers=problem["teachers"],
        n_steps_task1=10,
        n_steps_task2=10,
        lr=5e-2,
        low_rank_init_scale=0.1,
        eval_every=5,
        seed=0,
    )

    W_after = (
        model.W
        .detach()
        .cpu()
    )

    # This should be exact, not just approximately equal.
    assert torch.equal(
        W_before,
        W_after,
    )

    assert not model.W.requires_grad
    assert model.w.requires_grad
    assert model.use_low_rank


def test_adapter_changes_during_task1_and_task2():
    problem = make_tiny_problem()

    W_checkpoint = get_pretrained_checkpoint(
        problem
    )

    model = Student(
        d=problem["d"],
        kappa=problem["kappa"],
        dtype=DTYPE,
    )

    init_scale = 0.1
    seed = 42

    # Reconstruct exactly the initialization that
    # finetune_sequential() will use.
    generator = torch.Generator(
        device="cpu"
    )
    generator.manual_seed(seed)

    w_initial = (
        init_scale
        * torch.randn(
            problem["d"],
            generator=generator,
            dtype=DTYPE,
        )
    )

    result = finetune_sequential(
        model=model,
        W_checkpoint=W_checkpoint,
        pretraining_test=problem["pretrain_test"],
        task1_train=problem["task1_train"],
        task1_test=problem["task1_test"],
        task2_train=problem["task2_train"],
        task2_test=problem["task2_test"],
        teachers=problem["teachers"],
        n_steps_task1=20,
        n_steps_task2=20,
        lr=1e-1,
        low_rank_init_scale=init_scale,
        eval_every=10,
        seed=seed,
    )

    # Task 1 should modify the initial adapter.
    assert not torch.equal(
        result.w_after_task1,
        w_initial,
    )

    # Task 2 should further modify the SAME adapter.
    assert not torch.equal(
        result.w_after_task2,
        result.w_after_task1,
    )


def test_task2_does_not_reset_adapter():
    """
    If Task 2 is given zero training steps, the adapter after
    Task 2 must be exactly the adapter learned on Task 1.

    This directly checks that we do not reset w between tasks.
    """

    problem = make_tiny_problem()

    W_checkpoint = get_pretrained_checkpoint(
        problem
    )

    model = Student(
        d=problem["d"],
        kappa=problem["kappa"],
        dtype=DTYPE,
    )

    result = finetune_sequential(
        model=model,
        W_checkpoint=W_checkpoint,
        pretraining_test=problem["pretrain_test"],
        task1_train=problem["task1_train"],
        task1_test=problem["task1_test"],
        task2_train=problem["task2_train"],
        task2_test=problem["task2_test"],
        teachers=problem["teachers"],
        n_steps_task1=10,
        n_steps_task2=0,
        lr=5e-2,
        low_rank_init_scale=0.1,
        eval_every=5,
        seed=0,
    )

    assert torch.equal(
        result.w_after_task1,
        result.w_after_task2,
    )


# ============================================================
# Pre-training error during fine-tuning
# ============================================================


def test_pretraining_error_before_finetuning_is_checkpoint_error():
    """
    The baseline pre-training error should be measured using
    the frozen W checkpoint before the low-rank adapter is
    introduced.
    """

    problem = make_tiny_problem()

    W_checkpoint = get_pretrained_checkpoint(
        problem
    )

    # Compute the expected baseline independently.
    baseline_model = Student(
        d=problem["d"],
        kappa=problem["kappa"],
        dtype=DTYPE,
    )

    baseline_model.load_large_rank(
        W_checkpoint
    )

    baseline_model.set_pretraining_mode()

    expected_error = generalization_error(
        baseline_model,
        problem["pretrain_test"].X,
        problem["pretrain_test"].y,
    ).item()

    # Run sequential fine-tuning.
    model = Student(
        d=problem["d"],
        kappa=problem["kappa"],
        dtype=DTYPE,
    )

    result = finetune_sequential(
        model=model,
        W_checkpoint=W_checkpoint,
        pretraining_test=problem["pretrain_test"],
        task1_train=problem["task1_train"],
        task1_test=problem["task1_test"],
        task2_train=problem["task2_train"],
        task2_test=problem["task2_test"],
        teachers=problem["teachers"],
        n_steps_task1=5,
        n_steps_task2=5,
        lr=5e-2,
        low_rank_init_scale=0.1,
        eval_every=5,
        seed=0,
    )

    assert math.isclose(
        result.pretraining_error_before,
        expected_error,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_pretraining_forgetting_is_computed_correctly():
    problem = make_tiny_problem()

    W_checkpoint = get_pretrained_checkpoint(
        problem
    )

    model = Student(
        d=problem["d"],
        kappa=problem["kappa"],
        dtype=DTYPE,
    )

    result = finetune_sequential(
        model=model,
        W_checkpoint=W_checkpoint,
        pretraining_test=problem["pretrain_test"],
        task1_train=problem["task1_train"],
        task1_test=problem["task1_test"],
        task2_train=problem["task2_train"],
        task2_test=problem["task2_test"],
        teachers=problem["teachers"],
        n_steps_task1=10,
        n_steps_task2=10,
        lr=5e-2,
        low_rank_init_scale=0.1,
        eval_every=5,
        seed=0,
    )

    expected_after_task1 = (
        result.pretraining_error_after_task1
        - result.pretraining_error_before
    )

    expected_after_task2 = (
        result.pretraining_error_after_task2
        - result.pretraining_error_before
    )

    assert math.isclose(
        result.pretraining_forgetting_after_task1,
        expected_after_task1,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    assert math.isclose(
        result.pretraining_forgetting_after_task2,
        expected_after_task2,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_task1_forgetting_is_computed_correctly():
    problem = make_tiny_problem()

    W_checkpoint = get_pretrained_checkpoint(
        problem
    )

    model = Student(
        d=problem["d"],
        kappa=problem["kappa"],
        dtype=DTYPE,
    )

    result = finetune_sequential(
        model=model,
        W_checkpoint=W_checkpoint,
        pretraining_test=problem["pretrain_test"],
        task1_train=problem["task1_train"],
        task1_test=problem["task1_test"],
        task2_train=problem["task2_train"],
        task2_test=problem["task2_test"],
        teachers=problem["teachers"],
        n_steps_task1=10,
        n_steps_task2=10,
        lr=5e-2,
        low_rank_init_scale=0.1,
        eval_every=5,
        seed=0,
    )

    expected = (
        result.task1_error_after_task2
        - result.task1_error_after_task1
    )

    assert math.isclose(
        result.forgetting_task1,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


# ============================================================
# Full end-to-end sanity check
# ============================================================


def test_full_training_pipeline_returns_finite_metrics():
    problem = make_tiny_problem()

    W_checkpoint = get_pretrained_checkpoint(
        problem
    )

    model = Student(
        d=problem["d"],
        kappa=problem["kappa"],
        dtype=DTYPE,
    )

    result = finetune_sequential(
        model=model,
        W_checkpoint=W_checkpoint,
        pretraining_test=problem["pretrain_test"],
        task1_train=problem["task1_train"],
        task1_test=problem["task1_test"],
        task2_train=problem["task2_train"],
        task2_test=problem["task2_test"],
        teachers=problem["teachers"],
        n_steps_task1=10,
        n_steps_task2=10,
        lr=5e-2,
        low_rank_init_scale=0.1,
        eval_every=5,
        seed=0,
    )

    values = [
        result.pretraining_error_before,
        result.task1_error_before,
        result.task2_error_before,

        result.pretraining_error_after_task1,
        result.task1_error_after_task1,
        result.task2_error_after_task1,

        result.pretraining_error_after_task2,
        result.task1_error_after_task2,
        result.task2_error_after_task2,

        result.overlap_w1_after_task1,
        result.overlap_w2_after_task1,
        result.overlap_w1_after_task2,
        result.overlap_w2_after_task2,

        result.pretraining_forgetting_after_task1,
        result.pretraining_forgetting_after_task2,
        result.forgetting_task1,
    ]

    for value in values:
        assert math.isfinite(value)

    # We should have recorded dynamics from all three stages.
    stages = {
        metric.stage
        for metric in result.history
    }

    assert "before_task1" in stages
    assert "task1" in stages
    assert "task2" in stages

    # Every recorded trajectory metric should also be finite.
    for metric in result.history:

        assert math.isfinite(
            metric.pretraining_error
        )

        assert math.isfinite(
            metric.task1_error
        )

        assert math.isfinite(
            metric.task2_error
        )

        assert math.isfinite(
            metric.overlap_w1
        )

        assert math.isfinite(
            metric.overlap_w2
        )