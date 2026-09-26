from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from torch import Tensor

from src.data_model import Dataset, Teachers, teacher_forward
from src.model import Student
from src.observables import (
    representation_error,
    generalization_error,
    forgetting,
    vector_overlap,
)


@dataclass
class PretrainingMetric:
    step: int

    step_over_d: float
    step_over_d2: float

    representation_error: float
    generalization_error: float

@dataclass
class PretrainingCheckpoint:
    step: int
    W: Tensor
    representation_error: float
    generalization_error: float
    optimizer_state: dict
    generator_state: Tensor

@dataclass
class PretrainingRun:
    metrics: List[PretrainingMetric]
    checkpoints: List[PretrainingCheckpoint]

    final_step: int
    final_optimizer_state: dict
    final_generator_state: Tensor

@torch.no_grad()
def _evaluate_pretraining(model: Student, test_data: Dataset, S_star: Tensor, step: int,) -> PretrainingMetric:
    d = model.d

    rep_error = representation_error(model.S_large, S_star,).item()
    gen_error = generalization_error(model, test_data.X, test_data.y,).item()

    return PretrainingMetric(step=step, step_over_d=step / d, step_over_d2=step / (d * d),
        representation_error=rep_error, generalization_error=gen_error,)

@dataclass
class FineTuningMetric:
    stage: str
    step: int

    pretraining_error: float

    task1_error: float
    task2_error: float

    overlap_w1: float
    overlap_w2: float


@dataclass
class FineTuningResult:
    # State before any fine-tuning
    pretraining_error_before: float
    task1_error_before: float
    task2_error_before: float

    # State after Task 1
    pretraining_error_after_task1: float

    task1_error_after_task1: float
    task2_error_after_task1: float

    overlap_w1_after_task1: float
    overlap_w2_after_task1: float

    # State after Task 2
    pretraining_error_after_task2: float

    task1_error_after_task2: float
    task2_error_after_task2: float

    overlap_w1_after_task2: float
    overlap_w2_after_task2: float

    # Catastrophic forgetting
    pretraining_forgetting_after_task1: float
    pretraining_forgetting_after_task2: float
    forgetting_task1: float

    # Final adapters, useful for later diagnostics
    w_after_task1: Tensor
    w_after_task2: Tensor

    # Full fine-tuning trajectory
    history: List[FineTuningMetric]


def mse_loss(prediction: Tensor, target: Tensor,) -> Tensor:
    return torch.mean((prediction - target) ** 2)


def _sample_example(dataset: Dataset, generator: torch.Generator,) -> tuple[Tensor, Tensor]:
    n_samples = dataset.X.shape[0]

    index = torch.randint(low=0, high=n_samples, size=(1,), generator=generator, device=dataset.X.device,)

    return (dataset.X[index], dataset.y[index],)


def pretrain(model: Student, test_data: Dataset, S_star: Tensor, T: int, target_step: int, lr: float,
    checkpoint_steps: List[int], eval_every: int, seed: int = 0, start_step: int = 0,
    optimizer_state: Optional[dict] = None, generator_state: Optional[Tensor] = None, metric_callback: Optional[Callable[[PretrainingMetric], None]] = None,
    checkpoint_callback: Optional[Callable[[PretrainingCheckpoint], None]] = None,) -> PretrainingRun:

    checkpoint_steps = sorted(set(checkpoint_steps) | {target_step})
    checkpoint_steps = [step for step in checkpoint_steps if start_step <= step <= target_step]

    model.set_pretraining_mode()
    model.zero_grad(set_to_none=True)

    optimizer = torch.optim.SGD([model.W], lr=lr,)

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    generator = torch.Generator(device=model.W.device)

    if generator_state is not None:
        generator.set_state(generator_state.cpu())
    else:
        generator.manual_seed(seed)

    metrics = []
    checkpoints = []

    def record_metric(step: int,) -> PretrainingMetric:
        metric = _evaluate_pretraining(model=model, test_data=test_data, S_star=S_star, step=step,)
        metrics.append(metric)

        if metric_callback is not None:
            metric_callback(metric)

        return metric

    def save_checkpoint(step: int, metric: PretrainingMetric,) -> None:
        checkpoint = PretrainingCheckpoint(step=step,
            W=(model.W.detach().cpu().clone()),
            representation_error=metric.representation_error,
            generalization_error=metric.generalization_error,
            optimizer_state=deepcopy(optimizer.state_dict()),
            generator_state=(generator.get_state().cpu().clone()),)

        checkpoints.append(checkpoint)

        if checkpoint_callback is not None:
            checkpoint_callback(checkpoint)


    if start_step == 0:
        metric = record_metric(step=0)

        if 0 in checkpoint_steps:
            save_checkpoint(step=0, metric=metric,)

    for step in range(start_step + 1, target_step + 1,):
        X = torch.randn(1, T, model.d, generator=generator, dtype=model.W.dtype, device=model.W.device,)

        with torch.no_grad():
            y = teacher_forward(X, S_star,)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(X)
        loss = mse_loss(prediction, y,)

        loss.backward()
        optimizer.step()

        is_eval_step = (step % eval_every == 0)
        is_checkpoint_step = (step in checkpoint_steps)

        if (is_eval_step or is_checkpoint_step or step == target_step):
            metric = record_metric(step=step)

            if is_checkpoint_step:
                save_checkpoint(step=step, metric=metric,)


    return PretrainingRun(
        metrics=metrics,
        checkpoints=checkpoints,
        final_step=target_step,
        final_optimizer_state=deepcopy(optimizer.state_dict()),
        final_generator_state=(generator.get_state().cpu().clone()),
    )


@torch.no_grad()
def _evaluate_finetuning(model: Student, pretraining_test: Dataset, task1_test: Dataset,
    task2_test: Dataset, teachers: Teachers, stage: str, step: int,) -> FineTuningMetric:

    pretraining_error = generalization_error(model, pretraining_test.X, pretraining_test.y,).item()

    error1 = generalization_error(model, task1_test.X, task1_test.y,).item()
    error2 = generalization_error(model, task2_test.X, task2_test.y,).item()

    overlap1 = vector_overlap(model.w, teachers.w1_star,).item()
    overlap2 = vector_overlap(model.w, teachers.w2_star,).item()

    return FineTuningMetric(stage=stage, step=step, pretraining_error=pretraining_error,
        task1_error=error1, task2_error=error2, overlap_w1=overlap1, overlap_w2=overlap2,)


def _train_low_rank_on_task(model: Student, train_data: Dataset, pretraining_test: Dataset, task1_test: Dataset, task2_test: Dataset,
    teachers: Teachers, optimizer: torch.optim.Optimizer, n_steps: int, eval_every: int,
    generator: torch.Generator, stage: str,) -> List[FineTuningMetric]:
    """
    Train the low-rank adapter w on one task.
    """

    history = []

    for step in range(1, n_steps + 1):
        X_batch, y_batch = _sample_example(train_data, generator,)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(X_batch)
        loss = mse_loss(prediction, y_batch,)

        loss.backward()

        optimizer.step()

        if step % eval_every == 0 or step == n_steps:
            metric = _evaluate_finetuning(model=model, pretraining_test=pretraining_test, task1_test=task1_test, task2_test=task2_test,
                teachers=teachers, stage=stage, step=step,)

            history.append(metric)

    return history


def finetune_sequential(model: Student, W_checkpoint: Tensor, pretraining_test: Dataset, task1_train: Dataset, task1_test: Dataset,
    task2_train: Dataset, task2_test: Dataset, teachers: Teachers, n_steps_task1: int,
    n_steps_task2: int, lr: float, low_rank_init_scale: float = 1e-3, eval_every: int = 100,
    seed: int = 0,) -> FineTuningResult:
    """
    Perform the sequential fine-tuning protocol:
    """

    model.load_large_rank(W_checkpoint )
    model.set_pretraining_mode()
    model.zero_grad(set_to_none=True)

    history = []

    initial_metric = _evaluate_finetuning(model=model, pretraining_test=pretraining_test, task1_test=task1_test, task2_test=task2_test,
        teachers=teachers, stage="before_task1", step=0,)

    history.append(initial_metric)

    pretraining_error_before = initial_metric.pretraining_error
    task1_error_before = initial_metric.task1_error
    task2_error_before = initial_metric.task2_error     

    model.reset_low_rank(scale=low_rank_init_scale, seed=seed,)

    model.set_finetuning_mode()
    model.zero_grad(set_to_none=True)

    optimizer = torch.optim.SGD([model.w], lr=lr,)

    generator = torch.Generator(device=task1_train.X.device)
    generator.manual_seed(seed)

    # ========================================================
    # TASK 1
    # ========================================================

    history_task1 = _train_low_rank_on_task(model=model, train_data=task1_train, pretraining_test=pretraining_test,task1_test=task1_test, task2_test=task2_test,
        teachers=teachers, optimizer=optimizer, n_steps=n_steps_task1, eval_every=eval_every,
        generator=generator, stage="task1",)

    history.extend(history_task1)

    metric_after_task1 = _evaluate_finetuning(model=model, pretraining_test=pretraining_test, task1_test=task1_test, task2_test=task2_test,
        teachers=teachers, stage="after_task1", step=n_steps_task1,)

    w_after_task1 = (model.w.detach().cpu().clone())

    # ========================================================
    # TASK 2
    # ========================================================

    history_task2 = _train_low_rank_on_task(model=model, train_data=task2_train, pretraining_test=pretraining_test, task1_test=task1_test, task2_test=task2_test,
        teachers=teachers, optimizer=optimizer, n_steps=n_steps_task2, eval_every=eval_every,
        generator=generator, stage="task2",)

    history.extend(history_task2)

    metric_after_task2 = _evaluate_finetuning(model=model, pretraining_test=pretraining_test, task1_test=task1_test, task2_test=task2_test,
        teachers=teachers, stage="after_task2", step=n_steps_task2,)

    w_after_task2 = (model.w.detach().cpu().clone())

    F1 = forgetting(task1_error_after_task1=(metric_after_task1.task1_error), task1_error_after_task2=(metric_after_task2.task1_error),)
    pretraining_forgetting_after_task1 = metric_after_task1.pretraining_error - initial_metric.pretraining_error
    pretraining_forgetting_after_task2 = metric_after_task2.pretraining_error - initial_metric.pretraining_error
    
    return FineTuningResult(
        pretraining_error_before=initial_metric.pretraining_error,
        task1_error_before=task1_error_before,
        task2_error_before=task2_error_before,
        pretraining_error_after_task1=metric_after_task1.pretraining_error,
        task1_error_after_task1=metric_after_task1.task1_error,
        task2_error_after_task1=metric_after_task1.task2_error,
        overlap_w1_after_task1=metric_after_task1.overlap_w1,
        overlap_w2_after_task1=metric_after_task1.overlap_w2,
        pretraining_error_after_task2=metric_after_task2.pretraining_error,
        task1_error_after_task2=metric_after_task2.task1_error,
        task2_error_after_task2=metric_after_task2.task2_error,
        overlap_w1_after_task2=metric_after_task2.overlap_w1,
        overlap_w2_after_task2=metric_after_task2.overlap_w2,
        pretraining_forgetting_after_task1=pretraining_forgetting_after_task1,
        pretraining_forgetting_after_task2=pretraining_forgetting_after_task2,
        forgetting_task1=F1,
        w_after_task1=w_after_task1,
        w_after_task2=w_after_task2,
        history=history,
    )