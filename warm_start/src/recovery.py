"""Exact update replay and checkpointing for the legacy fresh_full trajectory."""

from __future__ import annotations

import os
import math
from pathlib import Path

import torch

from .data import Dataset
from .losses import output_mse
from .model import Student


class FreshTrainer:
    """Uses the same randperm/batch/update sequence as training.train.

    Checkpoints include the current permutation and cursor: RNG state alone
    cannot identify a minibatch in the middle of a shuffled epoch.
    """

    def __init__(self, model: Student, data: Dataset, optimizer: str, lr: float,
                 batch_size: int, order_seed: int):
        self.model, self.data, self.batch_size = model, data, batch_size
        if optimizer not in ("sgd", "adam") or batch_size < 1 or lr <= 0:
            raise ValueError("Invalid optimizer, batch size or learning rate.")
        optimizer_class = torch.optim.SGD if optimizer == "sgd" else torch.optim.Adam
        self.optimizer = optimizer_class([model.W], lr=lr)
        self.generator = torch.Generator(device=model.W.device).manual_seed(order_seed)
        self.order = None
        self.position = 0
        self.step = 0
        self.completed_epochs = 0
        self.examples_seen = 0

    def update(self) -> None:
        if self.order is None or self.position == len(self.data):
            self.order = torch.randperm(len(self.data), generator=self.generator,
                                        device=self.model.W.device)
            self.position = 0
        indices = self.order[self.position:self.position + self.batch_size]
        self.optimizer.zero_grad(set_to_none=True)
        loss = output_mse(self.model(self.data.X[indices]), self.data.y[indices])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at update {self.step + 1}.")
        loss.backward()
        if self.model.W.grad is None or not torch.isfinite(self.model.W.grad).all():
            raise FloatingPointError(f"Non-finite gradient at update {self.step + 1}.")
        self.optimizer.step()
        self.step += 1
        self.position += len(indices)
        self.examples_seen += len(indices)
        if self.position == len(self.data):
            self.completed_epochs += 1

    def state_dict(self) -> dict:
        return {
            "W": self.model.W.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
            "generator_state": self.generator.get_state().cpu(),
            "order": None if self.order is None else self.order.cpu(),
            "position": self.position, "step": self.step,
            "completed_epochs": self.completed_epochs, "examples_seen": self.examples_seen,
            "batch_size": self.batch_size, "n_samples": len(self.data),
        }

    def restore_sgd_sampler(self, step: int) -> None:
        """Recover cursor/RNG from a completed legacy plain-SGD weight snapshot.

        This does not replay weight updates. Vanilla SGD has no optimizer memory;
        optimizers with momentum/Adam must instead load a complete checkpoint.
        Call only on a newly initialized trainer using the original order seed.
        """
        if not isinstance(self.optimizer, torch.optim.SGD) or any(
            group.get("momentum", 0) != 0 for group in self.optimizer.param_groups
        ):
            raise ValueError("Weight-only continuation requires plain SGD without momentum.")
        if self.step != 0 or self.order is not None or step < 0:
            raise ValueError("Sampler restoration requires a new trainer and nonnegative step.")
        batches = math.ceil(len(self.data) / self.batch_size)
        for _ in range(math.ceil(step / batches)):
            self.order = torch.randperm(len(self.data), generator=self.generator,
                                        device=self.model.W.device)
        completed, remaining = divmod(step, batches)
        self.position = (remaining * self.batch_size if remaining
                         else len(self.data) if step else 0)
        self.step, self.completed_epochs = step, completed
        self.examples_seen = completed * len(self.data) + remaining * self.batch_size

    def load_state_dict(self, state: dict) -> None:
        if state["batch_size"] != self.batch_size or state["n_samples"] != len(self.data):
            raise ValueError("Checkpoint dataset size/batch size mismatch.")
        with torch.no_grad():
            self.model.W.copy_(state["W"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator_state"].cpu())
        self.order = (None if state["order"] is None
                      else state["order"].to(self.model.W.device))
        self.position, self.step = state["position"], state["step"]
        self.completed_epochs = state["completed_epochs"]
        self.examples_seen = state["examples_seen"]


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Publish only a complete checkpoint; retain the last one during writes."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate_train_partitions(model: Student, data: Dataset, indices_A: torch.Tensor,
                              batch_size: int) -> dict[str, dict[str, float]]:
    """Evaluate D once and derive A/B metrics without extra forward passes/copies."""
    membership = torch.zeros(len(data), dtype=torch.bool, device=data.X.device)
    membership[indices_A] = True
    totals = {name: torch.zeros(3, dtype=torch.float64, device=data.X.device)
              for name in ("full", "A", "B")}
    counts = {"full": len(data), "A": len(indices_A), "B": len(data) - len(indices_A)}
    for start in range(0, len(data), batch_size):
        X = data.X[start:start + batch_size]
        target = data.attention[start:start + batch_size]
        logits = model.attention_logits(X)
        attention = logits.softmax(-1)
        values = torch.stack((
            (attention @ X - data.y[start:start + batch_size]).square().mean((-2, -1)),
            (torch.special.xlogy(target, target)
             - target * logits.log_softmax(-1)).sum(-1).mean(-1),
            -torch.special.xlogy(attention, attention).sum(-1).mean(-1),
        ), dim=-1).double()
        mask = membership[start:start + len(X)]
        totals["full"] += values.sum(0)
        totals["A"] += values[mask].sum(0)
        totals["B"] += values[~mask].sum(0)
    names = ("mse", "attention_kl", "attention_entropy")
    return {name: dict(zip(names, (value / counts[name]).cpu().tolist()))
            for name, value in totals.items()}
