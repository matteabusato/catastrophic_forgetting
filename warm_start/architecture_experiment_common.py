"""Shared CIFAR-10 warm/cold protocol and metric logging for architecture runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms


FIT_FRACTION = 2.0 / 3.0
FIRST_STAGE_FRACTION = 0.5
WARM_EPOCHS_PER_STAGE = 350
COLD_EPOCHS = 350
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(
    dataset: datasets.CIFAR10,
    indices: Iterable[int],
    sampler_seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(sampler_seed)
    sampler = SubsetRandomSampler(list(indices), generator=generator)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, int]:
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        batch_count = labels.size(0)
        loss_sum += loss.item() * batch_count
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += batch_count
    return loss_sum / count, correct / count, len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += labels.size(0)
    return loss_sum / count, correct / count


def run_experiment(
    architecture: str,
    model_factory: Callable[[], nn.Module],
    model_config: Dict[str, object],
    script_path: str,
) -> None:
    script_dir = Path(script_path).resolve().parent
    parser = argparse.ArgumentParser(
        description=f"Warm-vs-cold CIFAR-10 run for {architecture}."
    )
    parser.add_argument("--seed", type=int, default=0, choices=range(5))
    parser.add_argument("--data-root", type=Path, default=script_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or (
        script_dir / "outputs" / "architecture_study" / architecture
        / f"seed_{args.seed:02d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
    trainset = datasets.CIFAR10(
        root=str(args.data_root), train=True, download=True, transform=transform
    )
    permutation = np.random.RandomState(args.seed).permutation(len(trainset))
    fit_count = int(len(trainset) * FIT_FRACTION)
    fit_indices = permutation[:fit_count].tolist()
    heldout_indices = permutation[fit_count:].tolist()
    half_count = int(len(fit_indices) * FIRST_STAGE_FRACTION)
    half_indices = fit_indices[:half_count]
    heldout_loader = make_loader(trainset, heldout_indices, args.seed + 301, device)

    model_for_count = model_factory()
    parameter_count = sum(parameter.numel() for parameter in model_for_count.parameters())
    run_config = {
        "dataset": "CIFAR-10 train split",
        "heldout_source": "reserved third of CIFAR-10 training set; not official test set",
        "seed": args.seed,
        "architecture": architecture,
        "model_config": model_config,
        "parameter_count": parameter_count,
        "fit_examples": len(fit_indices),
        "half_fit_examples": len(half_indices),
        "heldout_examples": len(heldout_indices),
        "warm_epochs_half_then_full": WARM_EPOCHS_PER_STAGE,
        "cold_epochs_full": COLD_EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "loss": "cross_entropy",
        "augmentation": "none",
        "normalization_mean": NORMALIZE_MEAN,
        "normalization_std": NORMALIZE_STD,
        "device": str(device),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    curve_path = output_dir / "curves.csv"
    fields = [
        "seed", "architecture", "method", "phase", "stage_epoch", "total_epoch",
        "train_examples", "optimizer_updates", "learning_rate", "train_loss",
        "train_accuracy", "heldout_loss", "heldout_accuracy", "elapsed_seconds",
    ]
    summary: Dict[str, object] = {
        "seed": args.seed,
        "architecture": architecture,
        "parameter_count": parameter_count,
    }

    def record_epoch(
        writer: csv.DictWriter,
        handle,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        method: str,
        phase: str,
        stage_epoch: int,
        total_epoch: int,
        train_examples: int,
        updates: int,
        start_time: float,
    ) -> int:
        train_loss, train_accuracy, batches = train_epoch(
            model, train_loader, optimizer, device
        )
        heldout_loss, heldout_accuracy = evaluate(model, heldout_loader, device)
        updates += batches
        elapsed = time.time() - start_time
        writer.writerow(
            {
                "seed": args.seed,
                "architecture": architecture,
                "method": method,
                "phase": phase,
                "stage_epoch": stage_epoch,
                "total_epoch": total_epoch,
                "train_examples": train_examples,
                "optimizer_updates": updates,
                "learning_rate": LEARNING_RATE,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "heldout_loss": heldout_loss,
                "heldout_accuracy": heldout_accuracy,
                "elapsed_seconds": elapsed,
            }
        )
        handle.flush()
        print(
            f"seed={args.seed} architecture={architecture} method={method} "
            f"phase={phase} epoch={stage_epoch} train_loss={train_loss:.5f} "
            f"heldout_loss={heldout_loss:.5f} train_acc={train_accuracy:.4f} "
            f"heldout_acc={heldout_accuracy:.4f} elapsed_s={elapsed:.1f}",
            flush=True,
        )
        return updates

    def save_checkpoint(path: Path, model: nn.Module, method: str, phase: str) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "seed": args.seed,
                "architecture": architecture,
                "method": method,
                "phase": phase,
                "parameter_count": parameter_count,
                "model_config": model_config,
            },
            path,
        )

    with curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        # Warm arm: half-data stage, followed by full data with fresh Adam state.
        seed_everything(args.seed + 101)
        warm_model = model_factory().to(device)
        warm_start_time = time.time()
        warm_updates = 0
        half_loader = make_loader(trainset, half_indices, args.seed + 401, device)
        optimizer = torch.optim.Adam(
            warm_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        for epoch in range(1, WARM_EPOCHS_PER_STAGE + 1):
            warm_updates = record_epoch(
                writer, handle, warm_model, half_loader, optimizer,
                "warm_start", "half_data", epoch, epoch, len(half_indices),
                warm_updates, warm_start_time,
            )
        save_checkpoint(output_dir / "warm_after_half.pt", warm_model, "warm_start", "half_data")
        half_train_loader = make_loader(trainset, half_indices, args.seed + 405, device)
        summary["warm_after_half_train"] = dict(
            zip(("loss", "accuracy"), evaluate(warm_model, half_train_loader, device))
        )

        full_warm_loader = make_loader(trainset, fit_indices, args.seed + 402, device)
        optimizer = torch.optim.Adam(
            warm_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        for stage_epoch in range(1, WARM_EPOCHS_PER_STAGE + 1):
            warm_updates = record_epoch(
                writer, handle, warm_model, full_warm_loader, optimizer,
                "warm_start", "full_data", stage_epoch,
                WARM_EPOCHS_PER_STAGE + stage_epoch, len(fit_indices),
                warm_updates, warm_start_time,
            )
        save_checkpoint(output_dir / "warm_final.pt", warm_model, "warm_start", "full_data")
        warm_train_loader = make_loader(trainset, fit_indices, args.seed + 404, device)
        warm_train_final = evaluate(warm_model, warm_train_loader, device)
        warm_heldout_final = evaluate(warm_model, heldout_loader, device)
        summary["warm_final"] = {
            "train_loss": warm_train_final[0],
            "train_accuracy": warm_train_final[1],
            "heldout_loss": warm_heldout_final[0],
            "heldout_accuracy": warm_heldout_final[1],
            "fit_reached_100_percent": warm_train_final[1] >= 1.0,
        }

        # Cold arm: 350 full-data epochs with a paired minibatch ordering.
        seed_everything(args.seed + 202)
        cold_model = model_factory().to(device)
        cold_start_time = time.time()
        cold_updates = 0
        cold_loader = make_loader(trainset, fit_indices, args.seed + 402, device)
        optimizer = torch.optim.Adam(
            cold_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        for epoch in range(1, COLD_EPOCHS + 1):
            cold_updates = record_epoch(
                writer, handle, cold_model, cold_loader, optimizer,
                "cold_start", "full_data", epoch, epoch, len(fit_indices),
                cold_updates, cold_start_time,
            )
        save_checkpoint(output_dir / "cold_final.pt", cold_model, "cold_start", "full_data")
        cold_train_loader = make_loader(trainset, fit_indices, args.seed + 404, device)
        cold_train_final = evaluate(cold_model, cold_train_loader, device)
        cold_heldout_final = evaluate(cold_model, heldout_loader, device)
        summary["cold_final"] = {
            "train_loss": cold_train_final[0],
            "train_accuracy": cold_train_final[1],
            "heldout_loss": cold_heldout_final[0],
            "heldout_accuracy": cold_heldout_final[1],
            "fit_reached_100_percent": cold_train_final[1] >= 1.0,
        }

    summary["warm_minus_cold_heldout_accuracy"] = (
        summary["warm_final"]["heldout_accuracy"]
        - summary["cold_final"]["heldout_accuracy"]
    )
    summary["warm_minus_cold_heldout_loss"] = (
        summary["warm_final"]["heldout_loss"] - summary["cold_final"]["heldout_loss"]
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary={json.dumps(summary, sort_keys=True)}", flush=True)
