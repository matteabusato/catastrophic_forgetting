#!/usr/bin/env python3
"""Run a seed-paired warm-vs-cold CIFAR-10 experiment with a smaller ResNet.

The warm arm trains 350 epochs on half of the fit split, then 350 epochs on
the full fit split. The cold arm trains 350 epochs on the full fit split from
a fresh initialization. Metrics are written after each epoch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms


SEED_COUNT = 5
FIT_FRACTION = 2.0 / 3.0
FIRST_STAGE_FRACTION = 0.5
WARM_EPOCHS_PER_STAGE = 350
COLD_EPOCHS = 350
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
CHANNELS = (16, 32, 64, 128)
BLOCKS_PER_STAGE = (1, 1, 1, 1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channels * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = F.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return F.relu(outputs + residual)


class SmallCifarResNet(nn.Module):
    """CIFAR ResNet with one 2-convolution BasicBlock in each stage."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = CHANNELS[0]
        self.conv1 = nn.Conv2d(3, CHANNELS[0], 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(CHANNELS[0])
        self.layer1 = self._make_stage(CHANNELS[0], BLOCKS_PER_STAGE[0], stride=1)
        self.layer2 = self._make_stage(CHANNELS[1], BLOCKS_PER_STAGE[1], stride=2)
        self.layer3 = self._make_stage(CHANNELS[2], BLOCKS_PER_STAGE[2], stride=2)
        self.layer4 = self._make_stage(CHANNELS[3], BLOCKS_PER_STAGE[3], stride=2)
        self.linear = nn.Linear(CHANNELS[-1], num_classes)

    def _make_stage(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_channels, channels, block_stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = F.relu(self.bn1(self.conv1(inputs)))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = F.avg_pool2d(outputs, kernel_size=4)
        outputs = torch.flatten(outputs, 1)
        return self.linear(outputs)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, choices=range(SEED_COUNT))
    parser.add_argument(
        "--data-root", type=Path, default=script_dir / "data",
        help="Torchvision CIFAR-10 location; downloaded if missing.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to outputs/small_resnet/seed_XX.",
    )
    return parser.parse_args()


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


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, int]:
    model.train()
    loss_total = 0.0
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
        n = labels.size(0)
        loss_total += loss.item() * n
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += n
    return loss_total / count, correct / count, len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    loss_total = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_total += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += labels.size(0)
    return loss_total / count, correct / count


def append_metric(
    writer: csv.DictWriter,
    handle,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seed: int,
    method: str,
    phase: str,
    stage_epoch: int,
    total_epoch: int,
    train_examples: int,
    optimizer_updates: int,
    elapsed: float,
) -> int:
    train_loss, train_accuracy, batches = run_train_epoch(
        model, train_loader, optimizer, device
    )
    heldout_loss, heldout_accuracy = evaluate(model, val_loader, device)
    optimizer_updates += batches
    writer.writerow(
        {
            "seed": seed,
            "method": method,
            "phase": phase,
            "stage_epoch": stage_epoch,
            "total_epoch": total_epoch,
            "train_examples": train_examples,
            "optimizer_updates": optimizer_updates,
            "learning_rate": LEARNING_RATE,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "heldout_loss": heldout_loss,
            "heldout_accuracy": heldout_accuracy,
            "elapsed_seconds": time.time() - elapsed,
        }
    )
    handle.flush()
    print(
        f"seed={seed} method={method} phase={phase} epoch={stage_epoch} "
        f"train_loss={train_loss:.5f} heldout_loss={heldout_loss:.5f} "
        f"train_acc={train_accuracy:.4f} heldout_acc={heldout_accuracy:.4f} "
        f"elapsed_s={time.time() - elapsed:.1f}",
        flush=True,
    )
    return optimizer_updates


def save_checkpoint(
    path: Path,
    model: nn.Module,
    seed: int,
    method: str,
    completed_full_epochs: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "method": method,
            "architecture": "CIFAR ResNet [1,1,1,1]",
            "channels": CHANNELS,
            "blocks_per_stage": BLOCKS_PER_STAGE,
            "completed_full_data_epochs": completed_full_epochs,
        },
        path,
    )


def train_warm_arm(
    seed: int,
    dataset: datasets.CIFAR10,
    fit_indices: List[int],
    half_indices: List[int],
    val_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    writer: csv.DictWriter,
    handle,
) -> Dict[str, float]:
    seed_everything(seed + 101)
    model = SmallCifarResNet().to(device)
    elapsed = time.time()
    updates = 0

    half_loader = make_loader(dataset, half_indices, seed + 401, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    for epoch in range(1, WARM_EPOCHS_PER_STAGE + 1):
        updates = append_metric(
            writer, handle, model, half_loader, val_loader, optimizer, device,
            seed, "warm_start", "half_data", epoch, epoch, len(half_indices),
            updates, elapsed,
        )

    save_checkpoint(
        output_dir / "warm_after_half.pt", model, seed, "warm_after_half", 0
    )

    full_loader = make_loader(dataset, fit_indices, seed + 402, device)
    # Reset Adam state at the half-data/full-data transition.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    for stage_epoch in range(1, WARM_EPOCHS_PER_STAGE + 1):
        updates = append_metric(
            writer, handle, model, full_loader, val_loader, optimizer, device,
            seed, "warm_start", "full_data", stage_epoch,
            WARM_EPOCHS_PER_STAGE + stage_epoch, len(fit_indices), updates, elapsed,
        )
    save_checkpoint(
        output_dir / "warm_final.pt", model, seed, "warm_start",
        WARM_EPOCHS_PER_STAGE,
    )
    final = evaluate(model, val_loader, device)
    return {"heldout_loss": final[0], "heldout_accuracy": final[1]}


def train_cold_arm(
    seed: int,
    dataset: datasets.CIFAR10,
    fit_indices: List[int],
    val_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    writer: csv.DictWriter,
    handle,
) -> Dict[str, float]:
    seed_everything(seed + 202)
    model = SmallCifarResNet().to(device)
    elapsed = time.time()
    # Match the warm arm's full-data minibatch order for paired comparison.
    train_loader = make_loader(dataset, fit_indices, seed + 402, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    updates = 0
    for epoch in range(1, COLD_EPOCHS + 1):
        updates = append_metric(
            writer, handle, model, train_loader, val_loader, optimizer, device,
            seed, "cold_start", "full_data", epoch, epoch, len(fit_indices),
            updates, elapsed,
        )
    save_checkpoint(
        output_dir / "cold_final.pt", model, seed, "cold_start", COLD_EPOCHS
    )
    final = evaluate(model, val_loader, device)
    return {"heldout_loss": final[0], "heldout_accuracy": final[1]}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        Path(__file__).resolve().parent / "outputs" / "small_resnet"
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
    indices = np.random.RandomState(args.seed).permutation(len(trainset))
    fit_size = int(len(trainset) * FIT_FRACTION)
    fit_indices = indices[:fit_size].tolist()
    val_indices = indices[fit_size:].tolist()
    half_size = int(len(fit_indices) * FIRST_STAGE_FRACTION)
    half_indices = fit_indices[:half_size]
    val_loader = make_loader(trainset, val_indices, args.seed + 301, device)

    model_for_count = SmallCifarResNet()
    parameter_count = sum(parameter.numel() for parameter in model_for_count.parameters())
    metadata = {
        "dataset": "CIFAR-10",
        "seed": args.seed,
        "architecture": "CIFAR ResNet [1,1,1,1]",
        "channels": CHANNELS,
        "blocks_per_stage": BLOCKS_PER_STAGE,
        "parameter_count": parameter_count,
        "fit_examples": len(fit_indices),
        "half_fit_examples": len(half_indices),
        "heldout_examples": len(val_indices),
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
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))

    csv_path = output_dir / "curves.csv"
    fields = [
        "seed", "method", "phase", "stage_epoch", "total_epoch",
        "train_examples", "optimizer_updates", "learning_rate", "train_loss",
        "train_accuracy", "heldout_loss", "heldout_accuracy", "elapsed_seconds",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        print(
            f"device={device} seed={args.seed} architecture=ResNet[1,1,1,1] "
            f"parameters={parameter_count} output={output_dir.resolve()}",
            flush=True,
        )
        warm_final = train_warm_arm(
            args.seed, trainset, fit_indices, half_indices, val_loader,
            output_dir, device, writer, handle,
        )
        cold_final = train_cold_arm(
            args.seed, trainset, fit_indices, val_loader,
            output_dir, device, writer, handle,
        )

    summary = {
        "seed": args.seed,
        "architecture": metadata["architecture"],
        "parameter_count": parameter_count,
        "warm_final": warm_final,
        "cold_final": cold_final,
        "warm_minus_cold_heldout_accuracy": (
            warm_final["heldout_accuracy"] - cold_final["heldout_accuracy"]
        ),
        "warm_minus_cold_heldout_loss": (
            warm_final["heldout_loss"] - cold_final["heldout_loss"]
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary={json.dumps(summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
