#!/usr/bin/env python3
"""Reproduce the two-stage CIFAR-10 warm-start experiment in Ash & Adams Fig. 1.

Training choices are deliberately constants rather than CLI options so the run
uses the protocol recorded below. Only data/output locations are configurable.
Requires PyTorch, torchvision, NumPy, and (optionally, for the plot) matplotlib.
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


# Figure 1 / paper protocol. Do not change these for the reference run.
DATASET = "CIFAR-10"
MODEL = "CIFAR ResNet-18 (Ash & Adams implementation)"
NUM_SEEDS = 5
FIRST_STAGE_FRACTION = 0.5
FIT_FRACTION_OF_CIFAR_TRAIN = 2.0 / 3.0
EPOCHS_PER_WARM_STAGE = 350
COLD_EPOCHS = 700
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
NUM_WORKERS = 0
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


class BasicBlock(nn.Module):
    """Basic residual block from the authors' CIFAR ResNet implementation."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """CIFAR ResNet-18: [2,2,2,2] basic blocks, widths 16/32/64/128."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(32, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(64, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(128, num_blocks=2, stride=2)
        self.linear = nn.Linear(128, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = torch.flatten(out, 1)
        return self.linear(out)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("CIFAR10_ROOT", here / "data")),
        help="Directory for torchvision's CIFAR-10 files (downloaded if absent).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("WARM_START_OUTPUT_DIR", here / "outputs")),
        help="Directory for metrics, checkpoints, and the summary plot.",
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
    generator_seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(generator_seed)
    sampler = SubsetRandomSampler(list(indices), generator=generator)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )


def train_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device
) -> Tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        n = labels.size(0)
        loss_sum += loss.item() * n
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += n
    return loss_sum / count, correct / count


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, labels).item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += labels.size(0)
    return loss_sum / count, correct / count


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    # A new Adam instance is made for each stage, resetting its moment state.
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def save_checkpoint(path: Path, model: nn.Module, seed: int, method: str) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "method": method,
            "protocol": protocol_metadata(),
        },
        path,
    )


def protocol_metadata() -> Dict[str, object]:
    return {
        "dataset": DATASET,
        "model": MODEL,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "loss": "cross_entropy",
        "augmentation": "none",
        "normalization_mean": NORMALIZE_MEAN,
        "normalization_std": NORMALIZE_STD,
        "fit_fraction_of_cifar_train": FIT_FRACTION_OF_CIFAR_TRAIN,
        "first_stage_fraction_of_fit_split": FIRST_STAGE_FRACTION,
        "warm_stage_epochs": EPOCHS_PER_WARM_STAGE,
        "cold_epochs": COLD_EPOCHS,
        "num_seeds": NUM_SEEDS,
        "optimizer_state_reset_between_warm_stages": True,
        "evaluation": "held-out third of CIFAR-10 training set",
    }


def run_seed(
    seed: int,
    trainset: datasets.CIFAR10,
    output_dir: Path,
    device: torch.device,
) -> List[Dict[str, object]]:
    seed_dir = output_dir / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    curves_path = seed_dir / "curves.csv"
    curves_path.write_text(
        "seed,method,epoch,stage,train_examples,train_loss,train_accuracy,heldout_loss,heldout_accuracy\n"
    )

    # Match the authors' released data protocol: hold out a random third of the
    # provided CIFAR-10 training set, then use half of the remaining fit split
    # for warm-start pretraining.
    indices = np.random.RandomState(seed).permutation(len(trainset))
    n_fit = int(len(trainset) * FIT_FRACTION_OF_CIFAR_TRAIN)
    fit_indices = indices[:n_fit].tolist()
    val_indices = indices[n_fit:].tolist()
    n_first = int(len(fit_indices) * FIRST_STAGE_FRACTION)
    first_indices = fit_indices[:n_first]

    val_loader = make_loader(trainset, val_indices, seed + 301, device)
    rows: List[Dict[str, object]] = []

    # Warm-start arm: 350 epochs on the first half, then 350 on all fit data.
    seed_everything(seed + 101)
    warm_model = ResNet(num_classes=10).to(device)
    warm_stage1_loader = make_loader(trainset, first_indices, seed + 401, device)
    optimizer = make_optimizer(warm_model)
    start_time = time.time()
    for epoch in range(1, EPOCHS_PER_WARM_STAGE + 1):
        train_loss, train_acc = train_epoch(warm_model, warm_stage1_loader, optimizer, device)
        val_loss, val_acc = evaluate(warm_model, val_loader, device)
        row = {
            "seed": seed,
            "method": "warm_start",
            "epoch": epoch,
            "stage": "half_data",
            "train_examples": len(first_indices),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "heldout_loss": val_loss,
            "heldout_accuracy": val_acc,
        }
        rows.append(row)
        append_row(curves_path, row)
        print_row(row, start_time)

    save_checkpoint(seed_dir / "warm_after_half.pt", warm_model, seed, "warm_after_half")

    full_warm_loader = make_loader(trainset, fit_indices, seed + 402, device)
    optimizer = make_optimizer(warm_model)  # deliberately reset optimizer state
    for stage_epoch in range(1, EPOCHS_PER_WARM_STAGE + 1):
        train_loss, train_acc = train_epoch(warm_model, full_warm_loader, optimizer, device)
        val_loss, val_acc = evaluate(warm_model, val_loader, device)
        row = {
            "seed": seed,
            "method": "warm_start",
            "epoch": EPOCHS_PER_WARM_STAGE + stage_epoch,
            "stage": "full_data",
            "train_examples": len(fit_indices),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "heldout_loss": val_loss,
            "heldout_accuracy": val_acc,
        }
        rows.append(row)
        append_row(curves_path, row)
        print_row(row, start_time)

    save_checkpoint(seed_dir / "warm_final.pt", warm_model, seed, "warm_start")

    # Cold-start arm: fresh initialization and full fit split from epoch 1.
    seed_everything(seed + 202)
    cold_model = ResNet(num_classes=10).to(device)
    cold_loader = make_loader(trainset, fit_indices, seed + 403, device)
    optimizer = make_optimizer(cold_model)
    cold_start_time = time.time()
    for epoch in range(1, COLD_EPOCHS + 1):
        train_loss, train_acc = train_epoch(cold_model, cold_loader, optimizer, device)
        val_loss, val_acc = evaluate(cold_model, val_loader, device)
        row = {
            "seed": seed,
            "method": "cold_start",
            "epoch": epoch,
            "stage": "full_data",
            "train_examples": len(fit_indices),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "heldout_loss": val_loss,
            "heldout_accuracy": val_acc,
        }
        rows.append(row)
        append_row(curves_path, row)
        print_row(row, cold_start_time)

    save_checkpoint(seed_dir / "cold_final.pt", cold_model, seed, "cold_start")
    return rows


def print_row(row: Dict[str, object], start_time: float) -> None:
    elapsed = time.time() - start_time
    print(
        f"seed={row['seed']} method={row['method']} epoch={row['epoch']} "
        f"stage={row['stage']} train_acc={row['train_accuracy']:.4f} "
        f"heldout_acc={row['heldout_accuracy']:.4f} elapsed_s={elapsed:.1f}",
        flush=True,
    )


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_row(path: Path, row: Dict[str, object]) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writerow(row)


def write_summary(output_dir: Path, all_rows: List[Dict[str, object]]) -> None:
    summaries = []
    for seed in range(NUM_SEEDS):
        seed_rows = [r for r in all_rows if r["seed"] == seed]
        if not seed_rows:
            continue
        for method in ("warm_start", "cold_start"):
            method_rows = [r for r in seed_rows if r["method"] == method]
            if not method_rows:
                continue
            final = max(method_rows, key=lambda r: int(r["epoch"]))
            summaries.append(
                {
                    "seed": seed,
                    "method": method,
                    "final_train_accuracy": final["train_accuracy"],
                    "final_heldout_accuracy": final["heldout_accuracy"],
                    "final_train_loss": final["train_loss"],
                    "final_heldout_loss": final["heldout_loss"],
                }
            )
    write_rows(output_dir / "summary_by_seed.csv", summaries)


def plot_curves(output_dir: Path, rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; CSV metrics were saved without a plot.", flush=True)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    series = (
        ("half_only", "50% data (pretraining stage)", "#78A9D1", "--"),
        ("warm_continued", "Warm start, then full data", "#165A91", "-"),
        ("cold_start", "Full data from scratch", "#E87500", "-"),
    )
    for ax, metric, ylabel in (
        (axes[0], "heldout_accuracy", "Held-out accuracy"),
        (axes[1], "train_accuracy", "Training accuracy"),
    ):
        for series_name, label, color, linestyle in series:
            if series_name == "half_only":
                method_rows = [
                    r for r in rows if r["method"] == "warm_start" and r["stage"] == "half_data"
                ]
            elif series_name == "warm_continued":
                method_rows = [
                    r for r in rows if r["method"] == "warm_start" and r["stage"] == "full_data"
                ]
            else:
                method_rows = [r for r in rows if r["method"] == "cold_start"]
            by_seed: Dict[int, Dict[int, float]] = {}
            for row in method_rows:
                epoch = int(row["epoch"])
                if series_name == "warm_continued":
                    epoch -= EPOCHS_PER_WARM_STAGE
                by_seed.setdefault(int(row["seed"]), {})[epoch] = float(row[metric])
            common_epochs = sorted(set.intersection(*(set(v) for v in by_seed.values())))
            values = np.asarray(
                [[by_seed[seed][epoch] for epoch in common_epochs] for seed in sorted(by_seed)]
            )
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
            if series_name == "warm_continued":
                common_epochs = [epoch + EPOCHS_PER_WARM_STAGE for epoch in common_epochs]
            ax.plot(common_epochs, mean, color=color, linestyle=linestyle, label=label)
            ax.fill_between(
                common_epochs,
                np.maximum(0.0, mean - std),
                np.minimum(1.0, mean + std),
                color=color,
                alpha=0.18,
            )
        if metric == "heldout_accuracy":
            ax.axvline(EPOCHS_PER_WARM_STAGE, color="0.5", linestyle="--", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Ash & Adams Figure 1 protocol: CIFAR-10 ResNet-18")
    fig.savefig(output_dir / "figure1_replication.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print(f"protocol={json.dumps(protocol_metadata(), sort_keys=True)}", flush=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
    trainset = datasets.CIFAR10(
        root=str(args.data_root), train=True, download=True, transform=transform
    )
    all_rows: List[Dict[str, object]] = []
    for seed in range(NUM_SEEDS):
        print(f"starting seed {seed}/{NUM_SEEDS - 1}", flush=True)
        rows = run_seed(seed, trainset, args.output_dir, device)
        all_rows.extend(rows)
        write_summary(args.output_dir, all_rows)
    plot_curves(args.output_dir, all_rows)
    print(f"outputs={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
