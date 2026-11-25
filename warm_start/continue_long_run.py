#!/usr/bin/env python3
"""Continue matched warm, cold, and LR-rewarmed CIFAR-10 runs.

Starting checkpoints are the Figure 1 endpoints and LR-rewarmed endpoints.
Warm has completed 350 full-data epochs; cold and LR-rewarmed have completed
700. Metrics are appended per epoch; continuation checkpoints include
optimizer state for recovery.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import replicate_figure1 as baseline


METHODS = ("warm_start", "cold_start", "lr_rewarm")
START_FULL_DATA_EPOCHS = {
    "warm_start": baseline.EPOCHS_PER_WARM_STAGE,
    "cold_start": baseline.COLD_EPOCHS,
    "lr_rewarm": baseline.EPOCHS_PER_WARM_STAGE * 2,
}
CHECKPOINT_INTERVAL = 50
CONTINUATION_LR = baseline.LEARNING_RATE


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=here / "data")
    parser.add_argument("--input-dir", type=Path, default=here / "outputs")
    parser.add_argument(
        "--output-dir", type=Path, default=here / "outputs" / "long_run"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument(
        "--target-full-data-epochs",
        type=int,
        default=3500,
        help="Total full-data epochs, including the 700-epoch starting run.",
    )
    return parser.parse_args()


def input_checkpoint(input_dir: Path, seed: int, method: str) -> Path:
    seed_dir = input_dir / f"seed_{seed:02d}"
    if method == "lr_rewarm":
        return seed_dir / "lr_rewarm" / "final.pt"
    return seed_dir / f"{method.replace('_start', '')}_final.pt"


def load_model(path: Path, seed: int, method: str, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {method} endpoint for seed {seed}: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("seed") != seed:
        raise ValueError(f"Checkpoint seed mismatch in {path}: expected {seed}")
    metadata_key = "variant" if method == "lr_rewarm" else "method"
    if checkpoint.get(metadata_key) != method:
        raise ValueError(
            f"Checkpoint {path} identifies {metadata_key}="
            f"{checkpoint.get(metadata_key)!r}; expected {method!r}."
        )
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise ValueError(f"No model_state_dict in {path}")
    model = baseline.ResNet(num_classes=10).to(device)
    model.load_state_dict(state, strict=True)
    return model, checkpoint


def save_resumable_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler_generator_state: torch.Tensor,
    seed: int,
    method: str,
    completed_epoch: int,
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "sampler_generator_state": sampler_generator_state,
            "seed": seed,
            "method": method,
            "completed_full_data_epoch": completed_epoch,
            "continuation_learning_rate": CONTINUATION_LR,
            "starting_checkpoint_full_data_epochs": START_FULL_DATA_EPOCHS[method],
        },
        temporary_path,
    )
    temporary_path.replace(path)


def make_full_loader(
    trainset: datasets.CIFAR10,
    indices: Iterable[int],
    seed: int,
    device: torch.device,
) -> DataLoader:
    # Every arm for a seed gets the same minibatch ordering.
    return baseline.make_loader(trainset, indices, seed + 402, device)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        n = labels.size(0)
        loss_sum += loss.item() * n
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += n
    return loss_sum / count, correct / count


def read_existing_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def keep_rows_through_epoch(
    path: Path, rows: List[Dict[str, object]], completed_epoch: int
) -> List[Dict[str, object]]:
    """Drop log entries beyond the last durable model checkpoint."""
    retained = [row for row in rows if int(row["full_data_epoch"]) <= completed_epoch]
    if rows:
        fields = list(rows[0].keys())
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(retained)
    return retained


def run_method(
    seed: int,
    method: str,
    trainset: datasets.CIFAR10,
    fit_indices: List[int],
    val_indices: List[int],
    output_dir: Path,
    input_dir: Path,
    device: torch.device,
    target_epoch: int,
) -> Dict[str, object]:
    method_dir = output_dir / f"seed_{seed:02d}" / method
    method_dir.mkdir(parents=True, exist_ok=True)
    curves_path = method_dir / "curves.csv"
    latest_path = method_dir / "checkpoint_latest.pt"
    rows = read_existing_rows(curves_path)

    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model = baseline.ResNet(num_classes=10).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        start_epoch = int(checkpoint["completed_full_data_epoch"])
        optimizer = baseline.make_optimizer(model)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        sampler_generator_state = checkpoint.get("sampler_generator_state")
        rows = keep_rows_through_epoch(curves_path, rows, start_epoch)
    else:
        start_path = input_checkpoint(input_dir, seed, method)
        model, checkpoint = load_model(start_path, seed, method, device)
        expected_start_epoch = START_FULL_DATA_EPOCHS[method]
        start_epoch = int(
            checkpoint.get("completed_full_data_epoch", expected_start_epoch)
        )
        if start_epoch != expected_start_epoch:
            raise ValueError(
                f"{start_path} ends at full-data epoch {start_epoch}; expected "
                f"{expected_start_epoch} for {method}."
            )
        optimizer = baseline.make_optimizer(model)
        sampler_generator_state = None
        # There is no durable continuation checkpoint, so discard any partial
        # continuation rows and restart cleanly from the input endpoint.
        with curves_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "seed", "method", "full_data_epoch", "learning_rate",
                    "train_loss", "train_accuracy", "heldout_loss",
                    "heldout_accuracy",
                ],
            )
            writer.writeheader()
        rows = []

    if not curves_path.exists():
        with curves_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "seed", "method", "full_data_epoch", "learning_rate",
                    "train_loss", "train_accuracy", "heldout_loss",
                    "heldout_accuracy",
                ],
            )
            writer.writeheader()

    if start_epoch > target_epoch:
        raise ValueError(
            f"{method} seed {seed} already reached epoch {start_epoch}, above "
            f"requested target {target_epoch}."
        )

    train_loader = make_full_loader(trainset, fit_indices, seed, device)
    if sampler_generator_state is not None:
        train_loader.sampler.generator.set_state(sampler_generator_state)
    val_loader = baseline.make_loader(trainset, val_indices, seed + 301, device)
    start_time = time.time()
    for epoch in range(start_epoch + 1, target_epoch + 1):
        for group in optimizer.param_groups:
            group["lr"] = CONTINUATION_LR
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device)
        heldout_loss, heldout_accuracy = baseline.evaluate(model, val_loader, device)
        row = {
            "seed": seed,
            "method": method,
            "full_data_epoch": epoch,
            "learning_rate": CONTINUATION_LR,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "heldout_loss": heldout_loss,
            "heldout_accuracy": heldout_accuracy,
        }
        with curves_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writerow(row)
            handle.flush()
        rows.append(row)

        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == target_epoch:
            save_resumable_checkpoint(
                latest_path,
                model,
                optimizer,
                train_loader.sampler.generator.get_state(),
                seed,
                method,
                epoch,
            )
        print(
            f"seed={seed} method={method} full_data_epoch={epoch} "
            f"train_loss={train_loss:.5f} heldout_loss={heldout_loss:.5f} "
            f"train_acc={train_accuracy:.4f} heldout_acc={heldout_accuracy:.4f} "
            f"elapsed_s={time.time() - start_time:.1f}",
            flush=True,
        )

    if rows:
        final_row = max(rows, key=lambda row: int(row["full_data_epoch"]))
        return {
            key: final_row[key]
            for key in (
                "seed", "method", "full_data_epoch", "train_loss",
                "train_accuracy", "heldout_loss", "heldout_accuracy",
            )
        }
    return {"seed": seed, "method": method, "full_data_epoch": start_epoch}


def write_summary(output_dir: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with (output_dir / "summary_by_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics = ("train_loss", "train_accuracy", "heldout_loss", "heldout_accuracy")
    aggregate = []
    for method in METHODS:
        method_rows = [row for row in rows if row.get("method") == method]
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in method_rows if metric in row])
            if values.size:
                aggregate.append(
                    {
                        "method": method,
                        "metric": metric,
                        "num_seeds": int(values.size),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    }
                )
    with (output_dir / "summary_across_seeds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)


def write_trajectory_summary(output_dir: Path, seeds: List[int]) -> None:
    """Aggregate the per-seed trajectories at each shared epoch."""
    output_rows = []
    metrics = (
        "learning_rate", "train_loss", "train_accuracy",
        "heldout_loss", "heldout_accuracy",
    )
    for method in METHODS:
        by_epoch: Dict[int, Dict[str, List[float]]] = {}
        for seed in seeds:
            curve_path = output_dir / f"seed_{seed:02d}" / method / "curves.csv"
            if not curve_path.exists():
                continue
            for row in read_existing_rows(curve_path):
                epoch = int(row["full_data_epoch"])
                bucket = by_epoch.setdefault(epoch, {metric: [] for metric in metrics})
                for metric in metrics:
                    bucket[metric].append(float(row[metric]))
        for epoch, values_by_metric in sorted(by_epoch.items()):
            row: Dict[str, object] = {
                "method": method,
                "full_data_epoch": epoch,
            }
            for metric, values_list in values_by_metric.items():
                values = np.asarray(values_list)
                row[f"{metric}_num_seeds"] = int(values.size)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = (
                    float(values.std(ddof=1)) if values.size > 1 else 0.0
                )
            output_rows.append(row)
    if output_rows:
        with (output_dir / "trajectory_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)


def main() -> None:
    args = parse_args()
    if args.target_full_data_epochs < max(START_FULL_DATA_EPOCHS.values()):
        raise ValueError(
            f"Target must be at least {max(START_FULL_DATA_EPOCHS.values())} "
            "full-data epochs."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "device": str(device),
                "seeds": args.seeds,
                "methods": METHODS,
                "starting_full_data_epochs": START_FULL_DATA_EPOCHS,
                "target_full_data_epochs": args.target_full_data_epochs,
                "continuation_learning_rate": CONTINUATION_LR,
                "outputs": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(baseline.NORMALIZE_MEAN, baseline.NORMALIZE_STD),
        ]
    )
    trainset = datasets.CIFAR10(
        root=str(args.data_root), train=True, download=False, transform=transform
    )
    summaries: List[Dict[str, object]] = []
    for seed in args.seeds:
        indices = np.random.RandomState(seed).permutation(len(trainset))
        n_fit = int(len(trainset) * baseline.FIT_FRACTION_OF_CIFAR_TRAIN)
        fit_indices = indices[:n_fit].tolist()
        val_indices = indices[n_fit:].tolist()
        for method in METHODS:
            result = run_method(
                seed,
                method,
                trainset,
                fit_indices,
                val_indices,
                args.output_dir,
                args.input_dir,
                device,
                args.target_full_data_epochs,
            )
            summaries = [
                row
                for row in summaries
                if not (row.get("seed") == seed and row.get("method") == method)
            ]
            summaries.append(result)
            write_summary(args.output_dir, summaries)
    write_trajectory_summary(args.output_dir, args.seeds)


if __name__ == "__main__":
    main()
