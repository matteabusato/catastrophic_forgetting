#!/usr/bin/env python3
"""Continue a seed from its half-data checkpoint with ELR re-warming.

Two variants are supported: a nominal learning-rate restart alone, or the
same restart with per-step projection of convolutional and linear weights to
their norms at random initialization.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torchvision import datasets

import replicate_figure1 as baseline


FULL_DATA_EPOCHS = 700
REWARM_LR = 1e-2
FINAL_LR = 1e-3
CHECKPOINT_INTERVAL = 50


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", choices=("lr_rewarm", "lr_rewarm_project"), required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=here / "data",
        help="Directory containing torchvision CIFAR-10 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Use a distinct output directory for each variant.",
    )
    return parser.parse_args()


def initial_weight_norms(device: torch.device, seed: int) -> Dict[str, torch.Tensor]:
    """Recreate the initialization used before the saved warm stage."""
    baseline.seed_everything(seed + 101)
    initial_model = baseline.ResNet(num_classes=10).to(device)
    return {
        name: parameter.detach().norm().clone()
        for name, parameter in initial_model.named_parameters()
        if name.endswith(".weight") and parameter.ndim >= 2
    }


@torch.no_grad()
def project_weight_norms(
    model: nn.Module, reference_norms: Dict[str, torch.Tensor]
) -> None:
    for name, parameter in model.named_parameters():
        if name not in reference_norms:
            continue
        current_norm = parameter.norm()
        if current_norm.item() > 0:
            parameter.mul_(reference_norms[name] / current_norm)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    variant: str,
    completed_epoch: int,
    reference_norms: Optional[Dict[str, torch.Tensor]],
    seed: int,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "seed": seed,
        "variant": variant,
        "completed_full_data_epoch": completed_epoch,
        "starting_checkpoint": "warm_after_half.pt",
        "learning_rate_schedule": {
            "type": "cosine",
            "start": REWARM_LR,
            "end": FINAL_LR,
            "epochs": FULL_DATA_EPOCHS,
        },
    }
    if reference_norms is not None:
        payload["initial_weight_norms"] = reference_norms
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    seed = args.seed
    if args.checkpoint is None:
        args.checkpoint = (
            Path(__file__).resolve().parent
            / "outputs"
            / f"seed_{seed:02d}"
            / "warm_after_half.pt"
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Seed-{seed} transition checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} variant={args.variant} seed={seed}", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)

    transform = baseline.transforms.Compose(
        [
            baseline.transforms.ToTensor(),
            baseline.transforms.Normalize(baseline.NORMALIZE_MEAN, baseline.NORMALIZE_STD),
        ]
    )
    trainset = datasets.CIFAR10(
        root=str(args.data_root), train=True, download=False, transform=transform
    )
    indices = baseline.np.random.RandomState(seed).permutation(len(trainset))
    n_fit = int(len(trainset) * baseline.FIT_FRACTION_OF_CIFAR_TRAIN)
    fit_indices = indices[:n_fit].tolist()
    val_indices = indices[n_fit:].tolist()
    val_loader = baseline.make_loader(trainset, val_indices, seed + 301, device)
    full_loader = baseline.make_loader(trainset, fit_indices, seed + 402, device)

    reference_norms = (
        initial_weight_norms(device, seed) if args.variant == "lr_rewarm_project" else None
    )
    baseline.seed_everything(seed + 101)
    model = baseline.ResNet(num_classes=10).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint.get("seed") != seed or checkpoint.get("method") != "warm_after_half":
        raise ValueError("Checkpoint metadata does not match the requested warm_after_half seed")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    # Match the reference experiment's optimizer reset at the full-data boundary.
    optimizer = baseline.make_optimizer(model)
    csv_path = args.output_dir / "curves.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed", "variant", "epoch", "stage", "learning_rate",
                "train_loss", "train_accuracy", "heldout_loss", "heldout_accuracy",
            ],
        )
        writer.writeheader()
        start_time = time.time()
        for epoch in range(FULL_DATA_EPOCHS):
            progress = epoch / max(FULL_DATA_EPOCHS - 1, 1)
            learning_rate = FINAL_LR + 0.5 * (REWARM_LR - FINAL_LR) * (
                1.0 + math.cos(math.pi * progress)
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_count = 0
            for images, labels in full_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                if reference_norms is not None:
                    project_weight_norms(model, reference_norms)
                count = labels.size(0)
                train_loss_sum += loss.item() * count
                train_correct += (logits.argmax(dim=1) == labels).sum().item()
                train_count += count

            train_loss = train_loss_sum / train_count
            train_accuracy = train_correct / train_count
            heldout_loss, heldout_accuracy = baseline.evaluate(model, val_loader, device)
            row = {
                "seed": seed,
                "variant": args.variant,
                "epoch": baseline.EPOCHS_PER_WARM_STAGE + epoch + 1,
                "stage": "full_data",
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "heldout_loss": heldout_loss,
                "heldout_accuracy": heldout_accuracy,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"seed={seed} variant={args.variant} epoch={row['epoch']} "
                f"lr={learning_rate:.6g} train_acc={train_accuracy:.4f} "
                f"heldout_acc={heldout_accuracy:.4f} elapsed_s={time.time()-start_time:.1f}",
                flush=True,
            )

            completed_epoch = epoch + 1
            if completed_epoch % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    args.output_dir / f"checkpoint_epoch_{completed_epoch:03d}.pt",
                    model,
                    args.variant,
                    completed_epoch,
                    reference_norms,
                    seed,
                )

    save_checkpoint(
        args.output_dir / "final.pt", model, args.variant, FULL_DATA_EPOCHS,
        reference_norms, seed
    )
    print(f"outputs={args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
