from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.common import append_csv, get_device, load_checkpoint, load_config, seed_everything
from src.data import task_loader
from src.models import CIFARResNet18


def main():
    parser = argparse.ArgumentParser(description="Compute representation diagnostics on fixed training-task images")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--checkpoint", action="append")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config, device = load_config(args.config), get_device(args.device)
    paths = [Path(p) for p in args.checkpoint] if args.checkpoint else []
    if not paths:
        for seed in args.seed or config["experiment"]["seeds"]:
            root = Path(config["experiment"]["checkpoint_dir"]) / f"seed_{seed}"
            paths.extend([root / "init.pt", *[root / f"after_task_{i}.pt" for i in range(1, 5)]])
    output = Path(config["experiment"]["output_dir"]) / "representation_diagnostics.csv"
    for path in paths:
        state = load_checkpoint(path, device)
        seed = int(state["seed"])
        seed_everything(seed)
        # Fixed deterministic examples drawn from the sequential training classes.
        name = config["data"]["training_tasks"][0]
        _, dataset = task_loader(config["experiment"]["data_dir"], name, True, 128,
            config["data"].get("num_workers", 0), config["data"].get("download", True), False,
            seed=seed, shuffle=False)
        indices = np.random.default_rng(seed).choice(len(dataset), min(args.max_images, len(dataset)), replace=False)
        loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=128, shuffle=False,
                            num_workers=config["data"].get("num_workers", 0))
        model = CIFARResNet18().to(device)
        model.backbone.load_state_dict(state["backbone"])
        model.eval()
        feats, zero_count, activation_count, norm_sum, n_images = [], 0, 0, 0.0, 0
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                x = model.backbone.conv1(images)
                x = model.backbone.bn1(x)
                x = model.backbone.relu(x)
                activation_count += x.numel()
                zero_count += (x <= 0).sum().item()
                features = model.features(images)
                feats.append(features.cpu().numpy())
                norm_sum += features.norm(dim=1).sum().item()
                n_images += len(images)
        matrix = np.concatenate(feats, axis=0).astype(np.float64)
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        eigenvalues = singular ** 2
        participation_ratio = float(eigenvalues.sum() ** 2 / max((eigenvalues ** 2).sum(), 1e-12))
        variance = matrix.var(axis=0)
        rank = int((variance > max(float(variance.max()) * 1e-6, 1e-12)).sum())
        row = {"seed": seed, "checkpoint": "init" if state["completed_tasks"] == 0 else f"after_task_{state['completed_tasks']}",
               "tasks_learned": state["completed_tasks"], "images": n_images,
               "participation_ratio": participation_ratio, "effective_rank_variance": rank,
               "relu_zero_fraction_first_stage": zero_count / max(activation_count, 1),
               "mean_feature_norm": norm_sum / max(n_images, 1),
               "backbone_parameter_norm": float(torch.linalg.vector_norm(torch.cat(
                   [p.detach().flatten().cpu() for p in model.backbone.parameters()]))) }
        append_csv(output, row)
        print(row)


if __name__ == "__main__":
    main()
