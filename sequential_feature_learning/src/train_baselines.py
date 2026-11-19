from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.common import (amp_enabled, append_csv, autocast_context, ensure_dirs, get_device,
                        load_config, make_scaler, seed_everything)
from src.data import task_loader
from src.models import CIFARResNet18, backbone_state, head_states


def main():
    parser = argparse.ArgumentParser(description="Train the last-task-only CIFAR-100 baseline")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config, device = load_config(args.config), get_device(args.device)
    exp = config["experiment"]
    epochs = config.get("baseline", {}).get("last_task_only_epochs", config["training"]["epochs_per_task"])
    task = config["data"]["training_tasks"][-1]
    for seed in args.seed or exp["seeds"]:
        seed_everything(int(seed))
        model = CIFARResNet18().to(device)
        model.add_head(task)
        directory = Path(exp["checkpoint_dir"]) / f"seed_{seed}" / "baselines"
        ensure_dirs(directory, Path(exp["output_dir"]) / "trajectories")
        torch.save({"format_version": 1, "model": model.state_dict(), "backbone": backbone_state(model),
                    "heads": head_states(model), "completed_tasks": 1, "task_order": [task],
                    "seed": int(seed), "config": config, "optimizer_steps": 0, "examples_seen": 0,
                    "metadata": {"architecture": "cifar_resnet18", "baseline": "last_task_only"}},
                   directory / "last_task_init.pt")
        loader, _ = task_loader(exp["data_dir"], task, True, config["training"]["batch_size"],
            config["data"].get("num_workers", 0), config["data"].get("download", True), True,
            config["data"].get("train_examples_per_task", 2500) // 5, int(seed) + 1003, shuffle=True)
        cfg = config["training"]
        use_amp = amp_enabled(device, cfg.get("amp", False))
        scaler = make_scaler(device, use_amp)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"], momentum=cfg["momentum"],
                                    weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        steps = examples = 0
        for epoch in range(epochs):
            model.train()
            total_loss = correct = count = 0
            for images, labels in loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, use_amp):
                    logits = model(images, task)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                n = len(labels)
                total_loss += loss.item() * n
                correct += (logits.argmax(1) == labels).sum().item()
                count += n
                steps += 1
                examples += n
            scheduler.step()
            row = {"seed": seed, "task": task, "task_index": 1, "epoch": epoch + 1,
                   "train_loss": total_loss / count, "train_accuracy": correct / count,
                   "learning_rate": scheduler.get_last_lr()[0], "optimizer_steps": steps,
                   "examples_seen": examples, "baseline": "last_task_only"}
            append_csv(Path(exp["output_dir"]) / "trajectories" / "last_task_only_train.csv", row)
            print(f"seed={seed} last_task_only epoch={epoch + 1}/{epochs} loss={row['train_loss']:.4f} "
                  f"acc={row['train_accuracy']:.3f}")
            payload = {"format_version": 1, "model": model.state_dict(), "backbone": backbone_state(model),
                "heads": head_states(model), "completed_tasks": 1, "task_order": [task], "seed": int(seed),
                "config": config, "optimizer_steps": steps, "examples_seen": examples,
                "metadata": {"architecture": "cifar_resnet18", "baseline": "last_task_only"}}
            torch.save(payload, directory / "last_task_only_last.pt")
        torch.save(payload, directory / "last_task_only.pt")


if __name__ == "__main__":
    main()
