from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.common import (amp_enabled, append_csv, append_jsonl, autocast_context, config_snapshot,
                        ensure_dirs, get_device, load_checkpoint, load_config, make_scaler,
                        seed_everything)
from src.data import TRAINING_TASKS, task_loader
from src.models import CIFARResNet18, backbone_state, head_states


def save_checkpoint(path: Path, model: CIFARResNet18, seed: int, config: dict,
                    completed_tasks: int, optimizer_steps: int, examples_seen: int,
                    **extra):
    payload = {
        "format_version": 1, "model": model.state_dict(),
        "backbone": backbone_state(model), "heads": head_states(model),
        "completed_tasks": completed_tasks, "task_order": list(config["data"]["training_tasks"]),
        "seed": seed, "config": config_snapshot(config),
        "optimizer_steps": optimizer_steps, "examples_seen": examples_seen,
        "metadata": {"architecture": "cifar_resnet18", "num_classes_per_task": 5}, **extra,
    }
    torch.save(payload, path)


def train_seed(config: dict, seed: int, device: torch.device, resume: bool = False):
    exp = config["experiment"]
    output = Path(exp["output_dir"])
    ckpt_dir = Path(exp["checkpoint_dir"]) / f"seed_{seed}"
    ensure_dirs(output / "trajectories", ckpt_dir)
    seed_everything(seed)
    model = CIFARResNet18().to(device)
    train_cfg = config["training"]
    use_amp = amp_enabled(device, train_cfg.get("amp", False))
    scaler = make_scaler(device, use_amp)
    optimizer_steps, examples_seen, completed_tasks = 0, 0, 0
    start_task, start_epoch = 0, 0
    state = {}
    resume_path = ckpt_dir / "last.pt"
    if resume and resume_path.exists():
        state = load_checkpoint(resume_path, device)
        model.load_state_dict(state["model"])
        optimizer_steps = state["optimizer_steps"]
        examples_seen = state["examples_seen"]
        completed_tasks = state["completed_tasks"]
        start_task = state.get("current_task_index", completed_tasks)
        start_epoch = state.get("current_epoch", 0) + (1 if "current_epoch" in state else 0)
        if start_task >= len(config["data"]["training_tasks"]):
            print(f"seed={seed}: all tasks already complete in {resume_path}")
            return
        print(f"Resuming seed {seed} at task {start_task + 1}, epoch {start_epoch + 1}")
    else:
        save_checkpoint(ckpt_dir / "init.pt", model, seed, config, 0, 0, 0)

    for task_index, task in enumerate(config["data"]["training_tasks"]):
        if task_index < start_task:
            continue
        if task not in model.heads:
            model.add_head(task)
        for head_name, head in model.heads.items():
            for parameter in head.parameters():
                parameter.requires_grad_(head_name == task)
        loader, _ = task_loader(
            exp["data_dir"], task, train=True, batch_size=train_cfg["batch_size"],
            workers=config["data"].get("num_workers", 0), download=config["data"].get("download", True),
            augment=True, limit_per_class=config["data"].get("train_examples_per_task", 2500) // 5,
            seed=seed + 1000 + task_index)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.SGD(trainable, lr=train_cfg["learning_rate"],
                                    momentum=train_cfg["momentum"], weight_decay=train_cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs_per_task"])
        epoch_begin = start_epoch if task_index == start_task else 0
        if resume and task_index == start_task and state.get("current_task_index") == task_index:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if state.get("scaler"):
                scaler.load_state_dict(state["scaler"])
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epoch_begin, train_cfg["epochs_per_task"]):
            model.train()
            loss_sum, correct, count = 0.0, 0, 0
            for images, labels in loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, use_amp):
                    logits = model(images, task)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                n = labels.size(0)
                loss_sum += loss.detach().item() * n
                correct += (logits.argmax(1) == labels).sum().item()
                count += n
                optimizer_steps += 1
                examples_seen += n
            scheduler.step()
            row = {"seed": seed, "task": task, "task_index": task_index + 1, "epoch": epoch + 1,
                   "train_loss": loss_sum / count, "train_accuracy": correct / count,
                   "learning_rate": scheduler.get_last_lr()[0], "optimizer_steps": optimizer_steps,
                   "examples_seen": examples_seen}
            append_csv(output / "trajectories" / "sequential_train.csv", row)
            append_jsonl(output / "trajectories" / "sequential_train.jsonl", row)
            save_checkpoint(ckpt_dir / "last.pt", model, seed, config, completed_tasks,
                            optimizer_steps, examples_seen, current_task_index=task_index,
                            current_epoch=epoch, optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), scaler=scaler.state_dict())
            print(f"seed={seed} task={task_index + 1}/{len(config['data']['training_tasks'])} "
                  f"epoch={epoch + 1}/{train_cfg['epochs_per_task']} loss={row['train_loss']:.4f} "
                  f"acc={row['train_accuracy']:.3f}")
        completed_tasks = task_index + 1
        save_checkpoint(ckpt_dir / f"after_task_{completed_tasks}.pt", model, seed, config,
                        completed_tasks, optimizer_steps, examples_seen)
        save_checkpoint(ckpt_dir / "last.pt", model, seed, config, completed_tasks,
                        optimizer_steps, examples_seen)
        # A new task always gets a new optimizer and schedule.
        start_epoch = 0
        resume = False


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-100 tasks sequentially")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--seed", type=int, action="append", help="Override configured seeds; repeatable")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--resume", action="store_true", help="Resume each seed from checkpoints/seed_N/last.pt")
    args = parser.parse_args()
    config = load_config(args.config)
    training_tasks = set(config["data"]["training_tasks"])
    probe_tasks = set(config["data"]["probe_tasks"])
    overlap = training_tasks & probe_tasks
    if overlap:
        raise ValueError(f"Probe tasks must be excluded from sequential training: {sorted(overlap)}")
    seeds = args.seed or config["experiment"]["seeds"]
    device = get_device(args.device)
    print(f"device={device}; seeds={seeds}")
    for seed in seeds:
        train_seed(config, int(seed), device, args.resume)


if __name__ == "__main__":
    main()
