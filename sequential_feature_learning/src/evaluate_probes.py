from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.common import (amp_enabled, append_csv, append_jsonl, autocast_context, ensure_dirs,
                        get_device, load_checkpoint, load_config, make_scaler, seed_everything)
from src.data import task_loader
from src.models import CIFARResNet18


def eval_epoch(model, task, loader, device, criterion, use_amp):
    model.eval()
    loss_sum = correct = count = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast_context(device, use_amp):
                logits = model(images, task)
                loss = criterion(logits, labels)
            n = labels.size(0)
            loss_sum += loss.item() * n
            correct += (logits.argmax(1) == labels).sum().item()
            count += n
    return loss_sum / count, correct / count


def evaluate_one(checkpoint_path: Path, config: dict, device: torch.device):
    state = load_checkpoint(checkpoint_path, device)
    seed = int(state["seed"])
    # Each method starts from a separately constructed model loaded from the immutable saved backbone.
    probe_cfg, data_cfg, exp = config["probe"], config["data"], config["experiment"]
    results_dir = Path(exp["output_dir"]) / "probes"
    ensure_dirs(results_dir / "trajectories")
    use_amp = amp_enabled(device, probe_cfg.get("amp", False))
    baseline_name = state.get("metadata", {}).get("baseline")
    checkpoint_label = (baseline_name if baseline_name else
                        ("init" if state["completed_tasks"] == 0 else f"after_task_{state['completed_tasks']}"))

    for probe in data_cfg["probe_tasks"]:
        train_loader, _ = task_loader(exp["data_dir"], probe, True, probe_cfg["batch_size"],
            data_cfg.get("num_workers", 0), data_cfg.get("download", True), True,
            data_cfg.get("probe_examples_per_class", 50), seed + 5000, shuffle=True)
        test_loader, _ = task_loader(exp["data_dir"], probe, False, probe_cfg["batch_size"],
            data_cfg.get("num_workers", 0), data_cfg.get("download", True), False,
            seed=seed, shuffle=False)

        # Common head initialization for this seed/probe across checkpoints and both methods.
        head_generator = torch.Generator().manual_seed(seed * 100003 + data_cfg["probe_tasks"].index(probe) * 97)
        initial_head = nn.Linear(512, 5)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_generator.initial_seed())
            initial_head.reset_parameters()
        head_state = {k: v.detach().clone() for k, v in initial_head.state_dict().items()}

        for method in ("linear", "finetune"):
            model = CIFARResNet18().to(device)
            model.backbone.load_state_dict(state["backbone"])
            model.add_head(probe, initial_state=head_state)
            for parameter in model.backbone.parameters():
                parameter.requires_grad_(method == "finetune")
            # eval() freezes BatchNorm running statistics even while the head is optimized.
            criterion = nn.CrossEntropyLoss()
            params = ([{"params": model.heads[probe].parameters(), "lr": probe_cfg["linear_learning_rate"]}]
                      if method == "linear" else [
                          {"params": model.backbone.parameters(), "lr": probe_cfg["finetune_backbone_learning_rate"]},
                          {"params": model.heads[probe].parameters(), "lr": probe_cfg["finetune_head_learning_rate"]}])
            optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=probe_cfg["weight_decay"])
            epochs = probe_cfg["linear_epochs"] if method == "linear" else probe_cfg["finetune_epochs"]
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            scaler = make_scaler(device, use_amp)
            best_test_acc = 0.0
            steps = 0
            # Reset data-loader shuffle order across all checkpoints for comparable trajectories.
            for epoch in range(epochs):
                if method == "linear":
                    model.eval()  # Freeze BatchNorm statistics for the linear probe.
                    model.heads[probe].train()
                else:
                    model.train()
                train_loss = train_correct = train_count = 0
                for images, labels in train_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(device, use_amp):
                        logits = model(images, probe)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    n = labels.size(0)
                    train_loss += loss.detach().item() * n
                    train_correct += (logits.argmax(1) == labels).sum().item()
                    train_count += n
                    steps += 1
                test_loss, test_acc = eval_epoch(model, probe, test_loader, device, criterion, use_amp)
                best_test_acc = max(best_test_acc, test_acc)
                scheduler.step()
                row = {"seed": seed, "checkpoint": checkpoint_label, "checkpoint_path": str(checkpoint_path),
                    "tasks_learned": state["completed_tasks"], "probe": probe, "method": method,
                    "epoch": epoch + 1, "train_loss": train_loss / train_count,
                    "train_accuracy": train_correct / train_count, "test_loss": test_loss,
                    "test_accuracy": test_acc, "best_test_accuracy": best_test_acc,
                    "optimizer_steps": steps, "source_optimizer_steps": state.get("optimizer_steps", 0),
                    "source_examples_seen": state.get("examples_seen", 0)}
                append_csv(results_dir / "trajectories" / "probe_trajectory.csv", row)
                append_jsonl(results_dir / "trajectories" / "probe_trajectory.jsonl", row)
                final_row = dict(row)
                if epoch == epochs - 1:
                    append_csv(results_dir / "probe_final.csv", final_row)
                    print(f"seed={seed} {checkpoint_label} {probe} {method}: final={test_acc:.3f} "
                          f"best={best_test_acc:.3f}")
            del model, optimizer, scheduler, scaler


def main():
    parser = argparse.ArgumentParser(description="Run linear-probe and fine-tuning evaluations")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--checkpoint", action="append", help="Checkpoint path; repeatable")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    device = get_device(args.device)
    paths = [Path(p) for p in args.checkpoint] if args.checkpoint else []
    if not paths:
        for seed in args.seed or config["experiment"]["seeds"]:
            directory = Path(config["experiment"]["checkpoint_dir"]) / f"seed_{seed}"
            paths.extend([directory / "init.pt", *[directory / f"after_task_{i}.pt" for i in range(1, 5)]])
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    for path in paths:
        evaluate_one(path, config, device)


if __name__ == "__main__":
    main()
