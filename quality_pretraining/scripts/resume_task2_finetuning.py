"""Continue a saved Task-2 run without reinitializing its adapter or data stream."""

import argparse
import csv
import json
from pathlib import Path

import torch

from scripts.run_task2_finetuning import (
    append_metric, evaluate, parse_dtype, save_task2_checkpoint, should_evaluate,
)
from src.data_model import generate_dataset
from src.io_utils import append_run_history, load_config, load_pretraining_checkpoint, load_teachers
from src.losses import resolve_loss_type, teacher_student_loss
from src.model import Student


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--additional-steps", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, required=True,
                        help="Checkpoint interval in SGD updates, relative to the resumed step.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.additional_steps <= 0 or args.checkpoint_every <= 0:
        parser.error("--additional-steps and --checkpoint-every must be positive")

    run_dir = args.run_dir.resolve()
    config = load_config(run_dir)
    if config.get("experiment") != "finetune_task2":
        raise ValueError("--run-dir must identify an existing Task-2 fine-tuning run")
    checkpoint_root = run_dir / "checkpoints"
    checkpoints = []
    for path in checkpoint_root.glob("step_*"):
        if path.name[5:].isdigit() and all(
            (path / file).is_file() for file in ("w.pt", "training_state.pt", "metrics.json")
        ):
            checkpoints.append((int(path.name[5:]), path))
    if not checkpoints:
        raise FileNotFoundError(f"No complete Task-2 checkpoints found in {checkpoint_root}")
    start, checkpoint = max(checkpoints)
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    if state["task2_step"] != start:
        raise ValueError("Checkpoint step does not match its training state")

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or max(int(row["task2_step"]) for row in rows) != start:
        raise ValueError(
            "The metrics endpoint must match the latest complete checkpoint. "
            "Resume a completed run; refusing to append overlapping or incomplete history."
        )

    source = Path(config["source_pretrain_run"])
    source_config = load_config(source)
    loss_type = resolve_loss_type(None, config, source_config, state)
    dtype = parse_dtype(config["dtype"])
    device = args.device
    d, T = config["d"], config["T"]
    teachers = load_teachers(source, device, dtype)
    base = load_pretraining_checkpoint(source, config["pretraining_step"], device, dtype)
    resolve_loss_type(loss_type, base)
    if any(config[key] != source_config[key] for key in ("d", "T", "kappa", "dtype")):
        raise ValueError("Task-2 and pre-training model configurations do not match")

    model = Student(d, config["kappa"], dtype=dtype, device=device)
    model.load_large_rank(base["W"])
    model.set_finetuning_mode()
    with torch.no_grad():
        model.w.copy_(torch.load(checkpoint / "w.pt", map_location=device, weights_only=True))
    optimizer = torch.optim.SGD([model.w], lr=config["lr"])
    optimizer.load_state_dict(state["optimizer_state"])
    if any(group["lr"] != config["lr"] for group in optimizer.param_groups):
        raise ValueError("Saved optimizer learning rate differs from the run configuration")
    generator = torch.Generator(device=device)
    generator.set_state(state["generator_state"].cpu())

    # Exactly the same fixed evaluation inputs and teachers as the original run.
    n_test = config["n_test"]
    pretraining_test = generate_dataset(n_test, T, d, teachers.S_star, config["pretraining_test_seed"])
    task1_test = generate_dataset(n_test, T, d, teachers.S1_star, config["task1_test_seed"])
    task2_test = generate_dataset(n_test, T, d, teachers.S2_star, config["task2_test_seed"])
    # Keep the ORIGINAL task-switch and pre-Task-1 baselines. Recomputing them
    # from the resumed adapter would incorrectly reset forgetting to zero.
    baseline_keys = ("pretraining_error_before_task1", "pretraining_error_before_task2",
                     "task1_error_before_task2")
    if any(key not in config for key in baseline_keys) or "attention_baselines" not in config:
        raise ValueError("Run configuration is missing original forgetting baselines")
    attention_keys = ("pretraining_attention_ce_before_task2", "task1_attention_ce_before_task2",
                      "pretraining_attention_ce_before_task1", "task1_attention_ce_before_task1")
    if any(key not in config["attention_baselines"] for key in attention_keys):
        raise ValueError("Run configuration is missing original attention forgetting baselines")
    target = start + args.additional_steps
    save_steps = set(range(start + args.checkpoint_every, target + 1, args.checkpoint_every)) | {target}
    updated_config = {
        **config,
        "target_steps": target,
        "target_steps_over_d": target / d,
        "checkpoint_steps": sorted(set(config.get("checkpoint_steps", [])) | save_steps),
    }
    temporary = run_dir / "config.tmp.json"
    temporary.write_text(json.dumps(updated_config, indent=2))
    temporary.replace(run_dir / "config.json")
    append_run_history(run_dir, start, target)
    print(f"Resuming Task 2: {run_dir}", flush=True)
    print(f"Steps: {start:,} -> {target:,}; t/d: {start/d:g} -> {target/d:g}", flush=True)
    print(f"Loss: {loss_type}; lr: {config['lr']}; new checkpoints: {sorted(save_steps)}", flush=True)

    for step in range(start + 1, target + 1):
        X = torch.randn(1, T, d, generator=generator, dtype=dtype, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = teacher_student_loss(model, X, teachers.S2_star, loss_type)
        loss.backward()
        optimizer.step()
        if step in save_steps or should_evaluate(step, d, config["eval_every"]):
            metric = evaluate(
                model, teachers, pretraining_test, task1_test, task2_test,
                config["pretraining_step"], config["task1_step"], step,
                config["pretraining_error_before_task1"],
                config["pretraining_error_before_task2"],
                config["task1_error_before_task2"],
                attention_baselines=config["attention_baselines"],
            )
            append_metric(metrics_path, metric)
            print(f"[step {step:,}] t/d={step/d:g} "
                  f"CE={metric['task2_attention_ce']:.6e} "
                  f"KL={metric['task2_attention_kl']:.6e} "
                  f"MSE={metric['task2_error']:.6e} "
                  f"F1_CE={metric['forgetting_task1_attention_ce']:.6e}", flush=True)
            if step in save_steps:
                save_task2_checkpoint(checkpoint_root, step, model, optimizer,
                                           generator, metric, loss_type=loss_type)
    print(f"Task-2 continuation completed. Metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
