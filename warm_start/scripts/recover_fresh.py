#!/usr/bin/env python3
"""Recover only fresh_full from a legacy run, then checkpoint its continuation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from warm_start.src.data import generate_dataset
from warm_start.src.experiment import experiment_seeds
from warm_start.src.io_utils import write_json
from warm_start.src.model import Student, Teacher
from warm_start.src.observables import evaluate, matrix_metrics
from warm_start.src.recovery import FreshTrainer, atomic_torch_save, evaluate_train_partitions


def latest_fresh_row(path: Path) -> dict:
    with path.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("phase") == "fresh_full"]
    valid = []
    for row in rows:
        try:
            step = int(row["updates"])
            values = [float(row[key]) for key in ("train_mse", "test_mse")]
        except (KeyError, ValueError, TypeError):
            continue  # A killed writer may leave a partial last row.
        if all(math.isfinite(value) for value in values):
            valid.append((step, row))
    if not valid:
        raise ValueError("No valid fresh_full rows found in the original metrics.csv.")
    return max(valid, key=lambda item: item[0])[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Original timestamped folder containing config.json and metrics.csv")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from fresh_recovery/latest.pt without replay")
    parser.add_argument("--target-steps", type=int, help="Absolute fresh_full target; defaults to original")
    parser.add_argument("--log-every", type=int, help="Defaults to original log interval")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--device", help="Defaults to original device; use same device type for replay")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    raw = json.loads((run_dir / "config.json").read_text())
    settings = raw["final_training"]
    if settings.get("max_steps") is None or settings.get("early_stopping", True):
        raise ValueError("Recovery supports the fixed-step, early-stopping-disabled runner.")
    target = args.target_steps if args.target_steps is not None else settings["max_steps"]
    interval = args.log_every if args.log_every is not None else settings.get("log_every_steps")
    if interval is None or min(target, interval, args.checkpoint_every, args.threads) < 1:
        raise ValueError("Target, logging/checkpoint intervals and threads must be positive.")
    torch.set_num_threads(args.threads)
    device = args.device or raw["device"]
    destination = run_dir / "fresh_recovery"
    checkpoint_path = destination / "latest.pt"
    if args.resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved["source_config"] != raw:
            raise ValueError("The original config differs from the checkpoint's source config.")
        replay_row = None
        start_step = saved["trainer"]["step"]
    else:
        replay_row = latest_fresh_row(run_dir / "metrics.csv")
        start_step = int(replay_row["updates"])
        saved = None
    if target < start_step:
        raise ValueError(f"Target {target} is below saved/logged step {start_step}.")
    if not args.resume:
        destination.mkdir(exist_ok=False)
    write_json(destination / "status.json", {"status": "initializing", "target_steps": target})
    print(f"Recovering only fresh_full: start/logged step={start_step}, target={target}.", flush=True)
    try:
        seeds = experiment_seeds(raw["seed"])
        dtype = getattr(torch, raw["dtype"])
        teacher = Teacher(raw["d"], raw["P_teacher"], seed=seeds["teacher"],
                          init_scale=raw["teacher_scale"], dtype=dtype, device=device)
        model = Student(raw["d"], raw["P"], seed=seeds["initialization"],
                        init_scale=raw["init_scale"], dtype=dtype, device=device)
        initial_W = model.W.detach().clone()
        print("Regenerating original training and held-out data from recorded seeds.", flush=True)
        full = generate_dataset(teacher, raw["n_train"], raw["T"], seeds["train_data"])
        test = generate_dataset(teacher, raw["n_test"], raw["T"], seeds["test_data"])
        split_generator = torch.Generator(device=device).manual_seed(seeds["split"])
        permutation = torch.randperm(len(full), generator=split_generator, device=device)
        indices_A = permutation[:int(len(full) * raw["initial_fraction"])]
        trainer = FreshTrainer(model, full, settings["optimizer"], settings["lr"],
                               settings["batch_size"], seeds["final_order"])
        teacher_S = teacher.S

        def metrics() -> dict:
            partitions = evaluate_train_partitions(model, full, indices_A, settings["eval_batch_size"])
            values = {"train": partitions["full"], **partitions,
                      "test": evaluate(model, test, settings["eval_batch_size"])}
            row = {"seed": raw["seed"], "regime": "fresh", "phase": "fresh_full",
                   "phase_step": trainer.step, "regime_step": trainer.step,
                   "updates": trainer.step, "epoch": trainer.completed_epochs,
                   "examples_seen": trainer.examples_seen}
            for name, scores in values.items():
                row.update({f"{name}_{key}": value for key, value in scores.items()})
                row[f"{name}_loss"] = scores["mse"]
                row[f"{name}_error"] = scores["mse"]
            row.update(matrix_metrics(model, teacher_S, initial_W))
            if not all(math.isfinite(value) for value in row.values() if isinstance(value, float)):
                raise FloatingPointError("Non-finite recovery metrics.")
            return row

        def checkpoint() -> None:
            atomic_torch_save({"format_version": 1, "source_config": raw,
                               "torch_version": str(torch.__version__),
                               "trainer": trainer.state_dict()}, checkpoint_path)

        if saved is not None:
            trainer.load_state_dict(saved["trainer"])
            print(f"Loaded resumable checkpoint at fresh_full step {trainer.step}.", flush=True)
        else:
            write_json(destination / "status.json", {"status": "replaying", "to_step": start_step})
            print("Replaying updates without evaluations; warm phases are not rerun.", flush=True)
            while trainer.step < start_step:
                trainer.update()
                if trainer.step % 10000 == 0:
                    print(f"Replay: {trainer.step}/{start_step}", flush=True)
            row = metrics()
            tolerance = 1e-5 if dtype == torch.float32 else 1e-8
            for key in ("train_mse", "test_mse"):
                reference = float(replay_row[key])
                if not math.isclose(row[key], reference, rel_tol=tolerance, abs_tol=1e-10):
                    raise RuntimeError(f"Replay does not match original {key}: "
                                       f"replayed={row[key]}, logged={reference}. "
                                       "Check original code, PyTorch version and device type.")
            print("Replayed training/test MSE match the original logged values.", flush=True)
            checkpoint()

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        with (destination / f"metrics_{stamp}.csv").open("w", newline="") as handle:
            writer = None

            def log() -> None:
                nonlocal writer
                row = metrics()
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                print(f"[fresh_full] step={trainer.step} train_loss={row['train_loss']:.8g} "
                      f"train_error={row['train_error']:.8g} test_loss={row['test_loss']:.8g} "
                      f"test_error={row['test_error']:.8g}", flush=True)

            write_json(destination / "status.json", {"status": "training", "target_steps": target})
            log()
            while trainer.step < target:
                trainer.update()
                # Save before expensive evaluation, so a timeout during evaluation
                # still leaves the most recent scheduled checkpoint intact.
                if trainer.step % args.checkpoint_every == 0 or trainer.step == target:
                    checkpoint()
                if trainer.step % interval == 0 or trainer.step == target:
                    log()
        atomic_torch_save({"W": model.W.detach().cpu(), "step": trainer.step},
                           destination / "fresh_final_W.pt")
        write_json(destination / "status.json", {"status": "complete", "step": trainer.step})
    except BaseException as error:
        write_json(destination / "status.json", {"status": "failed",
                                                "exception": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
