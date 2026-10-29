#!/usr/bin/env python3
"""Continue warm_full or fresh_full, stopping on training MSE or an absolute cap."""

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

from warm_start.src.data import Dataset, generate_dataset
from warm_start.src.experiment import experiment_seeds
from warm_start.src.io_utils import write_json
from warm_start.src.model import Student, Teacher
from warm_start.src.observables import evaluate, matrix_metrics
from warm_start.src.recovery import FreshTrainer, atomic_torch_save, evaluate_train_partitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--regime", choices=("warm", "fresh"), required=True)
    parser.add_argument("--target-steps", type=int, default=4000000,
                        help="Absolute full-data update cap, excluding A warm-up")
    parser.add_argument("--loss-tolerance", type=float, default=1e-8)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--device", help="Defaults to original device")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.target_steps, args.log_every, args.checkpoint_every, args.threads) < 1:
        parser.error("Step budgets, intervals and threads must be positive.")
    if not math.isfinite(args.loss_tolerance) or args.loss_tolerance < 0:
        parser.error("loss-tolerance must be nonnegative and finite.")
    torch.set_num_threads(args.threads)
    root = args.run_dir.expanduser().resolve()
    raw = json.loads((root / "config.json").read_text())
    settings = raw["final_training"]
    if settings.get("max_steps") is None or settings.get("early_stopping", True):
        raise ValueError("Expected the original fixed-update-budget experiment.")
    destination = root / "continuation" / args.regime
    latest = destination / "latest.pt"
    weights_file = root / f"seed_{raw['seed']}" / "weights.pt"
    recovery_file = root / "fresh_recovery" / "latest.pt"
    saved = None
    weights = None
    # Prefer the newest complete state; continuation always takes precedence.
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved.get("regime") != args.regime:
            raise ValueError("Continuation checkpoint regime mismatch.")
    elif args.regime == "fresh" and recovery_file.exists():
        candidate = torch.load(recovery_file, map_location="cpu", weights_only=True)
        if not weights_file.exists() or candidate["trainer"]["step"] >= settings["max_steps"]:
            saved = candidate
    if saved is not None:
        if saved["source_config"] != raw:
            raise ValueError("Checkpoint source configuration mismatch.")
        starting_step = saved["trainer"]["step"]
    else:
        if not weights_file.exists():
            raise FileNotFoundError(
                f"No saved {args.regime} weights/checkpoint found. "
                "An incomplete legacy run cannot be continued from CSV losses alone.")
        if settings["optimizer"] != "sgd":
            raise ValueError("Legacy weight-only continuation requires plain SGD; Adam state was not saved.")
        weights = torch.load(weights_file, map_location="cpu", weights_only=True)
        starting_step = settings["max_steps"]
    if args.target_steps < starting_step:
        raise ValueError(f"Target must be at least the existing step {starting_step}.")
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    write_json(destination / f"settings_{stamp}.json", {
        "regime": args.regime, "starting_step": starting_step,
        "target_steps": args.target_steps, "loss_tolerance": args.loss_tolerance,
        "log_every": args.log_every, "checkpoint_every": args.checkpoint_every,
    })
    try:
        device = args.device or raw["device"]
        dtype = getattr(torch, raw["dtype"])
        seeds = experiment_seeds(raw["seed"])
        model = Student(raw["d"], raw["P"], seed=seeds["initialization"],
                        init_scale=raw["init_scale"], dtype=dtype, device=device)
        teacher = Teacher(raw["d"], raw["P_teacher"], seed=seeds["teacher"],
                          init_scale=raw["teacher_scale"], dtype=dtype, device=device)
        initial_W = model.W.detach().clone()
        if weights is not None:
            with torch.no_grad():
                initial_W.copy_(weights["initial_W"])
                teacher.W.copy_(weights["teacher_W"])
                model.W.copy_(weights[f"{args.regime}_final_W"])
        data_file = root / f"seed_{raw['seed']}" / "datasets.pt"
        if data_file.exists():
            tensors = torch.load(data_file, map_location="cpu", weights_only=True, mmap=True)
            def dataset(prefix):
                return Dataset(*(tensors[f"{prefix}_{key}"].to(device=device, dtype=dtype)
                                 for key in ("X", "y", "attention")))
            full, test = dataset("train"), dataset("test")
            indices_A = tensors["indices_A"].to(device)
        else:
            full = generate_dataset(teacher, raw["n_train"], raw["T"], seeds["train_data"])
            test = generate_dataset(teacher, raw["n_test"], raw["T"], seeds["test_data"])
            generator = torch.Generator(device=device).manual_seed(seeds["split"])
            order = torch.randperm(len(full), generator=generator, device=device)
            indices_A = order[:int(len(full) * raw["initial_fraction"])]
        trainer = FreshTrainer(model, full, settings["optimizer"], settings["lr"],
                               settings["batch_size"], seeds["final_order"])
        if saved is not None:
            trainer.load_state_dict(saved["trainer"])
        else:
            trainer.restore_sgd_sampler(starting_step)
        teacher_S = teacher.S

        def checkpoint() -> None:
            atomic_torch_save({"format_version": 1, "source_config": raw,
                               "regime": args.regime, "torch_version": str(torch.__version__),
                               "trainer": trainer.state_dict()}, latest)

        def metrics() -> dict:
            partitions = evaluate_train_partitions(model, full, indices_A, settings["eval_batch_size"])
            values = {"train": partitions["full"], **partitions,
                      "test": evaluate(model, test, settings["eval_batch_size"])}
            offset = raw["pretraining"]["max_steps"] if args.regime == "warm" else 0
            row = {"seed": raw["seed"], "regime": args.regime, "phase": f"{args.regime}_full",
                   "phase_step": trainer.step, "regime_step": offset + trainer.step,
                   "updates": trainer.step, "epoch": trainer.completed_epochs,
                   "examples_seen": trainer.examples_seen}
            for name, scores in values.items():
                row.update({f"{name}_{key}": value for key, value in scores.items()})
                row[f"{name}_loss"] = row[f"{name}_error"] = scores["mse"]
            row.update(matrix_metrics(model, teacher_S, initial_W))
            if not all(math.isfinite(value) for value in row.values() if isinstance(value, float)):
                raise FloatingPointError("Non-finite continuation metrics.")
            return row

        first = metrics()
        # Legacy weight snapshots have no optimizer/RNG state: validate their
        # dataset and model evaluation against the original full-data endpoint.
        if saved is None:
            with (root / "metrics.csv").open() as handle:
                matches = [row for row in csv.DictReader(handle)
                           if row.get("phase") == f"{args.regime}_full"
                           and int(row["updates"]) == starting_step]
            if not matches:
                raise ValueError("Cannot validate saved weights: original endpoint CSV row missing.")
            for key in ("train_mse", "test_mse"):
                if not math.isclose(first[key], float(matches[-1][key]),
                                    rel_tol=1e-5 if dtype == torch.float32 else 1e-8, abs_tol=1e-10):
                    raise ValueError(f"Saved weights/data do not match original endpoint {key}.")
        checkpoint()
        write_json(destination / "status.json", {"status": "training", "step": trainer.step})
        with (destination / f"metrics_{stamp}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(first))
            writer.writeheader()

            def log(row):
                writer.writerow(row)
                handle.flush()
                print(f"[{args.regime}_full] step={trainer.step} "
                      f"train_loss={row['train_loss']:.10g} test_loss={row['test_loss']:.10g}", flush=True)

            row = first
            log(row)
            reason = "loss_tolerance" if row["train_loss"] <= args.loss_tolerance else "step_cap"
            while trainer.step < args.target_steps and reason != "loss_tolerance":
                trainer.update()
                if trainer.step % args.checkpoint_every == 0:
                    checkpoint()
                if trainer.step % args.log_every == 0 or trainer.step == args.target_steps:
                    row = metrics()
                    log(row)
                    if row["train_loss"] <= args.loss_tolerance:
                        reason = "loss_tolerance"
            checkpoint()
        write_json(destination / "summary.json", {
            "regime": args.regime, "step": trainer.step, "stop_reason": reason,
            "reached_loss_tolerance": reason == "loss_tolerance",
            "train_loss": row["train_loss"], "test_loss": row["test_loss"],
            "loss_tolerance": args.loss_tolerance,
        })
        write_json(destination / "status.json", {"status": "complete", "stop_reason": reason,
                                                "step": trainer.step})
        print(f"Stopped at {trainer.step}: {reason}.", flush=True)
    except BaseException as error:
        write_json(destination / "status.json", {"status": "failed", "exception": type(error).__name__,
                                                "message": str(error)})
        raise


if __name__ == "__main__":
    main()
