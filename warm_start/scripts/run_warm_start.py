#!/usr/bin/env python3
"""Single-seed paired warm-start experiment with live step-based CSV logging.

Can be invoked by absolute path from any working directory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Resolve the package independently of the shell's working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from warm_start.src.experiment import ExperimentConfig, run_experiment
from warm_start.src.io_utils import save_result, write_json
from warm_start.src.training import TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        help="New run directory; defaults to a timestamped warm_start/outputs path")
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--rank", type=int, default=32, help="Student P")
    parser.add_argument("--teacher-rank", type=int, default=32, help="Teacher P_star")
    parser.add_argument("--n-train", type=int, default=128, help="Total number of sequences in A union B")
    parser.add_argument("--initial-fraction", type=float, default=0.5, help="Fraction assigned to A")
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--teacher-scale", type=float, default=1.0)
    parser.add_argument("--pretrain-steps", type=int, default=2000, help="Optimizer updates on A")
    parser.add_argument("--train-steps", type=int, default=5000,
                        help="Optimizer updates on A union B, for each final regime")
    parser.add_argument("--log-every", type=int, default=100, help="Log every this many optimizer updates")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-lr", type=float, help="Defaults to --lr")
    parser.add_argument("--fit-tolerance", type=float, default=1e-8,
                        help="Reported fit threshold; does not stop these fixed-budget runs")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    shared = dict(batch_size=args.batch_size, optimizer=args.optimizer,
                  log_every_steps=args.log_every, eval_batch_size=args.eval_batch_size,
                  early_stopping=False, loss_tolerance=args.fit_tolerance)
    config = ExperimentConfig(
        d=args.d, T=args.tokens, P=args.rank, P_teacher=args.teacher_rank,
        n_train=args.n_train, n_test=args.n_test, initial_fraction=args.initial_fraction,
        init_scale=args.init_scale, teacher_scale=args.teacher_scale,
        seed=args.seed, dtype=args.dtype, device=args.device,
        pretraining=TrainingConfig(max_steps=args.pretrain_steps,
                                  lr=args.pretrain_lr if args.pretrain_lr is not None else args.lr,
                                  **shared),
        final_training=TrainingConfig(max_steps=args.train_steps, lr=args.lr, **shared),
    )
    # Fail before creating output if the requested split would be empty.
    if not 0 < int(config.n_train * config.initial_fraction) < config.n_train:
        raise ValueError("initial_fraction must produce nonempty A and B.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output or REPO_ROOT / "warm_start" / "outputs" /
              f"seed_{args.seed}_{stamp}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", asdict(config))
    write_json(output / "status.json", {"status": "running", "seed": args.seed})
    print(f"Output directory: {output}", flush=True)
    print("Loss and prediction error both denote output MSE averaged over N*T*d.", flush=True)
    print(f"Warm: A for {args.pretrain_steps} updates, then A union B for {args.train_steps}; "
          f"fresh: A union B for {args.train_steps} from the shared W0.", flush=True)

    try:
        with (output / "metrics.csv").open("w", newline="") as handle:
            writer = None

            def log_metric(metric: dict) -> None:
                nonlocal writer
                warm = metric["phase"] != "fresh_full"
                offset = args.pretrain_steps if metric["phase"] == "warm_full" else 0
                row = {"seed": args.seed, "regime": "warm" if warm else "fresh",
                       "phase_step": metric["updates"],
                       "regime_step": offset + metric["updates"], **metric}
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                print(f"[{row['phase']}] step={row['phase_step']} "
                      f"train_loss={row['train_loss']:.8g} train_error={row['train_error']:.8g} "
                      f"test_loss={row['test_loss']:.8g} test_error={row['test_error']:.8g}",
                      flush=True)

            result = run_experiment(config, metric_callback=log_metric)
        save_result(result, output / f"seed_{args.seed}")
        write_json(output / "summary.json", result.summary)
        write_json(output / "status.json", {"status": "complete", "seed": args.seed})
        print(json.dumps(result.summary, indent=2, allow_nan=False), flush=True)
    except BaseException as error:
        # Retain flushed rows and an explicit status for failed/interrupted jobs.
        write_json(output / "status.json", {"status": "failed", "seed": args.seed,
                                            "exception": type(error).__name__,
                                            "message": str(error)})
        raise


if __name__ == "__main__":
    main()
