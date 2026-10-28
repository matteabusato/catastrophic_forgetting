"""Run from the repository root: python -m warm_start --output PATH."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from .src.experiment import ExperimentConfig, run_experiment, summarize_pairs
from .src.io_utils import save_result, write_json
from .src.training import TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON overrides for ExperimentConfig")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Independent replicate seeds; overrides config seed")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    raw = json.loads(args.config.read_text()) if args.config else {}
    for key in ("pretraining", "final_training"):
        if key in raw:
            raw[key] = TrainingConfig(**raw[key])
    config = ExperimentConfig(**raw)
    seeds = args.seeds if args.seeds is not None else [config.seed]
    if len(set(seeds)) != len(seeds):
        parser.error("--seeds must be distinct")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for seed in seeds:
        result = run_experiment(replace(config, seed=seed))
        save_result(result, args.output / f"seed_{seed}")
        summaries.append(result.summary)
        print(json.dumps(result.summary, allow_nan=False), flush=True)
    aggregate = summarize_pairs(summaries)
    write_json(args.output / "aggregate.json", aggregate)
    print(json.dumps(aggregate, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
