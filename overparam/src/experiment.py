"""Paired width sweep, sequential Task-1/Task-2 orchestration and CLI entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import statistics
import torch
from .config import ExperimentConfig
from .data import GaussianStream, label_inputs, make_teachers, role_seed
from .io_utils import MetricWriter, save_checkpoint, write_json
from .model import Student
from .observables import evaluate, forgetting, matrix_metrics
from .training import train_phase


def run_experiment(config: ExperimentConfig, output: str | Path) -> list[dict]:
    """Teachers and diagnostics are sampled once per seed, outside the width loop."""
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", asdict(config))
    write_json(root / "metadata.json", dict(
        torch_version=str(torch.__version__), correlation=0,
        loss="mean squared output error over examples, tokens and features",
        training="fresh iid Gaussian minibatches, no replay, no early stopping",
        initialization="independent Gaussian factors; initial S is not matched across widths",
        pairing="same teachers, test/probe inputs and phase input streams across widths",
        optimizer="plain SGD, no momentum or weight decay; reset at task switch",
        teacher_rank_fixed=config.teacher_rank,
        warning="Fixed update budgets do not imply convergence."))
    write_json(root / "status.json", dict(status="running"))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    summaries = []
    try:
        dtype = getattr(torch, config.dtype)
        for seed in config.seeds:
            seed_dir = root / f"seed_{seed}"
            seed_dir.mkdir()
            teachers = make_teachers(config.d, config.teacher_rank, seed,
                                     config.teacher_std, dtype, config.device)
            seeds = {role: role_seed(seed, role) for role in
                     ("task1", "task2", "test", "probe", "train_task1", "train_task2")}
            write_json(seed_dir / "seeds.json", seeds)
            datasets = {}
            for split, n in (("test", config.n_test), ("probe", config.n_probe)):
                X = GaussianStream(seeds[split], config.tokens, config.d,
                                   dtype, config.device).sample(n)
                for name, teacher in teachers.items():
                    datasets[f"{split}_{name}"] = label_inputs(X, teacher, config.eval_batch_size)
            torch.save({name: teacher.W.detach().cpu() for name, teacher in teachers.items()},
                       seed_dir / "teachers.pt")
            # Targets are deterministic functions of teachers and these saved inputs.
            torch.save({split: datasets[f"{split}_task1"].X.cpu() for split in ("test", "probe")},
                       seed_dir / "evaluation_inputs.pt")
            for kappa in config.kappas:
                P = round(kappa * config.d)
                run_dir = seed_dir / f"P_{P}"
                run_dir.mkdir()
                init_seed = role_seed(seed, f"student_P_{P}")
                model = Student(config.d, P, init_seed, config.init_std, dtype, config.device)
                initial_W = model.W.detach().clone()
                lr = config.learning_rate(P)
                write_json(run_dir / "config.json", dict(seed=seed, init_seed=init_seed,
                           requested_kappa=kappa, actual_kappa=P / config.d, P=P,
                           max_rank=min(P, config.d), parameters=P * config.d, lr=lr))
                torch.save(initial_W.cpu(), run_dir / "initial_W.pt")
                initial_error = evaluate(model, datasets["test_task1"], config.eval_batch_size)["mse"]
                task1_error = None
                task1_S = None
                histories = {}
                for phase, steps in (("task1", config.task1_steps), ("task2", config.task2_steps)):
                    if phase == "task2" and steps == 0:
                        continue
                    stream = GaussianStream(seeds[f"train_{phase}"], config.tokens,
                                            config.d, dtype, config.device)

                    def record(step, minibatch_mse):
                        row = dict(seed=seed, P=P, kappa=P / config.d, lr=lr, phase=phase,
                                   phase_step=step,
                                   total_step=step + (config.task1_steps if phase == "task2" else 0),
                                   phase_examples=step * config.batch_size,
                                   last_minibatch_mse=minibatch_mse)
                        for name, data in datasets.items():
                            row.update({f"{name}_{key}": value for key, value in
                                        evaluate(model, data, config.eval_batch_size).items()})
                        row.update(matrix_metrics(model, teachers, initial_W, task1_S))
                        row.update(forgetting(row["test_task1_mse"], task1_error, initial_error)
                                   if phase == "task2" else
                                   dict(forgetting_abs=None, acquired_task1_improvement=None,
                                        forgetting_relative=None))
                        return row

                    writer = MetricWriter(run_dir / f"{phase}_metrics.csv")
                    try:
                        histories[phase] = train_phase(model, teachers[phase], stream, steps,
                                                      config.batch_size, lr, config.eval_every,
                                                      record, writer)
                    finally:
                        writer.close()
                    save_checkpoint(run_dir / f"{phase}_checkpoint.pt", model, stream, steps, phase)
                    if phase == "task1":
                        task1_error = histories[phase][-1]["test_task1_mse"]
                        task1_S = model.S.detach().clone()
                endpoint = histories.get("task2", histories["task1"])[-1]
                summary = dict(seed=seed, P=P, kappa=P / config.d,
                               initial_task1_mse=initial_error, after_task1_mse=task1_error,
                               after_task1_task2_mse=histories["task1"][-1]["test_task2_mse"],
                               final_task1_mse=endpoint["test_task1_mse"],
                               final_task2_mse=endpoint["test_task2_mse"],
                               task2_trained=config.task2_steps > 0,
                               **forgetting(endpoint["test_task1_mse"], task1_error, initial_error))
                write_json(run_dir / "summary.json", summary)
                summaries.append(summary)
                print(f"seed={seed} P={P} kappa={P/config.d:g} "
                      f"Task1 endpoint={task1_error:.6g} final Task2={endpoint['test_task2_mse']:.6g}",
                      flush=True)
        write_json(root / "summaries.json", summaries)
        aggregate = []
        for kappa in config.kappas:
            rows = [r for r in summaries if r["P"] == round(kappa * config.d)]
            metrics = {}
            for key in ("after_task1_mse", "final_task1_mse", "final_task2_mse", "forgetting_abs"):
                values = [r[key] for r in rows]
                metrics[key] = dict(mean=statistics.mean(values),
                                    sd=statistics.stdev(values) if len(values) > 1 else None)
            aggregate.append(dict(P=rows[0]["P"], kappa=rows[0]["kappa"], n=len(rows), metrics=metrics))
        write_json(root / "aggregate.json", aggregate)
        write_json(root / "status.json", dict(status="complete"))
        return summaries
    except Exception as exc:
        write_json(root / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        torch.set_num_threads(old_threads)


def main():
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--kappas", nargs="+", type=float, default=defaults.kappas)
    parser.add_argument("--seeds", nargs="+", type=int, default=defaults.seeds)
    parser.add_argument("--task1-only", action="store_true")
    for name in ("d", "tokens", "teacher_rank", "task1_steps", "task2_steps", "batch_size",
                 "n_test", "n_probe", "eval_every", "eval_batch_size", "threads"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("lr", "init_std", "teacher_std"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--lr-scaling", choices=("fixed", "sqrt_width"), default=defaults.lr_scaling)
    parser.add_argument("--dtype", choices=("float32", "float64"), default=defaults.dtype)
    parser.add_argument("--device", default=defaults.device)
    args = vars(parser.parse_args())
    output = args.pop("output") or Path("overparam/outputs") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    task1_only = args.pop("task1_only")
    config = ExperimentConfig(**args)
    if task1_only:
        config = replace(config, task2_steps=0)
    run_experiment(config, output)
    print(f"Saved to {output.resolve()}")


if __name__ == "__main__":
    main()
