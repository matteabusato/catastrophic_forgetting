"""Independent, restartable fixed-budget Task-1 and Task-2 jobs for one width."""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import time
import warnings

import torch

from .data import GaussianStream, label_inputs, role_seed
from .io_utils import write_json
from .losses import output_mse
from .model import Student, Teacher
from .observables import evaluate, forgetting, matrix_metrics


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _save(path, value):
    """Publish a complete checkpoint atomically; never expose partial torch.save files."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


@contextmanager
def _lock(directory):
    # Kernel releases flock even after SIGKILL; the persistent lock file is harmless.
    with (directory / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is using {directory}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def checkpoint_steps(d, target, alphas, every_alpha):
    """Initial milestones, then every every_alpha beyond the last milestone; always final."""
    if d < 1 or target < 1 or not alphas or min(alphas) <= 0 or every_alpha <= 0:
        raise ValueError("Checkpoint settings must be positive and milestones nonempty.")
    if not all(math.isfinite(x) for x in (*alphas, every_alpha)):
        raise ValueError("Checkpoint alphas must be finite.")
    steps = {round(alpha * d * d) for alpha in alphas}
    if min(steps) < 1 or round(every_alpha * d * d) < 1:
        raise ValueError("Checkpoint intervals must correspond to at least one update.")
    last = max(alphas)
    index = 1
    while (last + index * every_alpha) * d * d <= target:
        steps.add(round((last + index * every_alpha) * d * d))
        index += 1
    return {s for s in steps if s <= target} | {target}


def _latest(directory, step=None):
    if step is not None:
        path = directory / "checkpoints" / f"step_{step:012d}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted((directory / "checkpoints").glob("step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints in {directory}")
    return candidates[-1]


def _common(parser):
    parser.add_argument("--run-dir", type=Path, help="New output folder, or existing folder with --resume.")
    parser.add_argument("--resume", action="store_true", help="Restore latest checkpoint in --run-dir.")
    parser.add_argument("--target-alpha", type=float, help="Total phase horizon in units of d^2; default 40.")
    parser.add_argument("--target-steps", type=int, help="Total phase updates; overrides --target-alpha.")
    parser.add_argument("--lr", type=float, help="Constant SGD rate; fresh default .01; inherited on resume.")
    parser.add_argument("--batch-size", type=int, help="Fresh default 1; inherited on resume/Task 2.")
    parser.add_argument("--eval-every", type=int, help="Metric interval in updates; fresh default d.")
    parser.add_argument("--eval-batch-size", type=int, help="Diagnostic batch size; fresh default 256.")
    parser.add_argument("--checkpoint-alphas", nargs="+", type=float,
                        help="Phase-local milestones; fresh default 5 10 20 30 40.")
    parser.add_argument("--checkpoint-every-alpha", type=float,
                        help="Interval after last milestone; fresh default 10.")
    parser.add_argument("--device", help="Fresh default cpu; may change when resuming.")
    parser.add_argument("--threads", type=int, help="Torch CPU threads; fresh default 1.")
    parser.add_argument("--print-every", type=int, help="Console interval in updates; fresh default d^2.")
    parser.add_argument("--train-seed", type=int, help="Override paired phase input RNG seed on a fresh phase.")


def task1_parser():
    parser = argparse.ArgumentParser(description="Train one full-W student on Task 1 only (MSE).")
    _common(parser)
    parser.add_argument("--results-root", type=Path, default=Path("overparam/results/task1"))
    parser.add_argument("--d", type=int)
    parser.add_argument("--tokens", "--T", dest="tokens", type=int)
    parser.add_argument("--kappa", type=float)
    parser.add_argument("--teacher-rank", type=int, help="Fixed teacher factor width; fresh default d.")
    parser.add_argument("--seed", type=int, help="Replicate seed; fresh default 0.")
    parser.add_argument("--init-std", type=float)
    parser.add_argument("--teacher-std", type=float)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--n-probe", type=int)
    parser.add_argument("--dtype", choices=("float32", "float64"))
    for name in ("teacher1", "teacher2", "student", "test", "probe", "task2-train"):
        parser.add_argument(f"--{name}-seed", type=int, help="Optional role-specific RNG override.")
    return parser


def task2_parser():
    parser = argparse.ArgumentParser(description="Load a chosen Task-1 checkpoint and train Task 2 only (MSE).")
    _common(parser)
    parser.add_argument("--task1-run-dir", type=Path, help="Required for a fresh Task-2 run.")
    parser.add_argument("--task1-step", type=int, help="Source update count; default latest saved checkpoint.")
    parser.add_argument("--results-root", type=Path, default=Path("overparam/results/task2"))
    return parser


def _horizon(args, d, saved=None):
    if args.target_steps is not None:
        return args.target_steps
    if args.target_alpha is not None:
        if not math.isfinite(args.target_alpha) or args.target_alpha <= 0:
            raise ValueError("target-alpha must be positive and finite.")
        return round(args.target_alpha * d * d)
    return saved if saved is not None else 40 * d * d


def _validate(config):
    for name in ("d", "tokens", "P", "teacher_rank", "target_steps", "batch_size", "eval_every",
                 "eval_batch_size", "n_test", "n_probe", "threads", "print_every"):
        if config[name] < 1:
            raise ValueError(f"{name} must be positive.")
    if config["tokens"] < 2:
        raise ValueError("tokens must be >=2.")
    for name in ("lr", "init_std", "teacher_std", "kappa"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    if config["teacher1_seed"] == config["teacher2_seed"]:
        raise ValueError("Independent teachers require distinct seeds.")
    if len({config[name] for name in ("train_seed", "task2_train_seed", "test_seed", "probe_seed")}) < 4:
        # In Task 2 train_seed intentionally becomes task2_train_seed.
        keys = ("task1_train_seed", "task2_train_seed", "test_seed", "probe_seed")
        if len({config.get(name, config["train_seed"]) for name in keys}) < 4:
            raise ValueError("Training and diagnostic RNG streams must have distinct seeds.")
    checkpoint_steps(config["d"], config["target_steps"], config["checkpoint_alphas"],
                     config["checkpoint_every_alpha"])


def _fresh_config(args):
    def value(name, default):
        result = getattr(args, name, None)
        return default if result is None else result
    d, seed, kappa = value("d", 100), value("seed", 0), value("kappa", 1.)
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be positive and finite.")
    config = dict(phase="task1", d=d, seed=seed, kappa=kappa, P=round(kappa*d),
                  tokens=value("tokens", 5), teacher_rank=value("teacher_rank", d),
                  init_std=value("init_std", 1.), teacher_std=value("teacher_std", 1.),
                  lr=value("lr", .01), batch_size=value("batch_size", 1),
                  n_test=value("n_test", 2048), n_probe=value("n_probe", 256),
                  eval_every=value("eval_every", d), eval_batch_size=value("eval_batch_size", 256),
                  dtype=value("dtype", "float64"), device=value("device", "cpu"),
                  threads=value("threads", 1), print_every=value("print_every", d*d),
                  checkpoint_alphas=value("checkpoint_alphas", [5., 10., 20., 30., 40.]),
                  checkpoint_every_alpha=value("checkpoint_every_alpha", 10.),
                  target_steps=_horizon(args, d), loss="output_mse", optimizer="sgd",
                  momentum=0., weight_decay=0., teacher_correlation=0.)
    for arg, role in (("teacher1_seed", "task1"), ("teacher2_seed", "task2"),
                      ("student_seed", f"student_P_{config['P']}"), ("train_seed", "train_task1"),
                      ("task2_train_seed", "train_task2"), ("test_seed", "test"), ("probe_seed", "probe")):
        config[arg] = value(arg, role_seed(seed, role))
    config["task1_train_seed"] = config["train_seed"]
    _validate(config)
    return config


def _resume_config(args, directory):
    config = json.loads((directory / "config.json").read_text())
    # Scientific settings are inherited; changing them would invalidate exact continuation.
    mutable = {"device", "threads", "eval_batch_size", "print_every"}
    ignored = {"resume", "run_dir", "results_root", "target_steps", "target_alpha", *mutable}
    for key, value in vars(args).items():
        if key in ignored or value is None:
            continue
        if key not in config or value != config[key]:
            raise ValueError(f"Cannot change {key} on resume; settings are inherited from config.json.")
    for key in mutable:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    config["target_steps"] = _horizon(args, config["d"], config["target_steps"])
    _validate(config)
    return config


def _prepare_task1(directory, config):
    dtype, device = getattr(torch, config["dtype"]), config["device"]
    teachers = {name: Teacher(config["d"], config["teacher_rank"], config[f"teacher{i}_seed"],
                             config["teacher_std"], dtype, device)
                for i, name in enumerate(("task1", "task2"), 1)}
    _save(directory / "teachers.pt", {name: t.W.detach().cpu() for name, t in teachers.items()})
    inputs = {split: GaussianStream(config[f"{split}_seed"], config["tokens"], config["d"],
                                   dtype, "cpu").sample(config[f"n_{split}"])
              for split in ("test", "probe")}
    _save(directory / "evaluation_inputs.pt", inputs)
    model = Student(config["d"], config["P"], config["student_seed"], config["init_std"], dtype, device)
    _save(directory / "initial_W.pt", model.W.detach().cpu())


def _prepare_task2(directory, config, source_checkpoint):
    source = Path(config["task1_run_dir"])
    for name in ("teachers.pt", "evaluation_inputs.pt", "initial_W.pt"):
        _save(directory / name, _load(source / name))
    # Copy the switch checkpoint so a running Task-2 job is independent of source files.
    _save(directory / "task1_source.pt", source_checkpoint)


def _restore_objects(directory, config, checkpoint):
    dtype, device = getattr(torch, config["dtype"]), config["device"]
    teachers = {}
    for name, W in _load(directory / "teachers.pt").items():
        teacher = Teacher(config["d"], W.shape[1], 0, dtype=dtype, device=device)
        with torch.no_grad():
            teacher.W.copy_(W.to(device=device, dtype=dtype))
        teachers[name] = teacher
    inputs = _load(directory / "evaluation_inputs.pt")
    datasets = {f"{split}_{name}": label_inputs(X.to(device=device, dtype=dtype), teacher,
                                               config["eval_batch_size"])
                for split, X in inputs.items() for name, teacher in teachers.items()}
    initial = _load(directory / "initial_W.pt").to(device=device, dtype=dtype)
    model = Student(config["d"], config["P"], 0, dtype=dtype, device=device)
    with torch.no_grad():
        model.W.copy_(initial if checkpoint is None else checkpoint["W"].to(device=device, dtype=dtype))
    return model, teachers, datasets, initial


def _recover_csv(path, checkpoint):
    """Roll back uncheckpointed rows after interruption, retaining an audit backup."""
    if not path.exists():
        if checkpoint is not None:
            raise RuntimeError("Missing metrics.csv for saved checkpoint.")
        return
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    if checkpoint is None:
        keep = []
    else:
        keep = [row for row in rows if row.get("phase_step", "").isdigit()
                and int(row["phase_step"]) <= checkpoint["step"]]
        if not keep or int(keep[-1]["phase_step"]) != checkpoint["step"]:
            raise RuntimeError("Metrics do not contain the latest checkpoint's endpoint.")
    if len(keep) == len(rows):
        return
    backup = path.with_name(f"metrics_before_recovery_{time.time_ns()}.csv")
    backup.write_bytes(path.read_bytes())
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(keep)
    temporary.replace(path)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    elif torch.device(device).type == "mps":
        torch.mps.synchronize()


def _train(directory, config, resume):
    paths = list((directory / "checkpoints").glob("step_*.pt"))
    checkpoint = _load(_latest(directory)) if resume and paths else None
    if checkpoint is not None and checkpoint["step"] > config["target_steps"]:
        raise ValueError("target_steps precedes latest checkpoint.")
    _recover_csv(directory / "metrics.csv", checkpoint)
    model, teachers, datasets, initial = _restore_objects(directory, config, checkpoint)
    dtype = getattr(torch, config["dtype"])
    stream = GaussianStream(config["train_seed"], config["tokens"], config["d"], dtype, config["device"])
    optimizer = torch.optim.SGD([model.W], lr=config["lr"])
    start = checkpoint["step"] if checkpoint else 0
    train_seconds = checkpoint["train_seconds"] if checkpoint else 0.
    if checkpoint:
        stream.generator.set_state(checkpoint["generator_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    task1_source = _load(directory / "task1_source.pt") if config["phase"] == "task2" else None
    if task1_source:
        source_W = task1_source["W"].to(device=config["device"], dtype=dtype)
        task1_S = source_W @ source_W.T / math.sqrt(config["P"])
        initial_error = task1_source["initial_task1_mse"]
        switch_error = task1_source["metrics"]["test_task1_mse"]
        task1_offset = task1_source["step"]
        if checkpoint is None:
            with torch.no_grad():
                model.W.copy_(source_W)
    else:
        task1_S, switch_error, task1_offset = None, None, 0
        initial_error = checkpoint["initial_task1_mse"] if checkpoint else evaluate(
            model, datasets["test_task1"], config["eval_batch_size"])["mse"]
    milestone_steps = checkpoint_steps(config["d"], config["target_steps"],
                                      config["checkpoint_alphas"], config["checkpoint_every_alpha"])
    csv_path = directory / "metrics.csv"
    has_header = csv_path.exists() and csv_path.stat().st_size > 0
    handle = csv_path.open("a", newline="")
    writer = None

    @torch.no_grad()
    def record(step, minibatch_mse):
        nonlocal writer
        S = model.S
        row = dict(phase=config["phase"], phase_step=step, total_step=step + task1_offset,
                   alpha=step / config["d"]**2, seed=config["seed"], P=config["P"],
                   kappa=config["P"] / config["d"], lr=config["lr"],
                   examples_processed=step * config["batch_size"], train_seconds=train_seconds,
                   last_minibatch_mse=minibatch_mse)
        for name, data in datasets.items():
            row.update({f"{name}_{key}": value for key, value in
                        evaluate(model, data, config["eval_batch_size"], S=S).items()})
        row.update(matrix_metrics(model, teachers, initial, task1_S, S=S))
        row.update(forgetting(row["test_task1_mse"], switch_error, initial_error)
                   if task1_source else dict(forgetting_abs=None, acquired_task1_improvement=None,
                                             forgetting_relative=None))
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise FloatingPointError(f"Non-finite {key} at step {step}.")
        if writer is None:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not has_header:
                writer.writeheader()
        writer.writerow(row)
        handle.flush()
        if step % config["print_every"] == 0 or step == config["target_steps"]:
            print(f"{config['phase']} P={config['P']} step={step} alpha={row['alpha']:g} "
                  f"mse1={row['test_task1_mse']:.7g} mse2={row['test_task2_mse']:.7g}", flush=True)
        return row

    def save(step, metrics):
        _save(directory / "checkpoints" / f"step_{step:012d}.pt", dict(
            W=model.W.detach().cpu(), optimizer_state=optimizer.state_dict(),
            generator_state=stream.generator.get_state(), step=step, phase=config["phase"],
            train_seconds=train_seconds, initial_task1_mse=initial_error, metrics=metrics,
            config=config, torch_version=str(torch.__version__)))

    model.train()
    try:
        if checkpoint is None:
            metrics = record(0, None)
            save(0, metrics)
        else:
            metrics = checkpoint["metrics"]
        for step in range(start + 1, config["target_steps"] + 1):
            if step == start + 1:
                _sync(config["device"])
                started = time.perf_counter()
            X = stream.sample(config["batch_size"])
            with torch.no_grad():
                target = teachers[config["phase"]](X)
            optimizer.zero_grad(set_to_none=True)
            loss = output_mse(model(X), target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at update {step}.")
            loss.backward()
            if model.W.grad is None or not torch.isfinite(model.W.grad).all():
                raise FloatingPointError(f"Non-finite gradient at update {step}.")
            optimizer.step()
            log = step % config["eval_every"] == 0 or step in milestone_steps
            if log:
                _sync(config["device"])
                train_seconds += time.perf_counter() - started
                metrics = record(step, loss.item())
                if step in milestone_steps:
                    save(step, metrics)
                _sync(config["device"])
                started = time.perf_counter()
        _json(directory / "summary.json", dict(phase=config["phase"], steps=config["target_steps"],
                                               metrics=metrics, convergence_claim=False))
    finally:
        handle.close()


def run_job(config, directory, resume=False):
    """Run a preconfigured single phase. Fresh output folders must not exist."""
    _validate(config)
    directory = Path(directory).resolve()
    if not resume:
        directory.mkdir(parents=True, exist_ok=False)
    elif not (directory / "config.json").is_file():
        raise FileNotFoundError(f"No run configuration in {directory}")
    with _lock(directory):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(config["threads"])
        try:
            if not resume:
                (directory / "checkpoints").mkdir()
                if config["phase"] == "task1":
                    _prepare_task1(directory, config)
                else:
                    source = _load(Path(config["task1_checkpoint"]))
                    _prepare_task2(directory, config, source)
            _json(directory / "config.json", config)
            with (directory / "run_history.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(resume=resume, target_steps=config["target_steps"],
                                             device=config["device"], started_unix=time.time())) + "\n")
            _json(directory / "status.json", dict(status="running"))
            _train(directory, config, resume)
            _json(directory / "status.json", dict(status="complete", steps=config["target_steps"]))
        except Exception as exc:
            _json(directory / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
            raise
        finally:
            torch.set_num_threads(previous_threads)
    print(f"Saved to {directory}", flush=True)
    return directory


def task1_main(argv=None):
    args = task1_parser().parse_args(argv)
    if args.resume:
        if args.run_dir is None:
            raise ValueError("--resume requires --run-dir.")
        config = _resume_config(args, args.run_dir)
        if config["phase"] != "task1":
            raise ValueError("Task-1 script requires a Task-1 run.")
        directory = args.run_dir
    else:
        config = _fresh_config(args)
        directory = args.run_dir or args.results_root / f"d{config['d']}_seed{config['seed']}_P{config['P']}"
    if config["target_steps"] < 40 * config["d"]**2:
        warnings.warn("Explicit horizon is below the planned 40d^2 Task-1 budget (use for smoke tests only).")
    return run_job(config, directory, args.resume)


def task2_main(argv=None):
    args = task2_parser().parse_args(argv)
    if args.resume:
        if args.run_dir is None:
            raise ValueError("--resume requires --run-dir.")
        config = _resume_config(args, args.run_dir)
        if config["phase"] != "task2":
            raise ValueError("Task-2 script requires a Task-2 run.")
        directory = args.run_dir
    else:
        if args.task1_run_dir is None:
            raise ValueError("A fresh Task-2 job requires --task1-run-dir.")
        source_dir = args.task1_run_dir.resolve()
        source_path = _latest(source_dir, args.task1_step)
        source = _load(source_path)
        if source["phase"] != "task1":
            raise ValueError("Source checkpoint must belong to Task 1.")
        config = dict(source["config"])
        config.update(phase="task2", task1_run_dir=str(source_dir),
                      task1_checkpoint=str(source_path), task1_step=source["step"],
                      task1_train_seed=config["train_seed"], train_seed=config["task2_train_seed"],
                      target_steps=_horizon(args, config["d"]))
        for name in ("lr", "batch_size", "eval_every", "eval_batch_size", "checkpoint_alphas",
                     "checkpoint_every_alpha", "device", "threads", "print_every", "train_seed"):
            value = getattr(args, name)
            if value is not None:
                config[name] = value
        # Keep the role seed consistent if the Task-2 stream is explicitly overridden.
        config["task2_train_seed"] = config["train_seed"]
        directory = args.run_dir or args.results_root / (
            f"d{config['d']}_seed{config['seed']}_P{config['P']}_from_{source['step']}")
    return run_job(config, directory, args.resume)
