import argparse
import csv
import json
from pathlib import Path

import torch

from src.data_model import (
    generate_dataset,
    teacher_forward,
)
from src.model import Student
from src.observables import (
    generalization_error,
    representation_error,
    vector_overlap,
)
from src.training import mse_loss
from src.io_utils import (
    load_config,
    load_pretraining_checkpoint,
    load_teachers,
)


# ============================================================
# Utilities
# ============================================================

def parse_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64

    if name == "float32":
        return torch.float32

    raise ValueError(
        f"Unsupported dtype: {name}"
    )


def default_checkpoint_steps(
    d: int,
    target_steps: int,
) -> list[int]:

    candidates = {
        max(1, d // 2),
        d,
        2 * d,
        5 * d,
        target_steps,
    }

    return sorted(
        step
        for step in candidates
        if 0 < step <= target_steps
    )


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def evaluate(
    model: Student,
    teachers,
    pretraining_test,
    task1_test,
    pretraining_step: int,
    finetune_step: int,
) -> dict:

    rep_error = representation_error(
        model.S_large,
        teachers.S_star,
    ).item()

    pretraining_error = generalization_error(
        model,
        pretraining_test.X,
        pretraining_test.y,
    ).item()

    task1_error = generalization_error(
        model,
        task1_test.X,
        task1_test.y,
    ).item()

    overlap_w1 = vector_overlap(
        model.w,
        teachers.w1_star,
    ).item()

    w_norm = torch.linalg.vector_norm(
        model.w
    ).item()

    d = model.d

    return {
        "pretraining_step": pretraining_step,
        "finetune_step": finetune_step,
        "finetune_step_over_d": (
            finetune_step / d
        ),
        "representation_error": rep_error,
        "pretraining_error": pretraining_error,
        "task1_error": task1_error,
        "overlap_w1": overlap_w1,
        "w_norm": w_norm,
    }


METRIC_FIELDS = [
    "pretraining_step",
    "finetune_step",
    "finetune_step_over_d",
    "representation_error",
    "pretraining_error",
    "task1_error",
    "overlap_w1",
    "w_norm",
]


def append_metric(
    metrics_path: Path,
    metric: dict,
) -> None:

    write_header = not metrics_path.exists()

    with open(
        metrics_path,
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=METRIC_FIELDS,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(metric)


# ============================================================
# Checkpoint saving
# ============================================================

def save_finetuning_checkpoint(
    output_root: Path,
    pretraining_step: int,
    finetune_step: int,
    model: Student,
    optimizer,
    generator: torch.Generator,
    metric: dict,
    config: dict,
    overwrite: bool = False,
) -> Path:

    checkpoint_name = (
        f"step_{pretraining_step}"
        f"_finetune{finetune_step}"
    )

    checkpoint_dir = (
        output_root
        / checkpoint_name
    )

    if checkpoint_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Checkpoint already exists: "
            f"{checkpoint_dir}"
        )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    torch.save(
        {
            "W": (
                model.W
                .detach()
                .cpu()
                .clone()
            ),
            "w": (
                model.w
                .detach()
                .cpu()
                .clone()
            ),
        },
        checkpoint_dir / "model.pt",
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    torch.save(
        {
            "pretraining_step": (
                pretraining_step
            ),
            "finetune_step": (
                finetune_step
            ),
            "optimizer_state": (
                optimizer.state_dict()
            ),
            "generator_state": (
                generator
                .get_state()
                .cpu()
                .clone()
            ),
        },
        checkpoint_dir
        / "training_state.pt",
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    with open(
        checkpoint_dir / "metrics.json",
        "w",
    ) as f:
        json.dump(
            metric,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    with open(
        checkpoint_dir / "config.json",
        "w",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    return checkpoint_dir


# ============================================================
# Argument parser
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the rank-one adapter w "
            "on Task 1 starting from a chosen "
            "pre-training checkpoint."
        )
    )

    parser.add_argument(
        "--pretrain-run-dir",
        type=str,
        required=True,
        help=(
            "Folder containing the pretraining "
            "run and its checkpoints."
        ),
    )

    parser.add_argument(
        "--pretrain-step",
        type=int,
        required=True,
        help=(
            "Pretraining checkpoint to load."
        ),
    )

    parser.add_argument(
        "--results-root",
        type=str,
        default="results/task1_finetuning",
    )

    parser.add_argument(
        "--target-steps",
        type=int,
        default=None,
        help=(
            "Number of Task-1 SGD updates. "
            "Default: 5*d."
        ),
    )

    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Fine-tuning steps at which to save. "
            "Default: d/2, d, 2d, 5d."
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--low-rank-init-scale",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help=(
            "Default: d/10."
        ),
    )

    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--w-init-seed",
        type=int,
        default=21,
    )

    parser.add_argument(
        "--sgd-seed",
        type=int,
        default=22,
    )

    parser.add_argument(
        "--task1-test-seed",
        type=int,
        default=31,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


# ============================================================
# Main
# ============================================================

def main():

    args = build_parser().parse_args()

    # --------------------------------------------------------
    # Source pretraining run
    # --------------------------------------------------------

    pretrain_run_dir = Path(
        args.pretrain_run_dir
    ).resolve()

    if not pretrain_run_dir.exists():
        raise FileNotFoundError(
            f"Pretraining run not found: "
            f"{pretrain_run_dir}"
        )

    pretrain_config = load_config(
        pretrain_run_dir
    )

    dtype = parse_dtype(
        pretrain_config["dtype"]
    )

    device = args.device

    d = int(
        pretrain_config["d"]
    )

    T = int(
        pretrain_config["T"]
    )

    kappa = float(
        pretrain_config["kappa"]
    )

    n_test = (
        args.n_test
        if args.n_test is not None
        else int(
            pretrain_config.get(
                "n_test",
                1000,
            )
        )
    )

    # --------------------------------------------------------
    # Fine-tuning schedule
    # --------------------------------------------------------

    target_steps = (
        args.target_steps
        if args.target_steps is not None
        else 5 * d
    )

    eval_every = (
        args.eval_every
        if args.eval_every is not None
        else max(1, d // 10)
    )

    if args.checkpoint_steps is None:
        checkpoint_steps = (
            default_checkpoint_steps(
                d=d,
                target_steps=target_steps,
            )
        )
    else:
        checkpoint_steps = sorted(
            set(
                args.checkpoint_steps
                + [target_steps]
            )
        )

        checkpoint_steps = [
            step
            for step in checkpoint_steps
            if 0 < step <= target_steps
        ]

    # --------------------------------------------------------
    # Load exactly the same teacher
    # --------------------------------------------------------

    teachers = load_teachers(
        run_dir=pretrain_run_dir,
        device=device,
        dtype=dtype,
    )

    # --------------------------------------------------------
    # Load selected W checkpoint
    # --------------------------------------------------------

    loaded = load_pretraining_checkpoint(
        run_dir=pretrain_run_dir,
        step=args.pretrain_step,
        device=device,
        dtype=dtype,
    )

    if loaded["step"] != args.pretrain_step:
        raise RuntimeError(
            "Loaded checkpoint does not match "
            "requested pretraining step."
        )

    # --------------------------------------------------------
    # Student
    # --------------------------------------------------------

    model = Student(
        d=d,
        kappa=kappa,
        dtype=dtype,
        device=device,
    )

    model.load_large_rank(
        loaded["W"]
    )

    # First make sure W is the only active component
    model.set_pretraining_mode()

    # Initialize adapter slightly away from zero.
    #
    # This is essential because the model contains w w^T:
    # at exactly w = 0 the gradient with respect to w vanishes.
    model.reset_low_rank(
        scale=args.low_rank_init_scale,
        seed=args.w_init_seed,
    )

    # Freeze W, train only w.
    model.set_finetuning_mode()

    # --------------------------------------------------------
    # Task 1 teacher
    # --------------------------------------------------------

    S_task1 = (
        teachers.S_star
        + torch.outer(
            teachers.w1_star,
            teachers.w1_star,
        )
    )

    # --------------------------------------------------------
    # Fixed evaluation datasets
    # --------------------------------------------------------

    # Reuse the same pretraining test set definition.
    pretraining_test = generate_dataset(
        n_samples=n_test,
        T=T,
        d=d,
        S_teacher=teachers.S_star,
        seed=int(
            pretrain_config.get(
                "test_seed",
                11,
            )
        )
    )

    # Fixed Task-1 held-out set.
    task1_test = generate_dataset(
        n_samples=n_test,
        T=T,
        d=d,
        S_teacher=S_task1,
        seed=args.task1_test_seed
    )

    with torch.no_grad():

        pred = model(task1_test.X)
        target = task1_test.y

        diff2 = (pred - target) ** 2

        print("===== TASK 1 ERROR DIAGNOSTICS =====")

        print(
            "elementwise MSE:",
            diff2.mean().item()
        )

        print(
            "mean ||error||_F^2 per sample:",
            diff2.sum(dim=(-2, -1)).mean().item()
        )

        print(
            "(1/d) mean ||error||_F^2:",
            (
                diff2
                .sum(dim=(-2, -1))
                .mean()
                / d
            ).item()
        )

        print(
            "(1/(T*d)) mean ||error||_F^2:",
            (
                diff2
                .sum(dim=(-2, -1))
                .mean()
                / (T * d)
            ).item()
        )

        print(
            "generalization_error():",
            generalization_error(
                model,
                task1_test.X,
                task1_test.y,
            ).item()
        )

        print(
            "||w1_star||:",
            torch.linalg.vector_norm(
                teachers.w1_star
            ).item()
        )

        print(
            "||w||:",
            torch.linalg.vector_norm(
                model.w
            ).item()
        )

    # --------------------------------------------------------
    # Optimizer and online Task-1 sample stream
    # --------------------------------------------------------

    optimizer = torch.optim.SGD(
        [model.w],
        lr=args.lr,
    )

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        args.sgd_seed
    )

    # --------------------------------------------------------
    # Output structure
    # --------------------------------------------------------

    source_run_name = (
        pretrain_run_dir.name
    )

    output_root = (
        Path(args.results_root)
        / source_run_name
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_root
        / (
            f"step_{args.pretrain_step}"
            f"_task1_metrics.csv"
        )
    )

    config = {
        "experiment": "task1_finetuning",

        "source_pretrain_run": (
            str(pretrain_run_dir)
        ),

        "source_pretrain_run_name": (
            source_run_name
        ),

        "pretraining_step": (
            args.pretrain_step
        ),

        "d": d,
        "T": T,
        "kappa": kappa,

        "dtype": (
            pretrain_config["dtype"]
        ),

        "lr": args.lr,

        "low_rank_init_scale": (
            args.low_rank_init_scale
        ),

        "w_init_seed": (
            args.w_init_seed
        ),

        "sgd_seed": (
            args.sgd_seed
        ),

        "task1_test_seed": (
            args.task1_test_seed
        ),

        "n_test": n_test,

        "target_steps": target_steps,

        "eval_every": eval_every,

        "checkpoint_steps": (
            checkpoint_steps
        ),
    }

    config_path = (
        output_root
        / (
            f"step_{args.pretrain_step}"
            f"_task1_config.json"
        )
    )

    with open(
        config_path,
        "w",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    # Prevent accidentally mixing trajectories.
    if metrics_path.exists():
        if args.overwrite:
            metrics_path.unlink()
        else:
            raise FileExistsError(
                f"{metrics_path} already exists. "
                f"Use --overwrite if you really "
                f"want to restart this trajectory."
            )

    # --------------------------------------------------------
    # Print setup
    # --------------------------------------------------------

    print()
    print("Task-1 fine-tuning")
    print("----------------------------------------")
    print(
        f"source run      : {pretrain_run_dir}"
    )
    print(
        f"pretrain step   : {args.pretrain_step}"
    )
    print(
        f"d               : {d}"
    )
    print(
        f"T               : {T}"
    )
    print(
        f"device          : {device}"
    )
    print(
        f"learning rate   : {args.lr}"
    )
    print(
        f"target steps    : {target_steps}"
    )
    print(
        f"target / d      : {target_steps / d:.3f}"
    )
    print(
        f"eval every      : {eval_every}"
    )
    print(
        f"checkpoints     : {checkpoint_steps}"
    )
    print(
        f"output root     : {output_root}"
    )
    print()

    # --------------------------------------------------------
    # Step 0
    # --------------------------------------------------------

    metric = evaluate(
        model=model,
        teachers=teachers,
        pretraining_test=pretraining_test,
        task1_test=task1_test,
        pretraining_step=args.pretrain_step,
        finetune_step=0,
    )

    append_metric(
        metrics_path,
        metric,
    )

    print(
        f"[step 0] "
        f"task1={metric['task1_error']:.6e} "
        f"pre={metric['pretraining_error']:.6e} "
        f"overlap={metric['overlap_w1']:.6f}"
    )

    # --------------------------------------------------------
    # Task 1 online SGD
    # --------------------------------------------------------

    for step in range(
        1,
        target_steps + 1,
    ):

        # Fresh Gaussian training sample.
        X = torch.randn(
            1,
            T,
            d,
            generator=generator,
            dtype=dtype,
            device=device,
        )

        with torch.no_grad():
            y = teacher_forward(
                X,
                S_task1,
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(X)

        loss = mse_loss(
            prediction,
            y,
        )

        loss.backward()

        optimizer.step()

        is_eval_step = (
            step % eval_every == 0
        )

        is_checkpoint_step = (
            step in checkpoint_steps
        )

        is_final_step = (
            step == target_steps
        )

        if (
            is_eval_step
            or is_checkpoint_step
            or is_final_step
        ):

            metric = evaluate(
                model=model,
                teachers=teachers,
                pretraining_test=pretraining_test,
                task1_test=task1_test,
                pretraining_step=(
                    args.pretrain_step
                ),
                finetune_step=step,
            )

            append_metric(
                metrics_path,
                metric,
            )

            print(
                f"[step {step:8d}] "
                f"t/d={step / d:7.3f} | "
                f"task1="
                f"{metric['task1_error']:.6e} | "
                f"pre="
                f"{metric['pretraining_error']:.6e} | "
                f"overlap="
                f"{metric['overlap_w1']:.6f}"
            )

        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        if is_checkpoint_step:

            checkpoint_dir = (
                save_finetuning_checkpoint(
                    output_root=output_root,
                    pretraining_step=(
                        args.pretrain_step
                    ),
                    finetune_step=step,
                    model=model,
                    optimizer=optimizer,
                    generator=generator,
                    metric=metric,
                    config=config,
                    overwrite=args.overwrite,
                )
            )

            print(
                f"Saved checkpoint: "
                f"{checkpoint_dir}"
            )

    print()
    print("Task-1 fine-tuning completed.")
    print(
        f"Metrics: {metrics_path}"
    )


if __name__ == "__main__":
    main()