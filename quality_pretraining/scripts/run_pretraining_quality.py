import argparse
from pathlib import Path
import shutil

import torch

from src.data_model import (
    sample_teachers,
    generate_dataset,
)
from src.model import Student
from src.training import pretrain
from src.io_utils import (
    append_pretraining_metric,
    append_run_history,
    ensure_run_directories,
    get_latest_checkpoint_step,
    load_config,
    load_metric_steps,
    load_pretraining_checkpoint,
    load_teachers,
    save_config,
    save_pretraining_checkpoint,
    save_teachers,
)

def parse_dtype(
    name: str,
) -> torch.dtype:

    if name == "float64":
        return torch.float64

    if name == "float32":
        return torch.float32

    raise ValueError(
        f"Unsupported dtype: {name}"
    )


def float_tag(
    value: float,
) -> str:

    return (
        f"{value:g}"
        .replace(".", "p")
    )


def default_run_name(
    d: int,
    T: int,
    kappa: float,
    kappa_star: float,
    teacher_seed: int,
) -> str:

    return (
        f"d{d}"
        f"_T{T}"
        f"_k{float_tag(kappa)}"
        f"_ks{float_tag(kappa_star)}"
        f"_seed{teacher_seed}"
    )


def default_checkpoint_steps(
    d: int,
    target_step: int,
) -> list[int]:

    d2 = d * d

    candidates = {
        0,
        d,
        round(0.25 * d2),
        round(0.5 * d2),
        d2,
        2 * d2,
        5 * d2,
        10 * d2,
        target_step,
    }

    return sorted(
        step
        for step in candidates
        if 0 <= step <= target_step
    )
def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Run or resume the RQ1 pre-training "
            "quality experiment."
        )
    )

    parser.add_argument(
        "--results-root",
        type=str,
        default="results/quality_pretraining",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--target-steps",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--resume-step",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="*",
        default=None,
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

    # Fresh-run configuration
    parser.add_argument(
        "--d",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--T",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--kappa",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--kappa-star",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=[
            "float32",
            "float64",
        ],
    )

    parser.add_argument(
        "--teacher-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--student-seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--sgd-seed",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--test-seed",
        type=int,
        default=11,
    )

    parser.add_argument(
        "--n-test",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
    )

    return parser

def main():

    parser = build_parser()
    args = parser.parse_args()

    results_root = Path(
        args.results_root
    )

    # ========================================================
    # RESUME
    # ========================================================

    if args.resume:

        if args.run_name is None:
            raise ValueError(
                "--run-name is required when using --resume"
            )

        run_dir = (
            results_root
            / args.run_name
        )

        config = load_config(
            run_dir
        )

        dtype = parse_dtype(
            config["dtype"]
        )

        device = args.device

        d = config["d"]
        T = config["T"]
        kappa = config["kappa"]

        lr = config["lr"]
        sgd_seed = config["sgd_seed"]

        eval_every = (
            config["eval_every"]
        )

        # ----------------------------------------------
        # Reload exactly the same teachers
        # ----------------------------------------------

        teachers = load_teachers(
            run_dir=run_dir,
            device=device,
            dtype=dtype,
        )

        # ----------------------------------------------
        # Reconstruct exactly the same fixed test set
        # ----------------------------------------------

        pretraining_test = generate_dataset(
            n_samples=config["n_test"],
            T=T,
            d=d,
            S_teacher=teachers.S_star,
            seed=config["test_seed"],
        )

        # ----------------------------------------------
        # Load resume state
        # ----------------------------------------------

        latest_step = get_latest_checkpoint_step(run_dir)

        loaded = load_pretraining_checkpoint(
            run_dir=run_dir,
            step=latest_step,
            device=device,
            dtype=dtype,
        )

        start_step = loaded["step"]

        if args.target_steps <= start_step:
            raise ValueError(
                f"target_steps={args.target_steps} "
                f"must be larger than resumed step "
                f"{start_step}"
            )

        # Student initialization does not matter here,
        # because W is immediately replaced.
        model = Student(
            d=d,
            kappa=kappa,
            dtype=dtype,
            device=device,
        )

        model.load_large_rank(
            loaded["W"]
        )

        optimizer_state = (
            loaded["optimizer_state"]
        )

        generator_state = (
            loaded["generator_state"]
        )

    # ========================================================
    # FRESH RUN
    # ========================================================

    else:

        dtype = parse_dtype(
            args.dtype
        )

        device = args.device

        d = args.d
        T = args.T
        kappa = args.kappa

        eval_every = (
            args.eval_every
            if args.eval_every is not None
            else max(1, d // 2)
        )

        run_name = (
            args.run_name
            if args.run_name is not None
            else default_run_name(
                d=d,
                T=T,
                kappa=kappa,
                kappa_star=args.kappa_star,
                teacher_seed=args.teacher_seed,
            )
        )

        run_dir = (
            results_root
            / run_name
        )

        if run_dir.exists():
            print(
                f"Removing existing run directory: {run_dir}"
            )
            shutil.rmtree(run_dir)

        ensure_run_directories(
            run_dir
        )

        config = {
            "experiment": (
                "quality_pretraining"
            ),

            "d": d,
            "T": T,

            "kappa": kappa,
            "kappa_star": (
                args.kappa_star
            ),

            "lr": args.lr,

            "dtype": args.dtype,

            "teacher_seed": (
                args.teacher_seed
            ),

            "student_seed": (
                args.student_seed
            ),

            "sgd_seed": (
                args.sgd_seed
            ),

            "test_seed": (
                args.test_seed
            ),

            "n_test": (
                args.n_test
            ),

            "eval_every": (
                eval_every
            ),
        }

        save_config(
            run_dir,
            config,
        )

        # ----------------------------------------------
        # Teachers
        # ----------------------------------------------

        teachers = sample_teachers(
            d=d,
            kappa_star=args.kappa_star,
            device=device,
            dtype=dtype,
            seed=args.teacher_seed,
        )

        save_teachers(
            run_dir,
            teachers,
        )

        # ----------------------------------------------
        # Fixed held-out test set
        # ----------------------------------------------

        pretraining_test = generate_dataset(
            n_samples=args.n_test,
            T=T,
            d=d,
            S_teacher=teachers.S_star,
            seed=args.test_seed,
        )

        # ----------------------------------------------
        # Reproducible student initialization
        # ----------------------------------------------

        torch.manual_seed(
            args.student_seed
        )

        model = Student(
            d=d,
            kappa=kappa,
            dtype=dtype,
            device=device,
        )

        start_step = 0

        lr = args.lr
        sgd_seed = args.sgd_seed

        optimizer_state = None
        generator_state = None

    # ========================================================
    # Checkpoint schedule
    # ========================================================

    if args.checkpoint_steps is None:

        checkpoint_steps = (
            default_checkpoint_steps(
                d=d,
                target_step=args.target_steps,
            )
        )

    else:

        checkpoint_steps = sorted(
            set(
                args.checkpoint_steps
                + [args.target_steps]
            )
        )

    checkpoint_steps = [
        step
        for step in checkpoint_steps
        if start_step <= step <= args.target_steps
    ]

    print()
    print("Pre-training quality experiment")
    print("--------------------------------")
    print(f"run directory : {run_dir}")
    print(f"device        : {device}")
    print(f"d             : {d}")
    print(f"T             : {T}")
    print(f"start step    : {start_step}")
    print(f"target step   : {args.target_steps}")
    print(f"eval every    : {eval_every}")
    print(f"checkpoints   : {checkpoint_steps}")
    print()

    # ========================================================
    # Immediate persistent callbacks
    # ========================================================

    existing_metric_steps = (
        load_metric_steps(
            run_dir
        )
    )

    def metric_callback(metric):

        if metric.step in existing_metric_steps:
            return

        append_pretraining_metric(
            run_dir,
            metric,
        )

        existing_metric_steps.add(
            metric.step
        )

    def checkpoint_callback(
        checkpoint,
    ):

        save_pretraining_checkpoint(
            run_dir,
            checkpoint,
        )

        print(
            f"[checkpoint] "
            f"step={checkpoint.step:,} "
            f"t/d={checkpoint.step / d:.3f} "
            f"t/d²={checkpoint.step / (d*d):.5f} "
            f"eps_S={checkpoint.representation_error:.6e} "
            f"eps_pre={checkpoint.generalization_error:.6e} "
            f"q={checkpoint.q:.6e} "
            f"Q={checkpoint.Q:.6e} "
            f"Q*={checkpoint.Q_star:.6e} "
            f"trace_err={checkpoint.trace_mismatch:.6e}"
        )

    # ========================================================
    # Train / resume
    # ========================================================

    pretrain(
        model=model,
        test_data=pretraining_test,
        S_star=teachers.S_star,
        T=T,
        target_step=args.target_steps,
        lr=lr,
        checkpoint_steps=checkpoint_steps,
        eval_every=eval_every,
        seed=sgd_seed,
        start_step=start_step,
        optimizer_state=optimizer_state,
        generator_state=generator_state,
        metric_callback=metric_callback,
        checkpoint_callback=checkpoint_callback,
    )

    append_run_history(
        run_dir=run_dir,
        start_step=start_step,
        target_step=args.target_steps,
    )

    print()
    print(
        f"Finished at step {args.target_steps:,}."
    )

    print(
        f"Metrics: {run_dir / 'metrics.csv'}"
    )


if __name__ == "__main__":
    main()