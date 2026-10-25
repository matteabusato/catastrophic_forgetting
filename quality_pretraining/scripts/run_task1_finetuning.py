import argparse
import csv
import json
import shutil
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
    rank_one_reconstruction_error,
    rank_one_overlap,
    matrix_cosine_overlap,
)
from src.losses import LOSS_TYPES, ATTENTION_REDUCTION, resolve_loss_type, teacher_student_loss
from src.observables import task_attention_metrics
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


def float_tag(value: float) -> str:
    return (
        f"{value:g}"
        .replace(".", "p")
        .replace("-", "m")
    )


def default_checkpoint_steps(
    d: int,
    target_steps: int,
) -> list[int]:
    """
    Save a few physically meaningful points on the t/d scale.

    The metric trajectory itself is evaluated more often;
    these are the points where w is persisted.
    """

    candidates = {
        max(1, d // 2),
        d,
        2 * d,
        5 * d,
        10 * d,
        20 * d,
        50 * d,
        100 * d,
        200 * d,
        500 * d,
        target_steps,
    }

    return sorted(
        step
        for step in candidates
        if 0 < step <= target_steps
    )


def should_evaluate(
    step: int,
    d: int,
    eval_every: int,
) -> bool:
    """
    Fine resolution during the expected O(d) Task-1 regime.
    """

    if step <= 5 * d:
        interval = max(1, d // 20)

    elif step <= 20 * d:
        interval = max(1, d // 5)

    else:
        interval = eval_every

    return step % interval == 0


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
    task2_test=None,
    stage: str = "task1",
) -> dict:

    d = model.d

    # --------------------------------------------------------
    # Fixed large-rank representation
    # --------------------------------------------------------

    S_large = model.S_large
    S_star = teachers.S_star

    frozen_representation_error = (
        representation_error(
            S_large,
            S_star,
        ).item()
    )

    # --------------------------------------------------------
    # Prediction errors
    # --------------------------------------------------------

    pretraining_error = (
        generalization_error(
            model,
            pretraining_test.X,
            pretraining_test.y,
        ).item()
    )

    task1_error = (
        generalization_error(
            model,
            task1_test.X,
            task1_test.y,
        ).item()
    )

    # --------------------------------------------------------
    # Vector / adapter order parameters
    # --------------------------------------------------------

    signed_overlap_w1 = (
        vector_overlap(
            model.w,
            teachers.w1_star,
        ).item()
    )

    abs_overlap_w1 = abs(
        signed_overlap_w1
    )

    squared_overlap_w1 = (
        rank_one_overlap(
            model.w,
            teachers.w1_star,
        ).item()
    )

    w_norm = (
        torch.linalg.vector_norm(
            model.w
        ).item()
    )

    w_norm_sq_over_d = (
        torch.sum(model.w ** 2) / d
    ).item()

    w1_star_norm = (
        torch.linalg.vector_norm(
            teachers.w1_star
        ).item()
    )

    w1_star_norm_sq_over_d = (
        torch.sum(
            teachers.w1_star ** 2
        ) / d
    ).item()

    adapter_reconstruction_error = (
        rank_one_reconstruction_error(
            model.w,
            teachers.w1_star,
        ).item()
    )

    # --------------------------------------------------------
    # Full Task-1 matrix reconstruction
    # --------------------------------------------------------

    student_task1_matrix = (
        S_large
        + torch.outer(
            model.w,
            model.w,
        )
    )

    teacher_task1_matrix = (
        S_star
        + torch.outer(
            teachers.w1_star,
            teachers.w1_star,
        )
    )

    task1_matrix_error = (
        representation_error(
            student_task1_matrix,
            teacher_task1_matrix,
        ).item()
    )

    # --------------------------------------------------------
    # Does the adapter compensate pretraining residual?
    # --------------------------------------------------------

    pretraining_residual = (
        S_star - S_large
    )

    adapter_matrix = torch.outer(
        model.w,
        model.w,
    )

    residual_overlap = (
        matrix_cosine_overlap(
            adapter_matrix,
            pretraining_residual,
        ).item()
    )

    residual_norm = (
        torch.linalg.matrix_norm(
            pretraining_residual,
            ord="fro",
        ).item()
    )

    attention = {}
    attention.update(task_attention_metrics(model, pretraining_test, teachers.S_star, "pretraining"))
    attention.update(task_attention_metrics(model, task1_test, teachers.S1_star, "task1"))
    if task2_test is not None:
        attention.update(task_attention_metrics(model, task2_test, teachers.S2_star, "task2"))

    return {
        "pretraining_output_mse": pretraining_error,
        "task1_output_mse": task1_error,
        "task2_output_mse": generalization_error(model, task2_test.X, task2_test.y).item() if task2_test is not None else float("nan"),
        **attention,
        "stage": stage,
        "pretraining_step": (
            pretraining_step
        ),

        "pretraining_step_over_d2": (
            pretraining_step / (d * d)
        ),

        "finetune_step": (
            finetune_step
        ),

        "finetune_step_over_d": (
            finetune_step / d
        ),

        "frozen_representation_error": (
            frozen_representation_error
        ),

        "pretraining_error": (
            pretraining_error
        ),

        "task1_error": (
            task1_error
        ),

        "signed_overlap_w1": (
            signed_overlap_w1
        ),

        "abs_overlap_w1": (
            abs_overlap_w1
        ),

        "squared_overlap_w1": (
            squared_overlap_w1
        ),

        "w_norm": (
            w_norm
        ),

        "w_norm_sq_over_d": (
            w_norm_sq_over_d
        ),

        "w1_star_norm": (
            w1_star_norm
        ),

        "w1_star_norm_sq_over_d": (
            w1_star_norm_sq_over_d
        ),

        "adapter_reconstruction_error": (
            adapter_reconstruction_error
        ),

        "task1_matrix_error": (
            task1_matrix_error
        ),

        "residual_overlap": (
            residual_overlap
        ),

        "residual_norm": (
            residual_norm
        ),
    }


METRIC_FIELDS = [
    "pretraining_output_mse",
    "task1_output_mse",
    "task2_output_mse",
    "stage",
    "pretraining_attention_ce",
    "pretraining_attention_entropy",
    "pretraining_attention_kl",
    "task1_attention_ce",
    "task1_attention_entropy",
    "task1_attention_kl",
    "task2_attention_ce",
    "task2_attention_entropy",
    "task2_attention_kl",

    "pretraining_step",
    "pretraining_step_over_d2",

    "finetune_step",
    "finetune_step_over_d",

    "frozen_representation_error",

    "pretraining_error",
    "task1_error",

    "signed_overlap_w1",
    "abs_overlap_w1",
    "squared_overlap_w1",

    "w_norm",
    "w_norm_sq_over_d",

    "w1_star_norm",
    "w1_star_norm_sq_over_d",

    "adapter_reconstruction_error",
    "task1_matrix_error",

    "residual_overlap",
    "residual_norm",
]


def append_metric(
    metrics_path: Path,
    metric: dict,
) -> None:

    write_header = (
        not metrics_path.exists()
        or metrics_path.stat().st_size == 0
    )

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
    checkpoint_root: Path,
    finetune_step: int,
    model: Student,
    optimizer,
    generator: torch.Generator,
    metric: dict,
    loss_type: str = "output_mse",
) -> Path:

    checkpoint_dir = (
        checkpoint_root
        / f"step_{finetune_step:09d}"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.w
        .detach()
        .cpu()
        .clone(),
        checkpoint_dir / "w.pt",
    )

    torch.save(
        {
            "finetune_step": (
                finetune_step
            ),

            "loss_type": loss_type,
            "attention_reduction": ATTENTION_REDUCTION,

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

    with open(
        checkpoint_dir / "metrics.json",
        "w",
    ) as f:

        json.dump(
            metric,
            f,
            indent=2,
        )

    return checkpoint_dir


# ============================================================
# Arguments
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune only the rank-one vector w "
            "on Task 1 from a manually chosen "
            "pretraining checkpoint."
        )
    )

    parser.add_argument("--task2-test-seed", type=int, default=41)
    parser.add_argument("--loss", choices=LOSS_TYPES, default=None,
                        help="Training objective; defaults to the source checkpoint objective.")
    parser.add_argument("--results-root", type=Path, default=Path("results/quality_pretraining"))

    parser.add_argument(
        "--pretrain-run-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--pretrain-step",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--target-steps",
        type=int,
        required=True,
        help=(
            "Number of online Task-1 SGD updates."
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
        default=1.0,
        help=(
            "Std. dev. of w initialization. "
            "Baseline is O(1), matching w*_1."
        ),
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
        "--n-test",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help=(
            "Late-time evaluation interval. "
            "Default: d/2. Early evaluation "
            "is adaptive."
        ),
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

    loss_type = resolve_loss_type(args.loss, pretrain_config)

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

    eval_every = (
        args.eval_every
        if args.eval_every is not None
        else max(1, d // 2)
    )

    # --------------------------------------------------------
    # Checkpoint schedule
    # --------------------------------------------------------

    if args.checkpoint_steps is None:

        checkpoint_steps = (
            default_checkpoint_steps(
                d=d,
                target_steps=args.target_steps,
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
            if 0 < step <= args.target_steps
        ]

    # --------------------------------------------------------
    # Teacher + selected W
    # --------------------------------------------------------

    teachers = load_teachers(
        run_dir=pretrain_run_dir,
        device=device,
        dtype=dtype,
    )

    loaded = load_pretraining_checkpoint(
        run_dir=pretrain_run_dir,
        step=args.pretrain_step,
        device=device,
        dtype=dtype,
    )

    resolve_loss_type(loss_type, loaded)

    if loaded["step"] != args.pretrain_step:
        raise RuntimeError(
            "Loaded pretraining checkpoint "
            "does not match requested step."
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

    # Important:
    # set_pretraining_mode() zeros w.
    model.set_pretraining_mode()

    # Same random direction for a fixed seed;
    # scale is controlled independently.
    model.reset_low_rank(
        scale=args.low_rank_init_scale,
        seed=args.w_init_seed,
    )

    # Freeze W, activate w.
    model.set_finetuning_mode()

    # --------------------------------------------------------
    # Task-1 teacher
    # --------------------------------------------------------

    S_task1 = teachers.S1_star

    # --------------------------------------------------------
    # Fixed evaluation sets
    # --------------------------------------------------------

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
        ),
    )

    task1_test = generate_dataset(
        n_samples=n_test,
        T=T,
        d=d,
        S_teacher=S_task1,
        seed=args.task1_test_seed,
    )

    # --------------------------------------------------------
    # Evaluate all teachers on fixed inputs even before Task 2 is trained.
    task2_test = generate_dataset(n_test, T, d, teachers.S2_star, args.task2_test_seed)

    # Optimizer + online stream
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
    # Output layout
    # --------------------------------------------------------

    base_output_dir = args.results_root / (
        f"finetunetask1_d{d}_T{T}"
    )

    run_name = (
        f"prestep_{args.pretrain_step}"
        f"_lr{float_tag(args.lr)}"
        f"_init{float_tag(args.low_rank_init_scale)}"
        f"_wseed{args.w_init_seed}"
        f"_sgdseed{args.sgd_seed}"
    )

    if loss_type == "attention_ce":
        run_name += "_attention_ce"

    run_dir = (
        base_output_dir
        / run_name
    )

    if run_dir.exists():

        if args.overwrite:
            shutil.rmtree(run_dir)

        else:
            raise FileExistsError(
                f"Run already exists: {run_dir}\n"
                f"Use --overwrite to replace it."
            )

    checkpoint_root = (
        run_dir / "checkpoints"
    )

    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        run_dir / "metrics.csv"
    )

    config = {
        "loss_type": loss_type,
        "attention_reduction": ATTENTION_REDUCTION,
        "experiment": (
            "finetune_task1"
        ),

        "source_pretrain_run": (
            str(pretrain_run_dir)
        ),

        "pretraining_step": (
            args.pretrain_step
        ),

        "pretraining_step_over_d2": (
            args.pretrain_step / (d * d)
        ),

        "source_pretraining_metrics": (
            loaded["metrics"]
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

        "task2_test_seed": args.task2_test_seed,
        "task1_test_seed": (
            args.task1_test_seed
        ),

        "n_test": n_test,

        "target_steps": (
            args.target_steps
        ),

        "target_steps_over_d": (
            args.target_steps / d
        ),

        "eval_every": (
            eval_every
        ),

        "checkpoint_steps": (
            checkpoint_steps
        ),
    }

    with open(
        run_dir / "config.json",
        "w",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Print setup
    # --------------------------------------------------------

    print()
    print("==============================================")
    print("Task-1 fine-tuning")
    print("==============================================")
    print(f"source pretrain : {pretrain_run_dir}")
    print(f"pretrain step   : {args.pretrain_step:,}")
    print(
        f"pretrain t/d²   : "
        f"{args.pretrain_step / (d*d):.6f}"
    )
    print(f"d               : {d}")
    print(f"T               : {T}")
    print(f"device          : {device}")
    print(f"lr              : {args.lr}")
    print(f"training loss   : {loss_type}")
    print(
        f"w init scale    : "
        f"{args.low_rank_init_scale}"
    )
    print(f"target steps    : {args.target_steps:,}")
    print(
        f"target t/d      : "
        f"{args.target_steps / d:.3f}"
    )
    print(f"eval every late : {eval_every}")
    print(f"checkpoints     : {checkpoint_steps}")
    print(f"run directory   : {run_dir}")
    print()

    # --------------------------------------------------------
    # Initial evaluation
    # --------------------------------------------------------

    # The initialized vector is retained; only its contribution is disabled here.
    model.use_low_rank = False
    before_adapter = evaluate(model, teachers, pretraining_test, task1_test,
                              args.pretrain_step, 0, task2_test=task2_test,
                              stage="before_adapter_init")
    append_metric(metrics_path, before_adapter)
    model.use_low_rank = True

    metric = evaluate(
        model=model,
        teachers=teachers,
        pretraining_test=pretraining_test,
        task1_test=task1_test,
        pretraining_step=args.pretrain_step,
        finetune_step=0,
        stage="after_adapter_init",
        task2_test=task2_test,
    )

    append_metric(
        metrics_path,
        metric,
    )

    print(
        f"[step 0] "
        f"task1 MSE={metric['task1_error']:.6e} CE={metric['task1_attention_ce']:.6e} | "
        f"|m1|={metric['abs_overlap_w1']:.6f} | "
        f"m1²={metric['squared_overlap_w1']:.6f} | "
        f"||w||²/d="
        f"{metric['w_norm_sq_over_d']:.6f} | "
        f"eps_adapter="
        f"{metric['adapter_reconstruction_error']:.6e}"
    )

    # --------------------------------------------------------
    # Online Task-1 SGD
    # --------------------------------------------------------

    for step in range(
        1,
        args.target_steps + 1,
    ):

        X = torch.randn(
            1,
            T,
            d,
            generator=generator,
            dtype=dtype,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        loss = teacher_student_loss(model, X, teachers.S1_star, loss_type)

        loss.backward()
        optimizer.step()

        is_eval_step = (
            should_evaluate(
                step=step,
                d=d,
                eval_every=eval_every,
            )
        )

        is_checkpoint_step = (
            step in checkpoint_steps
        )

        is_final_step = (
            step == args.target_steps
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
                pretraining_step=args.pretrain_step,
                finetune_step=step,
                task2_test=task2_test,
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
                f"|m1|="
                f"{metric['abs_overlap_w1']:.6f} | "
                f"m1²="
                f"{metric['squared_overlap_w1']:.6f} | "
                f"||w||²/d="
                f"{metric['w_norm_sq_over_d']:.6f} | "
                f"eps_adapter="
                f"{metric['adapter_reconstruction_error']:.6e} | "
                f"residual_overlap="
                f"{metric['residual_overlap']:.6f}"
            )

        if is_checkpoint_step:

            checkpoint_dir = (
                save_finetuning_checkpoint(
                    checkpoint_root=checkpoint_root,
                    finetune_step=step,
                    model=model,
                    optimizer=optimizer,
                    generator=generator,
                    metric=metric,
                    loss_type=loss_type,
                )
            )

            print(
                f"Saved checkpoint: "
                f"{checkpoint_dir}"
            )

    print()
    print("Task-1 fine-tuning completed.")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()