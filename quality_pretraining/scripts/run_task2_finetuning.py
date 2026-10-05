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


def float_tag(value: float) -> str:

    return (
        f"{value:g}"
        .replace(".", "p")
        .replace("-", "m")
    )


def task1_checkpoint_dir(
    task1_run_dir: Path,
    step: int,
) -> Path:

    return (
        task1_run_dir
        / "checkpoints"
        / f"step_{step:09d}"
    )


def load_task1_checkpoint(
    task1_run_dir: Path,
    step: int,
    device: str,
    dtype: torch.dtype,
) -> dict:
    """
    Load the Task-1 adapter checkpoint that will initialize Task 2.
    """

    checkpoint_dir = task1_checkpoint_dir(
        task1_run_dir=task1_run_dir,
        step=step,
    )

    if not checkpoint_dir.exists():

        raise FileNotFoundError(
            f"Task-1 checkpoint does not exist:\n"
            f"{checkpoint_dir}"
        )

    w_path = checkpoint_dir / "w.pt"

    if not w_path.exists():

        raise FileNotFoundError(
            f"Missing Task-1 w checkpoint:\n"
            f"{w_path}"
        )

    w = torch.load(
        w_path,
        map_location="cpu",
        weights_only=True,
    )

    metrics_path = (
        checkpoint_dir
        / "metrics.json"
    )

    if metrics_path.exists():

        with open(
            metrics_path,
            "r",
        ) as f:

            metrics = json.load(f)

    else:

        metrics = None

    return {
        "step": step,
        "w": w.to(
            device=device,
            dtype=dtype,
        ),
        "metrics": metrics,
        "checkpoint_dir": checkpoint_dir,
    }


def load_task1_initial_metrics(
    task1_run_dir: Path,
) -> dict:
    """
    Read the Task-1 step-0 row.

    This gives the state before Task 1, needed for cumulative
    pretraining forgetting:

        F_pre^(2)(t)
          = eps_pre(t)
          - eps_pre(before Task 1)
    """

    metrics_path = (
        task1_run_dir
        / "metrics.csv"
    )

    if not metrics_path.exists():

        raise FileNotFoundError(
            f"Task-1 metrics.csv not found:\n"
            f"{metrics_path}"
        )

    with open(
        metrics_path,
        "r",
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

    if not rows:

        raise RuntimeError(
            f"Task-1 metrics file is empty: "
            f"{metrics_path}"
        )

    step0_rows = [
        row
        for row in rows
        if int(row["finetune_step"]) == 0
    ]

    if not step0_rows:

        raise RuntimeError(
            "Could not find Task-1 step 0 "
            f"in {metrics_path}"
        )

    row = step0_rows[0]

    return {
        "pretraining_error_before_task1": float(
            row["pretraining_error"]
        ),
        "task1_error_before_task1": float(
            row["task1_error"]
        ),
    }


def default_checkpoint_steps(
    d: int,
    target_steps: int,
) -> list[int]:
    """
    Physically meaningful Task-2 checkpoints on the t/d scale.
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
    Fine resolution early in Task 2, then regular late-time
    measurements.
    """

    if step <= 5 * d:

        interval = max(
            1,
            d // 20,
        )

    elif step <= 20 * d:

        interval = max(
            1,
            d // 5,
        )

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
    task2_test,
    pretraining_step: int,
    task1_step: int,
    task2_step: int,
    pretraining_error_before_task1: float,
    pretraining_error_before_task2: float,
    task1_error_before_task2: float,
) -> dict:

    d = model.d

    S_large = model.S_large
    S_star = teachers.S_star

    adapter_matrix = torch.outer(
        model.w,
        model.w,
    )

    # ========================================================
    # Prediction errors
    # ========================================================

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

    task2_error = (
        generalization_error(
            model,
            task2_test.X,
            task2_test.y,
        ).item()
    )

    # ========================================================
    # Forgetting
    # ========================================================

    # Catastrophic forgetting of Task 1 during Task 2.
    forgetting_task1 = (
        task1_error
        - task1_error_before_task2
    )

    # Additional pretraining forgetting caused by Task 2.
    pretraining_forgetting_since_task1 = (
        pretraining_error
        - pretraining_error_before_task2
    )

    # Total pretraining forgetting measured relative to
    # the state before Task 1.
    pretraining_forgetting_total = (
        pretraining_error
        - pretraining_error_before_task1
    )

    # ========================================================
    # Overlap with Task 1
    # ========================================================

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

    # ========================================================
    # Overlap with Task 2
    # ========================================================

    signed_overlap_w2 = (
        vector_overlap(
            model.w,
            teachers.w2_star,
        ).item()
    )

    abs_overlap_w2 = abs(
        signed_overlap_w2
    )

    squared_overlap_w2 = (
        rank_one_overlap(
            model.w,
            teachers.w2_star,
        ).item()
    )

    # ========================================================
    # Adapter norms
    # ========================================================

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

    w2_star_norm = (
        torch.linalg.vector_norm(
            teachers.w2_star
        ).item()
    )

    w2_star_norm_sq_over_d = (
        torch.sum(
            teachers.w2_star ** 2
        ) / d
    ).item()

    # ========================================================
    # Rank-one reconstruction errors
    # ========================================================

    adapter_reconstruction_error_task1 = (
        rank_one_reconstruction_error(
            model.w,
            teachers.w1_star,
        ).item()
    )

    adapter_reconstruction_error_task2 = (
        rank_one_reconstruction_error(
            model.w,
            teachers.w2_star,
        ).item()
    )

    # ========================================================
    # Full matrix reconstruction errors
    # ========================================================

    student_matrix = (
        S_large
        + adapter_matrix
    )

    task1_matrix_error = (
        representation_error(
            student_matrix,
            teachers.S1_star,
        ).item()
    )

    task2_matrix_error = (
        representation_error(
            student_matrix,
            teachers.S2_star,
        ).item()
    )

    # ========================================================
    # Frozen pretraining representation
    #
    # Useful as a sanity check: should remain constant.
    # ========================================================

    frozen_representation_error = (
        representation_error(
            S_large,
            S_star,
        ).item()
    )

    # ========================================================
    # Task similarity
    # ========================================================

    task_similarity_signed = (
        vector_overlap(
            teachers.w1_star,
            teachers.w2_star,
        ).item()
    )

    task_similarity_squared = (
        task_similarity_signed ** 2
    )

    # ========================================================
    # Return
    # ========================================================

    return {

        # ----------------------------------------------------
        # Source checkpoints
        # ----------------------------------------------------

        "pretraining_step": (
            pretraining_step
        ),

        "pretraining_step_over_d2": (
            pretraining_step
            / (d * d)
        ),

        "task1_step": (
            task1_step
        ),

        "task1_step_over_d": (
            task1_step / d
        ),

        # ----------------------------------------------------
        # Task-2 time
        # ----------------------------------------------------

        "task2_step": (
            task2_step
        ),

        "task2_step_over_d": (
            task2_step / d
        ),

        # ----------------------------------------------------
        # Prediction errors
        # ----------------------------------------------------

        "pretraining_error": (
            pretraining_error
        ),

        "task1_error": (
            task1_error
        ),

        "task2_error": (
            task2_error
        ),

        # ----------------------------------------------------
        # Fixed baselines
        # ----------------------------------------------------

        "pretraining_error_before_task1": (
            pretraining_error_before_task1
        ),

        "pretraining_error_before_task2": (
            pretraining_error_before_task2
        ),

        "task1_error_before_task2": (
            task1_error_before_task2
        ),

        # ----------------------------------------------------
        # Forgetting
        # ----------------------------------------------------

        "forgetting_task1": (
            forgetting_task1
        ),

        "pretraining_forgetting_since_task1": (
            pretraining_forgetting_since_task1
        ),

        "pretraining_forgetting_total": (
            pretraining_forgetting_total
        ),

        # ----------------------------------------------------
        # Task-1 overlaps
        # ----------------------------------------------------

        "signed_overlap_w1": (
            signed_overlap_w1
        ),

        "abs_overlap_w1": (
            abs_overlap_w1
        ),

        "squared_overlap_w1": (
            squared_overlap_w1
        ),

        # ----------------------------------------------------
        # Task-2 overlaps
        # ----------------------------------------------------

        "signed_overlap_w2": (
            signed_overlap_w2
        ),

        "abs_overlap_w2": (
            abs_overlap_w2
        ),

        "squared_overlap_w2": (
            squared_overlap_w2
        ),

        # ----------------------------------------------------
        # Adapter norm
        # ----------------------------------------------------

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

        "w2_star_norm": (
            w2_star_norm
        ),

        "w2_star_norm_sq_over_d": (
            w2_star_norm_sq_over_d
        ),

        # ----------------------------------------------------
        # Adapter reconstruction
        # ----------------------------------------------------

        "adapter_reconstruction_error_task1": (
            adapter_reconstruction_error_task1
        ),

        "adapter_reconstruction_error_task2": (
            adapter_reconstruction_error_task2
        ),

        # ----------------------------------------------------
        # Full matrix reconstruction
        # ----------------------------------------------------

        "task1_matrix_error": (
            task1_matrix_error
        ),

        "task2_matrix_error": (
            task2_matrix_error
        ),

        # ----------------------------------------------------
        # Sanity check
        # ----------------------------------------------------

        "frozen_representation_error": (
            frozen_representation_error
        ),

        # ----------------------------------------------------
        # Teacher similarity
        # ----------------------------------------------------

        "task_similarity_signed": (
            task_similarity_signed
        ),

        "task_similarity_squared": (
            task_similarity_squared
        ),
    }


METRIC_FIELDS = [

    "pretraining_step",
    "pretraining_step_over_d2",

    "task1_step",
    "task1_step_over_d",

    "task2_step",
    "task2_step_over_d",

    "pretraining_error",
    "task1_error",
    "task2_error",

    "pretraining_error_before_task1",
    "pretraining_error_before_task2",
    "task1_error_before_task2",

    "forgetting_task1",
    "pretraining_forgetting_since_task1",
    "pretraining_forgetting_total",

    "signed_overlap_w1",
    "abs_overlap_w1",
    "squared_overlap_w1",

    "signed_overlap_w2",
    "abs_overlap_w2",
    "squared_overlap_w2",

    "w_norm",
    "w_norm_sq_over_d",

    "w1_star_norm",
    "w1_star_norm_sq_over_d",

    "w2_star_norm",
    "w2_star_norm_sq_over_d",

    "adapter_reconstruction_error_task1",
    "adapter_reconstruction_error_task2",

    "task1_matrix_error",
    "task2_matrix_error",

    "frozen_representation_error",

    "task_similarity_signed",
    "task_similarity_squared",
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

def save_task2_checkpoint(
    checkpoint_root: Path,
    task2_step: int,
    model: Student,
    optimizer,
    generator: torch.Generator,
    metric: dict,
) -> Path:

    checkpoint_dir = (
        checkpoint_root
        / f"step_{task2_step:09d}"
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
        checkpoint_dir
        / "w.pt",
    )

    torch.save(
        {
            "task2_step": (
                task2_step
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

    with open(
        checkpoint_dir
        / "metrics.json",
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
            "Fine-tune the existing Task-1 adapter w "
            "on Task 2, starting from explicitly selected "
            "pretraining and Task-1 checkpoints."
        )
    )

    # --------------------------------------------------------
    # Pretraining source
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Task-1 source
    # --------------------------------------------------------

    parser.add_argument(
        "--task1-run-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--task1-step",
        type=int,
        required=True,
        help=(
            "Task-1 fine-tuning checkpoint used "
            "to initialize w for Task 2."
        ),
    )

    # --------------------------------------------------------
    # Task-2 training
    # --------------------------------------------------------

    parser.add_argument(
        "--target-steps",
        type=int,
        required=True,
        help=(
            "Number of online Task-2 SGD updates."
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--sgd-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--task2-test-seed",
        type=int,
        default=41,
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
            "Default: d/2."
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

    # ========================================================
    # Source directories
    # ========================================================

    pretrain_run_dir = Path(
        args.pretrain_run_dir
    ).resolve()

    task1_run_dir = Path(
        args.task1_run_dir
    ).resolve()

    if not pretrain_run_dir.exists():

        raise FileNotFoundError(
            f"Pretraining run not found:\n"
            f"{pretrain_run_dir}"
        )

    if not task1_run_dir.exists():

        raise FileNotFoundError(
            f"Task-1 run not found:\n"
            f"{task1_run_dir}"
        )

    # ========================================================
    # Configs
    # ========================================================

    pretrain_config = load_config(
        pretrain_run_dir
    )

    task1_config = load_config(
        task1_run_dir
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

    # ========================================================
    # Sanity checks: Task-1 run must belong to the same model
    # ========================================================

    if int(task1_config["d"]) != d:

        raise ValueError(
            "Dimension mismatch between pretraining "
            "and Task-1 runs."
        )

    if int(task1_config["T"]) != T:

        raise ValueError(
            "T mismatch between pretraining "
            "and Task-1 runs."
        )

    if float(task1_config["kappa"]) != kappa:

        raise ValueError(
            "kappa mismatch between pretraining "
            "and Task-1 runs."
        )

    if (
        int(task1_config["pretraining_step"])
        != args.pretrain_step
    ):

        raise ValueError(
            "The selected Task-1 run was not trained "
            "from the requested pretraining checkpoint.\n"
            f"Task-1 run pretraining step: "
            f"{task1_config['pretraining_step']}\n"
            f"Requested pretraining step: "
            f"{args.pretrain_step}"
        )

    # ========================================================
    # Test / evaluation settings
    # ========================================================

    n_test = (
        args.n_test
        if args.n_test is not None
        else int(
            task1_config.get(
                "n_test",
                pretrain_config.get(
                    "n_test",
                    1000,
                ),
            )
        )
    )

    eval_every = (
        args.eval_every
        if args.eval_every is not None
        else max(
            1,
            d // 2,
        )
    )

    # ========================================================
    # Checkpoint schedule
    # ========================================================

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

    # ========================================================
    # Teachers
    # ========================================================

    teachers = load_teachers(
        run_dir=pretrain_run_dir,
        device=device,
        dtype=dtype,
    )

    # ========================================================
    # Load chosen pretraining checkpoint W
    # ========================================================

    loaded_pretraining = (
        load_pretraining_checkpoint(
            run_dir=pretrain_run_dir,
            step=args.pretrain_step,
            device=device,
            dtype=dtype,
        )
    )

    if (
        loaded_pretraining["step"]
        != args.pretrain_step
    ):

        raise RuntimeError(
            "Loaded pretraining checkpoint does not "
            "match requested pretraining step."
        )

    # ========================================================
    # Load chosen Task-1 checkpoint w
    # ========================================================

    loaded_task1 = (
        load_task1_checkpoint(
            task1_run_dir=task1_run_dir,
            step=args.task1_step,
            device=device,
            dtype=dtype,
        )
    )

    # ========================================================
    # Student
    # ========================================================

    model = Student(
        d=d,
        kappa=kappa,
        dtype=dtype,
        device=device,
    )

    model.load_large_rank(
        loaded_pretraining["W"]
    )

    # Do NOT call reset_low_rank().
    #
    # Task 2 must start from the adapter learned on Task 1.

    with torch.no_grad():

        model.w.copy_(
            loaded_task1["w"]
        )

    # Freeze W and optimize only w.
    model.set_finetuning_mode()

    # ========================================================
    # Fixed evaluation datasets
    # ========================================================

    pretraining_test_seed = int(
        pretrain_config.get(
            "test_seed",
            11,
        )
    )

    task1_test_seed = int(
        task1_config.get(
            "task1_test_seed",
            31,
        )
    )

    pretraining_test = generate_dataset(
        n_samples=n_test,
        T=T,
        d=d,
        S_teacher=teachers.S_star,
        seed=pretraining_test_seed,
    )

    task1_test = generate_dataset(
        n_samples=n_test,
        T=T,
        d=d,
        S_teacher=teachers.S1_star,
        seed=task1_test_seed,
    )

    task2_test = generate_dataset(
        n_samples=n_test,
        T=T,
        d=d,
        S_teacher=teachers.S2_star,
        seed=args.task2_test_seed,
    )

    # ========================================================
    # Baseline BEFORE Task 1
    # ========================================================

    task1_initial = (
        load_task1_initial_metrics(
            task1_run_dir
        )
    )

    pretraining_error_before_task1 = (
        task1_initial[
            "pretraining_error_before_task1"
        ]
    )

    # ========================================================
    # Evaluate loaded state = state AFTER Task 1 / BEFORE Task 2
    # ========================================================

    pretraining_error_before_task2 = (
        generalization_error(
            model,
            pretraining_test.X,
            pretraining_test.y,
        ).item()
    )

    task1_error_before_task2 = (
        generalization_error(
            model,
            task1_test.X,
            task1_test.y,
        ).item()
    )

    task2_error_before_task2 = (
        generalization_error(
            model,
            task2_test.X,
            task2_test.y,
        ).item()
    )

    # ========================================================
    # Optimizer + fresh Task-2 online stream
    # ========================================================

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

    # ========================================================
    # Output layout
    # ========================================================

    base_output_dir = (
        Path("results")
        / "quality_pretraining"
        / f"finetunetask2_d{d}_T{T}"
    )

    task1_wseed = int(
        task1_config.get(
            "w_init_seed",
            -1,
        )
    )

    task1_sgdseed = int(
        task1_config.get(
            "sgd_seed",
            -1,
        )
    )

    run_name = (
        f"prestep_{args.pretrain_step}"
        f"_task1step_{args.task1_step}"
        f"_lr{float_tag(args.lr)}"
        f"_task1wseed{task1_wseed}"
        f"_task1sgdseed{task1_sgdseed}"
        f"_task2sgdseed{args.sgd_seed}"
    )

    run_dir = (
        base_output_dir
        / run_name
    )

    if run_dir.exists():

        if args.overwrite:

            shutil.rmtree(
                run_dir
            )

        else:

            raise FileExistsError(
                f"Run already exists: "
                f"{run_dir}\n"
                f"Use --overwrite to replace it."
            )

    checkpoint_root = (
        run_dir
        / "checkpoints"
    )

    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        run_dir
        / "metrics.csv"
    )

    # ========================================================
    # Config
    # ========================================================

    config = {

        "experiment": (
            "finetune_task2"
        ),

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        "source_pretrain_run": (
            str(pretrain_run_dir)
        ),

        "pretraining_step": (
            args.pretrain_step
        ),

        "pretraining_step_over_d2": (
            args.pretrain_step
            / (d * d)
        ),

        "source_task1_run": (
            str(task1_run_dir)
        ),

        "task1_step": (
            args.task1_step
        ),

        "task1_step_over_d": (
            args.task1_step / d
        ),

        "source_pretraining_metrics": (
            loaded_pretraining["metrics"]
        ),

        "source_task1_metrics": (
            loaded_task1["metrics"]
        ),

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        "d": d,
        "T": T,
        "kappa": kappa,

        "dtype": (
            pretrain_config["dtype"]
        ),

        # ----------------------------------------------------
        # Task-2 optimization
        # ----------------------------------------------------

        "lr": (
            args.lr
        ),

        "sgd_seed": (
            args.sgd_seed
        ),

        "task2_test_seed": (
            args.task2_test_seed
        ),

        "pretraining_test_seed": (
            pretraining_test_seed
        ),

        "task1_test_seed": (
            task1_test_seed
        ),

        "n_test": (
            n_test
        ),

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

        # ----------------------------------------------------
        # Starting errors
        # ----------------------------------------------------

        "pretraining_error_before_task1": (
            pretraining_error_before_task1
        ),

        "pretraining_error_before_task2": (
            pretraining_error_before_task2
        ),

        "task1_error_before_task2": (
            task1_error_before_task2
        ),

        "task2_error_before_task2": (
            task2_error_before_task2
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

    # ========================================================
    # Print setup
    # ========================================================

    print()
    print(
        "=============================================="
    )
    print(
        "Task-2 fine-tuning"
    )
    print(
        "=============================================="
    )

    print(
        f"source pretrain : "
        f"{pretrain_run_dir}"
    )

    print(
        f"pretrain step   : "
        f"{args.pretrain_step:,}"
    )

    print(
        f"pretrain t/d²   : "
        f"{args.pretrain_step / (d*d):.6f}"
    )

    print(
        f"source Task 1   : "
        f"{task1_run_dir}"
    )

    print(
        f"Task-1 step     : "
        f"{args.task1_step:,}"
    )

    print(
        f"Task-1 t/d      : "
        f"{args.task1_step / d:.3f}"
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
        f"Task-2 lr       : "
        f"{args.lr}"
    )

    print(
        f"Task-2 SGD seed : "
        f"{args.sgd_seed}"
    )

    print(
        f"target steps    : "
        f"{args.target_steps:,}"
    )

    print(
        f"target t/d      : "
        f"{args.target_steps / d:.3f}"
    )

    print(
        f"eval every late : "
        f"{eval_every}"
    )

    print(
        f"checkpoints     : "
        f"{checkpoint_steps}"
    )

    print(
        f"run directory   : "
        f"{run_dir}"
    )

    print()

    # ========================================================
    # Initial Task-2 evaluation
    #
    # This is exactly the selected Task-1 checkpoint.
    # ========================================================

    metric = evaluate(
        model=model,
        teachers=teachers,
        pretraining_test=pretraining_test,
        task1_test=task1_test,
        task2_test=task2_test,
        pretraining_step=args.pretrain_step,
        task1_step=args.task1_step,
        task2_step=0,
        pretraining_error_before_task1=(
            pretraining_error_before_task1
        ),
        pretraining_error_before_task2=(
            pretraining_error_before_task2
        ),
        task1_error_before_task2=(
            task1_error_before_task2
        ),
    )

    append_metric(
        metrics_path,
        metric,
    )

    print(
        f"[step 0] "
        f"task1="
        f"{metric['task1_error']:.6e} | "
        f"task2="
        f"{metric['task2_error']:.6e} | "
        f"F1="
        f"{metric['forgetting_task1']:.6e} | "
        f"m1²="
        f"{metric['squared_overlap_w1']:.6f} | "
        f"m2²="
        f"{metric['squared_overlap_w2']:.6f} | "
        f"||w||²/d="
        f"{metric['w_norm_sq_over_d']:.6f}"
    )

    # ========================================================
    # Online Task-2 SGD
    # ========================================================

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

        with torch.no_grad():

            y = teacher_forward(
                X,
                teachers.S2_star,
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
            step
            == args.target_steps
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
                task2_test=task2_test,
                pretraining_step=args.pretrain_step,
                task1_step=args.task1_step,
                task2_step=step,
                pretraining_error_before_task1=(
                    pretraining_error_before_task1
                ),
                pretraining_error_before_task2=(
                    pretraining_error_before_task2
                ),
                task1_error_before_task2=(
                    task1_error_before_task2
                ),
            )

            append_metric(
                metrics_path,
                metric,
            )

            print(
                f"[step {step:8d}] "
                f"t/d={step / d:8.3f} | "
                f"task1="
                f"{metric['task1_error']:.6e} | "
                f"task2="
                f"{metric['task2_error']:.6e} | "
                f"F1="
                f"{metric['forgetting_task1']:.6e} | "
                f"m1²="
                f"{metric['squared_overlap_w1']:.6f} | "
                f"m2²="
                f"{metric['squared_overlap_w2']:.6f} | "
                f"eps_w1="
                f"{metric['adapter_reconstruction_error_task1']:.6e} | "
                f"eps_w2="
                f"{metric['adapter_reconstruction_error_task2']:.6e}"
            )

        if is_checkpoint_step:

            checkpoint_dir = (
                save_task2_checkpoint(
                    checkpoint_root=checkpoint_root,
                    task2_step=step,
                    model=model,
                    optimizer=optimizer,
                    generator=generator,
                    metric=metric,
                )
            )

            print(
                f"Saved checkpoint: "
                f"{checkpoint_dir}"
            )

    print()
    print(
        "Task-2 fine-tuning completed."
    )
    print(
        f"Metrics: {metrics_path}"
    )


if __name__ == "__main__":
    main()