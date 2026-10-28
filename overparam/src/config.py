"""Validated settings for independent teachers and a paired student-width sweep."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExperimentConfig:
    d: int = 100
    tokens: int = 5
    teacher_rank: int = 100
    kappas: tuple[float, ...] = (1, 1.2, 1.5, 2, 3, 4, 5, 10, 20)
    seeds: tuple[int, ...] = (0,)
    task1_steps: int = 400_000
    task2_steps: int = 400_000
    batch_size: int = 1
    lr: float = 0.01
    lr_scaling: str = "fixed"
    init_std: float = 1.0
    teacher_std: float = 1.0
    n_test: int = 2048
    n_probe: int = 256
    eval_every: int = 100
    eval_batch_size: int = 256
    dtype: str = "float64"
    device: str = "cpu"
    threads: int = 1

    def __post_init__(self):
        for name in ("d", "teacher_rank", "task1_steps", "batch_size", "n_test",
                     "n_probe", "eval_every", "eval_batch_size", "threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.tokens < 2 or self.task2_steps < 0:
            raise ValueError("Need tokens>=2 and task2_steps>=0 (zero means Task 1 only).")
        if not self.kappas or any(not math.isfinite(k) or k <= 0 for k in self.kappas):
            raise ValueError("kappas must be nonempty, positive and finite.")
        widths = [round(k * self.d) for k in self.kappas]
        if min(widths) < 1 or len(set(widths)) != len(widths):
            raise ValueError("kappas must give distinct positive rounded student widths.")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and distinct.")
        for name in ("lr", "init_std", "teacher_std"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.lr_scaling not in ("fixed", "sqrt_width"):
            raise ValueError("lr_scaling must be fixed or sqrt_width.")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64.")

    def learning_rate(self, P: int) -> float:
        return self.lr * (math.sqrt(P / self.d) if self.lr_scaling == "sqrt_width" else 1)
