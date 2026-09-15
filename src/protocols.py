#!/usr/bin/env python3
"""
Training protocols:
1) joint
   - total length: N_large + N_small
   - every SGD step draws task S or S' with probability 1/2
   - both W and w are trainable throughout

2) W_then_w
   - phase 1: N_large steps, mixed S/S' data, train W only
   - phase 2: N_small steps, mixed S/S' data, train w only
   - both parameter blocks remain part of the forward model throughout;
     only requires_grad changes

3) w_then_W
   - phase 1: N_small steps, mixed S/S' data, train w only
   - phase 2: N_large steps, mixed S/S' data, train W only
   - both parameter blocks remain part of the forward model throughout

4) matched_two_stage
   - phase 1: N_large steps, task S only, train W only, LoRA switched off
   - phase 2: N_small steps, task S' only, freeze W, train w, LoRA switched on
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterator, Literal


ExperimentName = Literal[
    "joint",
    "W_then_w",
    "w_then_W",
    "matched_two_stage",
]

TaskName = Literal["S", "Sprime"]


@dataclass(frozen=True)
class ProtocolConfig:
    experiment: ExperimentName
    d: int = 200
    alpha: float = 2.0
    alpha_prime: float = 2.0

    # Probability of drawing a fine-tuning-task sample S' in the mixed
    # protocols. p=0.5 implements the setting discussed with the supervisor.
    p_finetune: float = 0.5

    # Kept separate from the model/data seed so the task ordering can be
    # changed independently of W_*, w_*, W, w, and Gaussian inputs.
    schedule_seed: int = 0

    def validate(self) -> None:
        if self.experiment not in (
            "joint",
            "W_then_w",
            "w_then_W",
            "matched_two_stage",
        ):
            raise ValueError(
                f"Unknown experiment {self.experiment!r}. "
                "Expected one of: joint, W_then_w, w_then_W, "
                "matched_two_stage."
            )

        if self.d <= 0:
            raise ValueError("d must be positive.")

        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")

        if self.alpha_prime <= 0:
            raise ValueError("alpha_prime must be positive.")

        if not (0.0 <= self.p_finetune <= 1.0):
            raise ValueError("p_finetune must lie in [0, 1].")

    @property
    def n_large(self) -> int:
        """N_large = round(alpha d^2)."""
        n = int(round(self.alpha * self.d * self.d))
        if n <= 0:
            raise ValueError("alpha*d^2 rounded to a non-positive number.")
        return n

    @property
    def n_small(self) -> int:
        """N_small = round(alpha_prime d)."""
        n = int(round(self.alpha_prime * self.d))
        if n <= 0:
            raise ValueError("alpha_prime*d rounded to a non-positive number.")
        return n

    @property
    def total_steps(self) -> int:
        """All four protocols use the same total number of SGD updates."""
        return self.n_large + self.n_small


@dataclass(frozen=True)
class StepInstruction:
    step: int
    phase: str
    phase_step: int
    task: TaskName
    train_W: bool
    train_w: bool
    include_lora: bool


class Protocol:
    def __init__(self, config: ProtocolConfig) -> None:
        config.validate()
        self.config = config

    def _draw_mixed_task(self, rng: Random) -> TaskName:
        return "Sprime" if rng.random() < self.config.p_finetune else "S"

    def __len__(self) -> int:
        return self.config.total_steps

    def __iter__(self) -> Iterator[StepInstruction]:
        rng = Random(self.config.schedule_seed)

        experiment = self.config.experiment

        if experiment == "joint":
            yield from self._joint(rng)
            return

        if experiment == "W_then_w":
            yield from self._W_then_w(rng)
            return

        if experiment == "w_then_W":
            yield from self._w_then_W(rng)
            return

        if experiment == "matched_two_stage":
            yield from self._matched_two_stage()
            return

        # Config validation should make this unreachable.
        raise RuntimeError(f"Unhandled experiment: {experiment}")

    def _joint(self, rng: Random) -> Iterator[StepInstruction]:
        for step in range(1, self.config.total_steps + 1):
            yield StepInstruction(
                step=step,
                phase="joint",
                phase_step=step,
                task=self._draw_mixed_task(rng),
                train_W=True,
                train_w=True,
                include_lora=True,
            )

    def _W_then_w(self, rng: Random) -> Iterator[StepInstruction]:
        # Phase 1: extensive-rank block only.
        for phase_step in range(1, self.config.n_large + 1):
            yield StepInstruction(
                step=phase_step,
                phase="train_W",
                phase_step=phase_step,
                task=self._draw_mixed_task(rng),
                train_W=True,
                train_w=False,
                include_lora=True,
            )

        # Phase 2: LoRA block only.
        offset = self.config.n_large
        for phase_step in range(1, self.config.n_small + 1):
            yield StepInstruction(
                step=offset + phase_step,
                phase="train_w",
                phase_step=phase_step,
                task=self._draw_mixed_task(rng),
                train_W=False,
                train_w=True,
                include_lora=True,
            )

    def _w_then_W(self, rng: Random) -> Iterator[StepInstruction]:
        # Phase 1: LoRA block only.
        for phase_step in range(1, self.config.n_small + 1):
            yield StepInstruction(
                step=phase_step,
                phase="train_w",
                phase_step=phase_step,
                task=self._draw_mixed_task(rng),
                train_W=False,
                train_w=True,
                include_lora=True,
            )

        # Phase 2: extensive-rank block only.
        offset = self.config.n_small
        for phase_step in range(1, self.config.n_large + 1):
            yield StepInstruction(
                step=offset + phase_step,
                phase="train_W",
                phase_step=phase_step,
                task=self._draw_mixed_task(rng),
                train_W=True,
                train_w=False,
                include_lora=True,
            )

    def _matched_two_stage(self) -> Iterator[StepInstruction]:
        # Paper-faithful pre-training:
        #   samples from S only
        #   y_hat(X; W, 0)
        #   optimize W only
        for phase_step in range(1, self.config.n_large + 1):
            yield StepInstruction(
                step=phase_step,
                phase="pretrain_W_on_S",
                phase_step=phase_step,
                task="S",
                train_W=True,
                train_w=False,
                include_lora=False,
            )

        # Fine-tuning:
        #   samples from S' only
        #   freeze W
        #   optimize w in the full W + ww^T architecture
        offset = self.config.n_large
        for phase_step in range(1, self.config.n_small + 1):
            yield StepInstruction(
                step=offset + phase_step,
                phase="finetune_w_on_Sprime",
                phase_step=phase_step,
                task="Sprime",
                train_W=False,
                train_w=True,
                include_lora=True,
            )


def make_protocol(
    experiment: ExperimentName,
    *,
    d: int = 200,
    alpha: float = 2.0,
    alpha_prime: float = 2.0,
    p_finetune: float = 0.5,
    schedule_seed: int = 0,
) -> Protocol:
    return Protocol(
        ProtocolConfig(
            experiment=experiment,
            d=d,
            alpha=alpha,
            alpha_prime=alpha_prime,
            p_finetune=p_finetune,
            schedule_seed=schedule_seed,
        )
    )


def protocol_summary(config: ProtocolConfig) -> dict[str, int | float | str]:
    config.validate()
    return {
        "experiment": config.experiment,
        "d": config.d,
        "alpha": config.alpha,
        "alpha_prime": config.alpha_prime,
        "p_finetune": config.p_finetune,
        "schedule_seed": config.schedule_seed,
        "n_large": config.n_large,
        "n_small": config.n_small,
        "total_steps": config.total_steps,
    }


def _sanity_check() -> None:
    d = 10
    alpha = 1.0
    alpha_prime = 1.0
    expected_large = 100
    expected_small = 10
    expected_total = 110

    experiments: tuple[ExperimentName, ...] = (
        "joint",
        "W_then_w",
        "w_then_W",
        "matched_two_stage",
    )

    for name in experiments:
        config = ProtocolConfig(
            experiment=name,
            d=d,
            alpha=alpha,
            alpha_prime=alpha_prime,
            schedule_seed=123,
        )
        protocol = Protocol(config)
        steps = list(protocol)

        assert config.n_large == expected_large
        assert config.n_small == expected_small
        assert len(steps) == expected_total
        assert steps[0].step == 1
        assert steps[-1].step == expected_total

        print(
            f"{name:18s} | total={len(steps):3d} | "
            f"first=({steps[0].phase}, {steps[0].task}, "
            f"W={steps[0].train_W}, w={steps[0].train_w}, "
            f"LoRA={steps[0].include_lora}) | "
            f"last=({steps[-1].phase}, {steps[-1].task}, "
            f"W={steps[-1].train_W}, w={steps[-1].train_w}, "
            f"LoRA={steps[-1].include_lora})"
        )

    # Exact matched schedule.
    matched = list(
        make_protocol(
            "matched_two_stage",
            d=d,
            alpha=alpha,
            alpha_prime=alpha_prime,
        )
    )
    assert all(s.task == "S" for s in matched[:expected_large])
    assert all(not s.include_lora for s in matched[:expected_large])
    assert all(s.train_W and not s.train_w for s in matched[:expected_large])

    assert all(s.task == "Sprime" for s in matched[expected_large:])
    assert all(s.include_lora for s in matched[expected_large:])
    assert all(not s.train_W and s.train_w for s in matched[expected_large:])

    # Mixed-task ordering is reproducible.
    p1 = list(
        make_protocol(
            "joint",
            d=d,
            alpha=alpha,
            alpha_prime=alpha_prime,
            schedule_seed=7,
        )
    )
    p2 = list(
        make_protocol(
            "joint",
            d=d,
            alpha=alpha,
            alpha_prime=alpha_prime,
            schedule_seed=7,
        )
    )
    assert [s.task for s in p1] == [s.task for s in p2]

    print("All protocol sanity checks passed.")


if __name__ == "__main__":
    _sanity_check()