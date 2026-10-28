"""Scientific invariants and a small persisted sequential-width experiment."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import torch
from overparam.src.config import ExperimentConfig
from overparam.src.data import GaussianStream, label_inputs
from overparam.src.experiment import run_experiment
from overparam.src.model import Student, Teacher, centered_logits
from overparam.src.observables import evaluate, forgetting
from overparam.src.training import train_phase


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_model_rank_centering_and_exact_fit(self):
        model = Student(4, 2, 1)
        self.assertEqual(torch.linalg.matrix_rank(model.S).item(), 2)
        X = torch.eye(4, dtype=torch.float64)
        expected = (model.S - model.S.trace() * X) / 4
        torch.testing.assert_close(centered_logits(X, model.S), expected)
        teacher = Teacher(4, 2, 1)
        data = label_inputs(X.unsqueeze(0), teacher)
        self.assertLess(evaluate(model, data)["mse"], 1e-25)
        self.assertFalse(teacher.W.requires_grad)
        model(X).sum().backward()
        self.assertTrue(torch.isfinite(model.W.grad).all())

    def test_matched_S_width_compensation(self):
        base = Student(4, 4, 4)
        wide = Student(4, 8, 5)
        with torch.no_grad():
            wide.W.copy_(torch.cat([base.W, base.W], dim=1) / 2**0.25)
        teacher = Teacher(4, 4, 6)
        X = GaussianStream(9, 3, 4, torch.float64, "cpu").sample(2)
        for model, lr in ((base, .01), (wide, .01 * 2**.5)):
            optimizer = torch.optim.SGD([model.W], lr=lr)
            for _ in range(3):
                optimizer.zero_grad()
                (model(X) - teacher(X)).square().mean().backward()
                optimizer.step()
        torch.testing.assert_close(base.S, wide.S)

    def test_stream_evaluation_and_forgetting(self):
        stream1 = GaussianStream(10, 3, 4, torch.float64, "cpu")
        stream2 = GaussianStream(10, 3, 4, torch.float64, "cpu")
        X = stream1.sample(7)
        torch.testing.assert_close(X, stream2.sample(7))
        data = label_inputs(X, Teacher(4, 4, 1))
        model = Student(4, 4, 2)
        for key, value in evaluate(model, data, 3).items():
            self.assertAlmostEqual(value, evaluate(model, data, 7)[key], places=12)
        self.assertIsNone(forgetting(2, 1, 1)["forgetting_relative"])
        self.assertEqual(forgetting(3, 1, 2)["forgetting_relative"], 2)

    def test_observation_callback_does_not_change_training(self):
        teacher = Teacher(4, 4, 6)
        a, b = Student(4, 4, 4), Student(4, 4, 4)
        for model, every in ((a, 1), (b, 3)):
            train_phase(model, teacher, GaussianStream(8, 3, 4, torch.float64, "cpu"),
                        5, 2, .01, every, lambda step, loss: dict(step=step))
        torch.testing.assert_close(a.W, b.W)

    def test_saved_sequential_protocol_and_task1_only(self):
        config = ExperimentConfig(d=4, tokens=3, teacher_rank=4, kappas=(.5, 1, 2),
                                  seeds=(0,), task1_steps=3, task2_steps=2,
                                  n_test=7, n_probe=3, eval_every=2, eval_batch_size=3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "full"
            summaries = run_experiment(config, root)
            self.assertEqual(len(summaries), 3)
            self.assertEqual(json.loads((root / "status.json").read_text())["status"], "complete")
            seed_dir = root / "seed_0"
            teachers = torch.load(seed_dir / "teachers.pt", weights_only=True)
            self.assertFalse(torch.equal(teachers["task1"], teachers["task2"]))
            for P in (2, 4, 8):
                directory = seed_dir / f"P_{P}"
                with (directory / "task1_metrics.csv").open() as f:
                    task1 = list(csv.DictReader(f))
                with (directory / "task2_metrics.csv").open() as f:
                    task2 = list(csv.DictReader(f))
                self.assertEqual([int(r["phase_step"]) for r in task1], [0, 2, 3])
                self.assertEqual(task1[-1]["test_task1_mse"], task2[0]["test_task1_mse"])
                self.assertEqual(float(task2[0]["forgetting_abs"]), 0)
                checkpoint = torch.load(directory / "task1_checkpoint.pt", weights_only=True)
                model = Student(4, P, 123)
                with torch.no_grad():
                    model.W.copy_(checkpoint["W"])
                # Replay Task 2 from the saved switch state and input RNG.
                stream = GaussianStream(0, 3, 4, torch.float64, "cpu")
                seeds = json.loads((seed_dir / "seeds.json").read_text())
                stream.generator.manual_seed(seeds["train_task2"])
                teacher = Teacher(4, 4, 0)
                with torch.no_grad():
                    teacher.W.copy_(teachers["task2"])
                train_phase(model, teacher, stream, 2, 1, .01, 2,
                            lambda step, loss: dict(step=step))
                final = torch.load(directory / "task2_checkpoint.pt", weights_only=True)
                torch.testing.assert_close(model.W, final["W"])
            with self.assertRaises(FileExistsError):
                run_experiment(config, root)
            only = Path(tmp) / "only"
            run_experiment(replace(config, kappas=(1,), task2_steps=0), only)
            self.assertFalse((only / "seed_0/P_4/task2_checkpoint.pt").exists())

    def test_invalid_config(self):
        for kwargs in (dict(kappas=(0,)), dict(kappas=(1, 1)), dict(lr=float("nan")),
                       dict(task1_steps=0), dict(seeds=())):
            with self.assertRaises(ValueError):
                ExperimentConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
