"""Scientific invariants of the experiment; no external data or network needed."""

from copy import deepcopy
from dataclasses import replace
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from warm_start.src.data import Dataset, generate_dataset, make_split
from warm_start.src.experiment import ExperimentConfig, run_experiment, summarize_pairs
from warm_start.src.io_utils import save_result
from warm_start.src.losses import output_mse
from warm_start.src.model import Student, Teacher, centered_logits
from warm_start.src.observables import evaluate
from warm_start.src.training import TrainingConfig, train


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gaussian_centering_and_gradient(self):
        generator = torch.Generator().manual_seed(42)
        X = torch.randn(30000, 3, 4, generator=generator, dtype=torch.float64)
        W = torch.randn(5, 4, generator=generator, dtype=torch.float64)
        S = W.T @ W / 5**0.5
        self.assertLess(centered_logits(X, S).mean(0).abs().max().item(), 0.03)
        W.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda factor: centered_logits(X[:2], factor.T @ factor / 5**0.5), (W,)))
        # Pure diagonal sequence tests the trace term's exact derivative.
        identity = torch.eye(4, dtype=torch.float64)
        logits = centered_logits(identity, S)
        torch.testing.assert_close(logits, (S - S.trace() * identity) / 4)

    def test_teacher_match_and_loss_normalization(self):
        teacher = Teacher(4, 5, seed=4)
        student = Student(4, 5, seed=5)
        student.load_state_dict(teacher.state_dict())
        data = generate_dataset(teacher, 7, 3, seed=6)
        self.assertFalse(teacher.W.requires_grad)
        self.assertFalse(data.y.requires_grad)
        self.assertEqual(output_mse(student(data.X), data.y).item(), 0)
        self.assertAlmostEqual(evaluate(student, data, 3)["attention_kl"], 0, places=14)
        self.assertEqual(output_mse(torch.ones(2, 3, 4), torch.zeros(2, 3, 4)), 1)
        torch.testing.assert_close(student.attention(data.X).sum(-1),
                                   torch.ones(7, 3, dtype=torch.float64))

    def test_split_retains_all_old_data(self):
        teacher = Teacher(4, 5)
        split = make_split(teacher, 11, 9, 3, 0.5, 2, 3, 4)
        indices = torch.cat((split.indices_A, split.indices_B))
        torch.testing.assert_close(indices.sort().values, torch.arange(11))
        torch.testing.assert_close(split.A.X, split.full.X[split.indices_A])
        self.assertFalse(torch.equal(split.full.X[:9], split.test.X))

    def test_evaluation_weighting_and_no_test_leakage(self):
        teacher, initial = Teacher(4, 5), Student(4, 5, seed=1)
        data = generate_dataset(teacher, 7, 3, 2)
        first, second = deepcopy(initial), deepcopy(initial)
        config = TrainingConfig(max_epochs=5, eval_every=1, batch_size=3)
        changed = Dataset(data.X, data.y + 100, data.attention)
        one = train(first, data, config, 3, "one", {"test": data})
        two = train(second, data, config, 3, "two", {"test": changed})
        torch.testing.assert_close(first.W, second.W, rtol=0, atol=0)
        self.assertEqual(one.stop_reason, two.stop_reason)
        self.assertLess(one.history[-1]["train_mse"], one.history[0]["train_mse"])
        self.assertAlmostEqual(evaluate(first, data, 3)["mse"],
                               evaluate(first, data, 7)["mse"], places=14)

    def test_paired_protocol_reproducibility_and_persistence(self):
        training = TrainingConfig(max_epochs=3, eval_every=1, batch_size=4)
        config = ExperimentConfig(d=4, T=3, P=5, P_teacher=5,
                                  n_train=10, n_test=9, pretraining=training,
                                  final_training=training)
        result, repeat = run_experiment(config), run_experiment(config)
        torch.testing.assert_close(result.warm_model.W, repeat.warm_model.W, rtol=0, atol=0)
        self.assertEqual(result.summary["delta_gen"], repeat.summary["delta_gen"])
        self.assertEqual(result.runs["warm"].examples_seen, 30)
        self.assertEqual(result.runs["fresh"].examples_seen, 30)
        self.assertEqual(result.runs["pretraining"].examples_seen, 15)
        self.assertEqual(result.runs["warm"].history[0]["train_mse"],
                          result.runs["pretraining"].history[-1]["full_mse"])
        # Fresh protocol equals a standalone run from W0 with final-phase seed.
        standalone = Student(config.d, config.P)
        with torch.no_grad():
            standalone.W.copy_(result.initial_W)
        train(standalone, result.data.full, training, result.seeds["final_order"], "fresh")
        torch.testing.assert_close(standalone.W, result.fresh_model.W, rtol=0, atol=0)
        # Warm phase starts with a fresh optimizer, not the first phase's moments.
        with torch.no_grad():
            standalone.W.copy_(result.pretrained_W)
        train(standalone, result.data.full, training, result.seeds["final_order"], "warm")
        torch.testing.assert_close(standalone.W, result.warm_model.W, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            save_result(result, path)
            weights = torch.load(path / "weights.pt", weights_only=True)
            torch.testing.assert_close(weights["warm_final_W"], result.warm_model.W)
            self.assertTrue((path / "warm_history.csv").is_file())
            with self.assertRaises(FileExistsError):
                save_result(result, path)

    def test_stopping_and_aggregation(self):
        teacher = Teacher(4, 5)
        data = generate_dataset(teacher, 4, 3, 1)
        config = TrainingConfig(max_epochs=10, eval_every=1)
        exact = Student(4, 5)
        exact.load_state_dict(teacher.state_dict())
        run = train(exact, data, config, 2, "exact")
        self.assertEqual(run.stop_reason, "loss_tolerance")
        self.assertEqual(run.epochs, 1)
        pair = dict(seed=0, delta_gen=0.1, comparable_training_error=True,
                    both_reached_loss_tolerance=False)
        aggregate = summarize_pairs([pair, {**pair, "seed": 1, "delta_gen": 0.3}])
        self.assertAlmostEqual(aggregate["mean_delta_gen"], 0.2)
        self.assertAlmostEqual(aggregate["se_delta_gen"], 0.1)
        self.assertIsNone(summarize_pairs([pair])["se_delta_gen"])
        with self.assertRaises(ValueError):
            summarize_pairs([pair, pair])
        with self.assertRaises(ValueError):
            replace(config, lr=float("nan"))

    def test_exact_update_budgets_and_step_callbacks(self):
        pretraining = TrainingConfig(max_steps=103, log_every_steps=100,
                                    batch_size=4, early_stopping=False)
        final = replace(pretraining, max_steps=205)
        config = ExperimentConfig(d=4, T=3, P=5, P_teacher=5, n_train=10, n_test=9,
                                  pretraining=pretraining, final_training=final)
        rows = []
        result = run_experiment(config, metric_callback=rows.append)
        expected = {"pretraining": [0, 100, 103],
                    "warm": [0, 100, 200, 205], "fresh": [0, 100, 200, 205]}
        for name, steps in expected.items():
            run = result.runs[name]
            self.assertEqual([row["updates"] for row in run.history], steps)
            self.assertEqual(run.updates, steps[-1])
            self.assertEqual(run.stop_reason, "max_steps")
        self.assertEqual(len(rows), 11)
        self.assertEqual(result.summary["warm_total_updates"], 308)
        for row in rows:
            self.assertEqual(row["train_loss"], row["train_error"])
            self.assertEqual(row["test_loss"], row["test_error"])

    def test_shell_runner_from_another_directory(self):
        launcher = Path(__file__).resolve().parents[1] / "scripts" / "run_warm_start.sh"
        with tempfile.TemporaryDirectory() as tmp:
            command = ["bash", str(launcher), "--output", "run", "--seed", "7",
                       "--d", "4", "--tokens", "3", "--rank", "5",
                       "--teacher-rank", "5", "--n-train", "10", "--n-test", "11",
                       "--pretrain-steps", "3", "--train-steps", "5", "--log-every", "2"]
            completed = subprocess.run(command, cwd=tmp, check=True, capture_output=True,
                                       text=True, env={**os.environ,
                                                      "WARM_START_PYTHON": sys.executable})
            output = Path(tmp) / "run"
            self.assertIn("train_loss=", completed.stdout)
            with (output / "metrics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            for phase, expected in {"pretraining_A": [0, 2, 3],
                                    "warm_full": [0, 2, 4, 5],
                                    "fresh_full": [0, 2, 4, 5]}.items():
                self.assertEqual([int(row["phase_step"]) for row in rows
                                  if row["phase"] == phase], expected)
            warm = [row for row in rows if row["phase"] == "warm_full"]
            self.assertEqual(int(warm[-1]["regime_step"]), 8)
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "complete")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["seed"], 7)
            self.assertEqual(summary["warm_final_updates"], 5)
            self.assertEqual(summary["fresh_updates"], 5)
            self.assertTrue((output / "seed_7" / "datasets.pt").is_file())
            # A second invocation must fail without truncating the first run.
            before = (output / "metrics.csv").read_bytes()
            duplicate = subprocess.run(command, cwd=tmp, capture_output=True,
                                       env={**os.environ, "WARM_START_PYTHON": sys.executable})
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertEqual((output / "metrics.csv").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
