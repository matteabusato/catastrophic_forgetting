"""Recovery must reproduce real updates, including permutation cursor and Adam state."""

from copy import deepcopy
import csv
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from warm_start.src.data import generate_dataset
from warm_start.src.experiment import ExperimentConfig, run_experiment
from warm_start.src.model import Student, Teacher
from warm_start.src.observables import evaluate
from warm_start.src.recovery import FreshTrainer, atomic_torch_save, evaluate_train_partitions
from warm_start.src.training import TrainingConfig, train


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_replay_and_checkpoint_match_original_across_epochs(self):
        data = generate_dataset(Teacher(4, 5), 10, 3, 2)
        for optimizer in ("sgd", "adam"):
            for cutoff in (2, 3, 5):  # includes mid-epoch and exact epoch boundary
                with self.subTest(optimizer=optimizer, cutoff=cutoff):
                    initial = Student(4, 5, seed=8)
                    uninterrupted = deepcopy(initial)
                    config = TrainingConfig(max_steps=8, log_every_steps=2,
                                            batch_size=4, optimizer=optimizer,
                                            early_stopping=False)
                    train(uninterrupted, data, config, 11, "fresh_full")
                    replay = FreshTrainer(deepcopy(initial), data, optimizer, config.lr, 4, 11)
                    for _ in range(cutoff):
                        replay.update()
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / "latest.pt"
                        atomic_torch_save(replay.state_dict(), path)
                        restored = FreshTrainer(deepcopy(initial), data, optimizer, config.lr, 4, 11)
                        restored.load_state_dict(torch.load(path, weights_only=True))
                    while restored.step < 8:
                        restored.update()
                    torch.testing.assert_close(restored.model.W, uninterrupted.W, rtol=0, atol=0)

    def test_partitioned_evaluation_matches_original(self):
        data = generate_dataset(Teacher(4, 5), 11, 3, 2)
        model = Student(4, 5, seed=8)
        a = torch.tensor([0, 3, 4, 7, 9])
        b = torch.tensor([1, 2, 5, 6, 8, 10])
        actual = evaluate_train_partitions(model, data, a, 3)
        for name, subset in (("full", data), ("A", data.subset(a)), ("B", data.subset(b))):
            expected = evaluate(model, subset, 3)
            for key in expected:
                self.assertAlmostEqual(actual[name][key], expected[key], places=14)

    def test_recovery_cli_replay_then_resume(self):
        training = TrainingConfig(max_steps=7, log_every_steps=1, batch_size=4,
                                  optimizer="sgd", lr=0.01, early_stopping=False)
        config = ExperimentConfig(d=4, T=3, P=5, P_teacher=5, n_train=10, n_test=9,
                                  pretraining=training, final_training=training)
        original = run_experiment(config)
        script = Path(__file__).resolve().parents[1] / "scripts" / "recover_fresh.py"
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "config.json").write_text(json.dumps(asdict(config)))
            history = original.runs["fresh"].history[:4]  # logged through step 3
            with (directory / "metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            old_log = (directory / "metrics.csv").read_bytes()
            command = [sys.executable, str(script), "--run-dir", tmp,
                       "--checkpoint-every", "2", "--log-every", "2"]
            replay = subprocess.run(command + ["--target-steps", "5"], check=True,
                                    text=True, capture_output=True)
            self.assertIn("match the original logged values", replay.stdout)
            subprocess.run(command + ["--resume", "--target-steps", "7"],
                           check=True, capture_output=True)
            recovered = torch.load(directory / "fresh_recovery" / "fresh_final_W.pt", weights_only=True)
            self.assertEqual(recovered["step"], 7)
            torch.testing.assert_close(recovered["W"], original.fresh_model.W, rtol=0, atol=0)
            self.assertEqual(old_log, (directory / "metrics.csv").read_bytes())
            self.assertEqual(len(list((directory / "fresh_recovery").glob("metrics_*.csv"))), 2)


if __name__ == "__main__":
    unittest.main()
