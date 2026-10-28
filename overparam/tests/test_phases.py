"""Checkpoint schedules, exact phase continuation and independent Task-2 handoff."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

import torch

from overparam.src.phases import checkpoint_steps, task1_main, task2_main


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


class PhaseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fresh = ["--d", "4", "--tokens", "3", "--kappa", "1.5", "--seed", "0",
                      "--n-test", "7", "--n-probe", "3", "--eval-every", "2",
                      "--eval-batch-size", "3", "--checkpoint-alphas", ".25", ".5",
                      "--checkpoint-every-alpha", ".25", "--print-every", "100"]

    def tearDown(self):
        self.temp.cleanup()

    def task1(self, folder, steps):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            task1_main(self.fresh + ["--target-steps", str(steps), "--run-dir", str(folder)])

    def test_schedule_and_defaults(self):
        self.assertEqual(checkpoint_steps(100, 650000, [5, 10, 20, 30, 40], 10),
                         {50000, 100000, 200000, 300000, 400000, 500000, 600000, 650000})

    def test_task1_resume_matches_uninterrupted_and_pairing(self):
        full, partial = self.root / "full", self.root / "partial"
        self.task1(full, 8)
        self.task1(partial, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            task1_main(["--resume", "--run-dir", str(partial), "--target-steps", "8"])
        for key in ("W", "generator_state"):
            torch.testing.assert_close(load(full / "checkpoints/step_000000000008.pt")[key],
                                       load(partial / "checkpoints/step_000000000008.pt")[key],
                                       rtol=0, atol=0)
        with (partial / "metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["phase_step"]) for row in rows], [0, 2, 4, 6, 8])
        wide = self.root / "wide"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            task1_main(self.fresh + ["--kappa", "2", "--target-steps", "4", "--run-dir", str(wide)])
        for name in ("task1", "task2"):
            torch.testing.assert_close(load(full / "teachers.pt")[name], load(wide / "teachers.pt")[name])
        torch.testing.assert_close(load(full / "evaluation_inputs.pt")["test"],
                                   load(wide / "evaluation_inputs.pt")["test"])
        self.assertFalse(torch.equal(load(full / "teachers.pt")["task1"],
                                     load(full / "teachers.pt")["task2"]))

    def test_task2_selected_checkpoint_and_resume(self):
        source = self.root / "source"
        self.task1(source, 8)
        before = (source / "checkpoints/step_000000000004.pt").read_bytes()
        common = ["--task1-run-dir", str(source), "--task1-step", "4"]
        a, b = self.root / "a", self.root / "b"
        task2_main(common + ["--target-steps", "8", "--run-dir", str(a)])
        task2_main(common + ["--target-steps", "4", "--run-dir", str(b)])
        task2_main(["--resume", "--run-dir", str(b), "--target-steps", "8"])
        torch.testing.assert_close(load(a / "checkpoints/step_000000000008.pt")["W"],
                                   load(b / "checkpoints/step_000000000008.pt")["W"], rtol=0, atol=0)
        switch = load(b / "checkpoints/step_000000000000.pt")
        selected = load(source / "checkpoints/step_000000000004.pt")
        torch.testing.assert_close(switch["W"], selected["W"], rtol=0, atol=0)
        self.assertEqual(switch["metrics"]["forgetting_abs"], 0.)
        self.assertEqual(switch["metrics"]["total_step"], 4)
        final = load(b / "checkpoints/step_000000000008.pt")
        self.assertEqual(final["metrics"]["forgetting_abs"],
                         final["metrics"]["test_task1_mse"] - selected["metrics"]["test_task1_mse"])
        self.assertEqual(before, (source / "checkpoints/step_000000000004.pt").read_bytes())
        self.assertEqual(json.loads((source / "config.json").read_text())["phase"], "task1")

    def test_recovery_removes_uncheckpointed_rows_and_refuses_overwrites(self):
        run = self.root / "recovery"
        self.task1(run, 4)
        with (run / "metrics.csv").open() as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        extra = dict(rows[-1], phase_step="6")
        with (run / "metrics.csv").open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(extra)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            task1_main(["--resume", "--run-dir", str(run), "--target-steps", "8"])
        self.assertEqual(len(list(run.glob("metrics_before_recovery_*.csv"))), 1)
        with (run / "metrics.csv").open() as handle:
            steps = [int(row["phase_step"]) for row in csv.DictReader(handle)]
        self.assertEqual(steps, [0, 2, 4, 6, 8])
        with self.assertRaises(FileExistsError):
            self.task1(run, 4)
        with self.assertRaises(ValueError):
            task1_main(["--resume", "--run-dir", str(run), "--lr", ".02"])

    def test_script_help_and_source_required(self):
        repo = Path(__file__).resolve().parents[2]
        for name in ("run_task1.py", "run_task2.py"):
            result = subprocess.run([sys.executable, str(repo / "overparam/scripts" / name), "--help"],
                                    capture_output=True, text=True, check=True)
            self.assertIn("--checkpoint-alphas", result.stdout)
        with self.assertRaises(ValueError):
            task2_main([])


if __name__ == "__main__":
    unittest.main()
