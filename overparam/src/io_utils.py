"""Strict JSON, flushed CSV logs and portable tensor checkpoints without overwrites."""

import csv
import json
from pathlib import Path
import torch


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class MetricWriter:
    def __init__(self, path: Path):
        self.handle = path.open("x", newline="")
        self.writer = None

    def __call__(self, row: dict):
        if self.writer is None:
            self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.handle.flush()

    def close(self):
        self.handle.close()


def save_checkpoint(path: Path, model, stream, step: int, phase: str) -> None:
    # Plain SGD has no momentum state; stream state supports future continuation.
    torch.save(dict(W=model.W.detach().cpu(), generator_state=stream.generator.get_state(),
                    step=step, phase=phase, d=model.d, P=model.P), path)
