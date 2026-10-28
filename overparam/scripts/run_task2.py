#!/usr/bin/env python3
"""Train Task 2 from a chosen Task-1 checkpoint, or extend an existing Task-2 run."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from overparam.src.phases import task2_main


if __name__ == "__main__":
    task2_main()
