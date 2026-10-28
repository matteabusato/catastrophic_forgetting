#!/usr/bin/env python3
"""Train or extend one Task-1 student; see --help for all exposed settings."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from overparam.src.phases import task1_main


if __name__ == "__main__":
    task1_main()
