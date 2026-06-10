#!/usr/bin/env python3
"""Run Experiment 1: single-value agreement-to-action (SVA)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.sva import main


if __name__ == "__main__":
    main()
