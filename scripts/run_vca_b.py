#!/usr/bin/env python3
"""Run Experiment 2b: VCA with explicit trade-off prompting (VCA-b)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.vca_b import main


if __name__ == "__main__":
    main()
