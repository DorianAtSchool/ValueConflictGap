#!/usr/bin/env python3
"""Run Experiment 2: value-conflict agreement-to-action (VCA)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.vca import main


if __name__ == "__main__":
    main()
