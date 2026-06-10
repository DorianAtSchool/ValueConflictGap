#!/usr/bin/env python3
"""Run Experiment 3: value-conflict preference-to-action (VCP)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.vcp import main


if __name__ == "__main__":
    main()
