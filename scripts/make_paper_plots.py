#!/usr/bin/env python3
"""Build paper plots from existing experiment result runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper_plots"))
    parser.add_argument("--ci-method", choices=["normal", "wilson"], default="wilson")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline" / "poster_value_action_gap_plots.py"),
        "--ci-method",
        args.ci_method,
        "--output-dir",
        str(args.output_dir),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
