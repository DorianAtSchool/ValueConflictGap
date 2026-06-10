#!/usr/bin/env python3
"""Build paper plots from existing experiment result runs."""

from __future__ import annotations

import argparse
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper_plots"))
    parser.add_argument("--ci-method", choices=["normal", "wilson"], default="wilson")
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise SystemExit(
        "Paper plot regeneration has not been ported into the clean src/ layout yet. "
        "Experiment runners are self-contained under src/experiments/."
    )


if __name__ == "__main__":
    main()
