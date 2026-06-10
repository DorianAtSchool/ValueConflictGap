"""Experiment 2b: VCA with explicit trade-off prompting (VCA-b)."""

from __future__ import annotations

import argparse

from .common import run_pipeline_script
from .vca import build_parser as build_vca_parser, command_args


def build_parser() -> argparse.ArgumentParser:
    parser = build_vca_parser()
    parser.description = __doc__
    return parser


def run(args: argparse.Namespace) -> None:
    run_pipeline_script(
        "run_scenario_value_action_gap_experiment.py",
        command_args(args, explicit_tradeoff=True),
        results_dir="results/vca_b",
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
