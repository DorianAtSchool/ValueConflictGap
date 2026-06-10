"""Experiment 1: single-value agreement-to-action (SVA)."""

from __future__ import annotations

import argparse

from .common import add_common_args, append_bool, append_if_present, run_pipeline_script


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--sampling-group",
        choices=["schwartz_group", "super_group", "value", "topic", "country"],
        default="schwartz_group",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    cmd = [
        "--models",
        *args.models,
        "--tasks",
        "task1",
        "task2",
        "--probe-modes",
        "mcq",
        "--interaction",
        "single",
        "--task1-max-rows",
        str(args.num_scenarios),
        "--task2-max-rows",
        str(args.num_scenarios),
        "--sampling-group",
        args.sampling_group,
        "--sampling-seed",
        str(args.sampling_seed),
        "--log-level",
        args.log_level,
    ]
    append_if_present(cmd, "--openai-api-key", args.openai_api_key)
    append_bool(cmd, "--use-vllm", args.use_vllm)
    run_pipeline_script("run_via_experiment.py", cmd, results_dir="results/sva")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
