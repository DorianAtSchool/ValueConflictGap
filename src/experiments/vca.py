"""Experiment 2: value-conflict agreement-to-action (VCA)."""

from __future__ import annotations

import argparse

from .common import add_common_args, append_bool, append_if_present, run_pipeline_script


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--value-set", default="personalprotective")
    parser.add_argument("--value-text", choices=["label", "description", "label_description"], default="label")
    return parser


def command_args(args: argparse.Namespace, *, explicit_tradeoff: bool = False) -> list[str]:
    cmd = [
        "--models",
        *args.models,
        "--value-sets",
        args.value_set,
        "--num-scenarios",
        str(args.num_scenarios),
        "--mode",
        "mcq",
        "--value-text",
        args.value_text,
        "--log-level",
        args.log_level,
    ]
    append_if_present(cmd, "--openai-api-key", args.openai_api_key)
    append_bool(cmd, "--use-vllm", args.use_vllm)
    append_bool(cmd, "--ask-prioritize-over-others", explicit_tradeoff)
    return cmd


def run(args: argparse.Namespace) -> None:
    run_pipeline_script("run_scenario_value_action_gap_experiment.py", command_args(args), results_dir="results/vca")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
