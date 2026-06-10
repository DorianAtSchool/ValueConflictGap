"""Experiment 3: value-conflict preference-to-action (VCP)."""

from __future__ import annotations

import argparse

from .common import add_common_args, append_bool, append_if_present, run_experiment_module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--value-set", default="personalprotective")
    parser.add_argument("--value-text", choices=["label", "description", "label_description"], default="label")
    return parser


def run(args: argparse.Namespace) -> None:
    cmd = [
        "--models",
        *args.models,
        "--value-sets",
        args.value_set,
        "--group-by",
        "pair",
        "--stances",
        "neutral",
        "--mode",
        "mcq",
        "--value-text",
        args.value_text,
        "--turn-counts",
        "0",
        "--num-scenarios",
        str(args.num_scenarios),
        "--log-level",
        args.log_level,
    ]
    append_if_present(cmd, "--openai-api-key", args.openai_api_key)
    append_bool(cmd, "--use-vllm", args.use_vllm)
    run_experiment_module("src.experiments.vcp_runner", cmd, results_dir="results/vcp")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
