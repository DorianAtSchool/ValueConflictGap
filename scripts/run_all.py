#!/usr/bin/env python3
"""Run any subset of the paper experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import sva, vca, vca_b, vcp
from src.experiments.common import DEFAULT_MODELS


EXPERIMENTS = {
    "sva": sva.run,
    "vca": vca.run,
    "vca_b": vca_b.run,
    "vcp": vcp.run,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--num-scenarios", type=int, default=500)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--value-set", default="personalprotective")
    parser.add_argument("--value-text", choices=["label", "description", "label_description"], default="label")
    parser.add_argument("--sampling-group", choices=["schwartz_group", "super_group", "value", "topic", "country"], default="schwartz_group")
    parser.add_argument("--openai-api-key", type=str, default=None)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for experiment in args.experiments:
        EXPERIMENTS[experiment](args)


if __name__ == "__main__":
    main()
