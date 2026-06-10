"""Shared helpers for clean paper experiment CLIs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = [
    "gpt-4o-mini",
    "gpt-5-mini",
    "gemma-2-9b-it",
    "qwen-2.5-7b-instruct",
    "tulu-3-sft",
]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--num-scenarios", type=int, default=500, help="Number of sampled scenarios per task.")
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--openai-api-key", type=str, default=None)
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for local Hugging Face models.")
    parser.add_argument("--log-level", default="INFO")


def run_experiment_module(module: str, args: list[str], *, results_dir: Path | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if results_dir is not None:
        env["PERSONA_DRIFTING_RESULTS_DIR"] = str(PROJECT_ROOT / results_dir)
    cmd = [sys.executable, "-m", module, *args]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def append_if_present(cmd: list[str], flag: str, value: str | None) -> None:
    if value:
        cmd.extend([flag, value])


def append_bool(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)
