#!/usr/bin/env python3
"""RunPod entry point for distributed scenario conversation experiment.

Auto-detects GPU configuration and runs the scenario conversation experiment
with sensible multi-GPU defaults. Can be run directly on a RunPod instance
with mounted storage for results.

Usage:
    # Auto-detect vLLM and run full experiment
    python run_runpod.py --models all --value-sets all --stances all --turn-counts 5 10

    # Explicit GPU count
    python run_runpod.py --models all --num-gpus 8 --turn-counts 5 10

    # Use OpenAI simulator
    python run_runpod.py --models tulu-3-sft tulu-3-dpo \\
        --simulator openai --simulator-model gpt-4o-mini \\
        --simulator-api-key sk-...
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


def setup_logging():
    """Configure logging for RunPod execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(__name__)


log = setup_logging()


def detect_gpu_config():
    """Detect GPU configuration on RunPod."""
    n_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]

    log.info(f"Detected {n_gpus} GPUs:")
    for i, name in enumerate(gpu_names):
        props = torch.cuda.get_device_properties(i)
        total_memory_gb = props.total_memory / 1e9
        log.info(f"  GPU {i}: {name} ({total_memory_gb:.1f}GB)")

    return n_gpus


def is_vllm_available():
    """Check if vLLM is installed."""
    try:
        import vllm
        log.info(f"vLLM {vllm.__version__} detected")
        return True
    except ImportError:
        log.warning("vLLM not found. Use: pip install vllm")
        return False


DEFAULT_MODELS = [
    "tulu-3-sft",
    "tulu-3-dpo",
    "tulu-3-rlvr",
    "llama-3.1-instruct",
    "llama-3.1-base",
]
DEFAULT_VALUE_SETS = ["HHH", "personalprotective", "modelspec"]
DEFAULT_STANCES = ["neutral", "pro_v1", "pro_v2"]


def resolve_models(args) -> list[str]:
    return args.models if args.models and args.models[0].lower() != "all" else DEFAULT_MODELS


def resolve_value_sets(args) -> list[str]:
    return (
        args.value_sets
        if args.value_sets and args.value_sets[0].lower() != "all"
        else DEFAULT_VALUE_SETS
    )


def resolve_stances(args) -> list[str]:
    return args.stances if args.stances and args.stances[0].lower() != "all" else DEFAULT_STANCES


def build_experiment_command(
    args,
    use_vllm: bool,
    *,
    models: list[str],
    stances: list[str],
    skip_postprocess: bool = False,
) -> list[str]:
    """Build the run_scenario_conversation_experiment.py command."""
    script_path = Path(__file__).resolve().parent / "run_scenario_conversation_experiment.py"

    cmd = [
        "python",
        str(script_path),
    ]

    # Models
    cmd.extend(["--models"] + models)

    # Value sets
    cmd.extend(["--value-sets"] + resolve_value_sets(args))

    # Stances
    cmd.extend(["--stances"] + stances)

    # Turn counts
    if args.turn_counts:
        cmd.extend(["--turn-counts"] + list(map(str, args.turn_counts)))

    # Number of scenarios
    cmd.extend(["--num-scenarios", str(args.num_scenarios)])

    # vLLM flag
    if use_vllm:
        cmd.append("--use-vllm")

    if skip_postprocess:
        cmd.append("--skip-postprocess")

    # User simulator
    if args.simulator:
        cmd.extend(["--simulator", args.simulator])
        if args.simulator_model:
            cmd.extend(["--simulator-model", args.simulator_model])
        if args.simulator_api_key:
            cmd.extend(["--simulator-api-key", args.simulator_api_key])

    # Mode
    if args.mode:
        cmd.extend(["--mode", args.mode])
        if args.mode == "openended":
            if args.judge:
                cmd.extend(["--judge", args.judge])
            if args.judge_model:
                cmd.extend(["--judge-model", args.judge_model])
            if args.judge_api_key:
                cmd.extend(["--judge-api-key", args.judge_api_key])

    # Group by
    if args.group_by:
        cmd.extend(["--group-by", args.group_by])

    return cmd


def run_experiment(cmd: list[str], *, env: dict[str, str] | None = None, log_path: Path | None = None):
    """Execute a single experiment command."""
    log.info(f"Running: {' '.join(cmd)}")
    log.info("=" * 80)
    if log_path is None:
        result = subprocess.run(cmd, env=env)
        return result.returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return result.returncode


def shard_specs(args) -> list[tuple[str, list[str]]]:
    models = resolve_models(args)
    stances = resolve_stances(args)
    if args.shard_by == "none":
        return [("all-models_all-stances", stances)]
    if args.shard_by == "model":
        return [(model, stances) for model in models]
    return [(f"{model}_{stance}", [stance]) for model in models for stance in stances]


def run_sharded_experiments(args, use_vllm: bool, available_gpus: list[int]) -> int:
    """Run shards in parallel, one shard per GPU, then return aggregate status."""
    models = resolve_models(args)
    specs = shard_specs(args)
    max_parallel = args.max_parallel or len(available_gpus)
    max_parallel = min(max_parallel, len(available_gpus))
    logs_dir = Path(__file__).resolve().parent / "results" / "scenario_conversation" / "shard_logs"
    processes: list[tuple[subprocess.Popen, str, Path]] = []
    pending = []

    for spec_name, stances in specs:
        if args.shard_by == "none":
            shard_models = models
        elif args.shard_by == "model":
            shard_models = [spec_name]
        else:
            shard_models = [modelspec for modelspec in models if spec_name.startswith(f"{modelspec}_")]
            if len(shard_models) != 1:
                raise ValueError(f"Could not infer model for shard {spec_name}")
        pending.append((spec_name, shard_models, stances))

    failed = False
    gpu_queue = list(available_gpus[:max_parallel])

    while pending or processes:
        while pending and gpu_queue:
            gpu_id = gpu_queue.pop(0)
            spec_name, shard_models, stances = pending.pop(0)
            cmd = build_experiment_command(
                args,
                use_vllm=use_vllm,
                models=shard_models,
                stances=stances,
                skip_postprocess=True,
            )
            shard_env = os.environ.copy()
            shard_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            log_path = logs_dir / f"{spec_name}.log"
            log.info("Launching shard %s on GPU %s", spec_name, gpu_id)
            log.info("  log: %s", log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            f = open(log_path, "w")
            proc = subprocess.Popen(cmd, env=shard_env, stdout=f, stderr=subprocess.STDOUT)
            processes.append((proc, spec_name, log_path))
            setattr(proc, "_gpu_id", gpu_id)
            setattr(proc, "_log_handle", f)

        next_processes = []
        for proc, spec_name, log_path in processes:
            ret = proc.poll()
            if ret is None:
                next_processes.append((proc, spec_name, log_path))
                continue
            proc._log_handle.close()
            gpu_queue.append(proc._gpu_id)
            if ret != 0:
                failed = True
                log.error("Shard failed: %s (exit=%s). See %s", spec_name, ret, log_path)
            else:
                log.info("Shard completed: %s", spec_name)
        processes = next_processes
        if pending or processes:
            import time
            time.sleep(2)

    return 1 if failed else 0


def collect_results_for_run(args) -> list[dict]:
    """Load the expected per-condition result JSONs for one combined summary."""
    results = []
    runs_dir = Path(__file__).resolve().parent / "results" / "scenario_conversation" / "runs"
    for model in resolve_models(args):
        for value_set in resolve_value_sets(args):
            for turns in args.turn_counts:
                for stance in resolve_stances(args):
                    suffix = f"_{stance}" if stance != "neutral" else ""
                    mode_suffix = f"_{args.mode}" if args.mode != "mcq" else ""
                    path = runs_dir / f"{model}_{value_set}_{turns}t{suffix}{mode_suffix}.json"
                    if not path.exists():
                        raise FileNotFoundError(f"Missing result JSON: {path}")
                    import json
                    with open(path) as f:
                        results.append(json.load(f))
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="RunPod wrapper for distributed scenario conversation experiment"
    )

    # Models
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["all"],
        help="Models to run (default: all). Use 'all' for all 5 models.",
    )

    # Value sets
    parser.add_argument(
        "--value-sets",
        type=str,
        nargs="+",
        default=["all"],
        help="Value sets to run (default: all). Use 'all' for HHH, personalprotective, modelspec.",
    )

    # Stances
    parser.add_argument(
        "--stances",
        type=str,
        nargs="+",
        default=["all"],
        help="Stances to run (default: all). Use 'all' for neutral, pro_v1, pro_v2.",
    )

    # Turn counts
    parser.add_argument(
        "--turn-counts",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Conversation turn counts (default: 5 10)",
    )

    # Number of scenarios
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=300,
        help="Max scenarios per value set (default: 300)",
    )

    # GPU configuration
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Override GPU count detection (default: auto-detect)",
    )

    # Force vLLM on/off
    parser.add_argument(
        "--force-vllm",
        action="store_true",
        help="Force use of vLLM even if only 1 GPU available",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Disable vLLM even if available",
    )

    # User simulator
    parser.add_argument(
        "--simulator",
        type=str,
        default="openai",
        choices=["anthropic", "openai"],
        help="User simulator provider (default: openai for gpt-4o-mini)",
    )
    parser.add_argument(
        "--simulator-model",
        type=str,
        default="gpt-4o-mini",
        help="Simulator model ID (default: gpt-4o-mini for speed and cost)",
    )
    parser.add_argument(
        "--simulator-api-key",
        type=str,
        default=None,
        help="API key for simulator (falls back to OPENAI_API_KEY env var)",
    )

    # Experiment options
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mcq", "openended"],
        default="mcq",
        help="Probing mode (default: mcq)",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="openai",
        choices=["anthropic", "openai"],
        help="Judge provider for open-ended mode (default: openai)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model ID for open-ended mode (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default=None,
        help="API key for the judge (falls back to OPENAI_API_KEY / ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=["pair", "scenario"],
        default="pair",
        help="Grouping mode (default: pair)",
    )
    parser.add_argument(
        "--shard-by",
        type=str,
        choices=["none", "model", "model-stance"],
        default="model-stance",
        help="Shard RunPod execution across GPUs (default: model-stance).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum number of concurrent shard workers (default: one per visible GPU).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    log.info("=" * 80)
    log.info("RunPod Scenario Conversation Experiment")
    log.info("=" * 80)

    # Detect GPU configuration
    n_gpus = args.num_gpus or detect_gpu_config()

    # Decide whether to use vLLM
    use_vllm = False
    if not args.no_vllm:
        has_vllm = is_vllm_available()
        # Use vLLM if:
        # 1. It's available AND
        # 2. Either --force-vllm is set OR we have multiple GPUs
        use_vllm = has_vllm and (args.force_vllm or n_gpus > 1)

    if use_vllm:
        log.info(f"Using vLLM for multi-GPU inference")
    else:
        log.info(f"Using standard AlignmentModel (single-GPU mode)")

    available_gpus = list(range(n_gpus))
    if args.shard_by != "none" and n_gpus > 1:
        exit_code = run_sharded_experiments(args, use_vllm=use_vllm, available_gpus=available_gpus)
    else:
        cmd = build_experiment_command(
            args,
            use_vllm=use_vllm,
            models=resolve_models(args),
            stances=resolve_stances(args),
            skip_postprocess=False,
        )
        exit_code = run_experiment(cmd)

    if exit_code == 0 and args.shard_by != "none" and n_gpus > 1:
        log.info("Combining shard outputs into one summary run...")
        from run_scenario_conversation_experiment import RESULTS_DIR, save_summary_csv
        from visualize import generate_scenario_experiment_plots

        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        all_results = collect_results_for_run(args)
        save_summary_csv(all_results, log, run_id=run_id)
        generate_scenario_experiment_plots(all_results, RESULTS_DIR, run_id=run_id)
        log.info("Combined summary saved under run_id=%s", run_id)

    log.info("=" * 80)
    if exit_code == 0:
        log.info("Experiment completed successfully!")
        log.info(
            "Results saved to: pipeline/results/scenario_conversation/all_results.csv"
        )
    else:
        log.error(f"Experiment failed with exit code {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
