"""Open-ended variant of the alignment target experiment.

Replaces MCQ probing with open-ended probing + a judge model (Anthropic or OpenAI).
Reuses canonical conversations from run_alignment_target_experiment.py if they exist,
otherwise generates them automatically (requires --simulator-api-key).

Usage:
    # With GPT-4o-mini as judge (cheapest):
    python run_alignment_target_experiment_openended.py \
        --judge openai --judge-model gpt-4o-mini \
        --judge-api-key sk-...

    # With Claude Haiku as judge:
    python run_alignment_target_experiment_openended.py \
        --judge anthropic --judge-model claude-haiku-4-5-20251001 \
        --judge-api-key sk-ant-...

    # Specific scope (matches your existing MCQ run):
    python run_alignment_target_experiment_openended.py \
        --models tulu-3-sft tulu-3-dpo tulu-3-rlvr llama-3.1-instruct llama-3.1-base \
        --value-sets personalprotective \
        --domains politics philosophy \
        --turn-counts 10 \
        --num-scenarios 300 \
        --judge openai --judge-model gpt-4o-mini \
        --judge-api-key sk-...
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Reuse model registry and AlignmentModel from the MCQ experiment
from src.shared.run_alignment_target_experiment import (
    ALIGNMENT_MODELS,
    DEFAULT_MODELS,
    DEFAULT_VALUE_SETS,
    DEFAULT_TURN_COUNTS,
    DEFAULT_REFERENCE_MODEL,
    AlignmentModel,
    get_domains,
    register_value_aligned_prompts,
    analyze_cross_method,
    generate_canonical_conversations,
    build_user_simulator,
)

# ---------------------------------------------------------------------------
# Results live in a separate directory so MCQ results are preserved
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "alignment_openended"

# Canonical conversations come from the MCQ experiment
CANONICAL_CONV_DIR = Path(__file__).resolve().parent / "results" / "alignment" / "canonical_conversations"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Open-ended alignment comparison (generates canonical conversations if missing)"
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help=f"Model keys to test (default: {DEFAULT_MODELS})",
    )

    # Judge model
    parser.add_argument("--judge", type=str, default="openai",
                        choices=["anthropic", "openai"],
                        help="Judge model provider")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini",
                        help="Judge model ID")
    parser.add_argument("--judge-api-key", type=str, default=None,
                        help="API key for judge model")

    # User simulator (for generating canonical conversations if missing)
    parser.add_argument("--simulator", type=str, default="openai",
                        choices=["anthropic", "openai"],
                        help="User simulator provider (only used if conversations need generating)")
    parser.add_argument("--simulator-model", type=str, default=None)
    parser.add_argument("--simulator-base-url", type=str, default=None)
    parser.add_argument("--simulator-api-key", type=str, default=None)
    parser.add_argument("--reference-model", type=str, default=DEFAULT_REFERENCE_MODEL,
                        help="HF model for generating canonical conversations")

    # Experiment scope
    parser.add_argument("--value-sets", type=str, nargs="+", default=None,
                        help=f"Value sets (default: {DEFAULT_VALUE_SETS})")
    parser.add_argument("--domains", type=str, nargs="+", default=None)
    parser.add_argument("--generic-only", action="store_true")
    parser.add_argument("--turn-counts", type=int, nargs="+", default=None,
                        help=f"Turn counts (default: {DEFAULT_TURN_COUNTS})")
    parser.add_argument("--num-scenarios", type=int, default=300,
                        help="Max scenarios per value set (default: 300)")

    return parser.parse_args()


def build_judge_client(args):
    """Build the judge API client."""
    if args.judge == "anthropic":
        if args.judge_api_key:
            os.environ["ANTHROPIC_API_KEY"] = args.judge_api_key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: No Anthropic API key for judge. Pass --judge-api-key or set ANTHROPIC_API_KEY.")
            sys.exit(1)
        import anthropic
        return anthropic.Anthropic()
    else:
        if args.judge_api_key:
            os.environ["OPENAI_API_KEY"] = args.judge_api_key
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: No OpenAI API key for judge. Pass --judge-api-key or set OPENAI_API_KEY.")
            sys.exit(1)
        import openai
        return openai.OpenAI()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def checkpoint_path(model_key: str, value_set: str, domain: str, num_turns: int) -> Path:
    return RESULTS_DIR / "checkpoints" / f"{model_key}_{value_set}_{domain}_{num_turns}.json"


def t0_checkpoint_path(model_key: str, value_set: str) -> Path:
    return RESULTS_DIR / "checkpoints" / f"{model_key}_{value_set}_t0.json"


def canonical_conv_path(value_set: str, domain: str, max_turns: int) -> Path:
    return CANONICAL_CONV_DIR / f"{value_set}_{domain}_{max_turns}turns.json"


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-model probing
# ---------------------------------------------------------------------------

def run_model_conditions(
    model_key: str,
    model_info: dict,
    value_sets: list[str],
    domains_per_vs: dict[str, list[str]],
    turn_counts: list[int],
    num_scenarios: int,
    judge_client,
    judge_model: str,
    log: logging.Logger,
) -> list[dict]:
    """Load one alignment model, run all open-ended T0/T1 conditions, unload."""

    from src.shared.probing import load_scenarios, probe_values_openended
    from analysis import fit_bradley_terry, compute_drift, compute_answer_flip_rate
    from visualize import plot_radar_t0_t1

    log.info(f"Loading model: {model_key} ({model_info['hf_id']})")
    model = AlignmentModel(
        model_info["hf_id"],
        is_base_model=model_info.get("is_base_model", False),
    )

    results = []

    for value_set in value_sets:
        scenarios = load_scenarios(value_set, max_scenarios=num_scenarios)
        log.info(f"  {value_set}: {len(scenarios)} scenarios")

        # Save scenario distribution report (once per value set)
        dist_path = RESULTS_DIR / f"scenario_distribution_{value_set}.json"
        if not dist_path.exists():
            from src.shared.probing import scenario_distribution_report
            save_json(dist_path, scenario_distribution_report(scenarios))

        # --- T0 (shared across domains) ---
        t0_cp = t0_checkpoint_path(model_key, value_set)
        if t0_cp.exists():
            log.info(f"  T0 checkpoint exists, loading")
            t0_data = load_json(t0_cp)
            outcomes_t0 = pd.DataFrame(t0_data["outcomes"])
            ranking_t0 = pd.DataFrame(t0_data["ranking"])
        else:
            log.info(f"  Running open-ended T0 probing...")
            outcomes_t0 = probe_values_openended(
                model, model_key, scenarios,
                judge_client=judge_client,
                judge_model=judge_model,
                context=None,
            )
            ranking_t0 = fit_bradley_terry(outcomes_t0)
            save_json(t0_cp, {
                "outcomes": outcomes_t0.to_dict(orient="records"),
                "ranking": ranking_t0.to_dict(orient="records"),
            })
            log.info(f"  T0: {len(outcomes_t0)} outcomes")

        # --- T1 per domain × turn count ---
        for domain in domains_per_vs[value_set]:
            for num_turns in turn_counts:
                conv_path = canonical_conv_path(value_set, domain, num_turns)
                if not conv_path.exists():
                    log.warning(
                        f"  Canonical conversation missing: {conv_path}. "
                        f"Run run_alignment_target_experiment.py first to generate it. Skipping."
                    )
                    continue

                cp = checkpoint_path(model_key, value_set, domain, num_turns)
                if cp.exists():
                    log.info(f"  {domain}/{num_turns}t: already done")
                    results.append(load_json(cp))
                    continue

                conversation = load_json(conv_path)
                log.info(f"  {domain}/{num_turns}t: open-ended T1 probing...")
                outcomes_t1 = probe_values_openended(
                    model, model_key, scenarios,
                    judge_client=judge_client,
                    judge_model=judge_model,
                    context=conversation,
                )
                ranking_t1 = fit_bradley_terry(outcomes_t1)
                drift = compute_drift(ranking_t0, ranking_t1)
                flips = compute_answer_flip_rate(outcomes_t0, outcomes_t1)

                # Radar chart
                plot_dir = RESULTS_DIR / "plots" / model_key / value_set
                plot_dir.mkdir(parents=True, exist_ok=True)
                plot_radar_t0_t1(
                    ranking_t0, ranking_t1,
                    f"{model_key} ({model_info['method']}) | {value_set} | {domain} | {num_turns}t [open-ended]",
                    plot_dir / f"radar_{domain}_{num_turns}t.png",
                )

                result = {
                    "model": model_key,
                    "method": model_info["method"],
                    "family": model_info["family"],
                    "value_set": value_set,
                    "domain": domain,
                    "num_turns": num_turns,
                    "ranking_t0": dict(zip(
                        ranking_t0["value"], ranking_t0["ability"].astype(float)
                    )),
                    "ranking_t1": dict(zip(
                        ranking_t1["value"], ranking_t1["ability"].astype(float)
                    )),
                    "l2_distance": drift["l2_distance"],
                    "rank_correlation": drift["rank_correlation"],
                    "per_value_delta": drift["per_value_delta"],
                    "overall_flip_rate": flips["overall_flip_rate"],
                    "flip_direction": flips.get("flip_direction", {}),
                    "per_pair_flip_rate": flips.get("per_pair_flip_rate", {}),
                    "n_matched": flips.get("n_matched", 0),
                    "probe_mode": "openended",
                    "judge_model": judge_model,
                }
                save_json(cp, result)
                results.append(result)
                log.info(
                    f"  {domain}/{num_turns}t: L2={drift['l2_distance']:.3f}, "
                    f"rho={drift['rank_correlation']:.3f}, "
                    f"flip={flips['overall_flip_rate']:.1%}"
                )

    model.unload()
    log.info(f"Model {model_key} unloaded.")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ["checkpoints", "runs", "plots"]:
        (RESULTS_DIR / subdir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RESULTS_DIR / "experiment.log"),
        ],
    )
    log = logging.getLogger(__name__)

    model_keys = args.models or DEFAULT_MODELS
    value_sets = args.value_sets or DEFAULT_VALUE_SETS
    turn_counts = args.turn_counts or DEFAULT_TURN_COUNTS

    # Validate model keys
    for mk in model_keys:
        if mk not in ALIGNMENT_MODELS:
            log.error(f"Unknown model key: {mk}. Available: {list(ALIGNMENT_MODELS.keys())}")
            sys.exit(1)

    # Check canonical conversations exist
    register_value_aligned_prompts(value_sets)
    domains_per_vs = {}
    for vs in value_sets:
        if args.domains:
            domains_per_vs[vs] = args.domains
        else:
            domains_per_vs[vs] = get_domains(vs, args.generic_only)

    missing_convs = []
    for vs in value_sets:
        for domain in domains_per_vs[vs]:
            for nt in turn_counts:
                cp = canonical_conv_path(vs, domain, nt)
                if not cp.exists():
                    missing_convs.append(str(cp))
    if missing_convs:
        log.info(
            f"Missing {len(missing_convs)} canonical conversations. "
            f"Generating with reference model..."
        )
        # Temporarily patch RESULTS_DIR so conversations are saved to the
        # canonical location (results/alignment/canonical_conversations/)
        from src.shared import run_alignment_target_experiment as rae
        original_results_dir = rae.RESULTS_DIR
        rae.RESULTS_DIR = CANONICAL_CONV_DIR.parent  # results/alignment/
        try:
            user_sim = build_user_simulator(args)
            generate_canonical_conversations(
                args.reference_model, user_sim, value_sets, domains_per_vs,
                turn_counts, log,
            )
        finally:
            rae.RESULTS_DIR = original_results_dir

    # Build judge client
    judge_client = build_judge_client(args)
    judge_model = args.judge_model

    # Experiment plan
    total_conditions = 0
    for vs in value_sets:
        total_conditions += len(model_keys) * len(domains_per_vs[vs]) * len(turn_counts)
    total_t0 = len(model_keys) * len(value_sets)
    total_judge_calls = (total_t0 + total_conditions) * args.num_scenarios
    log.info("=" * 60)
    log.info("OPEN-ENDED ALIGNMENT COMPARISON EXPERIMENT")
    log.info("=" * 60)
    log.info(f"  Models:          {model_keys}")
    log.info(f"  Value sets:      {value_sets}")
    log.info(f"  Turn counts:     {turn_counts}")
    log.info(f"  Scenarios:       {args.num_scenarios}")
    log.info(f"  T0 conditions:   {total_t0}")
    log.info(f"  T1 conditions:   {total_conditions}")
    log.info(f"  Judge:           {judge_model} ({args.judge})")
    log.info(f"  Est. judge calls: ~{total_judge_calls}")

    # --- Per-model probing ---
    all_results = []
    for model_key in model_keys:
        model_info = ALIGNMENT_MODELS[model_key]
        log.info(f"\n--- {model_key} ({model_info['method']}) ---")
        model_results = run_model_conditions(
            model_key, model_info, value_sets, domains_per_vs, turn_counts,
            args.num_scenarios, judge_client, judge_model, log,
        )
        all_results.extend(model_results)

    # --- Cross-method analysis (reuse from MCQ experiment) ---
    log.info("\n" + "=" * 60)
    log.info("CROSS-METHOD ANALYSIS (OPEN-ENDED)")
    log.info("=" * 60)
    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        log.warning("No results to analyze — all conditions were skipped (missing canonical conversations?).")
    else:
        results_df.to_csv(RESULTS_DIR / "all_results.csv", index=False)

        # Temporarily patch RESULTS_DIR in the imported function
        from src.shared import run_alignment_target_experiment as rae
        original_results_dir = rae.RESULTS_DIR
        rae.RESULTS_DIR = RESULTS_DIR
        try:
            analyze_cross_method(results_df, log)
        finally:
            rae.RESULTS_DIR = original_results_dir

    log.info(f"\nDone! {len(all_results)} conditions completed.")
    log.info(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
