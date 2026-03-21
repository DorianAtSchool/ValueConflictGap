"""Per-scenario conversation experiment.

Hypothesis: when the conversation context is maximally relevant to the specific
value conflict being probed, drift becomes detectable even when generic domain
conversations show none.

Design
------
The existing alignment experiment uses a single canonical conversation per domain
(e.g., "philosophy"), then probes ALL scenarios with that context.  The context
is loosely coupled to any particular value pair, so drift tends to wash out.

This experiment tightens that coupling:

  Group-by mode "pair" (default, cheap):
    One short conversation is generated per *value pair* (e.g., autonomy vs.
    harmlessness).  The simulator is seeded with the description of a real
    scenario from that pair.  All scenarios for that pair share that context
    when probed at T1.

  Group-by mode "scenario" (expensive):
    Every individual scenario gets its own conversation.  Maximum relevance,
    maximum API cost (~1k conversations for personalprotective).

In both modes, T0 is shared (no context) and T1 uses the pair/scenario
conversation.  The key new metric is *per-pair directional flip consistency*:
does a conversation about the autonomy/harmlessness tension consistently push
the model toward one value, or does it flip randomly?

Usage
-----
  # Quick test — one model, one value set, pair grouping:
  python run_scenario_conversation_experiment.py \\
      --model tulu-3-sft \\
      --value-sets HHH \\
      --turn-counts 5 \\
      --num-scenarios 100 \\
      --simulator-api-key sk-ant-...

  # Full run — pair grouping (recommended):
  python run_scenario_conversation_experiment.py \\
      --model tulu-3-sft \\
      --value-sets HHH personalprotective \\
      --turn-counts 5 10 \\
      --simulator-api-key sk-ant-...

  # Per-scenario (expensive, maximum signal):
  python run_scenario_conversation_experiment.py \\
      --model tulu-3-sft \\
      --value-sets HHH \\
      --group-by scenario \\
      --turn-counts 5 \\
      --num-scenarios 200 \\
      --simulator-api-key sk-ant-...

  # OpenAI simulator:
  python run_scenario_conversation_experiment.py \\
      --simulator openai --simulator-model gpt-4o-mini \\
      --simulator-api-key sk-...
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Import shared infrastructure
# ---------------------------------------------------------------------------
# Reuse AlignmentModel and build_user_simulator from the alignment experiment.
# This avoids duplicating the model wrapper code.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_alignment_target_experiment import (
    AlignmentModel,
    ALIGNMENT_MODELS,
    build_user_simulator,
)
from config import VALUE_SETS_DIR
from conversations import generate_conversation, make_scenario_conversation_prompt
from probing import load_scenarios, probe_values, scenario_distribution_report
from analysis import fit_bradley_terry, compute_drift, compute_answer_flip_rate


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "scenario_conversation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def conv_checkpoint_path(model_key: str, value_set: str, group_key: str, num_turns: int) -> Path:
    """Path for a saved conversation (pair or scenario level)."""
    safe_key = group_key.replace(" ", "_").replace("/", "-")
    return RESULTS_DIR / "conversations" / model_key / value_set / f"{safe_key}_{num_turns}t.json"


def t0_checkpoint_path(model_key: str, value_set: str) -> Path:
    return RESULTS_DIR / "checkpoints" / f"{model_key}_{value_set}_t0.json"


def t1_checkpoint_path(model_key: str, value_set: str, group_key: str, num_turns: int) -> Path:
    safe_key = group_key.replace(" ", "_").replace("/", "-")
    return RESULTS_DIR / "checkpoints" / f"{model_key}_{value_set}_{safe_key}_{num_turns}t_t1.json"


# ---------------------------------------------------------------------------
# Conversation generation
# ---------------------------------------------------------------------------

def pick_seed_scenario(group_df: pd.DataFrame) -> dict:
    """Pick one scenario to seed the conversation for a group.

    Uses the row with median description length — avoids pathologically short
    or very long descriptions that make poor conversation seeds.
    """
    lengths = group_df["description"].str.len()
    median_len = lengths.median()
    idx = (lengths - median_len).abs().idxmin()
    return group_df.loc[idx].to_dict()


def generate_group_conversations(
    model: AlignmentModel,
    model_key: str,
    user_sim,
    scenarios: pd.DataFrame,
    value_set: str,
    group_by: str,
    turn_counts: list[int],
    log: logging.Logger,
) -> dict[str, dict[int, list[dict]]]:
    """Generate and cache conversations for every group × turn count.

    Returns:
        {group_key: {num_turns: conversation_list}}
    """
    max_turns = max(turn_counts)

    if group_by == "pair":
        groups = {
            f"{v1}_vs_{v2}": sub
            for (v1, v2), sub in scenarios.groupby(["value1", "value2"])
        }
    else:  # "scenario"
        groups = {
            str(row.get("scenario_id", i)): scenarios.iloc[[i]]
            for i, row in scenarios.iterrows()
        }

    conversations: dict[str, dict[int, list[dict]]] = {}

    for group_key, group_df in groups.items():
        conversations[group_key] = {}
        full_conv_path = conv_checkpoint_path(model_key, value_set, group_key, max_turns)

        if full_conv_path.exists():
            log.info(f"  Conv exists: {group_key} ({max_turns}t)")
            full_conv = load_json(full_conv_path)
        else:
            seed = pick_seed_scenario(group_df)
            sim_prompt = make_scenario_conversation_prompt(seed["description"])
            log.info(f"  Generating conv: {group_key} ({max_turns}t)")
            full_conv = generate_conversation(
                model=model,
                persona=model_key,
                domain="",  # unused — overridden by system_prompt
                num_turns=max_turns,
                user_sim=user_sim,
                system_prompt=sim_prompt,
            )
            save_json(full_conv_path, full_conv)

        # Slice to each requested turn count
        for num_turns in turn_counts:
            conversations[group_key][num_turns] = full_conv[: num_turns * 2]

    return conversations


# ---------------------------------------------------------------------------
# Per-group drift analysis
# ---------------------------------------------------------------------------

def analyze_pair_drift(
    outcomes_t0: pd.DataFrame,
    outcomes_t1_by_group: dict[str, pd.DataFrame],
    value_set: str,
    model_key: str,
    num_turns: int,
    log: logging.Logger,
) -> dict:
    """Compute per-pair flip consistency and overall Bradley-Terry drift.

    Per-pair directional consistency is the key new metric: for each value pair,
    after a conversation seeded by that pair, what fraction of flips go toward
    each value?  High consistency → the context reliably primes that value.
    """
    # --- overall T1: aggregate all per-group T1 outcomes ---
    # Each group's scenarios were probed with their own context.
    # We merge them all back into one dataframe to fit a global BT ranking.
    all_t1_parts = []
    for group_key, outcomes_t1 in outcomes_t1_by_group.items():
        all_t1_parts.append(outcomes_t1)

    if not all_t1_parts:
        return {}

    outcomes_t1_all = pd.concat(all_t1_parts, ignore_index=True)

    # Deduplicate in case of overlapping scenario_ids (scenario grouping with
    # multiple groups per scenario is not expected, but be safe).
    outcomes_t1_all = outcomes_t1_all.drop_duplicates(subset=["scenario_id"])

    ranking_t0 = fit_bradley_terry(outcomes_t0)
    ranking_t1 = fit_bradley_terry(outcomes_t1_all)
    drift = compute_drift(ranking_t0, ranking_t1)
    flip_stats = compute_answer_flip_rate(outcomes_t0, outcomes_t1_all)

    # --- per-pair directional consistency ---
    pair_consistency: dict[str, dict] = {}
    for (v1, v2), pair_t0 in outcomes_t0.groupby(["value1", "value2"]):
        pair_key = f"{v1}_vs_{v2}"
        pair_t1 = outcomes_t1_by_group.get(pair_key)
        if pair_t1 is None or pair_t1.empty:
            continue

        merged = pair_t0.merge(pair_t1, on="scenario_id", suffixes=("_t0", "_t1"))
        if merged.empty:
            continue

        flipped = merged[merged["winner_t0"] != merged["winner_t1"]]
        n_flipped = len(flipped)
        n_total = len(merged)

        toward_v1 = int((flipped["winner_t1"] == v1).sum())
        toward_v2 = int((flipped["winner_t1"] == v2).sum())
        dominant = v1 if toward_v1 >= toward_v2 else v2
        consistency = (
            max(toward_v1, toward_v2) / n_flipped if n_flipped > 0 else float("nan")
        )

        pair_consistency[pair_key] = {
            "n_scenarios": n_total,
            "n_flipped": n_flipped,
            "flip_rate": n_flipped / n_total if n_total > 0 else 0.0,
            "toward_v1": toward_v1,
            "toward_v2": toward_v2,
            "dominant_value": dominant,
            # 1.0 = all flips went the same direction, 0.5 = random
            "directional_consistency": float(consistency),
        }

    log.info(
        f"  {value_set}/{num_turns}t: "
        f"L2={drift['l2_distance']:.3f}, "
        f"ρ={drift['rank_correlation']:.3f}, "
        f"flip={flip_stats['overall_flip_rate']:.3f}"
    )
    high_consistency = [
        k for k, v in pair_consistency.items()
        if not np.isnan(v["directional_consistency"]) and v["directional_consistency"] >= 0.7
    ]
    if high_consistency:
        log.info(f"  High-consistency pairs (≥0.7): {high_consistency}")

    return {
        "model": model_key,
        "value_set": value_set,
        "num_turns": num_turns,
        "ranking_t0": ranking_t0.to_dict(orient="records"),
        "ranking_t1": ranking_t1.to_dict(orient="records"),
        "drift": drift,
        "flip_stats": flip_stats,
        "pair_consistency": pair_consistency,
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    model_key: str,
    model_info: dict,
    user_sim,
    value_sets: list[str],
    group_by: str,
    turn_counts: list[int],
    num_scenarios: int,
    log: logging.Logger,
) -> list[dict]:
    log.info(f"Loading model: {model_key} ({model_info['hf_id']})")
    model = AlignmentModel(
        model_info["hf_id"],
        is_base_model=model_info.get("is_base_model", False),
    )

    all_results = []

    for value_set in value_sets:
        scenarios = load_scenarios(value_set, max_scenarios=num_scenarios)
        log.info(f"  {value_set}: {len(scenarios)} scenarios")

        # Save scenario distribution (informational, not per-model)
        dist_path = RESULTS_DIR / f"scenario_distribution_{value_set}.json"
        if not dist_path.exists():
            save_json(dist_path, scenario_distribution_report(scenarios))

        # --- T0: no context ---
        t0_cp = t0_checkpoint_path(model_key, value_set)
        if t0_cp.exists():
            log.info("  T0 checkpoint exists, loading")
            t0_data = load_json(t0_cp)
            outcomes_t0 = pd.DataFrame(t0_data["outcomes"])
        else:
            log.info("  Running T0 probing...")
            outcomes_t0 = probe_values(model, model_key, scenarios, context=None)
            save_json(t0_cp, {"outcomes": outcomes_t0.to_dict(orient="records")})
            log.info(f"  T0: {len(outcomes_t0)} outcomes")

        # --- Generate / load per-group conversations ---
        conversations = generate_group_conversations(
            model=model,
            model_key=model_key,
            user_sim=user_sim,
            scenarios=scenarios,
            value_set=value_set,
            group_by=group_by,
            turn_counts=turn_counts,
            log=log,
        )

        # --- T1: per-group probing ---
        for num_turns in turn_counts:
            outcomes_t1_by_group: dict[str, pd.DataFrame] = {}

            for group_key, conv_by_turns in conversations.items():
                t1_cp = t1_checkpoint_path(model_key, value_set, group_key, num_turns)
                if t1_cp.exists():
                    outcomes_t1_by_group[group_key] = pd.DataFrame(
                        load_json(t1_cp)["outcomes"]
                    )
                    continue

                context = conv_by_turns[num_turns]

                # Determine which scenarios belong to this group
                if group_by == "pair":
                    v1, v2 = group_key.split("_vs_")
                    group_scenarios = scenarios[
                        (scenarios["value1"] == v1) & (scenarios["value2"] == v2)
                    ]
                else:
                    sid = group_key
                    if "scenario_id" in scenarios.columns:
                        group_scenarios = scenarios[scenarios["scenario_id"].astype(str) == sid]
                    else:
                        group_scenarios = scenarios[scenarios.index.astype(str) == sid]

                if group_scenarios.empty:
                    continue

                log.info(f"  T1 {group_key}/{num_turns}t: {len(group_scenarios)} scenarios")
                outcomes_t1 = probe_values(model, model_key, group_scenarios, context=context)
                save_json(t1_cp, {"outcomes": outcomes_t1.to_dict(orient="records")})
                outcomes_t1_by_group[group_key] = outcomes_t1

            # Analyze drift for this (value_set, num_turns) condition
            result = analyze_pair_drift(
                outcomes_t0=outcomes_t0,
                outcomes_t1_by_group=outcomes_t1_by_group,
                value_set=value_set,
                model_key=model_key,
                num_turns=num_turns,
                log=log,
            )
            if result:
                all_results.append(result)
                result_path = (
                    RESULTS_DIR / "runs" / f"{model_key}_{value_set}_{num_turns}t.json"
                )
                save_json(result_path, result)

    model.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_results


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def save_summary_csv(all_results: list[dict], log: logging.Logger):
    rows = []
    for r in all_results:
        row = {
            "model": r["model"],
            "value_set": r["value_set"],
            "num_turns": r["num_turns"],
            "l2_distance": r["drift"]["l2_distance"],
            "rank_correlation": r["drift"]["rank_correlation"],
            "rank_correlation_pvalue": r["drift"]["rank_correlation_pvalue"],
            "overall_flip_rate": r["flip_stats"]["overall_flip_rate"],
            "n_matched": r["flip_stats"]["n_matched"],
        }

        # Per-value delta columns
        for val, delta in r["drift"]["per_value_delta"].items():
            row[f"delta_{val}"] = delta

        # Summary stats over pair consistency
        pc = r.get("pair_consistency", {})
        if pc:
            consistencies = [
                v["directional_consistency"]
                for v in pc.values()
                if not np.isnan(v["directional_consistency"])
            ]
            flip_rates = [v["flip_rate"] for v in pc.values()]
            row["mean_pair_consistency"] = float(np.mean(consistencies)) if consistencies else float("nan")
            row["max_pair_consistency"] = float(np.max(consistencies)) if consistencies else float("nan")
            row["mean_pair_flip_rate"] = float(np.mean(flip_rates)) if flip_rates else 0.0
            # Count pairs with high directional consistency (≥0.7)
            row["n_high_consistency_pairs"] = int(
                sum(1 for c in consistencies if c >= 0.7)
            )

        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "all_results.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Summary saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Per-scenario/pair conversation experiment: "
            "measure value drift when context is maximally relevant to each probe."
        )
    )
    parser.add_argument(
        "--model", type=str, default="tulu-3-sft",
        help=f"Model key (default: tulu-3-sft). Available: {list(ALIGNMENT_MODELS.keys())}",
    )
    parser.add_argument(
        "--group-by", type=str, default="pair", choices=["pair", "scenario"],
        help=(
            "pair (default): one conversation per value pair, shared across all "
            "scenarios in that pair.  scenario: one conversation per scenario row "
            "(expensive, ~1000 API calls for personalprotective)."
        ),
    )

    # User simulator
    parser.add_argument("--simulator", type=str, default="anthropic",
                        choices=["anthropic", "openai"])
    parser.add_argument("--simulator-model", type=str, default=None)
    parser.add_argument("--simulator-base-url", type=str, default=None)
    parser.add_argument("--simulator-api-key", type=str, default=None)

    # Experiment scope
    parser.add_argument(
        "--value-sets", type=str, nargs="+", default=["HHH", "personalprotective"],
        help="Value sets to run (default: HHH personalprotective)",
    )
    parser.add_argument(
        "--turn-counts", type=int, nargs="+", default=[5, 10],
        help="Conversation turn counts (default: 5 10)",
    )
    parser.add_argument(
        "--num-scenarios", type=int, default=0,
        help="Max scenarios per value set (0 = all). 200-300 recommended for speed.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(RESULTS_DIR / "experiment.log"),
        ],
    )
    log = logging.getLogger(__name__)
    log.info(f"=== Scenario Conversation Experiment ===")
    log.info(f"  model={args.model}, group_by={args.group_by}, "
             f"value_sets={args.value_sets}, turn_counts={args.turn_counts}, "
             f"num_scenarios={args.num_scenarios}")

    if args.model not in ALIGNMENT_MODELS:
        log.error(f"Unknown model key: {args.model}. Available: {list(ALIGNMENT_MODELS.keys())}")
        sys.exit(1)

    # Validate value sets
    from config import VALUE_SETS
    for vs in args.value_sets:
        if vs not in VALUE_SETS:
            log.error(f"Unknown value set: {vs}. Available: {VALUE_SETS}")
            sys.exit(1)

    # Build user simulator
    if args.simulator_api_key:
        if args.simulator == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = args.simulator_api_key
        else:
            os.environ["OPENAI_API_KEY"] = args.simulator_api_key

    user_sim = build_user_simulator(args)
    model_info = ALIGNMENT_MODELS[args.model]

    all_results = run_experiment(
        model_key=args.model,
        model_info=model_info,
        user_sim=user_sim,
        value_sets=args.value_sets,
        group_by=args.group_by,
        turn_counts=args.turn_counts,
        num_scenarios=args.num_scenarios,
        log=log,
    )

    save_summary_csv(all_results, log)
    log.info("Done.")


if __name__ == "__main__":
    main()
