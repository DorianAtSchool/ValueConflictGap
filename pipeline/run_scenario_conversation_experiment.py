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
      --models tulu-3-sft \\
      --value-sets HHH \\
      --turn-counts 5 \\
      --num-scenarios 100 \\
      --simulator-api-key sk-ant-...

  # Multiple models and stances in a single run:
  python run_scenario_conversation_experiment.py \\
      --models tulu-3-sft llama-3 \\
      --stances neutral pro_v1 pro_v2 \\
      --value-sets HHH personalprotective \\
      --turn-counts 5 10 \\
      --simulator-api-key sk-ant-...

  # Per-scenario (expensive, maximum signal):
  python run_scenario_conversation_experiment.py \\
      --models tulu-3-sft \\
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
import re
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
try:
    from alignmentmodel_vllm import AlignmentModelVLLM
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    AlignmentModelVLLM = None
from config import SCENARIO_PATHS, VALUE_SETS_DIR
from conversations import generate_conversation, make_scenario_conversation_prompt
from probing import (
    OPENENDED_ASSISTANT_SYSTEM_PROMPT,
    OPENENDED_JUDGE_STYLE,
    OPENENDED_PROMPT_STYLE,
    load_scenarios,
    probe_values,
    probe_values_openended,
    scenario_sample_signature,
    scenario_distribution_report,
)
from analysis import (
    compute_answer_flip_rate,
    compute_average_ranks_by_group,
    compute_drift,
    fit_bradley_terry,
)
from visualize import generate_scenario_experiment_plots


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


_REQUIRED_OUTCOME_COLUMNS = {"scenario_id", "value1", "value2", "choice", "winner"}
_EARLY_SIGNOFF_RE = re.compile(
    r"^(see you|talk soon|talk later|bye|goodbye|take care|catch you later|till next time|"
    r"until next time|farewell|you too)\b",
    re.IGNORECASE,
)
_ASSISTANT_REFUSAL_RE = re.compile(
    r"\b(i'm unable to provide the help that you need|as an ai|i don't have personal experience)\b",
    re.IGNORECASE,
)


def _validate_cached_conversation(conversation: list[dict], num_turns: int) -> list[str]:
    """Return a list of validation failures for a generated conversation."""
    issues: list[str] = []
    expected_len = num_turns * 2
    if len(conversation) != expected_len:
        issues.append(f"expected {expected_len} messages, found {len(conversation)}")
        return issues

    for i, msg in enumerate(conversation):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if msg.get("role") != expected_role:
            issues.append(f"message {i} has role={msg.get('role')} expected={expected_role}")
            break
        content = (msg.get("content") or "").strip()
        if not content:
            issues.append(f"message {i} is empty")
            break
        if msg.get("role") == "user":
            turn_idx = i // 2
            if turn_idx > 0 and content[0].islower():
                issues.append(f"user turn {turn_idx + 1} starts mid-sentence")
                break
            if turn_idx < num_turns - 2 and _EARLY_SIGNOFF_RE.match(content):
                issues.append(f"user turn {turn_idx + 1} ends conversation early")
                break

    assistant_text = "\n".join(
        (msg.get("content") or "") for msg in conversation if msg.get("role") == "assistant"
    )
    if len(_ASSISTANT_REFUSAL_RE.findall(assistant_text)) >= 2:
        issues.append("assistant contains repeated refusal/disclaimer boilerplate")

    duplicate_adjacent = 0
    for i in range(len(conversation) - 1):
        left = (conversation[i].get("content") or "").strip()
        right = (conversation[i + 1].get("content") or "").strip()
        if left and left == right:
            duplicate_adjacent += 1
    if duplicate_adjacent >= 2:
        issues.append(f"{duplicate_adjacent} adjacent duplicate messages")

    return issues


def _validate_outcomes_frame(
    outcomes: pd.DataFrame,
    *,
    require_nonempty: bool = True,
) -> list[str]:
    """Return a list of validation failures for probing outcomes."""
    issues: list[str] = []
    missing = sorted(_REQUIRED_OUTCOME_COLUMNS - set(outcomes.columns))
    if missing:
        issues.append(f"missing columns: {missing}")
    if require_nonempty and outcomes.empty:
        issues.append("contains no outcomes")
    return issues


def _load_outcomes_checkpoint(
    path: Path,
    *,
    require_nonempty: bool = True,
    expected_metadata: dict | None = None,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Load an outcomes checkpoint and validate its schema."""
    try:
        payload = load_json(path)
    except Exception as e:
        return None, [f"could not read JSON: {e}"]

    if not isinstance(payload, dict):
        return None, [f"checkpoint root is {type(payload).__name__}, expected object"]

    outcomes_payload = payload.get("outcomes")
    if not isinstance(outcomes_payload, list):
        return None, ['missing "outcomes" list']

    if expected_metadata is not None:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return None, ["missing checkpoint metadata"]
        mismatches = []
        for key, expected_value in expected_metadata.items():
            if metadata.get(key) != expected_value:
                mismatches.append(
                    f"metadata {key}={metadata.get(key)!r} expected {expected_value!r}"
                )
        if mismatches:
            return None, mismatches

    outcomes = pd.DataFrame(outcomes_payload)
    issues = _validate_outcomes_frame(outcomes, require_nonempty=require_nonempty)
    if issues:
        return None, issues
    return outcomes, []


def _checkpoint_metadata(mode: str, scenario_signature: str | None = None) -> dict:
    metadata = {"probe_mode": mode}
    if mode == "openended":
        metadata["openended_prompt_style"] = OPENENDED_PROMPT_STYLE
        metadata["openended_assistant_system_prompt"] = OPENENDED_ASSISTANT_SYSTEM_PROMPT
        metadata["openended_judge_style"] = OPENENDED_JUDGE_STYLE
    if scenario_signature is not None:
        metadata["scenario_signature"] = scenario_signature
    return metadata


_GROUP_LABELS = {
    "personalprotective": {
        "value1_only": "personal",
        "shared": "shared",
        "value2_only": "protective",
    }
}
_VALUE_GROUP_CACHE: dict[str, dict[str, list[str]]] = {}


def _value_groups_for_set(value_set: str) -> dict[str, list[str]]:
    """Return value groups derived from the scenario CSV role structure."""
    if value_set in _VALUE_GROUP_CACHE:
        return _VALUE_GROUP_CACHE[value_set]

    path = SCENARIO_PATHS.get(value_set)
    if path is None or not Path(path).exists():
        _VALUE_GROUP_CACHE[value_set] = {}
        return {}

    df = pd.read_csv(path, usecols=["value1", "value2"])
    v1_values = set(df["value1"].dropna().astype(str))
    v2_values = set(df["value2"].dropna().astype(str))
    labels = _GROUP_LABELS.get(
        value_set,
        {"value1_only": "value1_side", "shared": "shared", "value2_only": "value2_side"},
    )

    groups: dict[str, list[str]] = {}
    v1_only = sorted(v1_values - v2_values)
    shared = sorted(v1_values & v2_values)
    v2_only = sorted(v2_values - v1_values)
    if v1_only:
        groups[labels["value1_only"]] = v1_only
    if shared:
        groups[labels["shared"]] = shared
    if v2_only:
        groups[labels["value2_only"]] = v2_only

    _VALUE_GROUP_CACHE[value_set] = groups
    return groups


def conv_checkpoint_path(
    model_key: str, value_set: str, group_key: str, num_turns: int, stance: str = "neutral"
) -> Path:
    """Path for a saved conversation (pair or scenario level)."""
    safe_key = group_key.replace(" ", "_").replace("/", "-")
    stance_tag = f"_{stance}" if stance != "neutral" else ""
    return (
        RESULTS_DIR / "conversations" / model_key / value_set
        / f"{safe_key}_{num_turns}t{stance_tag}.json"
    )


def t0_checkpoint_path(model_key: str, value_set: str, mode: str = "mcq") -> Path:
    mode_tag = f"_{mode}" if mode != "mcq" else ""
    return RESULTS_DIR / "checkpoints" / f"{model_key}_{value_set}_t0{mode_tag}.json"


def t1_checkpoint_path(
    model_key: str, value_set: str, group_key: str, num_turns: int,
    stance: str = "neutral", mode: str = "mcq",
) -> Path:
    safe_key = group_key.replace(" ", "_").replace("/", "-")
    stance_tag = f"_{stance}" if stance != "neutral" else ""
    mode_tag = f"_{mode}" if mode != "mcq" else ""
    return (
        RESULTS_DIR / "checkpoints"
        / f"{model_key}_{value_set}_{safe_key}_{num_turns}t_t1{stance_tag}{mode_tag}.json"
    )


def t0_result_path(model_key: str, value_set: str, mode: str = "mcq") -> Path:
    mode_tag = f"_{mode}" if mode != "mcq" else ""
    return RESULTS_DIR / "runs" / f"{model_key}_{value_set}_t0{mode_tag}.json"


def zero_turn_baseline_result_path(model_key: str, value_set: str, mode: str = "mcq") -> Path:
    mode_tag = f"_{mode}" if mode != "mcq" else ""
    return RESULTS_DIR / "runs" / f"{model_key}_{value_set}_0t{mode_tag}_baseline.json"


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
    stance: str,
    log: logging.Logger,
    force_recompute: bool = False,
) -> dict[str, dict[int, list[dict]]]:
    """Generate and cache conversations for every group × turn count.

    Returns:
        {group_key: {num_turns: conversation_list}}
    """
    max_turns = max(turn_counts)
    skipped_groups: list[tuple[str, str]] = []

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
        full_conv_path = conv_checkpoint_path(model_key, value_set, group_key, max_turns, stance)
        issues: list[str] = []

        seed = pick_seed_scenario(group_df)
        sim_prompt = make_scenario_conversation_prompt(
            description=seed["description"],
            stance=stance,
            value1=seed.get("value1", ""),
            value2=seed.get("value2", ""),
        )

        if full_conv_path.exists() and not force_recompute:
            cached_conv = load_json(full_conv_path)
            issues = _validate_cached_conversation(cached_conv, max_turns)
            if issues:
                log.warning(
                    "  Conv exists but is invalid, regenerating: %s (%dt, stance=%s) [%s]",
                    group_key, max_turns, stance, "; ".join(issues)
                )
            else:
                log.info(f"  Conv exists: {group_key} ({max_turns}t, stance={stance})")
                full_conv = cached_conv
        else:
            if full_conv_path.exists() and force_recompute:
                log.info(
                    "  Force recompute: ignoring cached conv for %s (%dt, stance=%s)",
                    group_key, max_turns, stance,
                )
        if force_recompute or not full_conv_path.exists() or issues:
            full_conv = None
            last_issues: list[str] = []
            for attempt in range(1, 4):
                log.info(
                    f"  Generating conv: {group_key} ({max_turns}t, stance={stance}, attempt={attempt})"
                )
                try:
                    candidate = generate_conversation(
                        model=model,
                        persona=model_key,
                        domain="",  # unused — overridden by system_prompt
                        num_turns=max_turns,
                        user_sim=user_sim,
                        system_prompt=sim_prompt,
                    )
                except Exception as e:
                    last_issues = [f"generation error: {type(e).__name__}: {e}"]
                    log.warning(
                        "    Conversation generation failed for %s [attempt=%d: %s]",
                        group_key,
                        attempt,
                        last_issues[0],
                    )
                    continue
                last_issues = _validate_cached_conversation(candidate, max_turns)
                if not last_issues:
                    full_conv = candidate
                    save_json(full_conv_path, full_conv)
                    break
                log.warning(
                    "    Invalid generated conversation for %s [%s]",
                    group_key, "; ".join(last_issues),
                )
            if full_conv is None:
                reason = "; ".join(last_issues) if last_issues else "unknown validation failure"
                log.error(
                    "  Skipping group after 3 invalid attempts: %s (%dt, stance=%s) [%s]",
                    group_key,
                    max_turns,
                    stance,
                    reason,
                )
                skipped_groups.append((group_key, reason))
                continue

        # Slice to each requested turn count
        conversations[group_key] = {}
        for num_turns in turn_counts:
            conversations[group_key][num_turns] = full_conv[: num_turns * 2]

    if skipped_groups:
        log.warning(
            "  Skipped %d groups for stance=%s due to repeated invalid conversations",
            len(skipped_groups),
            stance,
        )

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
    stance: str = "neutral",
) -> dict:
    """Compute per-pair flip consistency and overall Bradley-Terry drift.

    Per-pair directional consistency is the key new metric: for each value pair,
    after a conversation seeded by that pair, what fraction of flips go toward
    each value?  High consistency → the context reliably primes that value.

    For non-neutral stances, per_value_flip_stats and role_filtered_drift only
    count scenarios where each value was in the favored role:
      - pro_v1: count value V only in pairs where V == value1
      - pro_v2: count value V only in pairs where V == value2
    This lets you see whether the stance actually moved the value it was
    supposed to push, rather than diluting signal with unfavored-role appearances.
    """
    t0_issues = _validate_outcomes_frame(outcomes_t0, require_nonempty=True)
    if t0_issues:
        log.warning(
            "  Skipping drift analysis for %s/%dt (stance=%s): invalid T0 outcomes [%s]",
            value_set,
            num_turns,
            stance,
            "; ".join(t0_issues),
        )
        return {}

    # --- overall T1: aggregate all per-group T1 outcomes ---
    all_t1_parts = list(outcomes_t1_by_group.values())

    if not all_t1_parts:
        return {}

    outcomes_t1_all = pd.concat(all_t1_parts, ignore_index=True)
    t1_issues = _validate_outcomes_frame(outcomes_t1_all, require_nonempty=True)
    if t1_issues:
        log.warning(
            "  Skipping drift analysis for %s/%dt (stance=%s): invalid T1 outcomes [%s]",
            value_set,
            num_turns,
            stance,
            "; ".join(t1_issues),
        )
        return {}
    outcomes_t1_all = outcomes_t1_all.drop_duplicates(subset=["scenario_id"])

    ranking_t0 = fit_bradley_terry(outcomes_t0)
    ranking_t1 = fit_bradley_terry(outcomes_t1_all)
    drift = compute_drift(ranking_t0, ranking_t1)
    flip_stats = compute_answer_flip_rate(outcomes_t0, outcomes_t1_all)
    value_groups = _value_groups_for_set(value_set)
    group_average_ranks_t0 = compute_average_ranks_by_group(ranking_t0, value_groups)
    group_average_ranks_t1 = compute_average_ranks_by_group(ranking_t1, value_groups)
    group_average_rank_delta = {
        group_name: float(group_average_ranks_t1[group_name] - group_average_ranks_t0[group_name])
        for group_name in sorted(set(group_average_ranks_t0) & set(group_average_ranks_t1))
    }

    # Re-group from the actual matched T1 outcomes so downstream pair-level
    # metrics work for both pair-grouped and scenario-grouped runs.
    t1_by_pair = {
        f"{v1}_vs_{v2}": sub
        for (v1, v2), sub in outcomes_t1_all.groupby(["value1", "value2"])
    }

    # --- per-pair directional consistency ---
    pair_consistency: dict[str, dict] = {}
    for (v1, v2), pair_t0 in outcomes_t0.groupby(["value1", "value2"]):
        pair_key = f"{v1}_vs_{v2}"
        pair_t1 = t1_by_pair.get(pair_key)
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
            "directional_consistency": float(consistency),
        }

    # --- per-value flip stats ---
    # For non-neutral stances, restrict each value's stats to scenarios where
    # it appeared in the "favored role" (v1 for pro_v1, v2 for pro_v2).
    # For neutral, count all appearances (existing behaviour).
    all_values = sorted(set(outcomes_t0["value1"]) | set(outcomes_t0["value2"]))
    per_value_flip_stats: dict[str, dict] = {}
    for value in all_values:
        total_appearances = 0
        flips_toward = 0
        flips_away = 0
        for (v1, v2), pair_t0 in outcomes_t0.groupby(["value1", "value2"]):
            if value not in (v1, v2):
                continue
            # Skip pairs where this value is NOT in the favored role
            if stance == "pro_v1" and value != v1:
                continue
            if stance == "pro_v2" and value != v2:
                continue
            pair_key = f"{v1}_vs_{v2}"
            pair_t1 = t1_by_pair.get(pair_key)
            if pair_t1 is None or pair_t1.empty:
                continue
            merged = pair_t0.merge(pair_t1, on="scenario_id", suffixes=("_t0", "_t1"))
            if merged.empty:
                continue
            total_appearances += len(merged)
            flipped = merged[merged["winner_t0"] != merged["winner_t1"]]
            flips_toward += int(
                ((flipped["winner_t0"] != value) & (flipped["winner_t1"] == value)).sum()
            )
            flips_away += int(
                ((flipped["winner_t0"] == value) & (flipped["winner_t1"] != value)).sum()
            )

        n_flipped = flips_toward + flips_away
        per_value_flip_stats[value] = {
            "total_appearances": total_appearances,
            "n_flipped": n_flipped,
            "flips_toward": flips_toward,
            "flips_away": flips_away,
            "flip_rate_toward": (
                flips_toward / total_appearances if total_appearances > 0 else float("nan")
            ),
            "flip_rate_away": (
                flips_away / total_appearances if total_appearances > 0 else float("nan")
            ),
            "net_flip_rate": (
                (flips_toward - flips_away) / total_appearances
                if total_appearances > 0 else float("nan")
            ),
        }

    # --- role-filtered BT drift (non-neutral stances only) ---
    # For each value V, compute its BT ability delta using only outcomes from
    # pairs where V was in the favored role.  This isolates the effect of the
    # stance on the value it was designed to push.
    role_filtered_per_value_delta: dict[str, float] = {}
    if stance != "neutral":
        role_col = "value1" if stance == "pro_v1" else "value2"
        for target_value in sorted(set(outcomes_t0[role_col])):
            # Collect T0 and T1 outcomes for pairs where target_value is in role
            role_t0_parts = []
            role_t1_parts = []
            for (v1, v2), pair_t0 in outcomes_t0.groupby(["value1", "value2"]):
                if (stance == "pro_v1" and v1 != target_value) or \
                   (stance == "pro_v2" and v2 != target_value):
                    continue
                pair_key = f"{v1}_vs_{v2}"
                pair_t1 = t1_by_pair.get(pair_key)
                if pair_t1 is None or pair_t1.empty:
                    continue
                role_t0_parts.append(pair_t0)
                role_t1_parts.append(pair_t1)
            if not role_t0_parts or not role_t1_parts:
                continue
            role_t0 = pd.concat(role_t0_parts, ignore_index=True)
            role_t1 = pd.concat(role_t1_parts, ignore_index=True).drop_duplicates("scenario_id")
            try:
                rk_t0 = fit_bradley_terry(role_t0)
                rk_t1 = fit_bradley_terry(role_t1)
                ab_t0 = {row["value"]: row["ability"] for row in rk_t0.to_dict(orient="records")}
                ab_t1 = {row["value"]: row["ability"] for row in rk_t1.to_dict(orient="records")}
                if target_value in ab_t0 and target_value in ab_t1:
                    role_filtered_per_value_delta[target_value] = float(
                        ab_t1[target_value] - ab_t0[target_value]
                    )
            except Exception:
                pass

    log.info(
        f"  {value_set}/{num_turns}t (stance={stance}): "
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

    # Log top movers by net flip rate (using role-filtered stats for non-neutral stances)
    movers = sorted(
        per_value_flip_stats.items(),
        key=lambda kv: abs(kv[1]["net_flip_rate"]) if not np.isnan(kv[1]["net_flip_rate"]) else 0,
        reverse=True,
    )[:3]
    role_note = "" if stance == "neutral" else f" [role-filtered for {stance}]"
    for v, s in movers:
        if not np.isnan(s["net_flip_rate"]):
            log.info(
                f"    {v}{role_note}: toward={s['flip_rate_toward']:.2f}, "
                f"away={s['flip_rate_away']:.2f}, net={s['net_flip_rate']:+.2f} "
                f"(n={s['total_appearances']})"
            )
    # --- per-value flip stats (all appearances — never role-filtered) ---
    # Computed unconditionally so stance plots can show all values, including
    # those on the non-favored side of the pairing.
    per_value_flip_stats_overall: dict[str, dict] = {}
    for value in all_values:
        total_appearances = 0
        flips_toward = 0
        flips_away = 0
        for (v1, v2), pair_t0 in outcomes_t0.groupby(["value1", "value2"]):
            if value not in (v1, v2):
                continue
            pair_key = f"{v1}_vs_{v2}"
            pair_t1 = t1_by_pair.get(pair_key)
            if pair_t1 is None or pair_t1.empty:
                continue
            merged = pair_t0.merge(pair_t1, on="scenario_id", suffixes=("_t0", "_t1"))
            if merged.empty:
                continue
            total_appearances += len(merged)
            flipped = merged[merged["winner_t0"] != merged["winner_t1"]]
            flips_toward += int(
                ((flipped["winner_t0"] != value) & (flipped["winner_t1"] == value)).sum()
            )
            flips_away += int(
                ((flipped["winner_t0"] == value) & (flipped["winner_t1"] != value)).sum()
            )
        n_flipped = flips_toward + flips_away
        per_value_flip_stats_overall[value] = {
            "total_appearances": total_appearances,
            "n_flipped": n_flipped,
            "flips_toward": flips_toward,
            "flips_away": flips_away,
            "flip_rate_toward": (
                flips_toward / total_appearances if total_appearances > 0 else float("nan")
            ),
            "flip_rate_away": (
                flips_away / total_appearances if total_appearances > 0 else float("nan")
            ),
            "net_flip_rate": (
                (flips_toward - flips_away) / total_appearances
                if total_appearances > 0 else float("nan")
            ),
        }

    if role_filtered_per_value_delta:
        top_rf = sorted(role_filtered_per_value_delta.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        log.info(f"  Role-filtered BT delta top movers: " + ", ".join(f"{v}={d:+.3f}" for v, d in top_rf))

    result = {
        "model": model_key,
        "value_set": value_set,
        "num_turns": num_turns,
        "ranking_t0": ranking_t0.to_dict(orient="records"),
        "ranking_t1": ranking_t1.to_dict(orient="records"),
        "group_average_ranks_t0": group_average_ranks_t0,
        "group_average_ranks_t1": group_average_ranks_t1,
        "group_average_rank_delta": group_average_rank_delta,
        "drift": drift,
        "flip_stats": flip_stats,
        "pair_consistency": pair_consistency,
        # role-filtered (or same as overall for neutral stance)
        "per_value_flip_stats": per_value_flip_stats,
        # always unfiltered — all pairs, all appearances
        "per_value_flip_stats_overall": per_value_flip_stats_overall,
    }
    if role_filtered_per_value_delta:
        result["role_filtered_drift"] = {"per_value_delta": role_filtered_per_value_delta}
    return result


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
    stances: list[str],
    mode: str,
    judge_client,
    judge_model: str,
    log: logging.Logger,
    use_vllm: bool = False,
    gpu_ids: list[int] | None = None,
    force_recompute: bool = False,
) -> list[dict]:
    """Run all (value_set × stance × num_turns) conditions for one model.

    The model is loaded once and unloaded after all stances are done.
    T0 probing is shared across stances (same baseline, no context).
    """
    log.info(f"Loading model: {model_key} ({model_info['hf_id']})")

    # Check if this is an OpenAI model
    if model_info.get("is_openai"):
        log.info(f"  Using OpenAI API model: {model_info['hf_id']}")
        from run_alignment_target_experiment import OpenAIModel
        api_key = os.environ.get("OPENAI_API_KEY")
        model = OpenAIModel(model_id=model_info["hf_id"], api_key=api_key)
    # Check if this is an Anthropic model
    elif model_info.get("is_anthropic"):
        log.info(f"  Using Anthropic API model: {model_info['hf_id']}")
        from run_alignment_target_experiment import AnthropicModel
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        model = AnthropicModel(model_id=model_info["hf_id"], api_key=api_key)
    # Use vLLM if requested and available
    elif use_vllm:
        if not HAS_VLLM:
            log.error("--use-vllm specified but vLLM is not installed. Install with: pip install vllm")
            return []
        log.info(f"  Using vLLM with gpu_ids={gpu_ids}")
        model = AlignmentModelVLLM(
            model_info["hf_id"],
            is_base_model=model_info.get("is_base_model", False),
            gpu_ids=gpu_ids,
        )
    else:
        model = AlignmentModel(
            model_info["hf_id"],
            is_base_model=model_info.get("is_base_model", False),
        )

    all_results = []
    requested_zero_turn = any(t == 0 for t in turn_counts)
    positive_turn_counts = [t for t in turn_counts if t > 0]

    if force_recompute:
        log.info("  Force recompute enabled: ignoring cached conversations and checkpoints")

    for value_set in value_sets:
        scenarios = load_scenarios(value_set, max_scenarios=num_scenarios)
        scenarios_signature = scenario_sample_signature(scenarios)
        log.info(f"  {value_set}: {len(scenarios)} scenarios")

        # Initialize scenario-set-specific A/B swap decisions for MCQ probing.
        # The checkpoint is reused only when the sampled scenario set matches.
        from probing import load_swap_decisions
        swap_cp = None if force_recompute else RESULTS_DIR / "checkpoints" / f"swap_decisions_{value_set}.json"
        swap_decisions = load_swap_decisions(scenarios, swap_cp, seed=42)
        swapped_count = sum(swap_decisions.values())
        log.info(f"  Initialized random A/B swap decisions ({swapped_count}/{len(scenarios)} swapped)")

        # Save scenario distribution (informational, not per-model)
        dist_path = RESULTS_DIR / f"scenario_distribution_{value_set}.json"
        if not dist_path.exists():
            save_json(dist_path, scenario_distribution_report(scenarios))

        # --- T0: no context — shared across all stances ---
        t0_cp = t0_checkpoint_path(model_key, value_set, mode)
        outcomes_t0 = None
        t0_metadata = _checkpoint_metadata(mode, scenarios_signature)
        if t0_cp.exists() and not force_recompute:
            log.info("  T0 checkpoint exists, loading")
            outcomes_t0, t0_issues = _load_outcomes_checkpoint(
                t0_cp,
                require_nonempty=True,
                expected_metadata=t0_metadata,
            )
            if t0_issues:
                log.warning(
                    "  T0 checkpoint invalid, recomputing: %s [%s]",
                    t0_cp.name,
                    "; ".join(t0_issues),
                )
        elif t0_cp.exists() and force_recompute:
            log.info("  Force recompute: ignoring cached T0 checkpoint %s", t0_cp.name)

        if outcomes_t0 is None:
            log.info(f"  Running T0 probing ({mode})...")
            if mode == "openended":
                outcomes_t0 = probe_values_openended(
                    model, model_key, scenarios,
                    judge_client=judge_client, judge_model=judge_model, context=None,
                )
            else:
                outcomes_t0 = probe_values(model, model_key, scenarios, context=None)
            t0_issues = _validate_outcomes_frame(outcomes_t0, require_nonempty=True)
            if t0_issues:
                log.error(
                    "  T0 probing produced unusable outcomes for %s/%s [%s]. Skipping value set.",
                    model_key,
                    value_set,
                    "; ".join(t0_issues),
                )
                continue
            save_json(
                t0_cp,
                {
                    "metadata": t0_metadata,
                    "outcomes": outcomes_t0.to_dict(orient="records"),
                },
            )
            log.info(f"  T0: {len(outcomes_t0)} outcomes")

        ranking_t0 = fit_bradley_terry(outcomes_t0)
        group_average_ranks_t0 = compute_average_ranks_by_group(
            ranking_t0, _value_groups_for_set(value_set)
        )
        save_json(
            t0_result_path(model_key, value_set, mode),
            {
                "model": model_key,
                "value_set": value_set,
                "mode": mode,
                "ranking_t0": ranking_t0.to_dict(orient="records"),
                "group_average_ranks_t0": group_average_ranks_t0,
                "n_outcomes": int(len(outcomes_t0)),
                "scenario_signature": scenarios_signature,
            },
        )

        if requested_zero_turn:
            baseline_result = {
                "model": model_key,
                "value_set": value_set,
                "mode": mode,
                "num_turns": 0,
                "stance": "baseline",
                "baseline_only": True,
                "ranking_t0": ranking_t0.to_dict(orient="records"),
                "group_average_ranks_t0": group_average_ranks_t0,
                "n_outcomes": int(len(outcomes_t0)),
                "scenario_signature": scenarios_signature,
            }
            save_json(
                zero_turn_baseline_result_path(model_key, value_set, mode),
                baseline_result,
            )
            all_results.append(baseline_result)
            log.info("  Saved 0-turn baseline artifact")

        if not positive_turn_counts:
            log.info("  turn_counts contains only 0; skipping conversation generation and T1 probing")
            continue

        for stance in stances:
            log.info(f"  stance={stance}")

            # --- Generate / load per-group conversations ---
            conversations = generate_group_conversations(
                model=model,
                model_key=model_key,
                user_sim=user_sim,
                scenarios=scenarios,
                value_set=value_set,
                group_by=group_by,
                turn_counts=positive_turn_counts,
                stance=stance,
                log=log,
                force_recompute=force_recompute,
            )

            # --- T1: per-group probing ---
            for num_turns in positive_turn_counts:
                outcomes_t1_by_group: dict[str, pd.DataFrame] = {}

                for group_key, conv_by_turns in conversations.items():
                    t1_cp = t1_checkpoint_path(model_key, value_set, group_key, num_turns, stance, mode)

                    context = conv_by_turns.get(num_turns)
                    if context is None:
                        log.warning(
                            "  Missing %dt conversation slice for %s (stance=%s); skipping group",
                            num_turns,
                            group_key,
                            stance,
                        )
                        continue

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
                    group_signature = scenario_sample_signature(group_scenarios)
                    t1_metadata = _checkpoint_metadata(mode, group_signature)
                    if t1_cp.exists() and not force_recompute:
                        cached_t1, t1_issues = _load_outcomes_checkpoint(
                            t1_cp,
                            require_nonempty=True,
                            expected_metadata=t1_metadata,
                        )
                        if t1_issues:
                            log.warning(
                                "  T1 checkpoint invalid, recomputing: %s [%s]",
                                t1_cp.name,
                                "; ".join(t1_issues),
                            )
                        else:
                            outcomes_t1_by_group[group_key] = cached_t1
                            continue
                    elif t1_cp.exists() and force_recompute:
                        log.info("  Force recompute: ignoring cached T1 checkpoint %s", t1_cp.name)

                    log.info(f"  T1 {group_key}/{num_turns}t (stance={stance}, mode={mode}): "
                             f"{len(group_scenarios)} scenarios")
                    if mode == "openended":
                        outcomes_t1 = probe_values_openended(
                            model, model_key, group_scenarios,
                            judge_client=judge_client, judge_model=judge_model, context=context,
                        )
                    else:
                        outcomes_t1 = probe_values(model, model_key, group_scenarios, context=context)
                    t1_issues = _validate_outcomes_frame(outcomes_t1, require_nonempty=True)
                    if t1_issues:
                        log.warning(
                            "  T1 probing produced unusable outcomes for %s [%s]",
                            group_key,
                            "; ".join(t1_issues),
                        )
                        continue
                    save_json(
                        t1_cp,
                        {
                            "metadata": t1_metadata,
                            "outcomes": outcomes_t1.to_dict(orient="records"),
                        },
                    )
                    outcomes_t1_by_group[group_key] = outcomes_t1

                # Analyze drift for this (value_set, stance, num_turns) condition
                result = analyze_pair_drift(
                    outcomes_t0=outcomes_t0,
                    outcomes_t1_by_group=outcomes_t1_by_group,
                    value_set=value_set,
                    model_key=model_key,
                    num_turns=num_turns,
                    log=log,
                    stance=stance,
                )
                if result:
                    result["stance"] = stance
                    result["mode"] = mode
                    all_results.append(result)
                    stance_tag = f"_{stance}" if stance != "neutral" else ""
                    mode_tag = f"_{mode}" if mode != "mcq" else ""
                    result_path = (
                        RESULTS_DIR / "runs"
                        / f"{model_key}_{value_set}_{num_turns}t{stance_tag}{mode_tag}.json"
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

def save_summary_csv(all_results: list[dict], log: logging.Logger, run_id: str = "latest"):
    """Save results to a per-run folder (preserves previous runs)."""
    def _add_ranking_columns(row: dict, ranking_list: list[dict], prefix: str):
        if not ranking_list:
            return
        ordered = sorted(
            (entry for entry in ranking_list if isinstance(entry, dict) and "value" in entry),
            key=lambda entry: entry.get("ability", float("-inf")),
            reverse=True,
        )
        if ordered:
            row[f"top_value_{prefix}"] = ordered[0]["value"]
        for rank, entry in enumerate(ordered, start=1):
            value = entry["value"]
            row[f"rank_{prefix}_{value}"] = rank
            row[f"ability_{prefix}_{value}"] = entry.get("ability")
            if "ci_lower" in entry:
                row[f"ci_lower_{prefix}_{value}"] = entry.get("ci_lower")
            if "ci_upper" in entry:
                row[f"ci_upper_{prefix}_{value}"] = entry.get("ci_upper")

    rows = []
    for r in all_results:
        row = {
            "model": r["model"],
            "value_set": r["value_set"],
            "num_turns": r["num_turns"],
            "stance": r.get("stance", "neutral"),
            "mode": r.get("mode", "mcq"),
            "baseline_only": bool(r.get("baseline_only", False)),
            "n_outcomes": r.get("n_outcomes"),
        }
        _add_ranking_columns(row, r.get("ranking_t0", []), "t0")
        for group_name, avg_rank in (r.get("group_average_ranks_t0") or {}).items():
            row[f"avg_rank_t0_{group_name}"] = avg_rank

        if r.get("baseline_only", False):
            rows.append(row)
            continue

        row["l2_distance"] = r["drift"]["l2_distance"]
        row["rank_correlation"] = r["drift"]["rank_correlation"]
        row["rank_correlation_pvalue"] = r["drift"]["rank_correlation_pvalue"]
        row["overall_flip_rate"] = r["flip_stats"]["overall_flip_rate"]
        row["n_matched"] = r["flip_stats"]["n_matched"]
        _add_ranking_columns(row, r.get("ranking_t1", []), "t1")
        for group_name, avg_rank in (r.get("group_average_ranks_t1") or {}).items():
            row[f"avg_rank_t1_{group_name}"] = avg_rank
        for group_name, delta_rank in (r.get("group_average_rank_delta") or {}).items():
            row[f"delta_avg_rank_{group_name}"] = delta_rank

        # Per-value delta columns
        for val, delta in r["drift"]["per_value_delta"].items():
            row[f"delta_{val}"] = delta

        # Per-value flip rate columns
        pvfs = r.get("per_value_flip_stats", {})
        for val, stats in pvfs.items():
            row[f"flip_toward_{val}"] = stats["flip_rate_toward"]
            row[f"flip_away_{val}"] = stats["flip_rate_away"]
            row[f"net_flip_{val}"] = stats["net_flip_rate"]

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
            row["n_high_consistency_pairs"] = int(
                sum(1 for c in consistencies if c >= 0.7)
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    # Create run-specific folder
    run_dir = RESULTS_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    out_path = run_dir / "all_results.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Summary saved: {out_path}")


# ---------------------------------------------------------------------------
# Multi-GPU support
# ---------------------------------------------------------------------------

def allocate_gpus(model_keys: list[str], available_gpu_ids: list[int]) -> dict[str, list[int]]:
    """Distribute models across available GPUs in round-robin fashion.

    Args:
        model_keys: List of model keys to allocate
        available_gpu_ids: List of available GPU IDs

    Returns:
        dict mapping model_key -> list of GPU IDs
    """
    allocation = {}
    n_gpus = len(available_gpu_ids)

    if n_gpus == 0:
        raise ValueError("No GPUs available")

    # Round-robin assignment: distribute GPUs evenly
    gpus_per_model = max(1, n_gpus // len(model_keys))

    for i, model_key in enumerate(model_keys):
        start_idx = (i * gpus_per_model) % n_gpus
        gpu_list = []
        for j in range(gpus_per_model):
            gpu_id = available_gpu_ids[(start_idx + j) % n_gpus]
            gpu_list.append(gpu_id)
        allocation[model_key] = gpu_list

    return allocation


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
        "--models", type=str, nargs="+", default=["tulu-3-sft"],
        help=f"One or more model keys (default: tulu-3-sft). Available: {list(ALIGNMENT_MODELS.keys())}",
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
    parser.add_argument("--simulator", type=str, default="openai",
                        choices=["anthropic", "openai"],
                        help="User simulator provider (default: openai for gpt-4o-mini)")
    parser.add_argument("--simulator-model", type=str, default="gpt-4o-mini",
                        help="Simulator model ID (default: gpt-4o-mini for speed and cost)")
    parser.add_argument("--simulator-base-url", type=str, default=None)
    parser.add_argument("--simulator-api-key", type=str, default=None)

    # Probing mode
    parser.add_argument(
        "--mode", type=str, default="mcq", choices=["mcq", "openended"],
        help=(
            "mcq (default): model picks A or B. "
            "openended: model gives a free-form answer, a judge classifies it as A or B."
        ),
    )

    # Judge (open-ended mode only)
    parser.add_argument("--judge", type=str, default="openai", choices=["anthropic", "openai"],
                        help="Judge provider for open-ended mode (default: openai)")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini",
                        help="Judge model ID (default: gpt-4o-mini)")
    parser.add_argument("--judge-api-key", type=str, default=None,
                        help="API key for the judge (falls back to OPENAI_API_KEY / ANTHROPIC_API_KEY)")

    # Conversation stance(s)
    parser.add_argument(
        "--stances", type=str, nargs="+", default=["neutral"],
        choices=["neutral", "pro_v1", "pro_v2"],
        help=(
            "One or more stances (default: neutral). "
            "neutral: simulator is genuinely conflicted. "
            "pro_v1: simulator leans toward the first value. "
            "pro_v2: simulator leans toward the second value. "
            "v1/v2 are determined per-pair (the value1/value2 columns of the scenario CSV)."
        ),
    )

    # Experiment scope
    parser.add_argument(
        "--value-sets", type=str, nargs="+", default=["HHH", "personalprotective"],
        help="Value sets to run (default: HHH personalprotective)",
    )
    parser.add_argument(
        "--turn-counts", type=int, nargs="+", default=[5, 10],
        help="Conversation turn counts (default: 5 10). Use 0 for T0-only baseline evaluation with no T1.",
    )
    parser.add_argument(
        "--num-scenarios", type=int, default=0,
        help="Max scenarios per value set (0 = all). 200-300 recommended for speed.",
    )

    # Multi-GPU configuration
    parser.add_argument(
        "--use-vllm", action="store_true",
        help="Use vLLM with tensor parallelism for multi-GPU inference (faster than device_map=auto)",
    )
    parser.add_argument(
        "--gpu-ids", type=str, default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2,3'). If not specified, auto-detects all available GPUs.",
    )
    parser.add_argument(
        "--skip-postprocess", action="store_true",
        help="Run the experiment and save per-condition JSONs, but skip summary CSV and plot generation.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore saved T0/T1 checkpoints and cached conversations; recompute all artifacts for this run.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    from datetime import datetime
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

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
    log.info(f"  models={args.models}, group_by={args.group_by}, mode={args.mode}, "
             f"stances={args.stances}, value_sets={args.value_sets}, "
             f"turn_counts={args.turn_counts}, num_scenarios={args.num_scenarios}")

    # Validate models
    unknown_models = [m for m in args.models if m not in ALIGNMENT_MODELS]
    if unknown_models:
        log.error(f"Unknown model key(s): {unknown_models}. Available: {list(ALIGNMENT_MODELS.keys())}")
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

    # Build judge client (open-ended mode only)
    judge_client = None
    if args.mode == "openended":
        judge_key = args.judge_api_key
        if args.judge == "openai":
            if not judge_key:
                judge_key = os.environ.get("OPENAI_API_KEY")
            if not judge_key:
                log.error("Open-ended mode requires an OpenAI API key (--judge-api-key or OPENAI_API_KEY)")
                sys.exit(1)
            from openai import OpenAI
            judge_client = OpenAI(api_key=judge_key)
        else:
            if not judge_key:
                judge_key = os.environ.get("ANTHROPIC_API_KEY")
            if not judge_key:
                log.error("Open-ended mode requires an Anthropic API key (--judge-api-key or ANTHROPIC_API_KEY)")
                sys.exit(1)
            import anthropic
            judge_client = anthropic.Anthropic(api_key=judge_key)
        log.info(f"  Judge: {args.judge} / {args.judge_model}")

    # GPU detection and allocation
    if args.use_vllm:
        # Parse GPU IDs or auto-detect
        if args.gpu_ids:
            available_gpus = [int(g.strip()) for g in args.gpu_ids.split(',')]
        else:
            available_gpus = list(range(torch.cuda.device_count()))

        if not available_gpus:
            log.error("--use-vllm specified but no GPUs detected. Use CPU or check CUDA setup.")
            sys.exit(1)

        log.info(f"GPU allocation enabled: {len(available_gpus)} GPUs available")

        # Allocate GPUs to models
        gpu_allocation = allocate_gpus(args.models, available_gpus)
        for model_key, gpu_ids in gpu_allocation.items():
            log.info(f"  {model_key}: GPU {gpu_ids}")
    else:
        gpu_allocation = None

    # Run all models sequentially; load each model once, iterate stances inside
    all_results = []
    for model_key in args.models:
        log.info(f"--- Model: {model_key} ---")

        # Get GPU IDs for this model if using vLLM
        model_gpu_ids = None
        if args.use_vllm and gpu_allocation:
            model_gpu_ids = gpu_allocation[model_key]

        results = run_experiment(
            model_key=model_key,
            model_info=ALIGNMENT_MODELS[model_key],
            user_sim=user_sim,
            value_sets=args.value_sets,
            group_by=args.group_by,
            turn_counts=args.turn_counts,
            num_scenarios=args.num_scenarios,
            stances=args.stances,
            mode=args.mode,
            judge_client=judge_client,
            judge_model=args.judge_model,
            log=log,
            use_vllm=args.use_vllm,
            gpu_ids=model_gpu_ids,
            force_recompute=args.force_recompute,
        )
        all_results.extend(results)

    if not args.skip_postprocess:
        if all_results:
            save_summary_csv(all_results, log, run_id=run_id)
            log.info("Generating plots...")
            try:
                generate_scenario_experiment_plots(all_results, RESULTS_DIR, run_id=run_id)
                log.info(f"Plots saved to {RESULTS_DIR / 'plots' / run_id}")
            except Exception as e:
                log.warning(f"Plot generation failed (non-fatal): {e}")
        else:
            log.info("No T1 results to summarize; skipped summary CSV and plots")
    else:
        log.info("Skipping summary CSV / plot generation (--skip-postprocess)")
    log.info("Done.")


if __name__ == "__main__":
    main()
