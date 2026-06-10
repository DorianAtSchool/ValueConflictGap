"""Run a scenario-grounded value-vs-action bridge experiment.

This treats ConflictScope action selection as a scenario-grounded "task 2" and
adds a matched "task 1" that asks which value should be supported more in the
same scenario. The main comparison is the per-value gap between value selection
and action selection under the same context.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from run_alignment_target_experiment import ALIGNMENT_MODELS, AlignmentModel, OpenAIModel, build_user_simulator
from run_alignment_target_experiment_openended import build_judge_client
try:
    from alignmentmodel_vllm import AlignmentModelVLLM
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    AlignmentModelVLLM = None

from config import VALUE_SETS
from probing import (
    probe_action_choices_openended,
    load_scenarios,
    load_swap_decisions,
    load_value_descriptions,
    probe_value_choices,
    probe_value_choices_openended,
    probe_values,
    probe_values_openended,
)
from run_scenario_conversation_experiment import generate_group_conversations
from scenario_value_action_analysis import (
    merge_value_action_outcomes,
    selection_gap_by_pair,
    selection_gap_by_value,
    summarize_value_action_consistency,
)


RESULTS_DIR = Path(
    os.environ.get(
        "PERSONA_DRIFTING_RESULTS_DIR",
        Path(__file__).resolve().parent / "results" / "scenario_value_action",
    )
)


def parse_args():
    parser = argparse.ArgumentParser(description="Scenario-grounded value-vs-action bridge experiment")
    parser.add_argument("--models", nargs="+", required=True, help=f"Available: {list(ALIGNMENT_MODELS.keys())}")
    parser.add_argument("--value-sets", nargs="+", default=["HHH", "personalprotective"], choices=VALUE_SETS)
    parser.add_argument("--group-by", choices=["pair", "scenario"], default="pair")
    parser.add_argument("--stances", nargs="+", choices=["neutral", "pro_v1", "pro_v2"], default=["neutral"])
    parser.add_argument("--mode", choices=["mcq", "openended"], default="mcq")
    parser.add_argument(
        "--value-text",
        choices=["label", "description", "label_description"],
        default=None,
        help=(
            "How to render Task 1 values in prompts. Defaults to label_description "
            "for MCQ compatibility and hidden for open-ended compatibility."
        ),
    )
    parser.add_argument(
        "--ask-prioritize-over-others",
        action="store_true",
        help="Add an explicit Task 1 line asking whether one value should take priority over other relevant values.",
    )
    parser.add_argument("--turn-counts", nargs="+", type=int, default=[0, 5])
    parser.add_argument("--num-scenarios", type=int, default=0)
    parser.add_argument("--simulator", type=str, default="openai", choices=["anthropic", "openai"])
    parser.add_argument("--simulator-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--simulator-api-key", type=str, default=None)
    parser.add_argument("--simulator-base-url", type=str, default=None)
    parser.add_argument("--judge", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--judge-api-key", type=str, default=None)
    parser.add_argument("--openai-api-key", type=str, default=None)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _build_target_model(model_key: str, args):
    info = ALIGNMENT_MODELS[model_key]
    if info.get("is_openai"):
        return OpenAIModel(model_id=info["hf_id"], api_key=args.openai_api_key)
    if info.get("is_anthropic"):
        from run_alignment_target_experiment import AnthropicModel

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        return AnthropicModel(model_id=info["hf_id"], api_key=api_key)
    if args.use_vllm:
        if not HAS_VLLM:
            raise ImportError("vLLM requested but not installed")
        return AlignmentModelVLLM(info["hf_id"], is_base_model=info.get("is_base_model", False))
    return AlignmentModel(info["hf_id"], is_base_model=info.get("is_base_model", False))


def _probe_target(
    *,
    probe_target: str,
    mode: str,
    model,
    model_key: str,
    scenarios: pd.DataFrame,
    value_descriptions: dict[str, str],
    judge_client,
    judge_model: str,
    context: list[dict] | None,
    value_text_mode: str | None,
    ask_prioritize_over_others: bool,
) -> pd.DataFrame:
    if probe_target == "value":
        if mode == "openended":
            return probe_value_choices_openended(
                model,
                model_key,
                scenarios,
                value_descriptions,
                judge_client,
                judge_model,
                context=context,
                value_text_mode=value_text_mode,
                ask_prioritize_over_others=ask_prioritize_over_others,
            )
        return probe_value_choices(
            model,
            model_key,
            scenarios,
            value_descriptions,
            context=context,
            value_text_mode=value_text_mode or "label_description",
            ask_prioritize_over_others=ask_prioritize_over_others,
        )

    if mode == "openended":
        return probe_action_choices_openended(
            model,
            model_key,
            scenarios,
            judge_client,
            judge_model,
            context=context,
        )
    return probe_values(model, model_key, scenarios, context=context)


def _rows_for_group(scenarios: pd.DataFrame, group_by: str, group_key: str) -> pd.DataFrame:
    if group_by == "pair":
        value1, value2 = group_key.split("_vs_", 1)
        return scenarios[(scenarios["value1"] == value1) & (scenarios["value2"] == value2)]

    if "scenario_id" in scenarios.columns:
        return scenarios[scenarios["scenario_id"].astype(str) == group_key]
    return scenarios[scenarios.index.astype(str) == group_key]


def _attach_metadata(
    outcomes: pd.DataFrame,
    *,
    model_key: str,
    value_set: str,
    stance: str,
    mode: str,
    num_turns: int,
    probe_target: str,
    value_text_mode: str,
    ask_prioritize_over_others: bool,
) -> pd.DataFrame:
    if outcomes.empty:
        return outcomes
    enriched = outcomes.copy()
    enriched["model"] = model_key
    enriched["value_set"] = value_set
    enriched["stance"] = stance
    enriched["mode"] = mode
    enriched["num_turns"] = num_turns
    enriched["probe_target"] = probe_target
    enriched["value_text_mode"] = value_text_mode
    enriched["ask_prioritize_over_others"] = ask_prioritize_over_others
    return enriched


def _build_condition_summaries(
    raw_outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detailed_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    by_value_parts: list[pd.DataFrame] = []
    by_pair_parts: list[pd.DataFrame] = []

    group_cols = [
        "model",
        "value_set",
        "stance",
        "mode",
        "num_turns",
        "value_text_mode",
        "ask_prioritize_over_others",
    ]
    for keys, sub in raw_outcomes.groupby(group_cols):
        value_df = sub[sub["probe_target"] == "value"]
        action_df = sub[sub["probe_target"] == "action"]
        merged = merge_value_action_outcomes(value_df, action_df)
        if merged.empty:
            continue

        metadata = dict(zip(group_cols, keys))

        detailed = merged.copy()
        for key, value in metadata.items():
            detailed[key] = value
        detailed_parts.append(detailed)

        summary = summarize_value_action_consistency(merged)
        for key, value in metadata.items():
            summary[key] = value
        summary_parts.append(summary)

        by_value = selection_gap_by_value(merged)
        for key, value in metadata.items():
            by_value[key] = value
        by_value_parts.append(by_value)

        by_pair = selection_gap_by_pair(merged)
        for key, value in metadata.items():
            by_pair[key] = value
        by_pair_parts.append(by_pair)

    detailed_df = pd.concat(detailed_parts, ignore_index=True) if detailed_parts else pd.DataFrame()
    summary_df = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    by_value_df = pd.concat(by_value_parts, ignore_index=True) if by_value_parts else pd.DataFrame()
    by_pair_df = pd.concat(by_pair_parts, ignore_index=True) if by_pair_parts else pd.DataFrame()
    return detailed_df, summary_df, by_value_df, by_pair_df


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    log = logging.getLogger(__name__)

    if args.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    judge_client = build_judge_client(args) if args.mode == "openended" else None
    user_sim = build_user_simulator(args) if any(t > 0 for t in args.turn_counts) else None
    task1_value_text_mode = args.value_text
    task1_value_text_label = task1_value_text_mode or ("hidden" if args.mode == "openended" else "label_description")

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = RESULTS_DIR / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_outcomes: list[pd.DataFrame] = []

    for model_key in args.models:
        model = _build_target_model(model_key, args)
        try:
            for value_set in args.value_sets:
                scenarios = load_scenarios(value_set, max_scenarios=args.num_scenarios)
                load_swap_decisions(scenarios, checkpoint_path=None, seed=42)
                value_descriptions = load_value_descriptions(value_set)

                if 0 in args.turn_counts:
                    log.info("Running baseline probes for %s / %s", model_key, value_set)
                    for probe_target in ["value", "action"]:
                        outcomes = _probe_target(
                            probe_target=probe_target,
                            mode=args.mode,
                            model=model,
                            model_key=model_key,
                            scenarios=scenarios,
                            value_descriptions=value_descriptions,
                            judge_client=judge_client,
                            judge_model=args.judge_model,
                            context=None,
                            value_text_mode=task1_value_text_mode,
                            ask_prioritize_over_others=args.ask_prioritize_over_others,
                        )
                        all_outcomes.append(
                            _attach_metadata(
                                outcomes,
                                model_key=model_key,
                                value_set=value_set,
                                stance="baseline",
                                mode=args.mode,
                                num_turns=0,
                                probe_target=probe_target,
                                value_text_mode=task1_value_text_label,
                                ask_prioritize_over_others=args.ask_prioritize_over_others,
                            )
                        )

                positive_turns = [t for t in args.turn_counts if t > 0]
                if not positive_turns:
                    continue

                for stance in args.stances:
                    log.info(
                        "Generating conversations for %s / %s / stance=%s",
                        model_key,
                        value_set,
                        stance,
                    )
                    conversations = generate_group_conversations(
                        model=model,
                        model_key=model_key,
                        user_sim=user_sim,
                        scenarios=scenarios,
                        value_set=value_set,
                        group_by=args.group_by,
                        turn_counts=positive_turns,
                        stance=stance,
                        log=log,
                    )

                    for num_turns in positive_turns:
                        for group_key, conv_by_turns in conversations.items():
                            context = conv_by_turns.get(num_turns)
                            if context is None:
                                continue
                            group_scenarios = _rows_for_group(scenarios, args.group_by, group_key)
                            if group_scenarios.empty:
                                continue

                            for probe_target in ["value", "action"]:
                                outcomes = _probe_target(
                                    probe_target=probe_target,
                                    mode=args.mode,
                                    model=model,
                                    model_key=model_key,
                                    scenarios=group_scenarios,
                                    value_descriptions=value_descriptions,
                                    judge_client=judge_client,
                                    judge_model=args.judge_model,
                                    context=context,
                                    value_text_mode=task1_value_text_mode,
                                    ask_prioritize_over_others=args.ask_prioritize_over_others,
                                )
                                all_outcomes.append(
                                    _attach_metadata(
                                        outcomes,
                                        model_key=model_key,
                                        value_set=value_set,
                                        stance=stance,
                                        mode=args.mode,
                                        num_turns=num_turns,
                                        probe_target=probe_target,
                                        value_text_mode=task1_value_text_label,
                                        ask_prioritize_over_others=args.ask_prioritize_over_others,
                                    )
                                )
        finally:
            if hasattr(model, "unload"):
                model.unload()
            del model
            gc.collect()

    raw_outcomes = pd.concat(all_outcomes, ignore_index=True) if all_outcomes else pd.DataFrame()
    raw_outcomes.to_csv(out_dir / "raw_outcomes.csv", index=False)

    detailed_df, summary_df, by_value_df, by_pair_df = _build_condition_summaries(raw_outcomes)
    if not detailed_df.empty:
        detailed_df.to_csv(out_dir / "value_action_detailed.csv", index=False)
    if not summary_df.empty:
        summary_df.to_csv(out_dir / "value_action_consistency.csv", index=False)
    if not by_value_df.empty:
        by_value_df.to_csv(out_dir / "value_action_selection_gap_by_value.csv", index=False)
    if not by_pair_df.empty:
        by_pair_df.to_csv(out_dir / "value_action_selection_gap_by_pair.csv", index=False)

    log.info("Saved scenario value-action outputs to %s", out_dir)


if __name__ == "__main__":
    main()
