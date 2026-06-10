"""Run VIA task1/task2 probes in single-turn and multi-turn modes."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from conversations import generate_conversation
from run_alignment_target_experiment import ALIGNMENT_MODELS, AlignmentModel, OpenAIModel, build_user_simulator
from run_alignment_target_experiment_openended import build_judge_client
try:
    from alignmentmodel_vllm import AlignmentModelVLLM
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    AlignmentModelVLLM = None

from via_analysis import (
    pairwise_comparisons,
    summarize_condition,
    task1_mcq_vs_openended_aggregates,
    task1_mcq_vs_openended_detailed,
    task1_openended_to_task2_openended_consistency,
    task24_mcq_vs_openended_detailed,
    task24_vs_task1_consistency,
    task24_vs_task1_consistency_detailed,
    task2_mcq_vs_openended_aggregates,
    task2_mcq_vs_openended_detailed,
    task2_vs_task1_consistency,
    task2_vs_task1_consistency_detailed,
)
from via_data import balanced_sample, task1_rows, task2_4way_rows, task2_rows
from via_extraction import classify_task1, classify_task2, classify_task2_4way
from via_prompts import (
    VIA_OPENENDED_ASSISTANT_SYSTEM_PROMPT,
    make_task1_conversation_prompt,
    make_task2_conversation_prompt,
    task1_judge_prompt,
    task1_prompt,
    task2_4way_judge_prompt,
    task2_4way_prompt,
    task2_judge_prompt,
    task2_prompt,
)
from via_visualize import generate_via_plots


RESULTS_DIR = Path(os.environ.get("PERSONA_DRIFTING_RESULTS_DIR", Path(__file__).resolve().parent / "results" / "via"))


def parse_args():
    parser = argparse.ArgumentParser(description="Run VIA task1/task2 experiments")
    parser.add_argument("--models", nargs="+", required=True, help=f"Available: {list(ALIGNMENT_MODELS.keys())}")
    parser.add_argument("--tasks", nargs="+", choices=["task1", "task2", "task2_4way"], default=["task1", "task2"])
    parser.add_argument("--probe-modes", nargs="+", choices=["mcq", "openended"], default=["mcq", "openended"])
    parser.add_argument("--interaction", nargs="+", choices=["single", "multi"], default=["single", "multi"])
    parser.add_argument("--turn-counts", nargs="+", type=int, default=[5])
    parser.add_argument("--task1-max-rows", type=int, default=0)
    parser.add_argument("--task2-max-rows", type=int, default=0)
    parser.add_argument(
        "--sampling-group",
        type=str,
        choices=["schwartz_group", "super_group", "value", "topic", "country"],
        default="schwartz_group",
        help="When max-rows is set, sample as evenly as possible across this grouping.",
    )
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--judge", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--judge-api-key", type=str, default=None)
    parser.add_argument("--use-pydantic-ai", action="store_true")
    parser.add_argument("--simulator", type=str, default="openai", choices=["anthropic", "openai"])
    parser.add_argument("--simulator-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--simulator-api-key", type=str, default=None)
    parser.add_argument("--simulator-base-url", type=str, default=None)
    parser.add_argument("--openai-api-key", type=str, default=None)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


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


def _conversation_cache_path(model_key: str, row, turns: int) -> Path:
    safe = row["scenario_id"].replace("::", "__").replace("/", "-")
    return RESULTS_DIR / "conversations" / model_key / f"{safe}_{turns}t.json"


def _load_or_generate_conversation(model, model_key: str, row, turns: int, user_sim):
    path = _conversation_cache_path(model_key, row, turns)
    if path.exists():
        with open(path) as handle:
            return json.load(handle)
    if row["task"] == "task1":
        system_prompt = make_task1_conversation_prompt(row["country"], row["topic"], row["value"], row["value_gloss"])
    elif row["task"] == "task2":
        system_prompt = make_task2_conversation_prompt(
            row["country"], row["topic"], row["value"], row["value_gloss"], row["option1"], row["option2"]
        )
    else:
        system_prompt = make_task2_conversation_prompt(
            row["country"], row["topic"], row["value"], row["value_gloss"], row["option1"], row["option4"]
        )
    conversation = generate_conversation(
        model=model,
        persona=model_key,
        domain="",
        num_turns=turns,
        user_sim=user_sim,
        system_prompt=system_prompt,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(conversation, handle, indent=2)
    return conversation


def _generate_response(model, model_key: str, row, probe_mode: str, interaction: str, turns: int | None, user_sim):
    model.load_persona(model_key)
    conversation = None
    if interaction == "multi":
        conversation = _load_or_generate_conversation(model, model_key, row, turns or 5, user_sim)
    if row["task"] == "task1":
        prompt = task1_prompt(row, interaction, probe_mode)
    elif row["task"] == "task2":
        prompt = task2_prompt(row, interaction, probe_mode)
    else:
        prompt = task2_4way_prompt(row, interaction, probe_mode)
    messages = []
    if probe_mode == "openended":
        messages.append({"role": "system", "content": VIA_OPENENDED_ASSISTANT_SYSTEM_PROMPT})
    if conversation:
        messages.extend(conversation)
    messages.append({"role": "user", "content": prompt})
    response = model.generate(messages, max_new_tokens=250, temperature=0.2 if probe_mode == "openended" else 0.0)
    return response, prompt, conversation


def _classify_response(row, response: str, prompt: str, judge_client, judge_model: str, pydantic_ai_model: str | None):
    if row["task"] == "task1":
        return classify_task1(
            response,
            task1_judge_prompt(row, response),
            judge_client=judge_client,
            judge_model=judge_model,
            pydantic_ai_model=pydantic_ai_model,
        )
    if row["task"] == "task2":
        return classify_task2(
            response,
            task2_judge_prompt(row, response),
            judge_client=judge_client,
            judge_model=judge_model,
            pydantic_ai_model=pydantic_ai_model,
        )
    return classify_task2_4way(
        response,
        task2_4way_judge_prompt(row, response),
        judge_client=judge_client,
        judge_model=judge_model,
        pydantic_ai_model=pydantic_ai_model,
    )


def _rows_for_tasks(args) -> pd.DataFrame:
    want_task1 = "task1" in args.tasks
    want_task2 = "task2" in args.tasks
    want_task24 = "task2_4way" in args.tasks

    task1 = task1_rows(max_rows=0) if want_task1 else None
    task2 = task2_rows(max_rows=0) if want_task2 else None
    task24 = task2_4way_rows(max_rows=0) if want_task24 else None

    if want_task1 and (want_task2 or want_task24):
        key_cols = ["country", "topic", "value"]
        task1_keys = task1[key_cols + ["schwartz_group", "super_group"]].drop_duplicates()
        compare_source = task2 if want_task2 else task24
        task2_keys = compare_source[key_cols + ["schwartz_group", "super_group"]].drop_duplicates()
        shared = task2_keys.merge(task1_keys, on=key_cols, suffixes=("_task2", "_task1"))
        if shared.empty:
            return pd.DataFrame()

        shared["schwartz_group"] = shared["schwartz_group_task2"]
        shared["super_group"] = shared["super_group_task2"]
        shared = shared[key_cols + ["schwartz_group", "super_group"]]

        target_rows = 0
        if args.task1_max_rows > 0 and args.task2_max_rows > 0:
            target_rows = min(args.task1_max_rows, args.task2_max_rows)
        elif args.task1_max_rows > 0:
            target_rows = args.task1_max_rows
        elif args.task2_max_rows > 0:
            target_rows = args.task2_max_rows

        if target_rows > 0:
            shared = balanced_sample(
                shared,
                max_rows=target_rows,
                group_col=args.sampling_group,
                seed=args.sampling_seed,
            )

        task1 = task1.merge(shared[key_cols], on=key_cols, how="inner")
        frames = [task1]
        if want_task2:
            task2 = task2.merge(shared[key_cols], on=key_cols, how="inner")
            frames.append(task2)
        if want_task24:
            task24 = task24.merge(shared[key_cols], on=key_cols, how="inner")
            frames.append(task24)
        return pd.concat(frames, ignore_index=True)

    frames = []
    if want_task1:
        if args.task1_max_rows > 0:
            task1 = balanced_sample(
                task1,
                max_rows=args.task1_max_rows,
                group_col=args.sampling_group,
                seed=args.sampling_seed,
            )
        frames.append(task1)
    if want_task2:
        if args.task2_max_rows > 0:
            task2 = balanced_sample(
                task2,
                max_rows=args.task2_max_rows,
                group_col=args.sampling_group,
                seed=args.sampling_seed,
            )
        frames.append(task2)
    if want_task24:
        if args.task2_max_rows > 0:
            task24 = balanced_sample(
                task24,
                max_rows=args.task2_max_rows,
                group_col=args.sampling_group,
                seed=args.sampling_seed,
            )
        frames.append(task24)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    log = logging.getLogger(__name__)

    rows = _rows_for_tasks(args)
    if rows.empty:
        raise SystemExit("No rows selected")

    judge_client = build_judge_client(args)
    user_sim = build_user_simulator(args) if "multi" in args.interaction else None
    pydantic_ai_model = f"{args.judge}:{args.judge_model}" if args.use_pydantic_ai else None

    all_results = []
    extraction_failures = []
    for model_key in args.models:
        model = _build_target_model(model_key, args)
        try:
            for _, row in rows.iterrows():
                for probe_mode in args.probe_modes:
                    for interaction in args.interaction:
                        turn_values = args.turn_counts if interaction == "multi" else [0]
                        for turns in turn_values:
                            try:
                                response, prompt, conversation = _generate_response(
                                    model, model_key, row, probe_mode, interaction, turns, user_sim
                                )
                                decision = _classify_response(
                                    row, response, prompt, judge_client, args.judge_model, pydantic_ai_model
                                )
                            except Exception as e:
                                log.warning(
                                    "Skipping VIA row after generation/classification failure: "
                                    "model=%s task=%s scenario_id=%s probe_mode=%s interaction=%s turns=%s [%s: %s]",
                                    model_key,
                                    row["task"],
                                    row["scenario_id"],
                                    probe_mode,
                                    interaction,
                                    turns,
                                    type(e).__name__,
                                    e,
                                )
                                extraction_failures.append({
                                    "model": model_key,
                                    "task": row["task"],
                                    "scenario_id": row["scenario_id"],
                                    "country": row["country"],
                                    "topic": row["topic"],
                                    "value": row["value"],
                                    "schwartz_group": row["schwartz_group"],
                                    "super_group": row["super_group"],
                                    "interaction": interaction,
                                    "probe_mode": probe_mode,
                                    "num_turns": turns,
                                    "error_type": type(e).__name__,
                                    "error": str(e),
                                })
                                continue
                            all_results.append({
                                "model": model_key,
                                "task": row["task"],
                                "scenario_id": row["scenario_id"],
                                "country": row["country"],
                                "topic": row["topic"],
                                "value": row["value"],
                                "schwartz_group": row["schwartz_group"],
                                "super_group": row["super_group"],
                                "option1": row.get("option1"),
                                "option2": row.get("option2"),
                                "option1_polarity": row.get("option1_polarity"),
                                "option2_polarity": row.get("option2_polarity"),
                                "positive_option_label": row.get("positive_option_label"),
                                "negative_option_label": row.get("negative_option_label"),
                                "interaction": interaction,
                                "probe_mode": probe_mode,
                                "num_turns": turns,
                                "raw_response": response,
                                "prompt": prompt,
                                "label": decision.label,
                                "judge_confidence": decision.confidence,
                                "judge_reason": decision.reason,
                                "conversation_len": len(conversation) if conversation else 0,
                            })
        finally:
            if hasattr(model, "unload"):
                model.unload()
            del model
            gc.collect()

    results_df = pd.DataFrame(all_results)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = RESULTS_DIR / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "__".join(args.models)[:120]
    results_path = out_dir / f"{stem}.csv"
    results_df.to_csv(results_path, index=False)
    if extraction_failures:
        pd.DataFrame(extraction_failures).to_csv(out_dir / f"{stem}_failures.csv", index=False)
    if results_df.empty:
        log.warning("No valid VIA results to summarize; saved failures to %s", out_dir)
        return

    summaries = {}
    for key, sub in results_df.groupby(["task", "interaction", "probe_mode"]):
        summaries["::".join(map(str, key))] = summarize_condition(sub)
    _save_json(out_dir / f"{stem}_summary.json", summaries)

    pairwise = pairwise_comparisons(results_df)
    if not pairwise.empty:
        pairwise.to_csv(out_dir / f"{stem}_comparisons.csv", index=False)

    consistency = task2_vs_task1_consistency(results_df)
    if not consistency.empty:
        consistency.to_csv(out_dir / f"{stem}_task2_vs_task1.csv", index=False)
    consistency_detailed = task2_vs_task1_consistency_detailed(results_df)
    if not consistency_detailed.empty:
        consistency_detailed.to_csv(out_dir / f"{stem}_task2_vs_task1_detailed.csv", index=False)

    task1_probe_detailed = task1_mcq_vs_openended_detailed(results_df)
    if not task1_probe_detailed.empty:
        task1_probe_detailed.to_csv(out_dir / f"{stem}_task1_mcq_vs_openended_detailed.csv", index=False)
        task1_mcq_vs_openended_aggregates(task1_probe_detailed).to_csv(
            out_dir / f"{stem}_task1_mcq_vs_openended_aggregates.csv", index=False
        )

    task2_probe_detailed = task2_mcq_vs_openended_detailed(results_df)
    if not task2_probe_detailed.empty:
        task2_probe_detailed.to_csv(out_dir / f"{stem}_task2_mcq_vs_openended_detailed.csv", index=False)
        task2_mcq_vs_openended_aggregates(task2_probe_detailed).to_csv(
            out_dir / f"{stem}_task2_mcq_vs_openended_aggregates.csv", index=False
        )

    task24_probe_detailed = task24_mcq_vs_openended_detailed(results_df)
    if not task24_probe_detailed.empty:
        task24_probe_detailed.to_csv(out_dir / f"{stem}_task2_4way_mcq_vs_openended_detailed.csv", index=False)

    openended_conditional = task1_openended_to_task2_openended_consistency(results_df)
    if not openended_conditional.empty:
        openended_conditional.to_csv(
            out_dir / f"{stem}_task1_openended_to_task2_openended_consistency.csv",
            index=False,
        )

    task24_consistency = task24_vs_task1_consistency(results_df)
    if not task24_consistency.empty:
        task24_consistency.to_csv(out_dir / f"{stem}_task2_4way_vs_task1.csv", index=False)
    task24_consistency_detailed = task24_vs_task1_consistency_detailed(results_df)
    if not task24_consistency_detailed.empty:
        task24_consistency_detailed.to_csv(out_dir / f"{stem}_task2_4way_vs_task1_detailed.csv", index=False)

    generate_via_plots(results_df, RESULTS_DIR / "plots" / run_id)

    log.info("Saved VIA results to %s", results_path)


if __name__ == "__main__":
    main()
