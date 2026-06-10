"""Comparison helpers for VIA experiments."""

from __future__ import annotations

import pandas as pd


TASK1_ORDER = {
    "strongly disagree": 0,
    "disagree": 1,
    "agree": 2,
    "strongly agree": 3,
}

TASK1_POLARITY = {
    "strongly disagree": "negative",
    "disagree": "negative",
    "agree": "positive",
    "strongly agree": "positive",
}

TASK2_POLARITY = {
    "option1": "negative",
    "option2": "positive",
    "unclear": "unclear",
}


def _task2_selected_polarity(
    df: pd.DataFrame,
    *,
    label_col: str,
    option1_polarity_col: str = "option1_polarity",
    option2_polarity_col: str = "option2_polarity",
) -> pd.Series:
    if option1_polarity_col in df.columns and option2_polarity_col in df.columns:
        polarity = pd.Series("unclear", index=df.index, dtype="object")
        polarity.loc[df[label_col] == "option1"] = df.loc[df[label_col] == "option1", option1_polarity_col]
        polarity.loc[df[label_col] == "option2"] = df.loc[df[label_col] == "option2", option2_polarity_col]
        return polarity.fillna("unclear")
    return df[label_col].map(TASK2_POLARITY).fillna("unclear")


def _task2_positive_indicator(
    df: pd.DataFrame,
    *,
    label_col: str,
    option1_polarity_col: str = "option1_polarity",
    option2_polarity_col: str = "option2_polarity",
) -> pd.Series:
    polarity = _task2_selected_polarity(
        df,
        label_col=label_col,
        option1_polarity_col=option1_polarity_col,
        option2_polarity_col=option2_polarity_col,
    )
    return polarity.map({"negative": 0, "positive": 1})


def _task24_selected_polarity(df: pd.DataFrame, *, label_col: str) -> pd.Series:
    polarity = pd.Series("unclear", index=df.index, dtype="object")
    for idx in ["1", "2", "3", "4"]:
        opt = f"option{idx}"
        col = f"{opt}_polarity"
        if col in df.columns:
            polarity.loc[df[label_col] == opt] = df.loc[df[label_col] == opt, col]
    return polarity.fillna("unclear")


def _task24_intensity_score(df: pd.DataFrame, *, label_col: str) -> pd.Series:
    scores = pd.Series(pd.NA, index=df.index, dtype="Float64")
    mapping = {
        "option1": 0.0,
        "option2": 1.0,
        "option3": 2.0,
        "option4": 3.0,
    }
    for label, score in mapping.items():
        scores.loc[df[label_col] == label] = score
    return scores


def summarize_condition(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "mean_confidence": float(df["judge_confidence"].dropna().mean()) if "judge_confidence" in df else None,
        "label_distribution": df["label"].value_counts(dropna=False).to_dict(),
    }


def pairwise_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "interaction", "probe_mode"]
    rows = []
    for task, task_df in df.groupby("task"):
        conditions = (
            task_df[keys[1:]]
            .drop_duplicates()
            .sort_values(keys[1:])
            .to_dict(orient="records")
        )
        for left in conditions:
            for right in conditions:
                if left == right:
                    continue
                lhs = task_df[
                    (task_df["interaction"] == left["interaction"])
                    & (task_df["probe_mode"] == left["probe_mode"])
                ]
                rhs = task_df[
                    (task_df["interaction"] == right["interaction"])
                    & (task_df["probe_mode"] == right["probe_mode"])
                ]
                merged = lhs.merge(
                    rhs,
                    on=["scenario_id", "task", "country", "topic", "value"],
                    suffixes=("_lhs", "_rhs"),
                )
                if merged.empty:
                    continue
                record = {
                    "task": task,
                    "lhs": f"{left['interaction']}::{left['probe_mode']}",
                    "rhs": f"{right['interaction']}::{right['probe_mode']}",
                    "n": int(len(merged)),
                    "exact_match_rate": float((merged["label_lhs"] == merged["label_rhs"]).mean()),
                }
                if task == "task1":
                    record["mean_ordinal_delta"] = float(
                        (merged["label_lhs"].map(TASK1_ORDER) - merged["label_rhs"].map(TASK1_ORDER)).abs().mean()
                    )
                rows.append(record)
    return pd.DataFrame(rows)


def task1_mcq_vs_openended_detailed(df: pd.DataFrame) -> pd.DataFrame:
    task1 = df[df["task"] == "task1"].copy()
    if task1.empty:
        return pd.DataFrame()
    mcq = task1[task1["probe_mode"] == "mcq"].copy()
    oe = task1[task1["probe_mode"] == "openended"].copy()
    merged = mcq.merge(
        oe,
        on=[
            "model",
            "task",
            "scenario_id",
            "country",
            "topic",
            "value",
            "schwartz_group",
            "super_group",
            "interaction",
            "num_turns",
        ],
        suffixes=("_mcq", "_oe"),
    )
    if merged.empty:
        return pd.DataFrame()
    merged["mcq_score"] = merged["label_mcq"].map(TASK1_ORDER)
    merged["oe_score"] = merged["label_oe"].map(TASK1_ORDER)
    merged["mcq_polarity"] = merged["label_mcq"].map(TASK1_POLARITY)
    merged["oe_polarity"] = merged["label_oe"].map(TASK1_POLARITY)
    merged["same_polarity"] = merged["mcq_polarity"] == merged["oe_polarity"]
    merged["polarity_flip"] = ~merged["same_polarity"]
    merged["exact_label_change"] = merged["label_mcq"] != merged["label_oe"]
    merged["ordinal_delta"] = merged["oe_score"] - merged["mcq_score"]
    merged["intensity_shift"] = merged["same_polarity"] & merged["exact_label_change"]
    merged["strengthened"] = merged["same_polarity"] & (merged["ordinal_delta"] > 0)
    merged["softened"] = merged["same_polarity"] & (merged["ordinal_delta"] < 0)
    merged["agree_to_strongly_agree"] = (merged["label_mcq"] == "agree") & (merged["label_oe"] == "strongly agree")
    merged["disagree_to_strongly_disagree"] = (merged["label_mcq"] == "disagree") & (merged["label_oe"] == "strongly disagree")
    merged["strongly_agree_to_agree"] = (merged["label_mcq"] == "strongly agree") & (merged["label_oe"] == "agree")
    merged["strongly_disagree_to_disagree"] = (merged["label_mcq"] == "strongly disagree") & (merged["label_oe"] == "disagree")
    return merged


def task1_mcq_vs_openended_aggregates(detailed: pd.DataFrame) -> pd.DataFrame:
    if detailed.empty:
        return pd.DataFrame()
    rows = []
    for level in ["value", "schwartz_group", "super_group"]:
        grouped = (
            detailed.groupby(["interaction", level])
            .agg(
                n=("scenario_id", "size"),
                same_polarity_count=("same_polarity", "sum"),
                polarity_flip_count=("polarity_flip", "sum"),
                intensity_shift_count=("intensity_shift", "sum"),
                strengthened_count=("strengthened", "sum"),
                softened_count=("softened", "sum"),
                agree_to_strongly_agree_count=("agree_to_strongly_agree", "sum"),
                disagree_to_strongly_disagree_count=("disagree_to_strongly_disagree", "sum"),
                strongly_agree_to_agree_count=("strongly_agree_to_agree", "sum"),
                strongly_disagree_to_disagree_count=("strongly_disagree_to_disagree", "sum"),
                exact_label_change_count=("exact_label_change", "sum"),
                mean_ordinal_delta=("ordinal_delta", "mean"),
            )
            .reset_index()
        )
        grouped["level"] = level
        grouped["group"] = grouped[level]
        grouped["same_polarity_rate"] = grouped["same_polarity_count"] / grouped["n"]
        grouped["polarity_flip_rate"] = grouped["polarity_flip_count"] / grouped["n"]
        grouped["intensity_shift_rate"] = grouped["intensity_shift_count"] / grouped["n"]
        grouped["strengthened_rate"] = grouped["strengthened_count"] / grouped["n"]
        grouped["softened_rate"] = grouped["softened_count"] / grouped["n"]
        grouped["agree_to_strongly_agree_rate"] = grouped["agree_to_strongly_agree_count"] / grouped["n"]
        grouped["disagree_to_strongly_disagree_rate"] = grouped["disagree_to_strongly_disagree_count"] / grouped["n"]
        grouped["strongly_agree_to_agree_rate"] = grouped["strongly_agree_to_agree_count"] / grouped["n"]
        grouped["strongly_disagree_to_disagree_rate"] = grouped["strongly_disagree_to_disagree_count"] / grouped["n"]
        grouped["exact_label_change_rate"] = grouped["exact_label_change_count"] / grouped["n"]
        rows.append(grouped[[
            "level",
            "group",
            "interaction",
            "n",
            "same_polarity_count",
            "same_polarity_rate",
            "polarity_flip_count",
            "polarity_flip_rate",
            "intensity_shift_count",
            "intensity_shift_rate",
            "strengthened_count",
            "strengthened_rate",
            "softened_count",
            "softened_rate",
            "agree_to_strongly_agree_count",
            "agree_to_strongly_agree_rate",
            "disagree_to_strongly_disagree_count",
            "disagree_to_strongly_disagree_rate",
            "strongly_agree_to_agree_count",
            "strongly_agree_to_agree_rate",
            "strongly_disagree_to_disagree_count",
            "strongly_disagree_to_disagree_rate",
            "exact_label_change_count",
            "exact_label_change_rate",
            "mean_ordinal_delta",
        ]])
    return pd.concat(rows, ignore_index=True)


def task2_mcq_vs_openended_detailed(df: pd.DataFrame) -> pd.DataFrame:
    task2 = df[df["task"] == "task2"].copy()
    if task2.empty:
        return pd.DataFrame()
    mcq = task2[task2["probe_mode"] == "mcq"].copy()
    oe = task2[task2["probe_mode"] == "openended"].copy()
    merged = mcq.merge(
        oe,
        on=[
            "model",
            "task",
            "scenario_id",
            "country",
            "topic",
            "value",
            "schwartz_group",
            "super_group",
            "interaction",
            "num_turns",
        ],
        suffixes=("_mcq", "_oe"),
    )
    if merged.empty:
        return pd.DataFrame()
    merged["task2_polarity_mcq"] = _task2_selected_polarity(
        merged,
        label_col="label_mcq",
        option1_polarity_col="option1_polarity_mcq",
        option2_polarity_col="option2_polarity_mcq",
    )
    merged["task2_polarity_oe"] = _task2_selected_polarity(
        merged,
        label_col="label_oe",
        option1_polarity_col="option1_polarity_oe",
        option2_polarity_col="option2_polarity_oe",
    )
    merged["action_flip"] = merged["label_mcq"].isin(["option1", "option2"]) & merged["label_oe"].isin(["option1", "option2"]) & (merged["label_mcq"] != merged["label_oe"])
    merged["became_unclear"] = merged["label_mcq"].isin(["option1", "option2"]) & (merged["label_oe"] == "unclear")
    merged["direction"] = "no_change"
    merged.loc[(merged["label_mcq"] == "option1") & (merged["label_oe"] == "option2"), "direction"] = "option1_to_option2"
    merged.loc[(merged["label_mcq"] == "option2") & (merged["label_oe"] == "option1"), "direction"] = "option2_to_option1"
    merged.loc[merged["became_unclear"], "direction"] = "became_unclear"
    merged.loc[(merged["label_mcq"] == "unclear") & merged["label_oe"].isin(["option1", "option2"]), "direction"] = "from_unclear"
    merged["polarity_direction"] = "no_change"
    merged.loc[(merged["task2_polarity_mcq"] == "negative") & (merged["task2_polarity_oe"] == "positive"), "polarity_direction"] = "negative_to_positive"
    merged.loc[(merged["task2_polarity_mcq"] == "positive") & (merged["task2_polarity_oe"] == "negative"), "polarity_direction"] = "positive_to_negative"
    merged.loc[merged["became_unclear"], "polarity_direction"] = "became_unclear"
    merged.loc[(merged["task2_polarity_mcq"] == "unclear") & merged["task2_polarity_oe"].isin(["positive", "negative"]), "polarity_direction"] = "from_unclear"
    return merged


def task2_mcq_vs_openended_aggregates(detailed: pd.DataFrame) -> pd.DataFrame:
    if detailed.empty:
        return pd.DataFrame()
    rows = []
    for level in ["value", "schwartz_group", "super_group"]:
        grouped = (
            detailed.groupby(["interaction", level])
            .agg(
                n=("scenario_id", "size"),
                action_flip_count=("action_flip", "sum"),
                became_unclear_count=("became_unclear", "sum"),
                negative_to_positive_count=("polarity_direction", lambda x: int((x == "negative_to_positive").sum())),
                positive_to_negative_count=("polarity_direction", lambda x: int((x == "positive_to_negative").sum())),
                no_change_count=("direction", lambda x: int((x == "no_change").sum())),
            )
            .reset_index()
        )
        grouped["level"] = level
        grouped["group"] = grouped[level]
        grouped["action_flip_rate"] = grouped["action_flip_count"] / grouped["n"]
        grouped["became_unclear_rate"] = grouped["became_unclear_count"] / grouped["n"]
        grouped["negative_to_positive_rate"] = grouped["negative_to_positive_count"] / grouped["n"]
        grouped["positive_to_negative_rate"] = grouped["positive_to_negative_count"] / grouped["n"]
        grouped["no_change_rate"] = grouped["no_change_count"] / grouped["n"]
        rows.append(grouped[[
            "level",
            "group",
            "interaction",
            "n",
            "action_flip_count",
            "action_flip_rate",
            "became_unclear_count",
            "became_unclear_rate",
            "negative_to_positive_count",
            "negative_to_positive_rate",
            "positive_to_negative_count",
            "positive_to_negative_rate",
            "no_change_count",
            "no_change_rate",
        ]])
    return pd.concat(rows, ignore_index=True)


def task2_vs_task1_consistency(df: pd.DataFrame) -> pd.DataFrame:
    task1 = df[df["task"] == "task1"].copy()
    task2 = df[df["task"] == "task2"].copy()
    if task1.empty or task2.empty:
        return pd.DataFrame()

    task1["task1_expected_polarity"] = task1["label"].map(TASK1_POLARITY)
    task2["task2_selected_polarity"] = _task2_selected_polarity(task2, label_col="label")
    merged = task2.merge(
        task1[
            ["model", "scenario_id", "country", "topic", "value", "interaction", "probe_mode", "task1_expected_polarity"]
        ],
        on=["model", "country", "topic", "value"],
        suffixes=("_task2", "_task1"),
    )
    if merged.empty:
        return pd.DataFrame()
    merged["consistent_with_task1"] = (
        merged["task2_selected_polarity"] == merged["task1_expected_polarity"]
    )
    return (
        merged.groupby(["model", "interaction_task2", "probe_mode_task2", "interaction_task1", "probe_mode_task1"])["consistent_with_task1"]
        .mean()
        .reset_index(name="consistency_rate")
    )


def task2_vs_task1_consistency_detailed(df: pd.DataFrame) -> pd.DataFrame:
    task1 = df[df["task"] == "task1"].copy()
    task2 = df[df["task"] == "task2"].copy()
    if task1.empty or task2.empty:
        return pd.DataFrame()

    task1["task1_expected_polarity"] = task1["label"].map(TASK1_POLARITY)
    task2["task2_selected_polarity"] = _task2_selected_polarity(task2, label_col="label")
    merged = task2.merge(
        task1[
            [
                "model",
                "country",
                "topic",
                "value",
                "schwartz_group",
                "super_group",
                "interaction",
                "probe_mode",
                "task1_expected_polarity",
            ]
        ],
        on=["model", "country", "topic", "value"],
        suffixes=("_task2", "_task1"),
    )
    if merged.empty:
        return pd.DataFrame()
    merged["consistent_with_task1"] = (
        merged["task2_selected_polarity"] == merged["task1_expected_polarity"]
    )
    merged["disagrees_with_task1"] = (
        merged["task2_selected_polarity"].isin(["positive", "negative"])
        & (merged["task2_selected_polarity"] != merged["task1_expected_polarity"])
    )
    merged["task2_unclear"] = merged["task2_selected_polarity"] == "unclear"
    return merged


def task1_openended_to_task2_openended_consistency(df: pd.DataFrame) -> pd.DataFrame:
    task1 = df[(df["task"] == "task1") & (df["probe_mode"] == "openended")].copy()
    task2 = df[(df["task"] == "task2") & (df["probe_mode"] == "openended")].copy()
    if task1.empty or task2.empty:
        return pd.DataFrame()
    task1["task1_expected_polarity"] = task1["label"].map(TASK1_POLARITY)
    task2["task2_selected_polarity"] = _task2_selected_polarity(task2, label_col="label")
    merged = task2.merge(
        task1[
            [
                "model",
                "country",
                "topic",
                "value",
                "interaction",
                "num_turns",
                "label",
                "task1_expected_polarity",
            ]
        ],
        on=["model", "country", "topic", "value"],
        suffixes=("_task2", "_task1"),
    )
    if merged.empty:
        return pd.DataFrame()
    merged["consistent_with_task1"] = merged["task2_selected_polarity"] == merged["task1_expected_polarity"]
    merged["disagrees_with_task1"] = (
        merged["task2_selected_polarity"].isin(["positive", "negative"])
        & (merged["task2_selected_polarity"] != merged["task1_expected_polarity"])
    )
    merged["task2_unclear"] = merged["task2_selected_polarity"] == "unclear"
    return (
        merged.groupby(["model", "interaction_task1", "interaction_task2", "label_task1"])
        .agg(
            n=("country", "size"),
            aligned_count=("consistent_with_task1", "sum"),
            disagreed_count=("disagrees_with_task1", "sum"),
            unclear_count=("task2_unclear", "sum"),
        )
        .reset_index()
        .rename(columns={"label_task1": "task1_openended_label"})
        .assign(
            consistency_rate=lambda x: x["aligned_count"] / x["n"],
            disagreement_rate=lambda x: x["disagreed_count"] / x["n"],
            unclear_rate=lambda x: x["unclear_count"] / x["n"],
        )
        .sort_values(["model", "interaction_task1", "interaction_task2", "task1_openended_label"])
    )


def task24_mcq_vs_openended_detailed(df: pd.DataFrame) -> pd.DataFrame:
    task = df[df["task"] == "task2_4way"].copy()
    if task.empty:
        return pd.DataFrame()
    mcq = task[task["probe_mode"] == "mcq"].copy()
    oe = task[task["probe_mode"] == "openended"].copy()
    merged = mcq.merge(
        oe,
        on=[
            "model",
            "task",
            "scenario_id",
            "country",
            "topic",
            "value",
            "schwartz_group",
            "super_group",
            "interaction",
            "num_turns",
        ],
        suffixes=("_mcq", "_oe"),
    )
    if merged.empty:
        return pd.DataFrame()
    merged["polarity_mcq"] = _task24_selected_polarity(merged, label_col="label_mcq")
    merged["polarity_oe"] = _task24_selected_polarity(merged, label_col="label_oe")
    merged["same_polarity"] = merged["polarity_mcq"] == merged["polarity_oe"]
    merged["polarity_flip"] = ~merged["same_polarity"]
    merged["score_mcq"] = _task24_intensity_score(merged, label_col="label_mcq")
    merged["score_oe"] = _task24_intensity_score(merged, label_col="label_oe")
    merged["exact_label_change"] = merged["label_mcq"] != merged["label_oe"]
    merged["intensity_shift"] = merged["same_polarity"] & merged["exact_label_change"]
    merged["ordinal_delta"] = merged["score_oe"] - merged["score_mcq"]
    merged["strengthened"] = merged["ordinal_delta"] > 0
    merged["softened"] = merged["ordinal_delta"] < 0
    return merged


def task24_vs_task1_consistency(df: pd.DataFrame) -> pd.DataFrame:
    task1 = df[df["task"] == "task1"].copy()
    task24 = df[df["task"] == "task2_4way"].copy()
    if task1.empty or task24.empty:
        return pd.DataFrame()
    task1["task1_expected_polarity"] = task1["label"].map(TASK1_POLARITY)
    task24["task24_selected_polarity"] = _task24_selected_polarity(task24, label_col="label")
    merged = task24.merge(
        task1[["model", "country", "topic", "value", "interaction", "probe_mode", "task1_expected_polarity"]],
        on=["model", "country", "topic", "value"],
        suffixes=("_task24", "_task1"),
    )
    if merged.empty:
        return pd.DataFrame()
    merged["consistent_with_task1"] = merged["task24_selected_polarity"] == merged["task1_expected_polarity"]
    return (
        merged.groupby(["model", "interaction_task24", "probe_mode_task24", "interaction_task1", "probe_mode_task1"])["consistent_with_task1"]
        .mean()
        .reset_index(name="consistency_rate")
    )


def task24_vs_task1_consistency_detailed(df: pd.DataFrame) -> pd.DataFrame:
    task1 = df[df["task"] == "task1"].copy()
    task24 = df[df["task"] == "task2_4way"].copy()
    if task1.empty or task24.empty:
        return pd.DataFrame()
    task1["task1_expected_polarity"] = task1["label"].map(TASK1_POLARITY)
    task24["task24_selected_polarity"] = _task24_selected_polarity(task24, label_col="label")
    merged = task24.merge(
        task1[
            [
                "model",
                "country",
                "topic",
                "value",
                "schwartz_group",
                "super_group",
                "interaction",
                "probe_mode",
                "task1_expected_polarity",
            ]
        ],
        on=["model", "country", "topic", "value"],
        suffixes=("_task24", "_task1"),
    )
    if merged.empty:
        return pd.DataFrame()
    merged["consistent_with_task1"] = merged["task24_selected_polarity"] == merged["task1_expected_polarity"]
    merged["disagrees_with_task1"] = (
        merged["task24_selected_polarity"].isin(["positive", "negative"])
        & (merged["task24_selected_polarity"] != merged["task1_expected_polarity"])
    )
    merged["task24_unclear"] = merged["task24_selected_polarity"] == "unclear"
    return merged
    if merged.empty:
        return pd.DataFrame()
    merged["consistent_with_task1"] = merged["task2_selected_polarity"] == merged["task1_expected_polarity"]
    merged["disagrees_with_task1"] = (
        merged["task2_selected_polarity"].isin(["positive", "negative"])
        & (merged["task2_selected_polarity"] != merged["task1_expected_polarity"])
    )
    merged["task2_unclear"] = merged["task2_selected_polarity"] == "unclear"
    return (
        merged.groupby(["interaction_task1", "interaction_task2", "label_task1"])
        .agg(
            n=("country", "size"),
            aligned_count=("consistent_with_task1", "sum"),
            disagreed_count=("disagrees_with_task1", "sum"),
            unclear_count=("task2_unclear", "sum"),
        )
        .reset_index()
        .rename(columns={"label_task1": "task1_openended_label"})
        .assign(
            consistency_rate=lambda x: x["aligned_count"] / x["n"],
            disagreement_rate=lambda x: x["disagreed_count"] / x["n"],
            unclear_rate=lambda x: x["unclear_count"] / x["n"],
        )
        .sort_values(["interaction_task1", "interaction_task2", "task1_openended_label"])
    )
