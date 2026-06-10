"""Visualization helpers for VIA experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from via_analysis import TASK1_POLARITY, task24_vs_task1_consistency_detailed, task2_vs_task1_consistency_detailed


sns.set_style("whitegrid")
sns.set_context("talk")


TASK1_ORDER = {
    "strongly disagree": 0,
    "disagree": 1,
    "agree": 2,
    "strongly agree": 3,
}

CONDITION_ORDER = [
    ("single", "mcq"),
    ("single", "openended"),
    ("multi", "mcq"),
    ("multi", "openended"),
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save(fig, path: Path) -> None:
    _ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _task2_positive_rate(series: pd.Series) -> float:
    vals = series.map({"option1": 0.0, "option2": 1.0})
    return float(vals.dropna().mean()) if vals.notna().any() else np.nan


def _task24_intensity_rate(df: pd.DataFrame, *, label_col: str = "label") -> float:
    mapping = {"option1": 0.0, "option2": 1.0, "option3": 2.0, "option4": 3.0}
    vals = df[label_col].map(mapping)
    return float(vals.dropna().mean()) if vals.notna().any() else np.nan


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
    return df[label_col].map({"option1": "negative", "option2": "positive", "unclear": "unclear"}).fillna("unclear")


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
    return polarity.map({"negative": 0.0, "positive": 1.0})


def _plot_heatmap(
    heat: pd.DataFrame,
    path: Path,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    annotate: bool = False,
    annotation_text: pd.DataFrame | None = None,
) -> None:
    fig_h = max(6, len(heat) * 0.28)
    fig_w = max(8, heat.shape[1] * 2.2) if heat.shape[1] > 1 else 8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    annot_arg = annotation_text if annotation_text is not None else annotate
    fmt = "" if annotation_text is not None else (".2f" if annotate else "")
    sns.heatmap(
        heat,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        annot=annot_arg,
        fmt=fmt,
        ax=ax,
        yticklabels=list(heat.index),
        xticklabels=list(heat.columns),
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9 if len(heat) > 20 else 11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, path)


def _rate_count_summary(
    merged: pd.DataFrame,
    group_col: str,
    columns_col: str,
    success_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        merged.groupby([group_col, columns_col])[success_col]
        .agg(successes="sum", n="size")
        .reset_index()
    )
    grouped["rate"] = grouped["successes"] / grouped["n"]
    rate_heat = grouped.pivot(index=group_col, columns=columns_col, values="rate")
    text = grouped.copy()
    text["label"] = text.apply(lambda row: f"{row['rate']:.2f}\n({int(row['successes'])}/{int(row['n'])})", axis=1)
    text_heat = text.pivot(index=group_col, columns=columns_col, values="label")
    return rate_heat, text_heat


def _save_table_csv(
    grouped: pd.DataFrame,
    path: Path,
) -> None:
    _ensure_dir(path.parent)
    grouped.to_csv(path, index=False)


def _task1_intensity_summary(
    merged: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    grouped = (
        merged.groupby([level, "interaction"])
        .agg(
            n=("scenario_id", "size"),
            intensity_shift_count=("intensity_shift", "sum"),
            strengthened_count=("strengthened", "sum"),
            softened_count=("softened", "sum"),
            agree_to_strongly_agree_count=("agree_to_strongly_agree", "sum"),
            disagree_to_strongly_disagree_count=("disagree_to_strongly_disagree", "sum"),
            strongly_agree_to_agree_count=("strongly_agree_to_agree", "sum"),
            strongly_disagree_to_disagree_count=("strongly_disagree_to_disagree", "sum"),
        )
        .reset_index()
    )
    for col in [
        "intensity_shift_count",
        "strengthened_count",
        "softened_count",
        "agree_to_strongly_agree_count",
        "disagree_to_strongly_disagree_count",
        "strongly_agree_to_agree_count",
        "strongly_disagree_to_disagree_count",
    ]:
        grouped[col.replace("_count", "_rate")] = grouped[col] / grouped["n"]
    return grouped


def plot_task1_label_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    task1 = df[df["task"] == "task1"].copy()
    if task1.empty:
        return
    out_dir = output_dir / "task1"
    grouped = (
        task1.groupby(["interaction", "probe_mode", "label"])
        .size()
        .reset_index(name="count")
    )
    pivot = grouped.pivot_table(
        index=["interaction", "probe_mode"], columns="label", values="count", fill_value=0
    )
    present_order = [idx for idx in CONDITION_ORDER if idx in pivot.index]
    if present_order:
        pivot = pivot.reindex(pd.MultiIndex.from_tuples(present_order, names=pivot.index.names))
    pivot = pivot.reindex(columns=list(TASK1_ORDER.keys()), fill_value=0)
    pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="RdYlBu_r")
    ax.set_title("Task 1 Label Distribution by Condition")
    ax.set_ylabel("Share of responses")
    ax.set_xlabel("Condition")
    ax.set_ylim(0, 1)
    ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, out_dir / "task1_label_distribution.png")


def plot_task2_label_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    task2 = df[df["task"] == "task2"].copy()
    if task2.empty:
        return
    out_dir = output_dir / "task2"
    grouped = (
        task2.groupby(["interaction", "probe_mode", "label"])
        .size()
        .reset_index(name="count")
    )
    pivot = grouped.pivot_table(
        index=["interaction", "probe_mode"], columns="label", values="count", fill_value=0
    )
    present_order = [idx for idx in CONDITION_ORDER if idx in pivot.index]
    if present_order:
        pivot = pivot.reindex(pd.MultiIndex.from_tuples(present_order, names=pivot.index.names))
    pivot = pivot.reindex(columns=["option1", "option2", "unclear"], fill_value=0)
    pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["tomato", "steelblue", "gray"])
    ax.set_title("Task 2 Label Distribution by Condition")
    ax.set_ylabel("Share of responses")
    ax.set_xlabel("Condition")
    ax.set_ylim(0, 1)
    ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, out_dir / "task2_label_distribution.png")


def plot_task24_label_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    task = df[df["task"] == "task2_4way"].copy()
    if task.empty:
        return
    out_dir = output_dir / "task2_4way"
    grouped = (
        task.groupby(["interaction", "probe_mode", "label"])
        .size()
        .reset_index(name="count")
    )
    pivot = grouped.pivot_table(
        index=["interaction", "probe_mode"], columns="label", values="count", fill_value=0
    )
    present_order = [idx for idx in CONDITION_ORDER if idx in pivot.index]
    if present_order:
        pivot = pivot.reindex(pd.MultiIndex.from_tuples(present_order, names=pivot.index.names))
    pivot = pivot.reindex(columns=["option1", "option2", "option3", "option4", "unclear"], fill_value=0)
    pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#8b1e3f", "#d97706", "#0ea5e9", "#1d4ed8", "gray"])
    ax.set_title("Task 2 4-Way Label Distribution by Condition")
    ax.set_ylabel("Share of responses")
    ax.set_xlabel("Condition")
    ax.set_ylim(0, 1)
    ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, out_dir / "task2_4way_label_distribution.png")


def plot_mcq_vs_openended_agreement(df: pd.DataFrame, output_dir: Path) -> None:
    for task in ["task1", "task2", "task2_4way"]:
        sub = df[df["task"] == task]
        if sub["probe_mode"].nunique() < 2:
            continue
        out_dir = output_dir / task
        merged_rows = []
        for interaction, group in sub.groupby("interaction"):
            mcq = group[group["probe_mode"] == "mcq"]
            oe = group[group["probe_mode"] == "openended"]
            merged = mcq.merge(
                oe,
                on=["model", "task", "scenario_id", "country", "topic", "value", "schwartz_group", "super_group", "interaction", "num_turns"],
                suffixes=("_mcq", "_oe"),
            )
            if merged.empty:
                continue
            merged["interaction"] = interaction
            merged["exact_match"] = merged["label_mcq"] == merged["label_oe"]
            if task == "task1":
                merged["mcq_score"] = merged["label_mcq"].map(TASK1_ORDER)
                merged["oe_score"] = merged["label_oe"].map(TASK1_ORDER)
                merged["mcq_polarity"] = merged["label_mcq"].map(TASK1_POLARITY)
                merged["oe_polarity"] = merged["label_oe"].map(TASK1_POLARITY)
                merged["polarity_match"] = merged["mcq_polarity"] == merged["oe_polarity"]
                merged["same_polarity"] = merged["polarity_match"]
                merged["ordinal_delta"] = merged["oe_score"] - merged["mcq_score"]
                merged["intensity_shift"] = merged["same_polarity"] & (merged["label_mcq"] != merged["label_oe"])
                merged["strengthened"] = merged["same_polarity"] & (merged["ordinal_delta"] > 0)
                merged["softened"] = merged["same_polarity"] & (merged["ordinal_delta"] < 0)
                merged["agree_to_strongly_agree"] = (merged["label_mcq"] == "agree") & (merged["label_oe"] == "strongly agree")
                merged["disagree_to_strongly_disagree"] = (merged["label_mcq"] == "disagree") & (merged["label_oe"] == "strongly disagree")
                merged["strongly_agree_to_agree"] = (merged["label_mcq"] == "strongly agree") & (merged["label_oe"] == "agree")
                merged["strongly_disagree_to_disagree"] = (merged["label_mcq"] == "strongly disagree") & (merged["label_oe"] == "disagree")
            merged_rows.append(merged)
        if not merged_rows:
            continue
        merged_all = pd.concat(merged_rows, ignore_index=True)
        heat, text = _rate_count_summary(merged_all, "value", "interaction", "exact_match")
        _plot_heatmap(
            heat.fillna(np.nan),
            out_dir / f"{task}_mcq_vs_openended_value_heatmap.png",
            f"{task.upper()} MCQ vs Open-Ended Exact Label Agreement by Value",
            ylabel="Value",
        )
        exact_table = (
            merged_all.groupby(["value", "interaction"])["exact_match"]
            .agg(matches="sum", n="size")
            .reset_index()
        )
        exact_table["rate"] = exact_table["matches"] / exact_table["n"]
        _save_table_csv(
            exact_table,
            out_dir / f"{task}_mcq_vs_openended_exact_label_agreement_by_value.csv",
        )
        if task == "task1":
            polarity_heat, polarity_text = _rate_count_summary(
                merged_all, "value", "interaction", "polarity_match"
            )
            _plot_heatmap(
                polarity_heat.fillna(np.nan),
                out_dir / "task1_mcq_vs_openended_polarity_value_heatmap.png",
                "TASK1 MCQ vs Open-Ended Polarity Agreement by Value",
                ylabel="Value",
            )
            polarity_table = (
                merged_all.groupby(["value", "interaction"])["polarity_match"]
                .agg(matches="sum", n="size")
                .reset_index()
            )
            polarity_table["rate"] = polarity_table["matches"] / polarity_table["n"]
            _save_table_csv(
                polarity_table,
                out_dir / "task1_mcq_vs_openended_polarity_agreement_by_value.csv",
            )
            for level in ["value", "schwartz_group", "super_group"]:
                intensity_table = _task1_intensity_summary(merged_all, level)
                _save_table_csv(
                    intensity_table,
                    out_dir / f"task1_mcq_vs_openended_intensity_by_{level}.csv",
                )

                for metric, title in [
                    ("strengthened_rate", "Within-Polarity Strengthening"),
                    ("softened_rate", "Within-Polarity Softening"),
                    ("agree_to_strongly_agree_rate", "Agree to Strongly Agree"),
                    ("disagree_to_strongly_disagree_rate", "Disagree to Strongly Disagree"),
                ]:
                    heat = intensity_table.pivot(index=level, columns="interaction", values=metric)
                    _plot_heatmap(
                        heat.fillna(np.nan),
                        out_dir / f"task1_mcq_vs_openended_{metric.replace('_rate', '')}_by_{level}.png",
                        f"TASK1 MCQ vs Open-Ended {title} by {level.replace('_', ' ').title()}",
                        ylabel=level.replace("_", " ").title(),
                    )
            for level in ["schwartz_group", "super_group"]:
                exact_heat, _ = _rate_count_summary(merged_all, level, "interaction", "exact_match")
                _plot_heatmap(
                    exact_heat.fillna(np.nan),
                    out_dir / f"task1_mcq_vs_openended_exact_label_agreement_by_{level}.png",
                    f"TASK1 MCQ vs Open-Ended Exact Label Agreement by {level.replace('_', ' ').title()}",
                    ylabel=level.replace("_", " ").title(),
                )
                exact_table = (
                    merged_all.groupby([level, "interaction"])["exact_match"]
                    .agg(matches="sum", n="size")
                    .reset_index()
                )
                exact_table["rate"] = exact_table["matches"] / exact_table["n"]
                _save_table_csv(
                    exact_table,
                    out_dir / f"task1_mcq_vs_openended_exact_label_agreement_by_{level}.csv",
                )

                level_polarity_heat, _ = _rate_count_summary(
                    merged_all, level, "interaction", "polarity_match"
                )
                _plot_heatmap(
                    level_polarity_heat.fillna(np.nan),
                    out_dir / f"task1_mcq_vs_openended_polarity_agreement_by_{level}.png",
                    f"TASK1 MCQ vs Open-Ended Polarity Agreement by {level.replace('_', ' ').title()}",
                    ylabel=level.replace("_", " ").title(),
                )
                level_polarity_table = (
                    merged_all.groupby([level, "interaction"])["polarity_match"]
                    .agg(matches="sum", n="size")
                    .reset_index()
                )
                level_polarity_table["rate"] = level_polarity_table["matches"] / level_polarity_table["n"]
                _save_table_csv(
                    level_polarity_table,
                    out_dir / f"task1_mcq_vs_openended_polarity_agreement_by_{level}.csv",
                )
        elif task == "task2_4way":
            merged_all["polarity_mcq"] = _task2_selected_polarity(
                merged_all,
                label_col="label_mcq",
                option1_polarity_col="option1_polarity_mcq",
                option2_polarity_col="option2_polarity_mcq",
            )
            merged_all["polarity_oe"] = _task2_selected_polarity(
                merged_all,
                label_col="label_oe",
                option1_polarity_col="option1_polarity_oe",
                option2_polarity_col="option2_polarity_oe",
            )
            merged_all["polarity_match"] = merged_all["polarity_mcq"] == merged_all["polarity_oe"]
            polarity_heat, _ = _rate_count_summary(merged_all, "value", "interaction", "polarity_match")
            _plot_heatmap(
                polarity_heat.fillna(np.nan),
                out_dir / "task2_4way_mcq_vs_openended_polarity_value_heatmap.png",
                "TASK2 4-Way MCQ vs Open-Ended Polarity Agreement by Value",
                ylabel="Value",
            )
            polarity_table = (
                merged_all.groupby(["value", "interaction"])["polarity_match"]
                .agg(matches="sum", n="size")
                .reset_index()
            )
            polarity_table["rate"] = polarity_table["matches"] / polarity_table["n"]
            _save_table_csv(
                polarity_table,
                out_dir / "task2_4way_mcq_vs_openended_polarity_agreement_by_value.csv",
            )


def plot_task1_t0_t1_drift(df: pd.DataFrame, output_dir: Path) -> None:
    task1 = df[df["task"] == "task1"].copy()
    if task1["interaction"].nunique() < 2:
        return
    single = task1[task1["interaction"] == "single"].copy()
    multi = task1[task1["interaction"] == "multi"].copy()
    if single.empty or multi.empty:
        return
    single["score"] = single["label"].map(TASK1_ORDER)
    multi["score"] = multi["label"].map(TASK1_ORDER)
    merged = single.merge(
        multi,
        on=["model", "task", "scenario_id", "country", "topic", "value", "schwartz_group", "super_group", "probe_mode"],
        suffixes=("_t0", "_t1"),
    )
    if merged.empty:
        return
    merged["delta"] = merged["score_t1"] - merged["score_t0"]
    merged["polarity_t0"] = merged["label_t0"].map(TASK1_POLARITY)
    merged["polarity_t1"] = merged["label_t1"].map(TASK1_POLARITY)
    merged["polarity_flip"] = merged["polarity_t0"] != merged["polarity_t1"]
    merged["exact_label_change"] = merged["label_t0"] != merged["label_t1"]
    merged["intensity_shift"] = (~merged["polarity_flip"]) & merged["exact_label_change"]
    value_delta = merged.groupby(["value", "probe_mode"])["delta"].mean().reset_index()

    fig, axes = plt.subplots(1, max(1, value_delta["probe_mode"].nunique()), figsize=(16, max(6, value_delta["value"].nunique() * 0.25)), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (mode, mode_df) in zip(axes, value_delta.groupby("probe_mode")):
        mode_df = mode_df.sort_values("delta")
        ax.barh(mode_df["value"], mode_df["delta"], color=["tomato" if x < 0 else "steelblue" for x in mode_df["delta"]])
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"Task 1 T0→T1 Drift by Value ({mode})")
        ax.set_xlabel("Average ordinal change")
    _save(fig, output_dir / "task1" / "task1_t0_t1_value_drift.png")

    for level in ["value", "schwartz_group", "super_group"]:
        rate_table = (
            merged.groupby([level, "probe_mode"])
            .agg(
                polarity_flip_count=("polarity_flip", "sum"),
                intensity_shift_count=("intensity_shift", "sum"),
                n=("scenario_id", "size"),
            )
            .reset_index()
        )
        rate_table["polarity_flip_rate"] = rate_table["polarity_flip_count"] / rate_table["n"]
        rate_table["intensity_shift_rate"] = rate_table["intensity_shift_count"] / rate_table["n"]
        _save_table_csv(
            rate_table,
            output_dir / "task1" / f"task1_t0_t1_rates_by_{level}.csv",
        )

        for metric, title in [
            ("polarity_flip_rate", "Polarity Flip Rate"),
            ("intensity_shift_rate", "Intensity Shift Rate"),
        ]:
            order = (
                rate_table.groupby(level)[metric]
                .mean()
                .sort_values(ascending=False)
                .index
            )
            fig_w = max(12, len(order) * 0.35 if level == "value" else 12)
            fig, ax = plt.subplots(figsize=(fig_w, 6))
            sns.barplot(
                data=rate_table,
                x=level,
                y=metric,
                hue="probe_mode",
                order=order,
                ax=ax,
            )
            ax.set_title(f"Task 1 T0→T1 {title} by {level.replace('_', ' ').title()}")
            ax.set_xlabel(level.replace("_", " ").title())
            ax.set_ylabel(title)
            ax.set_ylim(0, 1)
            plt.setp(
                ax.get_xticklabels(),
                rotation=90 if level == "value" else 45,
                ha="center" if level == "value" else "right",
            )
            _save(
                fig,
                output_dir / "task1" / f"task1_t0_t1_{metric.replace('_rate', '')}_by_{level}.png",
            )


def plot_task2_t0_t1_positive_rate_drift(df: pd.DataFrame, output_dir: Path) -> None:
    task2 = df[df["task"] == "task2"].copy()
    if task2["interaction"].nunique() < 2:
        return
    single = task2[task2["interaction"] == "single"].copy()
    multi = task2[task2["interaction"] == "multi"].copy()
    if single.empty or multi.empty:
        return
    merged = single.merge(
        multi,
        on=["model", "task", "scenario_id", "country", "topic", "value", "schwartz_group", "super_group", "probe_mode"],
        suffixes=("_t0", "_t1"),
    )
    if merged.empty:
        return
    merged["positive_t0"] = _task2_positive_indicator(
        merged,
        label_col="label_t0",
        option1_polarity_col="option1_polarity_t0",
        option2_polarity_col="option2_polarity_t0",
    )
    merged["positive_t1"] = _task2_positive_indicator(
        merged,
        label_col="label_t1",
        option1_polarity_col="option1_polarity_t1",
        option2_polarity_col="option2_polarity_t1",
    )
    merged["delta"] = merged["positive_t1"] - merged["positive_t0"]
    merged["action_flip"] = (
        merged["label_t0"].isin(["option1", "option2"])
        & merged["label_t1"].isin(["option1", "option2"])
        & (merged["label_t0"] != merged["label_t1"])
    )
    agg = merged.groupby(["value", "probe_mode"])["delta"].mean().reset_index()

    fig, axes = plt.subplots(1, max(1, agg["probe_mode"].nunique()), figsize=(16, max(6, agg["value"].nunique() * 0.25)), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (mode, mode_df) in zip(axes, agg.groupby("probe_mode")):
        mode_df = mode_df.sort_values("delta")
        ax.barh(mode_df["value"], mode_df["delta"], color=["tomato" if x < 0 else "steelblue" for x in mode_df["delta"]])
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"Task 2 T0→T1 Positive-Action Drift ({mode})")
        ax.set_xlabel("Average change toward positive action")
    _save(fig, output_dir / "task2" / "task2_t0_t1_positive_rate_drift.png")

    for level in ["value", "schwartz_group", "super_group"]:
        flip_table = (
            merged.groupby([level, "probe_mode"])
            .agg(
                flip_count=("action_flip", "sum"),
                n=("scenario_id", "size"),
            )
            .reset_index()
        )
        flip_table["flip_rate"] = flip_table["flip_count"] / flip_table["n"]
        _save_table_csv(
            flip_table,
            output_dir / "task2" / f"task2_t0_t1_flip_rate_by_{level}.csv",
        )

        order = (
            flip_table.groupby(level)["flip_rate"]
            .mean()
            .sort_values(ascending=False)
            .index
        )
        fig_w = max(12, len(order) * 0.35 if level == "value" else 12)
        fig, ax = plt.subplots(figsize=(fig_w, 6))
        sns.barplot(
            data=flip_table,
            x=level,
            y="flip_rate",
            hue="probe_mode",
            order=order,
            ax=ax,
        )
        ax.set_title(f"Task 2 T0→T1 Flip Rate by {level.replace('_', ' ').title()}")
        ax.set_xlabel(level.replace("_", " ").title())
        ax.set_ylabel("Flip rate")
        ax.set_ylim(0, 1)
        plt.setp(
            ax.get_xticklabels(),
            rotation=90 if level == "value" else 45,
            ha="center" if level == "value" else "right",
        )
        _save(fig, output_dir / "task2" / f"task2_t0_t1_flip_rate_by_{level}.png")


def plot_group_level_summaries(df: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for group_col in ["schwartz_group", "super_group"]:
        for task, sub in df.groupby("task"):
            for (interaction, probe_mode, group), g in sub.groupby(["interaction", "probe_mode", group_col]):
                if task == "task2":
                    metric = float(_task2_positive_indicator(g, label_col="label").dropna().mean())
                elif task == "task2_4way":
                    metric = _task24_intensity_rate(g, label_col="label")
                else:
                    metric = float(g["label"].map(TASK1_ORDER).mean())
                rows.append({
                    "group_type": group_col,
                    "task": task,
                    "interaction": interaction,
                    "probe_mode": probe_mode,
                    "group": group,
                    "metric": metric,
                })
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return
    for (group_type, task), sub in plot_df.groupby(["group_type", "task"]):
        out_dir = output_dir / task
        fig, ax = plt.subplots(figsize=(12, max(5, sub["group"].nunique() * 0.45)))
        sns.barplot(data=sub, y="group", x="metric", hue="probe_mode", ax=ax)
        ax.set_title(f"{task.upper()} Summary by {group_type.replace('_', ' ').title()}")
        ax.set_xlabel("Average score / positive-action rate")
        ax.set_ylabel("")
        _save(fig, out_dir / f"{task}_{group_type}_summary_bars.png")


def plot_task2_unclear_rates(df: pd.DataFrame, output_dir: Path) -> None:
    task2 = df[df["task"] == "task2"].copy()
    if task2.empty:
        return
    rows = []
    for group_col in ["value", "schwartz_group", "super_group"]:
        agg = (
            task2.groupby([group_col, "interaction", "probe_mode"])["label"]
            .apply(lambda x: float((x == "unclear").mean()))
            .reset_index(name="unclear_rate")
        )
        agg["level"] = group_col
        rows.append(agg)
    plot_df = pd.concat(rows, ignore_index=True)
    for level, sub in plot_df.groupby("level"):
        fig_h = max(5, sub[level].nunique() * 0.22)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        sns.barplot(data=sub, y=level, x="unclear_rate", hue="probe_mode", ax=ax)
        ax.set_title(f"Task 2 Unclear Rate by {level.replace('_', ' ').title()}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Unclear rate")
        ax.set_ylabel("")
        _save(fig, output_dir / "task2" / f"task2_unclear_rate_{level}.png")


def plot_task2_vs_task1_consistency(df: pd.DataFrame, output_dir: Path) -> None:
    merged = task2_vs_task1_consistency_detailed(df)
    if merged.empty:
        return
    out_dir = output_dir / "task1_vs_task2"

    merged["task1_condition"] = merged["interaction_task1"] + " / " + merged["probe_mode_task1"]
    merged["task2_condition"] = merged["interaction_task2"] + " / " + merged["probe_mode_task2"]
    merged["condition_pair"] = merged["task2_condition"] + " vs " + merged["task1_condition"]

    value_rates = (
        merged.groupby(["model", "value", "condition_pair"])["consistent_with_task1"]
        .mean()
        .reset_index()
    )
    if not value_rates.empty:
        for model_key, model_merged in merged.groupby("model"):
            heat, text = _rate_count_summary(model_merged, "value", "condition_pair", "consistent_with_task1")
            _plot_heatmap(
                heat,
                out_dir / f"{model_key}__task2_vs_task1_consistency_value_heatmap.png",
                f"Task 2 Choice Alignment with Task 1 by Value ({model_key})",
                xlabel="Task 2 condition vs Task 1 anchor",
                ylabel="Value",
            )
        value_table = (
            merged.groupby(["model", "value", "condition_pair"])
            .agg(
                aligned=("consistent_with_task1", "sum"),
                disagreed=("disagrees_with_task1", "sum"),
                unclear=("task2_unclear", "sum"),
                n=("consistent_with_task1", "size"),
            )
            .reset_index()
        )
        value_table["rate"] = value_table["aligned"] / value_table["n"]
        _save_table_csv(
            value_table,
            out_dir / "task2_vs_task1_consistency_by_value.csv",
        )

    for group_col, out_name, title in [
        ("schwartz_group_task2", "task2_vs_task1_consistency_schwartz_group_heatmap.png", "Task 2 Choice Alignment with Task 1 by Schwartz Group"),
        ("super_group_task2", "task2_vs_task1_consistency_super_group_heatmap.png", "Task 2 Choice Alignment with Task 1 by Super Group"),
    ]:
        for model_key, model_merged in merged.groupby("model"):
            heat, text = _rate_count_summary(model_merged, group_col, "condition_pair", "consistent_with_task1")
            if heat.empty:
                continue
            _plot_heatmap(
                heat,
                out_dir / f"{model_key}__{out_name}",
                f"{title} ({model_key})",
                xlabel="Task 2 condition vs Task 1 anchor",
            )
        group_table = (
            merged.groupby(["model", group_col, "condition_pair"])
            .agg(
                aligned=("consistent_with_task1", "sum"),
                disagreed=("disagrees_with_task1", "sum"),
                unclear=("task2_unclear", "sum"),
                n=("consistent_with_task1", "size"),
            )
            .reset_index()
        )
        group_table["rate"] = group_table["aligned"] / group_table["n"]
        csv_name = out_name.replace("_heatmap.png", ".csv")
        _save_table_csv(group_table, out_dir / csv_name)

    for group_col in ["schwartz_group_task2", "super_group_task2"]:
        display_col = group_col.replace("_task2", "")
        group_rates = (
            merged.groupby(["model", group_col, "task1_condition", "task2_condition"])
            .agg(
                aligned_count=("consistent_with_task1", "sum"),
                total_n=("consistent_with_task1", "size"),
                consistency_rate=("consistent_with_task1", "mean"),
                unclear_rate=("task2_unclear", "mean"),
            )
            .reset_index()
        )
        if group_rates.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(18, max(5, group_rates[group_col].nunique() * 0.45)), sharey=True)
        sns.barplot(
            data=group_rates,
            y=group_col,
            x="consistency_rate",
            hue="task1_condition",
            ax=axes[0],
        )
        axes[0].set_title(f"Task 2 vs Task 1 Consistency by {display_col.replace('_', ' ').title()}")
        axes[0].set_xlim(0, 1)
        axes[0].set_xlabel("Consistency rate")
        axes[0].set_ylabel("")

        sns.barplot(
            data=group_rates,
            y=group_col,
            x="unclear_rate",
            hue="task1_condition",
            ax=axes[1],
        )
        axes[1].set_title(f"Task 2 Unclear Rate vs Task 1 by {display_col.replace('_', ' ').title()}")
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("Unclear rate")
        axes[1].set_ylabel("")

        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, title="Task 1 anchor", bbox_to_anchor=(1.02, 1), loc="upper left")
        if axes[1].legend_:
            axes[1].legend_.remove()
        _save(fig, out_dir / f"task2_vs_task1_consistency_{display_col}.png")


    summary = (
        merged.groupby(["model", "condition_pair"])
        .agg(
            n=("consistent_with_task1", "size"),
            aligned_count=("consistent_with_task1", "sum"),
            disagreement_count=("disagrees_with_task1", "sum"),
            unclear_count=("task2_unclear", "sum"),
            consistency_rate=("consistent_with_task1", "mean"),
            unclear_rate=("task2_unclear", "mean"),
            unique_values=("value", "nunique"),
            unique_topics=("topic", "nunique"),
            unique_countries=("country", "nunique"),
        )
        .reset_index()
        .sort_values("condition_pair")
    )
    if not summary.empty:
        fig_h = max(3, len(summary) * 0.45 + 1.5)
        fig, ax = plt.subplots(figsize=(14, fig_h))
        ax.axis("off")
        display = summary.copy()
        for col in ["consistency_rate", "unclear_rate"]:
            display[col] = display[col].map(lambda x: f"{x:.2f}")
        table = ax.table(
            cellText=display.values,
            colLabels=[
                "Model",
                "Condition Pair",
                "N",
                "Aligned",
                "Disagreed",
                "Unclear Count",
                "Consistency",
                "Unclear",
                "Values",
                "Topics",
                "Countries",
            ],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        ax.set_title("Task 2 vs Task 1 Consistency Summary Table", pad=18)
        _save(fig, out_dir / "task2_vs_task1_consistency_summary_table.png")


def plot_task24_vs_task1_consistency(df: pd.DataFrame, output_dir: Path) -> None:
    merged = task24_vs_task1_consistency_detailed(df)
    if merged.empty:
        return
    out_dir = output_dir / "task1_vs_task2_4way"
    merged["task1_condition"] = merged["interaction_task1"] + " / " + merged["probe_mode_task1"]
    merged["task24_condition"] = merged["interaction_task24"] + " / " + merged["probe_mode_task24"]
    merged["condition_pair"] = merged["task24_condition"] + " vs " + merged["task1_condition"]

    heat, _ = _rate_count_summary(merged, "value", "condition_pair", "consistent_with_task1")
    _plot_heatmap(
        heat,
        out_dir / "task2_4way_vs_task1_consistency_value_heatmap.png",
        "Task 2 4-Way Choice Alignment with Task 1 by Value",
        xlabel="Task 2 4-way condition vs Task 1 anchor",
        ylabel="Value",
    )
    value_table = (
        merged.groupby(["value", "condition_pair"])
        .agg(
            aligned=("consistent_with_task1", "sum"),
            disagreed=("disagrees_with_task1", "sum"),
            unclear=("task24_unclear", "sum"),
            n=("consistent_with_task1", "size"),
        )
        .reset_index()
    )
    value_table["rate"] = value_table["aligned"] / value_table["n"]
    _save_table_csv(value_table, out_dir / "task2_4way_vs_task1_consistency_by_value.csv")


def generate_via_plots(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    _ensure_dir(output_dir)
    plot_task1_label_distributions(df, output_dir)
    plot_task2_label_distributions(df, output_dir)
    plot_task24_label_distributions(df, output_dir)
    plot_mcq_vs_openended_agreement(df, output_dir)
    plot_task1_t0_t1_drift(df, output_dir)
    plot_task2_t0_t1_positive_rate_drift(df, output_dir)
    plot_group_level_summaries(df, output_dir)
    plot_task2_unclear_rates(df, output_dir)
    plot_task2_vs_task1_consistency(df, output_dir)
    plot_task24_vs_task1_consistency(df, output_dir)
