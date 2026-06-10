"""Visualization helpers for scenario-grounded value-action gap runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import VALUE_SETS_DIR


sns.set_style("whitegrid")
sns.set_context("talk")


PERSONAL_PROTECTIVE_GROUPS = {
    "authenticity": "personal",
    "autonomy": "personal",
    "creativity": "personal",
    "empowerment": "personal",
    "compliance": "protective",
    "harmlessness": "protective",
    "privacy": "protective",
    "responsibility": "protective",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save(fig, path: Path) -> None:
    _ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _condition_label(df: pd.DataFrame) -> pd.Series:
    return df["value_text_mode"].astype(str)


def _value_order(value_sets: pd.Series | list[str], observed_values: pd.Series | list[str]) -> list[str]:
    observed = {str(v) for v in observed_values if pd.notna(v)}
    ordered: list[str] = []
    seen: set[str] = set()
    for value_set in pd.Series(value_sets).dropna().astype(str).drop_duplicates():
        path = VALUE_SETS_DIR / f"{value_set}.json"
        if not path.exists():
            continue
        with open(path) as f:
            definitions = json.load(f)
        for value in definitions:
            if value in observed and value not in seen:
                ordered.append(value)
                seen.add(value)
    ordered.extend(sorted(observed - seen))
    return ordered


def _sort_values_stably(df: pd.DataFrame, value_set: str) -> pd.DataFrame:
    order = _value_order([value_set], df["value"])
    if not order:
        return df.sort_values("value").reset_index(drop=True)
    return (
        df.assign(value=pd.Categorical(df["value"], categories=order, ordered=True))
        .sort_values("value")
        .assign(value=lambda x: x["value"].astype(str))
        .reset_index(drop=True)
    )


def plot_gap_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return

    df = summary.copy()
    df["condition"] = _condition_label(df)
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.9)))
    y = np.arange(len(df))
    ax.barh(y, df["aligned_rate"], color="#4C956C", label="Aligned")
    ax.barh(y, df["mismatch_rate"], left=df["aligned_rate"], color="#C75146", label="Mismatch")
    ax.set_yticks(y)
    ax.set_yticklabels(df["condition"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of value-scenario probes")
    ax.set_title("Value-Action Gap Alignment")
    ax.legend(loc="lower right")
    _save(fig, output_dir / "gap_summary.png")


def plot_endorsement_action_dumbbell(by_value: pd.DataFrame, output_dir: Path) -> None:
    if by_value.empty:
        return

    for (model, value_set, mode), sub in by_value.groupby(["model", "value_set", "value_text_mode"]):
        ordered = _sort_values_stably(sub, value_set)
        fig_h = max(5, len(ordered) * 0.55)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        y = np.arange(len(ordered))
        ax.hlines(y, ordered["endorsement_rate"], ordered["action_support_rate"], color="#B8B8B8", linewidth=2)
        ax.scatter(ordered["endorsement_rate"], y, color="#2A6F97", s=80, label="Value endorsed")
        ax.scatter(ordered["action_support_rate"], y, color="#EE6C4D", s=80, label="Action supports value")
        ax.set_yticks(y)
        ax.set_yticklabels(ordered["value"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Rate")
        ax.set_title(f"Value Endorsement vs Action Support\n{model} | {value_set} | {mode}")
        ax.legend(loc="lower right")
        safe = str(mode).replace("/", "_").replace(" ", "")
        _save(fig, output_dir / "by_condition" / f"{model}__{value_set}__{safe}__endorsement_vs_action_support.png")


def plot_mismatch_by_value_grouped(by_value: pd.DataFrame, output_dir: Path) -> None:
    if by_value.empty or "mismatch_rate" not in by_value.columns:
        return

    palette = {
        "personal": "#2A6F97",
        "protective": "#C75146",
        "other": "#808080",
    }
    for (model, value_set, mode), sub in by_value.groupby(["model", "value_set", "value_text_mode"]):
        ordered = _sort_values_stably(sub, value_set).copy()
        ordered["value_group"] = ordered["value"].map(PERSONAL_PROTECTIVE_GROUPS).fillna("other")
        fig_h = max(5, len(ordered) * 0.55)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        sns.barplot(
            data=ordered,
            x="mismatch_rate",
            y="value",
            hue="value_group",
            dodge=False,
            palette=palette,
            ax=ax,
        )
        for group, line_style in [("personal", "--"), ("protective", ":")]:
            group_sub = ordered[ordered["value_group"] == group]
            if group_sub.empty:
                continue
            ax.axvline(
                group_sub["mismatch_rate"].mean(),
                color=palette[group],
                linestyle=line_style,
                linewidth=2,
                alpha=0.85,
                label=f"{group} mean",
            )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Task 1 / Task 2 mismatch rate")
        ax.set_ylabel("Value")
        ax.set_title(f"Value-Action Mismatch by Value\n{model} | {value_set} | {mode}")
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        ax.legend([h for h, _ in unique], [l for _, l in unique], loc="lower right")
        safe = str(mode).replace("/", "_").replace(" ", "")
        _save(fig, output_dir / "by_condition" / f"{model}__{value_set}__{safe}__mismatch_by_value_grouped.png")


def plot_gap_by_value_heatmap(by_value: pd.DataFrame, output_dir: Path) -> None:
    if by_value.empty:
        return

    df = by_value.copy()
    df["endorsement_action_gap"] = df["endorsement_rate"] - df["action_support_rate"]
    heat = df.pivot_table(
        index="value",
        columns="value_text_mode",
        values="endorsement_action_gap",
        aggfunc="mean",
    )
    if heat.empty:
        return
    order = _value_order(df["value_set"], heat.index.to_series())
    if order:
        heat = heat.reindex([v for v in order if v in heat.index])

    fig_h = max(5, len(heat) * 0.45)
    fig_w = max(8, heat.shape[1] * 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(heat, cmap="RdBu_r", center=0, annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Endorsement Minus Action Support by Value")
    ax.set_xlabel("Value Text Mode")
    ax.set_ylabel("Value")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, output_dir / "endorsement_action_gap_heatmap.png")


def plot_top_pair_mismatches(by_pair: pd.DataFrame, output_dir: Path, top_n: int = 12) -> None:
    if by_pair.empty:
        return

    for (model, value_set, mode), sub in by_pair.groupby(["model", "value_set", "value_text_mode"]):
        ordered = sub.sort_values("mismatch_rate", ascending=False).head(top_n).copy()
        if ordered.empty:
            continue
        ordered["pair"] = ordered["value1"] + " vs " + ordered["value2"]
        fig_h = max(5, len(ordered) * 0.55)
        fig, ax = plt.subplots(figsize=(11, fig_h))
        sns.barplot(data=ordered, x="mismatch_rate", y="pair", color="#C75146", ax=ax)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mismatch rate")
        ax.set_ylabel("Value pair")
        ax.set_title(f"Top Pair-Level Value-Action Gaps\n{model} | {value_set} | {mode}")
        safe = str(mode).replace("/", "_").replace(" ", "")
        _save(fig, output_dir / "by_condition" / f"{model}__{value_set}__{safe}__top_pair_mismatches.png")


def generate_scenario_value_action_gap_plots(run_dir: Path, output_dir: Path | None = None) -> None:
    run_dir = Path(run_dir)
    out_dir = Path(output_dir) if output_dir is not None else run_dir / "plots"

    summary_path = run_dir / "value_action_gap_summary.csv"
    by_value_path = run_dir / "value_action_gap_by_value.csv"
    by_pair_path = run_dir / "value_action_gap_by_pair.csv"

    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    by_value = pd.read_csv(by_value_path) if by_value_path.exists() else pd.DataFrame()
    by_pair = pd.read_csv(by_pair_path) if by_pair_path.exists() else pd.DataFrame()

    plot_gap_summary(summary, out_dir)
    plot_endorsement_action_dumbbell(by_value, out_dir)
    plot_mismatch_by_value_grouped(by_value, out_dir)
    plot_gap_by_value_heatmap(by_value, out_dir)
    plot_top_pair_mismatches(by_pair, out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots for a scenario value-action gap run directory")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    generate_scenario_value_action_gap_plots(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
