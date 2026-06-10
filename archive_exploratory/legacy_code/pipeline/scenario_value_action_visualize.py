"""Visualization helpers for scenario-grounded value-action selection runs."""

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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save(fig, path: Path) -> None:
    _ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _condition_label(df: pd.DataFrame) -> pd.Series:
    stance = df["stance"].astype(str)
    turns = df["num_turns"].astype(int).astype(str) + "t"
    mode = df["mode"].astype(str)
    return stance + " / " + turns + " / " + mode


def _value_order(value_sets: pd.Series | list[str], observed_values: pd.Series | list[str]) -> list[str]:
    """Return stable value ordering from value-set JSON definitions."""
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


def plot_consistency_summary(consistency: pd.DataFrame, output_dir: Path) -> None:
    if consistency.empty:
        return

    df = consistency.copy()
    df["condition"] = _condition_label(df)
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.9)))
    y = np.arange(len(df))
    ax.barh(y, df["aligned_rate"], color="#4C956C", label="Aligned")
    ax.barh(y, df["disagreed_rate"], left=df["aligned_rate"], color="#C75146", label="Disagreed")
    if "value_unclear_rate" in df:
        left = df["aligned_rate"] + df["disagreed_rate"]
        ax.barh(y, df["value_unclear_rate"], left=left, color="#A3A3A3", label="Value unclear")
    ax.set_yticks(y)
    ax.set_yticklabels(df["condition"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of scenarios")
    ax.set_title("Value-Action Consistency by Condition")
    ax.legend(loc="lower right")
    _save(fig, output_dir / "consistency_summary.png")


def plot_selection_gap_heatmap(by_value: pd.DataFrame, output_dir: Path) -> None:
    if by_value.empty:
        return

    df = by_value.copy()
    df["condition"] = _condition_label(df)
    heat = df.pivot_table(index="value", columns="condition", values="selection_gap", aggfunc="mean")
    if heat.empty:
        return
    order = _value_order(df["value_set"], heat.index.to_series())
    if order:
        heat = heat.reindex([v for v in order if v in heat.index])

    fig_h = max(5, len(heat) * 0.45)
    fig_w = max(8, heat.shape[1] * 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        heat,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Selection Gap by Value")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Value")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, output_dir / "selection_gap_heatmap.png")


def plot_value_rate_dumbbell(by_value: pd.DataFrame, output_dir: Path) -> None:
    if by_value.empty:
        return

    for (model, value_set, condition), sub in by_value.assign(condition=_condition_label(by_value)).groupby(
        ["model", "value_set", "condition"]
    ):
        ordered = _sort_values_stably(sub, value_set)
        fig_h = max(5, len(ordered) * 0.55)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        y = np.arange(len(ordered))
        ax.hlines(y, ordered["value_selected_rate"], ordered["action_selected_rate"], color="#B8B8B8", linewidth=2)
        ax.scatter(ordered["value_selected_rate"], y, color="#2A6F97", s=80, label="Value selected")
        ax.scatter(ordered["action_selected_rate"], y, color="#EE6C4D", s=80, label="Action selected")
        ax.set_yticks(y)
        ax.set_yticklabels(ordered["value"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Selection rate")
        ax.set_title(f"Value vs Action Selection Rates\n{model} | {value_set} | {condition}")
        ax.legend(loc="lower right")
        safe = condition.replace("/", "_").replace(" ", "")
        _save(fig, output_dir / "by_condition" / f"{model}__{value_set}__{safe}__value_vs_action_rates.png")


def plot_mismatch_by_value(by_value: pd.DataFrame, output_dir: Path) -> None:
    if by_value.empty or "consistency_rate" not in by_value:
        return

    df = by_value.copy()
    df["mismatch_rate"] = 1 - df["consistency_rate"]
    for (model, value_set, condition), sub in df.assign(condition=_condition_label(df)).groupby(
        ["model", "value_set", "condition"]
    ):
        ordered = _sort_values_stably(sub, value_set)
        fig_h = max(5, len(ordered) * 0.55)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        sns.barplot(data=ordered, x="mismatch_rate", y="value", color="#C75146", ax=ax)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mismatch rate")
        ax.set_ylabel("Value")
        ax.set_title(f"Task 1 vs Task 2 Mismatch by Value\n{model} | {value_set} | {condition}")
        for patch, rate in zip(ax.patches, ordered["mismatch_rate"]):
            ax.text(
                min(rate + 0.02, 0.98),
                patch.get_y() + patch.get_height() / 2,
                f"{rate:.0%}",
                va="center",
                ha="left" if rate < 0.9 else "right",
                fontsize=10,
            )
        safe = condition.replace("/", "_").replace(" ", "")
        _save(fig, output_dir / "by_condition" / f"{model}__{value_set}__{safe}__mismatch_by_value.png")


def plot_top_pair_disagreements(by_pair: pd.DataFrame, output_dir: Path, top_n: int = 12) -> None:
    if by_pair.empty:
        return

    for (model, value_set, condition), sub in by_pair.assign(condition=_condition_label(by_pair)).groupby(
        ["model", "value_set", "condition"]
    ):
        ordered = sub.sort_values("disagreed_rate", ascending=False).head(top_n).copy()
        if ordered.empty:
            continue
        ordered["pair"] = ordered["value1"] + " vs " + ordered["value2"]
        fig_h = max(5, len(ordered) * 0.55)
        fig, ax = plt.subplots(figsize=(11, fig_h))
        sns.barplot(data=ordered, x="disagreed_rate", y="pair", color="#C75146", ax=ax)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Disagreement rate")
        ax.set_ylabel("Value pair")
        ax.set_title(f"Top Pair-Level Disagreements\n{model} | {value_set} | {condition}")
        safe = condition.replace("/", "_").replace(" ", "")
        _save(fig, output_dir / "by_condition" / f"{model}__{value_set}__{safe}__top_pair_disagreements.png")


def generate_scenario_value_action_plots(run_dir: Path, output_dir: Path | None = None) -> None:
    run_dir = Path(run_dir)
    out_dir = Path(output_dir) if output_dir is not None else run_dir / "plots"

    consistency_path = run_dir / "value_action_consistency.csv"
    by_value_path = run_dir / "value_action_selection_gap_by_value.csv"
    by_pair_path = run_dir / "value_action_selection_gap_by_pair.csv"

    consistency = pd.read_csv(consistency_path) if consistency_path.exists() else pd.DataFrame()
    by_value = pd.read_csv(by_value_path) if by_value_path.exists() else pd.DataFrame()
    by_pair = pd.read_csv(by_pair_path) if by_pair_path.exists() else pd.DataFrame()

    plot_consistency_summary(consistency, out_dir)
    plot_selection_gap_heatmap(by_value, out_dir)
    plot_value_rate_dumbbell(by_value, out_dir)
    plot_mismatch_by_value(by_value, out_dir)
    plot_top_pair_disagreements(by_pair, out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots for a scenario value-action run directory")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    generate_scenario_value_action_plots(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
