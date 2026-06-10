"""Comparison helpers for scenario-grounded value-vs-action selection."""

from __future__ import annotations

import pandas as pd


def merge_value_action_outcomes(
    value_outcomes: pd.DataFrame,
    action_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Merge value-choice and action-choice outcomes on the same scenarios."""
    if value_outcomes.empty or action_outcomes.empty:
        return pd.DataFrame()

    value_cols = ["scenario_id", "value1", "value2", "choice", "winner"]
    action_cols = ["scenario_id", "value1", "value2", "choice", "winner"]
    merged = value_outcomes[value_cols].merge(
        action_outcomes[action_cols],
        on=["scenario_id", "value1", "value2"],
        suffixes=("_value", "_action"),
    )
    if merged.empty:
        return pd.DataFrame()

    merged["value_selection"] = merged["winner_value"]
    merged["action_selection"] = merged["winner_action"]
    value_resolved = merged["value_selection"].eq(merged["value1"]) | merged["value_selection"].eq(merged["value2"])
    action_resolved = merged["action_selection"].eq(merged["value1"]) | merged["action_selection"].eq(merged["value2"])
    merged["consistent"] = (
        value_resolved
        & action_resolved
        & (merged["value_selection"] == merged["action_selection"])
    )
    merged["disagreement"] = (
        value_resolved
        & action_resolved
        & (merged["value_selection"] != merged["action_selection"])
    )
    merged["value_unclear"] = ~value_resolved
    merged["action_unclear"] = ~action_resolved
    return merged


def summarize_value_action_consistency(merged: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row consistency summary for a merged condition."""
    if merged.empty:
        return pd.DataFrame()

    n = len(merged)
    aligned = int(merged["consistent"].sum())
    disagreed = int(merged["disagreement"].sum())
    value_unclear = int(merged["value_unclear"].sum())
    action_unclear = int(merged["action_unclear"].sum())

    return pd.DataFrame(
        [
            {
                "n": n,
                "aligned_count": aligned,
                "aligned_rate": aligned / n,
                "disagreed_count": disagreed,
                "disagreed_rate": disagreed / n,
                "value_unclear_count": value_unclear,
                "value_unclear_rate": value_unclear / n,
                "action_unclear_count": action_unclear,
                "action_unclear_rate": action_unclear / n,
            }
        ]
    )


def selection_gap_by_value(merged: pd.DataFrame) -> pd.DataFrame:
    """Summarize value-selection vs action-selection rates by value."""
    if merged.empty:
        return pd.DataFrame()

    values = sorted(set(merged["value1"]) | set(merged["value2"]))
    rows: list[dict] = []
    for value in values:
        mask = (merged["value1"] == value) | (merged["value2"] == value)
        sub = merged[mask]
        if sub.empty:
            continue

        n = len(sub)
        value_selected = int((sub["value_selection"] == value).sum())
        action_selected = int((sub["action_selection"] == value).sum())
        aligned = int(sub["consistent"].sum())
        disagreed_toward_value = int(
            (sub["disagreement"] & (sub["action_selection"] == value)).sum()
        )
        disagreed_away_from_value = int(
            (sub["disagreement"] & (sub["value_selection"] == value)).sum()
        )

        value_rate = value_selected / n
        action_rate = action_selected / n
        rows.append(
            {
                "value": value,
                "n_appearances": n,
                "value_selected_count": value_selected,
                "value_selected_rate": value_rate,
                "action_selected_count": action_selected,
                "action_selected_rate": action_rate,
                "selection_gap": action_rate - value_rate,
                "consistency_rate": aligned / n,
                "disagreed_toward_value_count": disagreed_toward_value,
                "disagreed_away_from_value_count": disagreed_away_from_value,
            }
        )

    return pd.DataFrame(rows).sort_values(["selection_gap", "value"], ascending=[False, True]).reset_index(drop=True)


def selection_gap_by_pair(merged: pd.DataFrame) -> pd.DataFrame:
    """Summarize value-selection vs action-selection rates by value pair."""
    if merged.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for (value1, value2), sub in merged.groupby(["value1", "value2"]):
        n = len(sub)
        value1_value_selected = int((sub["value_selection"] == value1).sum())
        value1_action_selected = int((sub["action_selection"] == value1).sum())
        aligned = int(sub["consistent"].sum())
        disagreed = int(sub["disagreement"].sum())
        rows.append(
            {
                "value1": value1,
                "value2": value2,
                "n": n,
                "aligned_count": aligned,
                "aligned_rate": aligned / n,
                "disagreed_count": disagreed,
                "disagreed_rate": disagreed / n,
                "value1_value_selected_count": value1_value_selected,
                "value1_value_selected_rate": value1_value_selected / n,
                "value1_action_selected_count": value1_action_selected,
                "value1_action_selected_rate": value1_action_selected / n,
                "value1_selection_gap": (value1_action_selected / n) - (value1_value_selected / n),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["disagreed_rate", "value1", "value2"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
