"""Bradley-Terry fitting and drift metrics."""

import numpy as np
import pandas as pd
import choix
from scipy.stats import spearmanr


def fit_bradley_terry(outcomes: pd.DataFrame, n_bootstrap: int = 1000) -> pd.DataFrame:
    """Fit Bradley-Terry model to pairwise outcomes.

    Args:
        outcomes: DataFrame with columns value1, value2, choice (A or B).
        n_bootstrap: Number of bootstrap samples for confidence intervals.

    Returns:
        DataFrame with columns: value, ability, ci_lower, ci_upper.
    """
    unique_values = sorted(set(outcomes["value1"].unique()) | set(outcomes["value2"].unique()))
    n_values = len(unique_values)
    value_to_idx = {val: i for i, val in enumerate(unique_values)}

    def make_comparisons(data):
        comps = []
        for _, row in data.iterrows():
            if row["choice"] == "A":
                comps.append((value_to_idx[row["value1"]], value_to_idx[row["value2"]]))
            else:
                comps.append((value_to_idx[row["value2"]], value_to_idx[row["value1"]]))
        return comps

    comparisons = make_comparisons(outcomes)
    params = choix.ilsr_pairwise(n_values, comparisons, alpha=0.01)

    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        boot_data = outcomes.sample(n=len(outcomes), replace=True)
        boot_comps = make_comparisons(boot_data)
        boot_params = choix.ilsr_pairwise(n_values, boot_comps, alpha=0.01)
        bootstrap_estimates.append(boot_params)

    bootstrap_estimates = np.array(bootstrap_estimates)
    ci_lower = np.percentile(bootstrap_estimates, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_estimates, 97.5, axis=0)

    rankings_df = pd.DataFrame({
        "value": unique_values,
        "ability": params,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    })
    return rankings_df.sort_values("ability", ascending=False).reset_index(drop=True)


def compute_drift(ranking_t0: pd.DataFrame, ranking_t1: pd.DataFrame) -> dict:
    """Compute drift metrics between T0 and T1 rankings.

    Returns dict with:
        - per_value_delta: {value: ability_t1 - ability_t0}
        - l2_distance: L2 norm of ability vector difference
        - rank_correlation: Spearman rho between T0 and T1 ranks
    """
    t0 = ranking_t0.set_index("value")["ability"]
    t1 = ranking_t1.set_index("value")["ability"]
    common = sorted(set(t0.index) & set(t1.index))

    per_value_delta = {v: float(t1[v] - t0[v]) for v in common}
    l2 = float(np.linalg.norm([t1[v] - t0[v] for v in common]))

    # Spearman on ranks
    t0_ranks = t0[common].rank(ascending=False)
    t1_ranks = t1[common].rank(ascending=False)
    rho, p_value = spearmanr(t0_ranks, t1_ranks)

    return {
        "per_value_delta": per_value_delta,
        "l2_distance": l2,
        "rank_correlation": float(rho),
        "rank_correlation_pvalue": float(p_value),
    }


def compute_answer_flip_rate(outcomes_t0: pd.DataFrame, outcomes_t1: pd.DataFrame) -> dict:
    """Compute % of scenarios where MCQ choice flipped between T0 and T1.

    Returns:
        - overall_flip_rate: float
        - per_pair_flip_rate: {(v1,v2): float}
    """
    merged = outcomes_t0.merge(
        outcomes_t1, on="scenario_id", suffixes=("_t0", "_t1")
    )
    merged["flipped"] = merged["choice_t0"] != merged["choice_t1"]

    overall = float(merged["flipped"].mean()) if len(merged) > 0 else 0.0

    per_pair = {}
    for (v1, v2), group in merged.groupby(["value1_t0", "value2_t0"]):
        per_pair[(v1, v2)] = float(group["flipped"].mean())

    return {"overall_flip_rate": overall, "per_pair_flip_rate": per_pair}
