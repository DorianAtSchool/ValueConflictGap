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
            if "winner" in row and pd.notna(row["winner"]):
                winner = row["winner"]
                loser = row["value2"] if winner == row["value1"] else row["value1"]
                comps.append((value_to_idx[winner], value_to_idx[loser]))
                continue

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


def compute_average_ranks_by_group(
    ranking: pd.DataFrame, groups: dict[str, list[str]]
) -> dict[str, float]:
    """Compute average ordinal rank per group (1 = highest priority).

    Args:
        ranking: DataFrame with columns `value` and `ability`.
        groups: Mapping from group name -> list of values.

    Returns:
        Dict mapping group name -> average ordinal rank across member values
        present in the ranking.
    """
    if ranking.empty or not groups:
        return {}

    ordered = ranking.sort_values("ability", ascending=False)["value"].tolist()
    rank_map = {value: idx for idx, value in enumerate(ordered, start=1)}

    averages: dict[str, float] = {}
    for group_name, values in groups.items():
        ranks = [rank_map[v] for v in values if v in rank_map]
        if ranks:
            averages[group_name] = float(np.mean(ranks))
    return averages


def compute_answer_flip_rate(outcomes_t0: pd.DataFrame, outcomes_t1: pd.DataFrame) -> dict:
    """Compute flip rates and flip directions between T0 and T1.

    Returns:
        - overall_flip_rate: float
        - per_pair_flip_rate: {"v1 vs v2": float}
        - flip_direction: {"value_name": net_gain} — positive means the value
          was chosen MORE often at T1 than T0 (net winner from flips)
        - per_pair_flip_direction: {"v1 vs v2": {"toward_v1": int, "toward_v2": int}}
        - n_matched: number of scenarios matched between T0 and T1
    """
    merged = outcomes_t0.merge(
        outcomes_t1, on="scenario_id", suffixes=("_t0", "_t1")
    )
    # Flip detection should compare winners, not A/B letters. Letter comparison is
    # confounded if prompt option order differs between probes.
    merged["flipped"] = merged["winner_t0"] != merged["winner_t1"]

    overall = float(merged["flipped"].mean()) if len(merged) > 0 else 0.0

    per_pair = {}
    per_pair_direction = {}
    flip_direction = {}  # value -> net gains from flips

    for (v1, v2), group in merged.groupby(["value1_t0", "value2_t0"]):
        pair_key = f"{v1} vs {v2}"
        flipped = group[group["flipped"]]
        per_pair[pair_key] = float(group["flipped"].mean())

        # Count which direction flips went
        toward_v1 = int((flipped["winner_t1"] == v1).sum())
        toward_v2 = int((flipped["winner_t1"] == v2).sum())
        per_pair_direction[pair_key] = {
            f"toward_{v1}": toward_v1,
            f"toward_{v2}": toward_v2,
        }

        # Accumulate net gains per value
        flip_direction[v1] = flip_direction.get(v1, 0) + toward_v1 - toward_v2
        flip_direction[v2] = flip_direction.get(v2, 0) + toward_v2 - toward_v1

    return {
        "overall_flip_rate": overall,
        "per_pair_flip_rate": per_pair,
        "flip_direction": flip_direction,
        "per_pair_flip_direction": per_pair_direction,
        "n_matched": len(merged),
    }
