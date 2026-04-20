"""Tests for analysis.py — Bradley-Terry fitting and drift metrics."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from analysis import fit_bradley_terry, compute_drift, compute_answer_flip_rate


def _make_outcomes(bias_a=0.7, n=200):
    """Generate synthetic pairwise outcomes where value1 wins ~bias_a of the time."""
    import random
    random.seed(42)
    rows = []
    values = ["honesty", "helpfulness", "harmlessness"]
    pairs = [("honesty", "helpfulness"), ("honesty", "harmlessness"), ("helpfulness", "harmlessness")]
    for i in range(n):
        v1, v2 = pairs[i % len(pairs)]
        choice = "A" if random.random() < bias_a else "B"
        rows.append({"scenario_id": i, "value1": v1, "value2": v2, "choice": choice})
    return pd.DataFrame(rows)


def test_fit_bradley_terry_basic():
    outcomes = _make_outcomes(bias_a=0.8, n=300)
    ranking = fit_bradley_terry(outcomes, n_bootstrap=50)
    assert "value" in ranking.columns
    assert "ability" in ranking.columns
    assert "ci_lower" in ranking.columns
    assert len(ranking) == 3
    # With bias_a=0.8, value1 (A side) should generally rank higher
    # (value1 is always the "A" option in our synthetic data)


def test_fit_bradley_terry_confidence_intervals():
    outcomes = _make_outcomes(n=300)
    ranking = fit_bradley_terry(outcomes, n_bootstrap=50)
    for _, row in ranking.iterrows():
        assert row["ci_lower"] <= row["ability"] <= row["ci_upper"]


def test_compute_drift():
    r_t0 = pd.DataFrame({"value": ["a", "b", "c"], "ability": [1.0, 0.0, -1.0]})
    r_t1 = pd.DataFrame({"value": ["a", "b", "c"], "ability": [0.5, 0.5, -1.0]})
    drift = compute_drift(r_t0, r_t1)
    assert drift["l2_distance"] > 0
    assert "per_value_delta" in drift
    assert abs(drift["per_value_delta"]["a"] - (-0.5)) < 1e-6
    assert abs(drift["per_value_delta"]["b"] - 0.5) < 1e-6


def test_compute_drift_identical():
    r = pd.DataFrame({"value": ["a", "b"], "ability": [1.0, -1.0]})
    drift = compute_drift(r, r)
    assert drift["l2_distance"] == 0.0
    assert abs(drift["rank_correlation"] - 1.0) < 1e-6


def test_compute_answer_flip_rate():
    t0 = pd.DataFrame({
        "scenario_id": [0, 1, 2, 3],
        "value1": ["a", "a", "b", "b"],
        "value2": ["b", "b", "c", "c"],
        "choice": ["A", "B", "A", "A"],
        "winner": ["a", "b", "b", "b"],
    })
    t1 = pd.DataFrame({
        "scenario_id": [0, 1, 2, 3],
        "value1": ["a", "a", "b", "b"],
        "value2": ["b", "b", "c", "c"],
        "choice": ["B", "B", "A", "B"],  # flipped: 0 and 3
        "winner": ["b", "b", "b", "c"],
    })
    flips = compute_answer_flip_rate(t0, t1)
    assert abs(flips["overall_flip_rate"] - 0.5) < 1e-6


def test_compute_answer_flip_rate_no_flips():
    t0 = pd.DataFrame({
        "scenario_id": [0, 1],
        "value1": ["a", "a"],
        "value2": ["b", "b"],
        "choice": ["A", "B"],
        "winner": ["a", "b"],
    })
    flips = compute_answer_flip_rate(t0, t0)
    assert flips["overall_flip_rate"] == 0.0
