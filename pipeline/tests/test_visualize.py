"""Tests for visualize.py — verify plots generate without errors."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from visualize import plot_radar_t0_t1, plot_drift_heatmap, plot_flip_rate_bars


def test_radar_plot(tmp_path):
    values = ["honesty", "helpfulness", "harmlessness"]
    t0 = pd.DataFrame({"value": values, "ability": [1.0, 0.0, -1.0], "ci_lower": [-0.1, -0.1, -1.1], "ci_upper": [1.1, 0.1, -0.9]})
    t1 = pd.DataFrame({"value": values, "ability": [0.5, 0.5, -1.0], "ci_lower": [-0.1, 0.4, -1.1], "ci_upper": [0.6, 0.6, -0.9]})
    out = tmp_path / "radar.png"
    plot_radar_t0_t1(t0, t1, "Test Radar", out)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_drift_heatmap(tmp_path):
    df = pd.DataFrame({
        "persona": ["a", "a", "b", "b"],
        "value_set": ["HHH", "HHH", "HHH", "HHH"],
        "domain": ["politics", "therapy", "politics", "therapy"],
        "l2_distance": [0.5, 0.3, 0.8, 0.1],
    })
    out = tmp_path / "heatmap.png"
    plot_drift_heatmap(df, out)
    # Output is per value set: heatmap_HHH.png
    assert (tmp_path / "heatmap_HHH.png").exists()


def test_flip_rate_bars(tmp_path):
    df = pd.DataFrame({
        "persona": ["a", "a", "b", "b"],
        "overall_flip_rate": [0.1, 0.2, 0.3, 0.15],
    })
    out = tmp_path / "flips.png"
    plot_flip_rate_bars(df, out)
    assert out.exists()


def test_empty_dataframe_no_crash(tmp_path):
    empty = pd.DataFrame()
    plot_drift_heatmap(empty, tmp_path / "empty_heat.png")
    plot_flip_rate_bars(empty, tmp_path / "empty_flips.png")
    # Should not crash, just skip
