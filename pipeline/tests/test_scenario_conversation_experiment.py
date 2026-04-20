"""Tests for scenario conversation checkpoint validation helpers."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from run_scenario_conversation_experiment import (
    _load_outcomes_checkpoint,
    _validate_outcomes_frame,
)


def test_validate_outcomes_frame_rejects_empty():
    issues = _validate_outcomes_frame(pd.DataFrame(), require_nonempty=True)
    assert any("contains no outcomes" in issue for issue in issues)
    assert any("missing columns" in issue for issue in issues)


def test_load_outcomes_checkpoint_rejects_empty_cache(tmp_path):
    path = tmp_path / "bad_checkpoint.json"
    with open(path, "w") as f:
        json.dump({"outcomes": []}, f)

    outcomes, issues = _load_outcomes_checkpoint(path, require_nonempty=True)

    assert outcomes is None
    assert any("contains no outcomes" in issue for issue in issues)


def test_load_outcomes_checkpoint_accepts_valid_cache(tmp_path):
    path = tmp_path / "good_checkpoint.json"
    payload = {
        "outcomes": [
            {
                "scenario_id": "s1",
                "value1": "helpfulness",
                "value2": "honesty",
                "choice": "A",
                "winner": "helpfulness",
            }
        ]
    }
    with open(path, "w") as f:
        json.dump(payload, f)

    outcomes, issues = _load_outcomes_checkpoint(path, require_nonempty=True)

    assert not issues
    assert list(outcomes.columns) == ["scenario_id", "value1", "value2", "choice", "winner"]
    assert len(outcomes) == 1
