"""Tests for probing.py — scenario loading and MCQ prompt creation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from probing import load_scenarios, create_scenario_prompt_mcq, parse_mcq_response


def test_load_scenarios_hhh():
    df = load_scenarios("HHH")
    assert len(df) > 1000
    assert "description" in df.columns
    assert "value1" in df.columns
    assert "action1" in df.columns


def test_load_scenarios_all_value_sets():
    for vs in ["HHH", "modelspec", "personalprotective"]:
        df = load_scenarios(vs)
        assert len(df) > 0
        # All rows should be filtered (keep_scenario=True)
        assert df["keep_scenario"].all()


def test_create_scenario_prompt_mcq():
    row = pd.Series({
        "description": "A test scenario about privacy vs helpfulness.",
        "action1": "Help the user fully",
        "action2": "Protect private info",
        "value1": "helpfulness",
        "value2": "privacy",
    })
    prompt, action_map = create_scenario_prompt_mcq(row)
    assert "A)" in prompt
    assert "B)" in prompt
    assert "ONLY the letter A or B" in prompt
    assert set(action_map.keys()) == {"A", "B"}
    assert set(action_map.values()) == {"action1", "action2"}


def test_parse_mcq_response():
    assert parse_mcq_response("A") == "A"
    assert parse_mcq_response("B") == "B"
    assert parse_mcq_response("a") == "A"
    assert parse_mcq_response("B) because...") == "B"
    assert parse_mcq_response("A\n") == "A"
    assert parse_mcq_response("") is None
    assert parse_mcq_response("I think") is None
