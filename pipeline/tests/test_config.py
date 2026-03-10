"""Tests for config.py — verify paths and parameters are well-formed."""

import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    PERSONAS, VALUE_SETS, GENERIC_DOMAINS, TURN_COUNTS,
    SCENARIO_PATHS, VALUE_SETS_DIR, DOMAIN_SYSTEM_PROMPTS,
)


def test_personas_nonempty():
    assert len(PERSONAS) == 10


def test_value_sets_exist():
    for vs in VALUE_SETS:
        assert SCENARIO_PATHS[vs].exists(), f"Missing scenario CSV: {SCENARIO_PATHS[vs]}"


def test_value_set_jsons_exist():
    for vs in VALUE_SETS:
        path = VALUE_SETS_DIR / f"{vs}.json"
        assert path.exists(), f"Missing value set JSON: {path}"
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict) and len(data) > 0


def test_domain_prompts():
    for domain in GENERIC_DOMAINS:
        assert domain in DOMAIN_SYSTEM_PROMPTS
        assert len(DOMAIN_SYSTEM_PROMPTS[domain]) > 20


def test_turn_counts():
    assert TURN_COUNTS == [5, 10, 20]
