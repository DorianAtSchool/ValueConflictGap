"""Tests for run_runpod.py command construction."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_runpod import build_experiment_command


def _args(**overrides):
    defaults = dict(
        models=["all"],
        value_sets=["HHH", "modelspec"],
        stances=["neutral"],
        turn_counts=[5],
        num_scenarios=3,
        simulator="openai",
        simulator_model="gpt-4o-mini",
        simulator_api_key=None,
        mode="mcq",
        judge="openai",
        judge_model="gpt-4o-mini",
        judge_api_key=None,
        group_by="scenario",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_experiment_command_forwards_openended_judge_flags():
    args = _args(
        mode="openended",
        judge="anthropic",
        judge_model="claude-haiku-4-5-20251001",
        judge_api_key="sk-ant-test",
    )

    cmd = build_experiment_command(
        args,
        use_vllm=True,
        models=["gpt-4o-mini"],
        stances=["neutral"],
        skip_postprocess=True,
    )

    assert "--mode" in cmd
    assert "openended" in cmd
    assert "--judge" in cmd
    assert "anthropic" in cmd
    assert "--judge-model" in cmd
    assert "claude-haiku-4-5-20251001" in cmd
    assert "--judge-api-key" in cmd
    assert "sk-ant-test" in cmd
    assert "--use-vllm" in cmd
    assert "--skip-postprocess" in cmd


def test_build_experiment_command_omits_judge_api_key_when_unset():
    args = _args(mode="openended", judge_api_key=None)

    cmd = build_experiment_command(
        args,
        use_vllm=False,
        models=["gpt-4o-mini"],
        stances=["neutral"],
        skip_postprocess=False,
    )

    assert "--judge" in cmd
    assert "--judge-model" in cmd
    assert "--judge-api-key" not in cmd
