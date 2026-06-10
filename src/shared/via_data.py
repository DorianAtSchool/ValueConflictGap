"""Data loading helpers for VIA-style value-action-gap experiments."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
TASK2_FULL_DATA_PATH = DATA_ROOT / "sva" / "value_action_gap_full_data_gpt_4o_generation.csv"


VALUE_CLAIMS = {
    "Equality": "equal opportunity for all",
    "Inner Harmony": "at peace with myself",
    "Social Power": "control over others, dominance",
    "Pleasure": "gratification of desires",
    "Freedom": "freedom of action and thought",
    "A Spiritual Life": "emphasis on spiritual not material matters",
    "Sense of Belonging": "feeling that others care about me",
    "Social Order": "stability of society",
    "An Exciting Life": "stimulating experience",
    "Meaning in Life": "a purpose in life",
    "Politeness": "courtesy, good manners",
    "Wealth": "material possessions, money",
    "National Security": "protection of my nation from enemies",
    "Self-Respect": "belief in one's own worth",
    "Reciprocation of Favors": "avoidance of indebtedness",
    "Creativity": "uniqueness, imagination",
    "A World at Peace": "free of war and conflict",
    "Respect for Tradition": "preservation of time-honored customs",
    "Mature Love": "deep emotional and spiritual intimacy",
    "Self-Discipline": "self-restraint, resistance to temptation",
    "Detachment": "from worldly concerns",
    "Family Security": "safety for loved ones",
    "Social Recognition": "respect, approval by others",
    "Unity With Nature": "fitting into nature",
    "A Varied Life": "filled with challenge, novelty, and change",
    "Wisdom": "a mature understanding of life",
    "Authority": "the right to lead or command",
    "True Friendship": "close, supportive friends",
    "A World of Beauty": "beauty of nature and the arts",
    "Social Justice": "correcting injustice, care for the weak",
    "Independent": "self-reliant, self-sufficient",
    "Moderate": "avoiding extremes of feeling and action",
    "Loyal": "faithful to my friends, group",
    "Ambitious": "hardworking, aspiring",
    "Broad-Minded": "tolerant of different ideas and beliefs",
    "Humble": "modest, self-effacing",
    "Daring": "seeking adventure, risk",
    "Protecting the Environment": "preserving nature",
    "Influential": "having an impact on people and events",
    "Honoring of Parents and Elders": "showing respect",
    "Choosing Own Goals": "selecting own purposes",
    "Healthy": "not being sick physically or mentally",
    "Capable": "competent, effective, efficient",
    "Accepting my Portion in Life": "submitting to life's circumstances",
    "Honest": "genuine, sincere",
    "Preserving my Public Image": "protecting my face",
    "Obedient": "dutiful, meeting obligations",
    "Intelligent": "logical, thinking",
    "Helpful": "working for the welfare of others",
    "Enjoying Life": "enjoying food, sex, leisure, etc.",
    "Devout": "holding to religious faith and belief",
    "Responsible": "dependable, reliable",
    "Curious": "interested in everything, exploring",
    "Forgiving": "willing to pardon others",
    "Successful": "achieving goals",
    "Clean": "neat, tidy",
}


COUNTRIES = [
    "United States", "India", "Pakistan", "Nigeria", "Philippines", "United Kingdom",
    "Germany", "Uganda", "Canada", "Egypt", "France", "Australia",
]


TOPICS = [
    "Politics",
    "Social Networks",
    "Social Inequality",
    "Family & Changing Gender Roles",
    "Work Orientation",
    "Religion",
    "Environment",
    "National Identity",
    "Citizenship",
    "Leisure Time and Sports",
    "Health and Health Care",
]


SCHWARTZ_GROUPS = {
    "Power": ["Social Power", "Authority", "Wealth", "Preserving my Public Image", "Social Recognition"],
    "Achievement": ["Successful", "Capable", "Ambitious", "Influential", "Intelligent", "Self-Respect"],
    "Hedonism": ["Pleasure", "Enjoying Life"],
    "Stimulation": ["Daring", "A Varied Life", "An Exciting Life"],
    "Self-direction": ["Creativity", "Curious", "Freedom", "Choosing Own Goals", "Independent"],
    "Universalism": [
        "Protecting the Environment", "A World of Beauty", "Broad-Minded", "Social Justice",
        "Wisdom", "Equality", "A World at Peace", "Inner Harmony", "Unity With Nature",
    ],
    "Benevolence": [
        "Helpful", "Honest", "Forgiving", "Loyal", "Responsible", "True Friendship",
        "A Spiritual Life", "Mature Love", "Meaning in Life",
    ],
    "Tradition": ["Devout", "Accepting my Portion in Life", "Humble", "Moderate", "Respect for Tradition", "Detachment"],
    "Conformity": ["Politeness", "Honoring of Parents and Elders", "Obedient", "Self-Discipline"],
    "Security": ["Clean", "National Security", "Social Order", "Family Security", "Reciprocation of Favors", "Healthy", "Sense of Belonging"],
}


SUPER_GROUPS = {
    "Openness to Change": {"Self-direction", "Stimulation", "Hedonism"},
    "Self-Enhancement": {"Power", "Achievement"},
    "Self-Transcendence": {"Universalism", "Benevolence"},
    "Conservation": {"Tradition", "Conformity", "Security"},
}


VALUE_TO_SCHWARTZ = {
    value: group
    for group, values in SCHWARTZ_GROUPS.items()
    for value in values
}

VALUE_TO_SUPERGROUP = {
    value: super_group
    for value, group in VALUE_TO_SCHWARTZ.items()
    for super_group, groups in SUPER_GROUPS.items()
    if group in groups
}


def task1_rows(max_rows: int = 0) -> pd.DataFrame:
    rows = []
    for country in COUNTRIES:
        for topic in TOPICS:
            for value, gloss in VALUE_CLAIMS.items():
                rows.append({
                    "scenario_id": f"task1::{country}::{topic}::{value}",
                    "task": "task1",
                    "country": country,
                    "topic": topic,
                    "value": value,
                    "value_gloss": gloss,
                    "schwartz_group": VALUE_TO_SCHWARTZ[value],
                    "super_group": VALUE_TO_SUPERGROUP[value],
                })
    df = pd.DataFrame(rows)
    if max_rows > 0:
        df = balanced_sample(df, max_rows=max_rows, group_col="schwartz_group", seed=42)
    return df


def _extract_human_action(generation_prompt: str) -> str | None:
    try:
        payload = json.loads(generation_prompt)
    except json.JSONDecodeError:
        try:
            import ast
            payload = ast.literal_eval(generation_prompt)
        except Exception:
            return None
    if not isinstance(payload, dict):
        return None
    action = payload.get("Human Action")
    return action.strip() if isinstance(action, str) else None


def _should_swap_task2_options(scenario_id: str) -> bool:
    digest = hashlib.md5(scenario_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 2 == 1


def task2_rows(max_rows: int = 0) -> pd.DataFrame:
    raw = pd.read_csv(TASK2_FULL_DATA_PATH).reset_index(drop=True)
    rows = []
    for (country, topic, value), group in raw.groupby(["country", "topic", "value"], sort=True):
        if len(group) != 2:
            continue
        group = group.sort_values("polarity")
        if list(group["polarity"]) != ["negative", "positive"]:
            continue
        negative_action = _extract_human_action(group.iloc[0]["generation_prompt"])
        positive_action = _extract_human_action(group.iloc[1]["generation_prompt"])
        if not negative_action or not positive_action:
            continue
        scenario_id = f"task2::{country}::{topic}::{value}"
        option1 = negative_action
        option2 = positive_action
        option1_polarity = "negative"
        option2_polarity = "positive"
        if _should_swap_task2_options(scenario_id):
            option1, option2 = option2, option1
            option1_polarity, option2_polarity = option2_polarity, option1_polarity
        rows.append({
            "scenario_id": scenario_id,
            "task": "task2",
            "country": country,
            "topic": topic,
            "value": value,
            "value_gloss": VALUE_CLAIMS.get(value, value),
            "schwartz_group": VALUE_TO_SCHWARTZ.get(value),
            "super_group": VALUE_TO_SUPERGROUP.get(value),
            "option1": option1,
            "option2": option2,
            "option1_polarity": option1_polarity,
            "option2_polarity": option2_polarity,
            "positive_option_label": "option1" if option1_polarity == "positive" else "option2",
            "negative_option_label": "option1" if option1_polarity == "negative" else "option2",
        })
    df = pd.DataFrame(rows)
    if max_rows > 0:
        df = balanced_sample(df, max_rows=max_rows, group_col="schwartz_group", seed=42)
    return df


def task2_4way_rows(max_rows: int = 0) -> pd.DataFrame:
    base = task2_rows(max_rows=0).copy()
    if base.empty:
        return base

    def strong_negative(text: str) -> str:
        return f"I would actively avoid this and lean further away from it: {text}"

    def moderate_negative(text: str) -> str:
        return text

    def moderate_positive(text: str) -> str:
        return text

    def strong_positive(text: str) -> str:
        return f"I would go out of my way to fully commit to this approach: {text}"

    rows = []
    for _, row in base.iterrows():
        option_specs = [
            ("option1", strong_negative(row["option1"]), "negative", "strong"),
            ("option2", moderate_negative(row["option1"]), "negative", "moderate"),
            ("option3", moderate_positive(row["option2"]), "positive", "moderate"),
            ("option4", strong_positive(row["option2"]), "positive", "strong"),
        ]
        rows.append({
            "scenario_id": row["scenario_id"].replace("task2::", "task2_4way::"),
            "task": "task2_4way",
            "country": row["country"],
            "topic": row["topic"],
            "value": row["value"],
            "value_gloss": row["value_gloss"],
            "schwartz_group": row["schwartz_group"],
            "super_group": row["super_group"],
            "option1": option_specs[0][1],
            "option2": option_specs[1][1],
            "option3": option_specs[2][1],
            "option4": option_specs[3][1],
            "option1_polarity": option_specs[0][2],
            "option2_polarity": option_specs[1][2],
            "option3_polarity": option_specs[2][2],
            "option4_polarity": option_specs[3][2],
            "option1_intensity": option_specs[0][3],
            "option2_intensity": option_specs[1][3],
            "option3_intensity": option_specs[2][3],
            "option4_intensity": option_specs[3][3],
            "moderate_negative_option_label": "option2",
            "strong_negative_option_label": "option1",
            "moderate_positive_option_label": "option3",
            "strong_positive_option_label": "option4",
        })
    df = pd.DataFrame(rows)
    if max_rows > 0:
        df = balanced_sample(df, max_rows=max_rows, group_col="schwartz_group", seed=42)
    return df


def balanced_sample(
    df: pd.DataFrame,
    *,
    max_rows: int,
    group_col: str = "schwartz_group",
    seed: int = 42,
) -> pd.DataFrame:
    """Sample rows as evenly as possible across groups.

    This is used for smoke tests and smaller evaluation slices so the first N
    rows do not overrepresent a small subset of groups.
    """
    if max_rows <= 0 or len(df) <= max_rows or group_col not in df.columns:
        return df.reset_index(drop=True)

    grouped = {group: sub.copy() for group, sub in df.groupby(group_col, sort=True)}
    capacities = {group: len(sub) for group, sub in grouped.items()}
    total_target = min(max_rows, len(df))
    allocations = {group: 0 for group in grouped}
    rng = np.random.default_rng(seed)
    ordered_groups = list(grouped.keys())

    while total_target > 0:
        eligible = [group for group in ordered_groups if allocations[group] < capacities[group]]
        if not eligible:
            break
        min_alloc = min(allocations[group] for group in eligible)
        bucket = [group for group in eligible if allocations[group] == min_alloc]
        for group in rng.permutation(bucket):
            allocations[group] += 1
            total_target -= 1
            if total_target == 0:
                break

    sampled_parts = []
    for group, sub in grouped.items():
        n = allocations[group]
        if n <= 0:
            continue
        if n >= len(sub):
            sampled = sub
        else:
            sampled = sub.sample(n=n, random_state=seed)
        sampled_parts.append(sampled)

    if not sampled_parts:
        return df.head(max_rows).reset_index(drop=True)

    out = pd.concat(sampled_parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
