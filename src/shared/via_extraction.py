"""Structured extraction and judging for VIA experiments."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

try:
    import json_repair
except ImportError:
    json_repair = None

try:
    from pydantic_ai import Agent
    HAS_PYDANTIC_AI = True
except ImportError:
    Agent = None
    HAS_PYDANTIC_AI = False


TASK1_CANONICAL = {
    "strongly disagree": "strongly disagree",
    "disagree": "disagree",
    "agree": "agree",
    "strongly agree": "strongly agree",
}
TASK2_CANONICAL = {
    "option 1": "option1",
    "option1": "option1",
    "option 2": "option2",
    "option2": "option2",
    "unclear": "unclear",
}
TASK2_4WAY_CANONICAL = {
    "option 1": "option1",
    "option1": "option1",
    "option 2": "option2",
    "option2": "option2",
    "option 3": "option3",
    "option3": "option3",
    "option 4": "option4",
    "option4": "option4",
    "unclear": "unclear",
}


class Task1Decision(BaseModel):
    label: Literal["strongly disagree", "disagree", "agree", "strongly agree"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class Task2Decision(BaseModel):
    label: Literal["option1", "option2", "unclear"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class Task24WayDecision(BaseModel):
    label: Literal["option1", "option2", "option3", "option4", "unclear"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _unique_mentioned_label(text: str, labels: list[str]) -> str | None:
    """Return the only mentioned label, ignoring labels contained inside longer labels."""
    matches: list[tuple[int, int, str]] = []
    for label in sorted(labels, key=len, reverse=True):
        pattern = r"(?<![a-z0-9])" + re.escape(label) + r"(?![a-z0-9])"
        for match in re.finditer(pattern, text):
            start, end = match.span()
            if any(start >= prev_start and end <= prev_end for prev_start, prev_end, _ in matches):
                continue
            matches.append((start, end, label))
    found = {label for _, _, label in matches}
    return next(iter(found)) if len(found) == 1 else None


def direct_parse_task1(text: str) -> Task1Decision | None:
    normalized = _normalize_spaces(text)
    labels = [
        "strongly disagree",
        "strongly agree",
        "disagree",
        "agree",
    ]
    for label in labels:
        if normalized == label or normalized.startswith(label):
            return Task1Decision(label=label, confidence=1.0, reason="direct_parse")
    label = _unique_mentioned_label(normalized, labels)
    if label is not None:
        return Task1Decision(label=label, confidence=0.95, reason="direct_parse_mentioned_label")
    return None


def direct_parse_task2(text: str) -> Task2Decision | None:
    normalized = _normalize_spaces(text)
    for raw, label in TASK2_CANONICAL.items():
        if normalized == raw or normalized.startswith(raw):
            return Task2Decision(label=label, confidence=1.0, reason="direct_parse")
    label = _unique_mentioned_label(normalized, list(TASK2_CANONICAL))
    if label is not None:
        return Task2Decision(label=TASK2_CANONICAL[label], confidence=0.95, reason="direct_parse_mentioned_label")
    return None


def direct_parse_task2_4way(text: str) -> Task24WayDecision | None:
    normalized = _normalize_spaces(text)
    for raw, label in TASK2_4WAY_CANONICAL.items():
        if normalized == raw or normalized.startswith(raw):
            return Task24WayDecision(label=label, confidence=1.0, reason="direct_parse")
    label = _unique_mentioned_label(normalized, list(TASK2_4WAY_CANONICAL))
    if label is not None:
        return Task24WayDecision(
            label=TASK2_4WAY_CANONICAL[label],
            confidence=0.95,
            reason="direct_parse_mentioned_label",
        )
    return None


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        if json_repair is None:
            return None
        try:
            repaired = json_repair.loads(match.group(0))
        except Exception:
            return None
        return repaired if isinstance(repaired, dict) else None


def _json_model_call(judge_client, judge_model: str, prompt: str) -> str:
    if hasattr(judge_client, "messages"):
        resp = judge_client.messages.create(
            model=judge_model,
            max_tokens=200,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    resp = judge_client.chat.completions.create(
        model=judge_model,
        max_tokens=200,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def _pydantic_ai_call(model_name: str, prompt: str, output_model):
    if not HAS_PYDANTIC_AI:
        raise RuntimeError("pydantic_ai is not installed")
    agent = Agent(model_name, output_type=output_model)
    return agent.run_sync(prompt).output


def classify_task1(
    response: str,
    prompt: str,
    judge_client=None,
    judge_model: str | None = None,
    pydantic_ai_model: str | None = None,
) -> Task1Decision:
    direct = direct_parse_task1(response)
    if direct is not None:
        return direct
    if pydantic_ai_model and HAS_PYDANTIC_AI:
        return _pydantic_ai_call(pydantic_ai_model, prompt, Task1Decision)
    if judge_client is None or judge_model is None:
        raise ValueError("Need judge_client/judge_model or pydantic_ai_model for non-trivial Task 1 extraction")
    payload = _extract_json_object(_json_model_call(judge_client, judge_model, prompt))
    if payload is None:
        raise ValueError("Judge did not return valid JSON for Task 1")
    return Task1Decision.model_validate(payload)


def classify_task2(
    response: str,
    prompt: str,
    judge_client=None,
    judge_model: str | None = None,
    pydantic_ai_model: str | None = None,
) -> Task2Decision:
    direct = direct_parse_task2(response)
    if direct is not None:
        return direct
    if pydantic_ai_model and HAS_PYDANTIC_AI:
        return _pydantic_ai_call(pydantic_ai_model, prompt, Task2Decision)
    if judge_client is None or judge_model is None:
        raise ValueError("Need judge_client/judge_model or pydantic_ai_model for non-trivial Task 2 extraction")
    payload = _extract_json_object(_json_model_call(judge_client, judge_model, prompt))
    if payload is None:
        raise ValueError("Judge did not return valid JSON for Task 2")
    return Task2Decision.model_validate(payload)


def classify_task2_4way(
    response: str,
    prompt: str,
    judge_client=None,
    judge_model: str | None = None,
    pydantic_ai_model: str | None = None,
) -> Task24WayDecision:
    direct = direct_parse_task2_4way(response)
    if direct is not None:
        return direct
    if pydantic_ai_model and HAS_PYDANTIC_AI:
        return _pydantic_ai_call(pydantic_ai_model, prompt, Task24WayDecision)
    if judge_client is None or judge_model is None:
        raise ValueError("Need judge_client/judge_model or pydantic_ai_model for non-trivial Task 2 4-way extraction")
    payload = _extract_json_object(_json_model_call(judge_client, judge_model, prompt))
    if payload is None:
        raise ValueError("Judge did not return valid JSON for Task 2 4-way")
    return Task24WayDecision.model_validate(payload)
