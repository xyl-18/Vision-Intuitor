"""
Accuracy reward with auxiliary format and length scores for Intuitor.

- `overall` is a weighted combination of available components.
- `format` and `length` are exposed as auxiliary signals for advantage fusion.
"""

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer

REWARD_NAME = "vision_intuitor_accuracy_with_format_reward"
REWARD_TYPE = "sequential"


def accuracy_reward(response: str, ground_truth: str) -> float:
    try:
        answer = extract_boxed_content(response)
        return 1.0 if grade_answer(answer, ground_truth) else 0.0
    except Exception:
        return 0.0


def format_reward(response: str) -> float:
    # Match: <think>...</think> followed by final \boxed{...}
    pattern = re.compile(r"<think>.*?</think>.*?\\boxed\{.*?\}\s*$", re.DOTALL)
    return 1.0 if pattern.search(response) else 0.0


def length_penalty(response_length: int, max_response_length: int) -> float:
    if max_response_length <= 0:
        return 0.0
    clamped_length = min(max(response_length, 0), max_response_length)
    return -float(clamped_length) / float(max_response_length)


def compute_score(
    reward_input: dict[str, Any],
    accuracy_weight: float = 1.0,
    format_weight: float = 0.0,
    length_weight: float = 0.0,
) -> dict[str, float]:
    response = reward_input["response"]
    response_length = int(reward_input.get("response_length", 0))
    max_response_length = int(reward_input.get("max_response_length", 4096))
    ground_truth = reward_input["ground_truth"]

    acc = accuracy_reward(response, ground_truth)
    fmt = format_reward(response)
    length = length_penalty(response_length, max_response_length)

    overall = accuracy_weight * acc + format_weight * fmt + length_weight * length

    return {
        "overall": overall,
        "accuracy": acc,
        "format": fmt,
        "length": length,
    }
