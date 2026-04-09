"""Reward function for Vision-Intuitor-SR1.

Validation uses `overall` as the model-selection metric.
Training can consume each component independently with configurable weights.
"""

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer

REWARD_NAME = "vision_intuitor_sr1_self_reward"
REWARD_TYPE = "sequential"


def format_reward(response: str) -> float:
    pattern = re.compile(
        r"^\s*<description>.*?</description>\s*"
        r"<think>.*?</think>\s*"
        r"\\boxed\{.*?\}\s*$",
        re.DOTALL,
    )
    return 1.0 if pattern.fullmatch(response) else 0.0


def description_format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*\\boxed\{.*?\}\s*$", re.DOTALL)
    return 1.0 if pattern.fullmatch(response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    try:
        answer = extract_boxed_content(response)
        return 1.0 if grade_answer(answer, ground_truth) else 0.0
    except Exception:
        return 0.0


def length_penalty(response_length: int, max_response_length: int) -> float:
    if max_response_length <= 0:
        return 0.0
    clamped_length = min(max(response_length, 0), max_response_length)
    return -float(clamped_length) / float(max_response_length)


def compute_score(
    reward_input: dict[str, Any],
    format_weight: float = 0.1,
    description_accuracy_weight: float = 1.0,
    description_format_weight: float = 0.0,
) -> dict[str, float]:
    response = reward_input["response"]
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
    ground_truth = reward_input["ground_truth"]
    description_answer = reward_input.get("description_answer", "")
    response_length = int(reward_input.get("response_length", 0))
    description_response_length = int(reward_input.get("description_response_length", 0))
    max_response_length = int(reward_input.get("max_response_length", 4096))
    description_max_response_length = int(
        reward_input.get("description_max_response_length", max_response_length)
    )

    fmt_score = format_reward(response)
    acc_score = accuracy_reward(response, ground_truth)

    desc_acc = 0.0
    desc_fmt = 0.0
    if description_answer:
        try:
            desc_acc = accuracy_reward(description_answer, ground_truth)
        except Exception:
            pass
        try:
            desc_fmt = description_format_reward(description_answer)
        except Exception:
            pass

    len_score = length_penalty(response_length, max_response_length)
    desc_len_score = length_penalty(description_response_length, description_max_response_length)

    overall = (
        (1.0 - format_weight) * acc_score
        + format_weight * fmt_score
        + description_accuracy_weight * desc_acc
        + description_format_weight * desc_fmt
    )

    return {
        "overall": overall,
        "format": fmt_score,
        "accuracy": acc_score,
        "length": len_score,
        "description_format": desc_fmt,
        "description_accuracy": desc_acc,
        "description_length": desc_len_score,
    }
