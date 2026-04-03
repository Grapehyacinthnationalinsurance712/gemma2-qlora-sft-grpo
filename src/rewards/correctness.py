"""
Correctness reward for GRPO: exact numeric match against GSM8K ground truth.
"""

import re
from typing import List


def extract_gsm8k_final_answer(text: str) -> str | None:
    """Extract the numeric answer following the '####' marker.

    Args:
        text: Model response or ground-truth string.

    Returns:
        Extracted numeric string, or None if no marker found.
    """
    match = re.search(r"####\s*([\d,.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def correctness_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Reward function: 1.0 if extracted answer matches ground truth, else 0.0.

    Args:
        prompts: List of input prompt strings (unused, required by GRPO interface).
        completions: List of model-generated response strings.
        **kwargs: Must include 'answer' — list of ground-truth answer strings.

    Returns:
        List of reward scores in {0.0, 1.0}.
    """
    ground_truths = kwargs.get("answer", [""] * len(completions))
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        predicted = extract_gsm8k_final_answer(completion)
        expected  = extract_gsm8k_final_answer(gt)
        if predicted is not None and expected is not None and predicted == expected:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards