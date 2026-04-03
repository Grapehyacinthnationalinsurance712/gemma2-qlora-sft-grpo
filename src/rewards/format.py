"""
Format reward for GRPO: checks for the required '#### <number>' answer marker.
"""

import re
from typing import List


def format_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Reward function: 1.0 if response contains a valid answer marker, else 0.0.

    The required format is: #### <number>
    This ensures the model produces a parseable final answer.

    Args:
        prompts: List of input prompt strings (unused).
        completions: List of model-generated response strings.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        List of reward scores in {0.0, 1.0}.
    """
    pattern = re.compile(r"####\s*[\d,.\-]+")
    return [1.0 if pattern.search(c) else 0.0 for c in completions]