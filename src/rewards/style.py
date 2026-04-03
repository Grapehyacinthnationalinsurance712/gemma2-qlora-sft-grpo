"""
Style reward for GRPO: uses the trained DistilBERT classifier to score
the probability that a response is written in Yoda style.
"""

from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_style_classifier(
    model_path: str,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load the trained Yoda-style binary classifier.

    Args:
        model_path: Path or HuggingFace identifier for the classifier checkpoint.

    Returns:
        Tuple of (classifier_model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def style_reward(
    prompts: List[str],
    completions: List[str],
    classifier_model: Optional[AutoModelForSequenceClassification] = None,
    classifier_tokenizer: Optional[AutoTokenizer] = None,
    **kwargs,
) -> List[float]:
    """Reward function: classifier probability of Yoda-style for each completion.

    Returns 0.0 for all completions if classifier is not provided.

    Args:
        prompts: List of input prompt strings (unused).
        completions: List of model-generated response strings.
        classifier_model: Trained DistilBERT binary classifier.
        classifier_tokenizer: Corresponding tokenizer.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        List of reward scores in [0.0, 1.0].
    """
    if classifier_model is None or classifier_tokenizer is None:
        return [0.0] * len(completions)

    device = next(classifier_model.parameters()).device
    rewards = []

    for text in completions:
        inputs = classifier_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = classifier_model(**inputs).logits
            prob = torch.softmax(logits, dim=-1)[0, 1].item()

        rewards.append(prob)

    return rewards