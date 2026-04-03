"""
Dataset loading and formatting for GSM8K mathematical reasoning.

Dataset: openai/gsm8k (HuggingFace)
"""

from typing import Dict

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from src.config import REASONING_DATASET_NAME, SEED


def load_gsm8k_dataset(dataset_name: str = REASONING_DATASET_NAME) -> DatasetDict:
    """Load the GSM8K grade-school math dataset.

    Args:
        dataset_name: HuggingFace dataset identifier.

    Returns:
        DatasetDict with 'train' and 'test' splits.
    """
    return load_dataset(dataset_name, "main")


def format_gsm8k_for_grpo(example: Dict, tokenizer: AutoTokenizer) -> Dict:
    """Format a GSM8K example as a GRPO prompt (question only, no answer).

    The model generates a chain-of-thought response ending with #### <number>.

    Args:
        example: Dataset row with 'question' and 'answer' fields.
        tokenizer: Tokenizer with apply_chat_template support.

    Returns:
        Dict with 'prompt' key containing the formatted input string.
    """
    content = (
        f"{example['question']}\n\n"
        f"Show your reasoning step by step, "
        f"then write your final answer as: #### <number>"
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {"prompt": prompt, "answer": example["answer"]}