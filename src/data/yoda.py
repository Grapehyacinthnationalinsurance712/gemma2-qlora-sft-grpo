"""
Dataset loading and formatting for Yoda-style translation SFT.

Dataset: dvgodoy/yoda_sentences (HuggingFace)
"""

from typing import Dict

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from src.config import YODA_DATASET_NAME, SEED


def load_yoda_dataset(dataset_name: str = YODA_DATASET_NAME) -> DatasetDict:
    """Load the Yoda sentences dataset and create a train/validation split.

    Args:
        dataset_name: HuggingFace dataset identifier.

    Returns:
        DatasetDict with 'train' (90%) and 'test' (10%) splits.
    """
    raw = load_dataset(dataset_name)
    return raw["train"].train_test_split(test_size=0.1, seed=SEED)


def format_yoda_translation_example(
    example: Dict, tokenizer: AutoTokenizer
) -> Dict:
    """Format a single Yoda dataset example into a chat-template prompt.

    The model learns: given standard English input, produce Yoda-style output.

    Args:
        example: Dataset row with 'normal' and 'yoda' fields.
        tokenizer: Tokenizer with apply_chat_template support.

    Returns:
        Dict with 'text' key containing the formatted training string.
    """
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": f"Translate to Yoda syntax:\n<text>{example['normal']}</text>",
            },
            {
                "role": "model",
                "content": example["yoda"],
            },
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": prompt}