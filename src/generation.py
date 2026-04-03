"""
Inference utilities for Gemma-2 with chat template formatting.
"""

import torch
from transformers import AutoTokenizer, PreTrainedModel


def format_prompt_yoda(text: str, tokenizer: AutoTokenizer) -> str:
    """Format an English sentence for Yoda-style translation inference.

    Args:
        text: Input English text.
        tokenizer: Tokenizer with chat template support.

    Returns:
        Formatted prompt string ready for tokenization.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Translate to Yoda syntax:\n<text>{text}</text>"}],
        tokenize=False,
        add_generation_prompt=True,
    )


def format_prompt_gsm8k(question: str, tokenizer: AutoTokenizer) -> str:
    """Format a GSM8K question for chain-of-thought reasoning inference.

    Args:
        question: Math word problem string.
        tokenizer: Tokenizer with chat template support.

    Returns:
        Formatted prompt string ready for tokenization.
    """
    content = (
        f"{question}\n\nShow your reasoning step by step, "
        f"then write your final answer as: #### <number>"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_response(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """Run inference and return the decoded model response.

    Args:
        model: Loaded language model (base or PEFT).
        tokenizer: Corresponding tokenizer.
        prompt: Formatted prompt string.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature. Use 0.0 for greedy decoding.

    Returns:
        Decoded response string (prompt tokens excluded).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-6),
            do_sample=temperature > 0,
        )

    return tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
    )