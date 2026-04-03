"""
Model loading utilities for QLoRA fine-tuning with Gemma-2.

Supports 4-bit and 8-bit quantization via BitsAndBytes.
"""

import gc
from pathlib import Path
from typing import Literal, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

from src.config import PATHS, BASE_MODEL_NAME


def get_quantization_config(n_bits: Literal[4, 8] = 4) -> BitsAndBytesConfig:
    """Return BitsAndBytes quantization config for QLoRA.

    Args:
        n_bits: Quantization precision. 4 uses NF4 double quantization,
                8 uses standard 8-bit quantization.

    Returns:
        Configured BitsAndBytesConfig instance.
    """
    if n_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def load_base_model_and_tokenizer(
    model_name: str = BASE_MODEL_NAME,
    quantize: bool = True,
    quantize_n_bits: Literal[4, 8] = 4,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal language model and its tokenizer.

    Args:
        model_name: HuggingFace model identifier.
        quantize: Whether to apply BitsAndBytes quantization.
        quantize_n_bits: Quantization precision (4 or 8 bits).

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = get_quantization_config(quantize_n_bits) if quantize else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not quantize else None,
    )
    return model, tokenizer


def load_peft_model(
    adapter_path: Path,
    base_model_name: str = BASE_MODEL_NAME,
    quantize: bool = True,
    quantize_n_bits: Literal[4, 8] = 4,
) -> Tuple[PeftModel, AutoTokenizer]:
    """Load a LoRA adapter on top of the quantized base model.

    Args:
        adapter_path: Path to the saved adapter directory.
        base_model_name: HuggingFace model identifier for the base model.
        quantize: Whether to quantize the base model before loading the adapter.
        quantize_n_bits: Quantization precision.

    Returns:
        Tuple of (peft_model, tokenizer).
    """
    model, tokenizer = load_base_model_and_tokenizer(
        base_model_name, quantize, quantize_n_bits
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def checkpoint_exists(path: Path) -> bool:
    """Return True if a checkpoint directory exists and is non-empty."""
    return path.exists() and any(path.iterdir())


def get_vram_usage() -> str:
    """Return formatted string of current VRAM allocation and reservation."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved() / 1e9
    return f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB"


def free_memory(*models) -> None:
    """Delete model references and clear CUDA cache.

    Args:
        *models: Any number of model objects to delete.
    """
    for m in models:
        del m
    torch.cuda.empty_cache()
    gc.collect()