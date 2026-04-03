"""
CLI inference script for Yoda translation or GSM8K reasoning.

Usage:
    python scripts/inference.py --task yoda --text "The stars are bright tonight."
    python scripts/inference.py --adapter checkpoints/sft_yoda --task yoda \
        --text "Knowledge is power."
"""

import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import login

from src.model import load_base_model_and_tokenizer, load_peft_model
from src.generation import format_prompt_yoda, format_prompt_gsm8k, generate_response

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference.")
    parser.add_argument("--adapter", default=None,
                        help="Path to LoRA adapter directory. Uses base model if omitted.")
    parser.add_argument("--task", choices=["yoda", "gsm8k"], default="yoda",
                        help="Task type for prompt formatting.")
    parser.add_argument("--text", required=True,
                        help="Input text.")
    parser.add_argument("--max_new_tokens", type=int,   default=256)
    parser.add_argument("--temperature",    type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    login(os.environ.get("HF_TOKEN", ""))

    if args.adapter:
        print(f"Loading adapter from: {args.adapter}")
        model, tokenizer = load_peft_model(args.adapter)
    else:
        print("Loading base model...")
        model, tokenizer = load_base_model_and_tokenizer(quantize=True, quantize_n_bits=8)
        model.eval()

    if args.task == "yoda":
        prompt = format_prompt_yoda(args.text, tokenizer)
    else:
        prompt = format_prompt_gsm8k(args.text, tokenizer)

    response = generate_response(
        model, tokenizer, prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(f"\nInput:  {args.text}")
    print(f"Output: {response}")


if __name__ == "__main__":
    main()