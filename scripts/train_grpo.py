"""
CLI script for running GRPO training.

Usage:
    export CHECKPOINT_DIR=./checkpoints
    export HF_TOKEN=your_token
    export WANDB_API_KEY=your_key

    # GRPO from SFT (correctness + format only)
    python scripts/train_grpo.py --start sft --rewards correctness format

    # GRPO from base (full composite reward)
    python scripts/train_grpo.py --start base --rewards correctness format style
"""

import argparse
import os

import wandb
from huggingface_hub import login
from dotenv import load_dotenv

from src.config import PATHS
from src.model import load_base_model_and_tokenizer, load_peft_model, checkpoint_exists
from src.data.gsm8k import load_gsm8k_dataset, format_gsm8k_for_grpo
from src.rewards.correctness import correctness_reward
from src.rewards.format import format_reward
from src.rewards.style import load_style_classifier, style_reward
from src.training.grpo import build_grpo_trainer

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO training.")
    parser.add_argument("--start", choices=["sft", "base"], default="sft",
                        help="Starting checkpoint: 'sft' or 'base'.")
    parser.add_argument("--rewards", nargs="+",
                        choices=["correctness", "format", "style"],
                        default=["correctness", "format"],
                        help="Reward components to include.")
    parser.add_argument("--epochs", type=int,   default=1)
    parser.add_argument("--lr",     type=float, default=5e-6)
    parser.add_argument("--batch",  type=int,   default=2)
    return parser.parse_args()


def main():
    args = parse_args()

    login(os.environ["HF_TOKEN"])
    wandb.login(os.environ.get("WANDB_API_KEY"))

    output_dir = (PATHS["rl_yoda_answ_from_sft"]
                  if args.start == "sft"
                  else PATHS["rl_yoda_answ_from_base"])

    if checkpoint_exists(output_dir):
        print(f"Checkpoint exists at {output_dir}. Skipping.")
        return

    if args.start == "sft":
        print("Loading SFT checkpoint as starting point...")
        model, tokenizer = load_peft_model(PATHS["sft_yoda_answ"])
    else:
        print("Loading base model as starting point...")
        model, tokenizer = load_base_model_and_tokenizer(quantize=True, quantize_n_bits=4)

    print("Loading GSM8K dataset...")
    dataset = load_gsm8k_dataset()
    train_ds = dataset["train"].map(
        lambda ex: format_gsm8k_for_grpo(ex, tokenizer)
    )

    reward_functions = []
    if "correctness" in args.rewards:
        reward_functions.append(correctness_reward)
    if "format" in args.rewards:
        reward_functions.append(format_reward)
    if "style" in args.rewards:
        cls_model, cls_tok = load_style_classifier(str(PATHS["classifier_yoda"]))
        reward_functions.append(
            lambda p, c, **kw: style_reward(p, c,
                                            classifier_model=cls_model,
                                            classifier_tokenizer=cls_tok,
                                            **kw)
        )

    trainer = build_grpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        reward_functions=reward_functions,
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
    )

    print("Starting GRPO training...")
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()