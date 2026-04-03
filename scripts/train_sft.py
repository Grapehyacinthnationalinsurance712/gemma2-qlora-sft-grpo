"""
CLI script for running SFT training.

Usage:
    export CHECKPOINT_DIR=./checkpoints
    export HF_TOKEN=your_token
    export WANDB_API_KEY=your_key

    python scripts/train_sft.py --task yoda --epochs 3 --lr 2e-4
    python scripts/train_sft.py --task qa   --epochs 3 --lr 2e-4
"""

import argparse
import os

import wandb
from huggingface_hub import login
from dotenv import load_dotenv

from src.config import PATHS, SEED
from src.model import load_base_model_and_tokenizer, checkpoint_exists
from src.data.yoda import load_yoda_dataset, format_yoda_translation_example
from src.training.sft import build_sft_trainer, get_lora_config

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Run SFT training.")
    parser.add_argument("--task",   choices=["yoda", "qa"], default="yoda",
                        help="Training task: 'yoda' for translation, 'qa' for Yoda QA.")
    parser.add_argument("--epochs", type=int,   default=3)
    parser.add_argument("--lr",     type=float, default=2e-4)
    parser.add_argument("--batch",  type=int,   default=4)
    parser.add_argument("--lora_r", type=int,   default=16)
    return parser.parse_args()


def main():
    args = parse_args()

    login(os.environ["HF_TOKEN"])
    wandb.login(os.environ.get("WANDB_API_KEY"))

    output_dir = PATHS["sft_yoda"] if args.task == "yoda" else PATHS["sft_yoda_answ"]

    if checkpoint_exists(output_dir):
        print(f"Checkpoint already exists at {output_dir}. Skipping training.")
        return

    print(f"Loading base model...")
    model, tokenizer = load_base_model_and_tokenizer(quantize=True, quantize_n_bits=4)

    print(f"Loading dataset for task: {args.task}")
    dataset = load_yoda_dataset()
    train_ds = dataset["train"].map(
        lambda ex: format_yoda_translation_example(ex, tokenizer)
    )
    eval_ds = dataset["test"].map(
        lambda ex: format_yoda_translation_example(ex, tokenizer)
    )

    lora_config = get_lora_config(r=args.lora_r)
    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        lora_config=lora_config,
    )

    print("Starting SFT training...")
    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()