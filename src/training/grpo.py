"""
GRPO training configuration using TRL GRPOTrainer.

Supports composite rewards: correctness, format, and optional style.
"""

from pathlib import Path
from typing import Callable, List

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from src.config import SEED


def get_grpo_lora_config(r: int = 16, lora_alpha: int = 32) -> LoraConfig:
    """Return a LoRA configuration for GRPO training.

    Args:
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.

    Returns:
        Configured LoraConfig instance.
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )


def build_grpo_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    reward_functions: List[Callable],
    output_dir: Path,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 5e-6,
    max_new_tokens: int = 512,
    lora_config: LoraConfig = None,
) -> GRPOTrainer:
    """Build and return a configured GRPOTrainer instance.

    Args:
        model: Base or SFT-initialised model.
        tokenizer: Corresponding tokenizer.
        train_dataset: Training dataset with 'prompt' and 'answer' fields.
        reward_functions: List of reward callables. Each takes (prompts, completions)
                          and returns a list of scalar rewards.
        output_dir: Directory for saving checkpoints.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per GPU.
        learning_rate: Initial learning rate.
        max_new_tokens: Maximum tokens to generate per rollout.
        lora_config: LoRA configuration. Uses default if None.

    Returns:
        Configured GRPOTrainer ready to call .train().
    """
    if lora_config is None:
        lora_config = get_grpo_lora_config()

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        seed=SEED,
        report_to="wandb",
    )

    return GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_functions,
    )