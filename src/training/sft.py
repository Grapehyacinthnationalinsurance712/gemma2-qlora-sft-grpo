"""
Supervised Fine-Tuning configuration using TRL SFTTrainer with LoRA.
"""

from pathlib import Path

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

from src.config import SEED


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> LoraConfig:
    """Return a LoRA configuration for causal LM fine-tuning.

    Args:
        r: LoRA rank — controls the number of trainable parameters.
        lora_alpha: Scaling factor for LoRA updates.
        lora_dropout: Dropout probability applied to LoRA layers.

    Returns:
        Configured LoraConfig instance.
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )


def build_sft_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: Path,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    lora_config: LoraConfig = None,
) -> SFTTrainer:
    """Build and return a configured SFTTrainer instance.

    Args:
        model: Base model (quantized).
        tokenizer: Corresponding tokenizer.
        train_dataset: Training split.
        eval_dataset: Validation split.
        output_dir: Directory for saving checkpoints.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per GPU.
        learning_rate: Initial learning rate.
        max_seq_length: Maximum token length per sample.
        lora_config: LoRA configuration. Uses default if None.

    Returns:
        Configured SFTTrainer ready to call .train().
    """
    if lora_config is None:
        lora_config = get_lora_config()

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        seed=SEED,
        report_to="wandb",
    )

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )