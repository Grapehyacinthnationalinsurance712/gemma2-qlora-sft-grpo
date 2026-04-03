"""
Global configuration for gemma2-qlora-sft-grpo.

All paths and model names are centralised here.
Set CHECKPOINT_DIR via environment variable before running:

    export CHECKPOINT_DIR=/path/to/your/checkpoints
"""

import os
from pathlib import Path

# ── Models ─────────────────────────────────────────────────────────────────
BASE_MODEL_NAME       = "google/gemma-2-2b-it"
CLASSIFIER_MODEL_NAME = "distilbert-base-uncased"

# ── Datasets ───────────────────────────────────────────────────────────────
YODA_DATASET_NAME      = "dvgodoy/yoda_sentences"
QA_DATASET_NAME        = "MuskumPillerum/General-Knowledge"
REASONING_DATASET_NAME = "openai/gsm8k"

# ── Paths ───────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoints"))

PATHS = {
    "base_model":             BASE_MODEL_NAME,
    "sft_yoda":               CHECKPOINT_DIR / "sft_yoda",
    "sft_yoda_answ":          CHECKPOINT_DIR / "sft_yoda_answ",
    "rl_yoda_answ_from_sft":  CHECKPOINT_DIR / "rl_yoda_answ_from_sft",
    "rl_yoda_answ_from_base": CHECKPOINT_DIR / "rl_yoda_answ_from_base",
    "synthetic_qa":           CHECKPOINT_DIR / "synthetic_yoda_qa",
    "classifier_data":        CHECKPOINT_DIR / "classifier_dataset",
    "classifier_yoda":        CHECKPOINT_DIR / "classifier_yoda",
}

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42