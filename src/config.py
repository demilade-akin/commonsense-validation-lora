"""Project-wide settings.

Every path, model name, sample size and training setting lives here, so one
edit changes the whole pipeline. To fit the one-hour limit on a slower
machine, lower the ``train_n`` values below (``python run_all.py --quick``
prints an estimate of the full run time).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Each entry is a HuggingFace Hub ID or a path to a local folder holding the
# model files. Use a local path if the machine cannot download from the Hub.
MODEL_NAME = "Qwen/Qwen3-0.6B"
ENCODER_MODELS = {
    "roberta-base": "FacebookAI/roberta-base",
    "distilbert": "distilbert/distilbert-base-uncased",
}

# The project runs on CPU so results and timings match the course VM.
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Official ComVE split sizes. The loaders fail if a split does not match.
EXPECTED_COUNTS = {"train": 10_000, "dev": 997, "test": 1_000}

# Split used for the reported results. Use "dev" while tuning settings.
EVAL_SPLIT = "test"

# Examples scored per subtask, sampled with SEED. None scores the whole split.
# 400 keeps the full run under one hour on the course VM.
EVAL_N: int | None = 400

# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
# Solved training examples shown in few-shot prompts. The counts give each
# answer letter the same number of examples (A/B once for Subtask A, A/B/C
# once for Subtask B), which avoids teaching the model to favour one letter.
FEW_SHOT_K = {"A": 2, "B": 3}

# Prompts scored per forward pass.
SCORING_BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# LoRA fine-tuning of the language model (proposed approach)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoRASettings:
    rank: int = 8                          # width of the low-rank matrices
    alpha: int = 16                        # scale of the update (alpha / rank)
    dropout: float = 0.05                  # dropout on the LoRA input
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    train_n: int | None = 400              # training examples per subtask
    epochs: int = 1
    batch_size: int = 8
    lr: float = 3e-4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0


# ---------------------------------------------------------------------------
# Full fine-tuning of encoder benchmarks (RoBERTa-base, DistilBERT)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EncoderSettings:
    train_n: int | None = 2000             # training examples per subtask
    epochs: int = 1
    batch_size: int = 16
    lr: float = 3e-5
    warmup_ratio: float = 0.1
    max_length: int = 128                  # tokens per choice
    max_grad_norm: float = 1.0


LORA = LoRASettings()
ENCODER = EncoderSettings()

# ---------------------------------------------------------------------------
# Quick mode (python run_all.py --quick)
# ---------------------------------------------------------------------------
# Tiny sizes that check every stage runs and measure its speed.
QUICK_EVAL_N = 20
QUICK_TRAIN_N = 32
