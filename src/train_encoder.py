"""Benchmarks: full fine-tuning of encoder models (RoBERTa-base, DistilBERT).

Each question becomes a multiple-choice problem. The encoder reads every
option separately and gives it one score, and the highest score wins:

* Subtask A: each option is one statement. The model learns to give the
  nonsensical statement the highest score.
* Subtask B: each option pairs the false statement with one reason. The
  model learns to give the correct reason the highest score.

This is the standard multiple-choice setup for BERT-style models.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Callable, Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMultipleChoice, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import DEVICE, ENCODER, SEED, EncoderSettings
from src.data import LETTERS, N_OPTIONS, SubtaskAExample, SubtaskBExample
from src.prompts import Example
from src.utils import Timer, accuracy, set_seed


def load_encoder(model_name: str, device: str = DEVICE):
    """Load an encoder with a freshly initialised multiple-choice head."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer


def choice_texts(example: Example) -> tuple[list[str], list[str] | None]:
    """The text of every option: (first segments, second segments or None)."""
    if isinstance(example, SubtaskAExample):
        return [example.sent0, example.sent1], None
    if isinstance(example, SubtaskBExample):
        return [example.false_sent] * len(example.options), list(example.options)
    raise TypeError(f"Unsupported example type: {type(example).__name__}")


def encode_choices(tokenizer, examples: Sequence[Example], max_length: int, device) -> dict:
    """Tokenize a batch into tensors of shape [batch, n_options, seq_len]."""
    firsts: list[str] = []
    seconds: list[str] = []
    for ex in examples:
        first, second = choice_texts(ex)
        firsts.extend(first)
        if second is not None:
            seconds.extend(second)

    encoded = tokenizer(
        firsts,
        seconds or None,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    n_options = len(examples[0].options)
    return {k: v.view(len(examples), n_options, -1).to(device) for k, v in encoded.items()}


def train_encoder(
    model,
    tokenizer,
    train_examples: Sequence[Example],
    cfg: EncoderSettings = ENCODER,
    seed: int = SEED,
    desc: str = "Encoder",
    show_progress: bool = True,
) -> list[float]:
    """Fine-tune every weight of the encoder. Returns the loss of each step."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    steps_per_epoch = math.ceil(len(train_examples) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total_steps), total_steps
    )

    losses: list[float] = []
    model.train()
    progress = tqdm(total=total_steps, desc=desc, disable=not show_progress)
    for epoch in range(cfg.epochs):
        order = list(train_examples)
        random.Random(seed + epoch).shuffle(order)

        for start in range(0, len(order), cfg.batch_size):
            batch_examples = order[start:start + cfg.batch_size]
            batch = encode_choices(tokenizer, batch_examples, cfg.max_length, device)
            labels = torch.tensor([ex.label for ex in batch_examples], device=device)

            loss = model(**batch, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            progress.update(1)
            progress.set_postfix(loss=f"{sum(losses[-10:]) / len(losses[-10:]):.3f}")
    progress.close()
    model.eval()
    return losses


@torch.inference_mode()
def predict_encoder(
    model,
    tokenizer,
    examples: Sequence[Example],
    batch_size: int = 32,
    max_length: int = ENCODER.max_length,
) -> list[int]:
    """Index of the highest-scoring option for each example."""
    device = next(model.parameters()).device
    model.eval()
    predictions: list[int] = []
    for start in range(0, len(examples), batch_size):
        batch = encode_choices(tokenizer, examples[start:start + batch_size], max_length, device)
        predictions.extend(model(**batch).logits.argmax(dim=-1).tolist())
    return predictions


def run_encoder(
    model_name: str,
    subtask: str,
    train_examples: Sequence[Example],
    eval_examples: Sequence[Example],
    cfg: EncoderSettings = ENCODER,
    seed: int = SEED,
    load_fn: Callable = load_encoder,
) -> dict:
    """Load an encoder, fine-tune it on one subtask and evaluate it."""
    set_seed(seed)  # also fixes the random initialisation of the new head
    model, tokenizer = load_fn(model_name)
    total = sum(p.numel() for p in model.parameters())

    with Timer(verbose=False) as t_train:
        losses = train_encoder(model, tokenizer, train_examples, cfg, seed, desc=f"{subtask}")
    with Timer(verbose=False) as t_eval:
        predictions = predict_encoder(model, tokenizer, eval_examples, max_length=cfg.max_length)

    counts = Counter(predictions)
    first, last = losses[: max(1, len(losses) // 10)], losses[-max(1, len(losses) // 10):]
    result = {
        "accuracy": accuracy(predictions, [ex.label for ex in eval_examples]),
        "n_eval": len(eval_examples),
        "prediction_counts": {LETTERS[i]: counts.get(i, 0) for i in range(N_OPTIONS[subtask])},
        "eval_seconds": t_eval.seconds,
        "n_train": len(train_examples),
        "train_seconds": t_train.seconds,
        "trainable_params": total,
        "total_params": total,
        "loss_start": sum(first) / len(first),
        "loss_end": sum(last) / len(last),
        "predictions": predictions,
    }
    del model
    return result
