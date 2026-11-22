"""Fine-tune the language model with LoRA on one subtask, then evaluate it.

Training reuses letter scoring: for each prompt we read the model's scores
for the answer letters at the first reply position and apply cross-entropy
against the correct letter. Only the LoRA weights receive updates.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import LORA, SEED, LoRASettings
from src.data import N_OPTIONS
from src.lora import count_parameters, inject_lora
from src.modeling import load_model_and_tokenizer
from src.prompting import evaluate_lm
from src.prompts import Example, build_messages
from src.scoring import encode_messages, letter_logits, letter_token_ids
from src.utils import Timer, set_seed


def train_lora(
    model,
    tokenizer,
    subtask: str,
    train_examples: Sequence[Example],
    cfg: LoRASettings = LORA,
    seed: int = SEED,
    show_progress: bool = True,
) -> list[float]:
    """Train the LoRA weights already injected into ``model``.

    Returns the loss of every training step, so the caller can check that
    training made progress.
    """
    device = next(model.parameters()).device
    letter_ids = letter_token_ids(tokenizer, N_OPTIONS[subtask])

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters: call inject_lora first.")
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr)

    steps_per_epoch = math.ceil(len(train_examples) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total_steps), total_steps
    )

    losses: list[float] = []
    model.train()
    progress = tqdm(total=total_steps, desc=f"LoRA {subtask}", disable=not show_progress)
    for epoch in range(cfg.epochs):
        order = list(train_examples)
        random.Random(seed + epoch).shuffle(order)

        for start in range(0, len(order), cfg.batch_size):
            batch_examples = order[start:start + cfg.batch_size]
            batch = encode_messages(tokenizer, [build_messages(ex) for ex in batch_examples], device)
            labels = torch.tensor([ex.label for ex in batch_examples], device=device)

            letters, _ = letter_logits(model, batch, letter_ids)
            loss = F.cross_entropy(letters, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            progress.update(1)
            progress.set_postfix(loss=f"{sum(losses[-10:]) / len(losses[-10:]):.3f}")
    progress.close()
    model.eval()
    return losses


def run_lora(
    subtask: str,
    train_examples: Sequence[Example],
    eval_examples: Sequence[Example],
    cfg: LoRASettings = LORA,
    seed: int = SEED,
    load_fn: Callable = load_model_and_tokenizer,
) -> dict:
    """Load a fresh base model, add LoRA, train on one subtask and evaluate.

    Each subtask starts from the original model, so nothing learned on one
    subtask carries over to the other.
    """
    set_seed(seed)
    model, tokenizer = load_fn()
    wrapped = inject_lora(model, cfg.target_modules, cfg.rank, cfg.alpha, cfg.dropout)
    trainable, total = count_parameters(model)

    with Timer(verbose=False) as t:
        losses = train_lora(model, tokenizer, subtask, train_examples, cfg, seed)

    result = evaluate_lm(model, tokenizer, subtask, eval_examples)
    first, last = losses[: max(1, len(losses) // 10)], losses[-max(1, len(losses) // 10):]
    result.update(
        {
            "n_train": len(train_examples),
            "train_seconds": t.seconds,
            "trainable_params": trainable,
            "total_params": total,
            "lora_layers": len(wrapped),
            "loss_start": sum(first) / len(first),
            "loss_end": sum(last) / len(last),
        }
    )
    del model
    return result
