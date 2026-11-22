"""Letter scoring: read the model's answer without letting it generate text.

For each question we run one forward pass and look at the model's
prediction for the first token of its reply. We compare the scores of the
answer letters only ("A"/"B", or "A"/"B"/"C") and pick the highest. This
always gives a valid answer and needs no parsing of free text.

The same ``letter_logits`` function serves evaluation (here) and LoRA
training (later), so both read the model's answer in exactly the same way.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import torch
from tqdm.auto import tqdm

from src.data import LETTERS


def letter_token_ids(tokenizer, n_options: int) -> list[int]:
    """Token IDs of the first ``n_options`` answer letters.

    Fails if a letter is not a single token, because one next-token
    probability could not then represent it.
    """
    ids = []
    for letter in LETTERS[:n_options]:
        tokens = tokenizer.encode(letter, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Letter {letter!r} is {len(tokens)} tokens: {tokens}")
        ids.append(tokens[0])
    return ids


def encode_messages(tokenizer, messages_batch: Sequence[list[dict]], device: str):
    """Apply the chat template (thinking off) and tokenize a batch of prompts.

    The template ends each prompt where the assistant's reply would begin,
    so the model's prediction at the last position is its first reply token.
    """
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for messages in messages_batch
    ]
    return tokenizer(texts, return_tensors="pt", padding=True).to(device)


def letter_logits(model, batch, letter_ids: Sequence[int]):
    """Next-token scores at the last position.

    Returns ``(letters, full)``: the scores of the answer letters only,
    shape ``[batch, n_options]``, and the scores over the whole vocabulary,
    shape ``[batch, vocab]``. ``logits_to_keep=1`` asks the model for the
    last position only, which saves memory and time.

    This function keeps gradients, so LoRA training can call it directly.
    """
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        logits_to_keep=1,
    )
    full = outputs.logits[:, -1, :]
    return full[:, list(letter_ids)], full


@dataclass
class ScoreResult:
    """Predictions and diagnostics from scoring a list of questions."""

    predictions: list[int]
    # Probabilities over the answer letters only (each row sums to 1).
    letter_probs: list[list[float]]
    # Share of the model's full next-token probability that went to the
    # answer letters. Low values mean the model barely follows the format.
    letter_mass: list[float]

    @property
    def mean_letter_mass(self) -> float:
        return sum(self.letter_mass) / len(self.letter_mass) if self.letter_mass else 0.0

    def prediction_counts(self) -> dict[str, int]:
        """How often each letter was predicted (shows position bias)."""
        n_options = len(self.letter_probs[0]) if self.letter_probs else 0
        counts = Counter(self.predictions)
        return {LETTERS[i]: counts.get(i, 0) for i in range(n_options)}


@torch.inference_mode()
def score_messages(
    model,
    tokenizer,
    messages_list: Sequence[list[dict]],
    n_options: int,
    batch_size: int = 8,
    show_progress: bool = True,
) -> ScoreResult:
    """Score a list of chat prompts and return the predicted letter indices."""
    device = next(model.parameters()).device
    letter_ids = letter_token_ids(tokenizer, n_options)

    predictions: list[int] = []
    letter_probs: list[list[float]] = []
    letter_mass: list[float] = []

    batches = range(0, len(messages_list), batch_size)
    for start in tqdm(batches, desc="Scoring", disable=not show_progress):
        batch = encode_messages(tokenizer, messages_list[start:start + batch_size], device)
        letters, full = letter_logits(model, batch, letter_ids)

        predictions.extend(letters.argmax(dim=-1).tolist())
        letter_probs.extend(torch.softmax(letters, dim=-1).tolist())
        letter_mass.extend(torch.softmax(full, dim=-1)[:, list(letter_ids)].sum(dim=-1).tolist())

    return ScoreResult(predictions, letter_probs, letter_mass)
