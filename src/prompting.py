"""Evaluate the language model with letter scoring, zero-shot or few-shot.

``evaluate_lm`` scores any version of the model: the base model with or
without solved examples in the prompt (zero-shot and few-shot prompting),
and the LoRA fine-tuned model after training.

Quick check from the repository root:
    python -m src.prompting --n 50             # zero-shot on 50 dev examples
    python -m src.prompting --n 50 --few-shot  # few-shot (counts from config)
"""

from __future__ import annotations

import argparse
from typing import Sequence

from src.config import FEW_SHOT_K, SCORING_BATCH_SIZE, SEED
from src.data import N_OPTIONS, downsample, load_subtask
from src.modeling import load_model_and_tokenizer
from src.prompts import Example, build_messages, pick_shots
from src.scoring import score_messages
from src.utils import Timer, accuracy, set_seed


def evaluate_lm(
    model,
    tokenizer,
    subtask: str,
    examples: Sequence[Example],
    shots: Sequence[Example] = (),
    batch_size: int = SCORING_BATCH_SIZE,
) -> dict:
    """Score ``examples`` with letter scoring and return accuracy and diagnostics."""
    messages = [build_messages(ex, shots) for ex in examples]
    with Timer(verbose=False) as t:
        result = score_messages(model, tokenizer, messages, N_OPTIONS[subtask], batch_size)

    return {
        "accuracy": accuracy(result.predictions, [ex.label for ex in examples]),
        "n_eval": len(examples),
        "prediction_counts": result.prediction_counts(),
        "mean_letter_mass": result.mean_letter_mass,
        "eval_seconds": t.seconds,
    }


def few_shot_examples(subtask: str, seed: int = SEED) -> list[Example]:
    """The solved training examples used in few-shot prompts for ``subtask``."""
    return pick_shots(load_subtask(subtask, "train"), FEW_SHOT_K[subtask], seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot or few-shot prompting.")
    parser.add_argument("--n", type=int, default=50, help="examples per subtask (default 50)")
    parser.add_argument("--few-shot", action="store_true", help="add solved examples to the prompt")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--subtasks", default="AB", help="which subtasks to run, e.g. A or AB")
    args = parser.parse_args()

    set_seed(SEED)
    model, tokenizer = load_model_and_tokenizer()

    for subtask in args.subtasks:
        examples = downsample(load_subtask(subtask, args.split), args.n, SEED)
        shots = few_shot_examples(subtask) if args.few_shot else []
        r = evaluate_lm(model, tokenizer, subtask, examples, shots)
        method = f"{len(shots)}-shot" if shots else "zero-shot"
        print(
            f"Subtask {subtask} | {method} | {args.split} n={r['n_eval']} | "
            f"accuracy {r['accuracy']:.3f} | picks {r['prediction_counts']} | "
            f"letter mass {r['mean_letter_mass']:.3f} | "
            f"{r['eval_seconds'] / r['n_eval']:.2f}s/example"
        )


if __name__ == "__main__":
    main()
