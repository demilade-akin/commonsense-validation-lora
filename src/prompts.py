"""Turn ComVE examples into chat messages for the language model.

Each question lists its options with letters (A/B, or A/B/C) and asks for
a letter only. Few-shot prompts put solved examples first as earlier chat
turns: the user asks a question and the assistant answers with a letter.
"""

from __future__ import annotations

import random
from typing import Sequence, Union

from src.data import LETTERS, SubtaskAExample, SubtaskBExample

Example = Union[SubtaskAExample, SubtaskBExample]


def format_question(example: Example) -> str:
    """Build the question text for one example of either subtask."""
    if isinstance(example, SubtaskAExample):
        return (
            "Which of these two statements does not make sense?\n"
            f"A: {example.sent0}\n"
            f"B: {example.sent1}\n"
            "Answer with A or B only."
        )
    if isinstance(example, SubtaskBExample):
        return (
            "This statement does not make sense:\n"
            f'"{example.false_sent}"\n'
            "Which reason best explains why?\n"
            f"A: {example.options[0]}\n"
            f"B: {example.options[1]}\n"
            f"C: {example.options[2]}\n"
            "Answer with A, B or C only."
        )
    raise TypeError(f"Unsupported example type: {type(example).__name__}")


def build_messages(example: Example, shots: Sequence[Example] = ()) -> list[dict]:
    """Chat messages for one question, preceded by any solved examples."""
    messages: list[dict] = []
    for shot in shots:
        messages.append({"role": "user", "content": format_question(shot)})
        messages.append({"role": "assistant", "content": LETTERS[shot.label]})
    messages.append({"role": "user", "content": format_question(example)})
    return messages


def pick_shots(train_examples: Sequence[Example], k: int, seed: int) -> list[Example]:
    """Pick ``k`` training examples whose answers cover the letters evenly.

    The same seed always returns the same examples. Mixing answer letters
    stops the prompt from teaching the model to repeat one letter.
    """
    if k <= 0:
        return []
    shuffled = list(train_examples)
    random.Random(seed).shuffle(shuffled)

    n_options = len(shuffled[0].options)
    by_label: dict[int, list[Example]] = {label: [] for label in range(n_options)}
    for ex in shuffled:
        by_label[ex.label].append(ex)

    shots: list[Example] = []
    label = 0
    while len(shots) < k:
        if by_label[label]:
            shots.append(by_label[label].pop())
        label = (label + 1) % n_options
    return shots
