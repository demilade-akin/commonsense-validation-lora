"""Load the ComVE dataset (SemEval-2020 Task 4) for Subtasks A and B.

The official release stores each split as two CSV files: one with the
questions (with a header row) and one with the answers (no header row).
The loaders join the two by ``id`` and return a list of small dataclasses
that every method in the project shares.

Subtask A: two similar statements; the label is the index (0 or 1) of the
statement that does not make sense.

Subtask B: the nonsensical statement and three candidate reasons; the
label is the index (0, 1 or 2) of the reason that explains why.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeVar

from src.config import DATA_DIR, EXPECTED_COUNTS, SEED

# Letters used for answer options. Subtask B answer files store these.
LETTERS = ("A", "B", "C")

SPLITS = ("train", "dev", "test")

# File names differ between splits in the official release.
_FILES = {
    "A": {
        "train": ("subtaskA_data_all.csv", "subtaskA_answers_all.csv"),
        "dev": ("subtaskA_dev_data.csv", "subtaskA_gold_answers.csv"),
        "test": ("subtaskA_test_data.csv", "subtaskA_gold_answers.csv"),
    },
    "B": {
        "train": ("subtaskB_data_all.csv", "subtaskB_answers_all.csv"),
        "dev": ("subtaskB_dev_data.csv", "subtaskB_gold_answers.csv"),
        "test": ("subtaskB_test_data.csv", "subtaskB_gold_answers.csv"),
    },
}


# ---------------------------------------------------------------------------
# Example types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SubtaskAExample:
    """One validation question: which statement does not make sense?"""

    id: str
    sent0: str
    sent1: str
    label: int  # 0 or 1: index of the nonsensical statement

    @property
    def options(self) -> tuple[str, str]:
        return (self.sent0, self.sent1)


@dataclass(frozen=True)
class SubtaskBExample:
    """One explanation question: which reason explains the false statement?"""

    id: str
    false_sent: str
    options: tuple[str, str, str]  # reasons A, B and C
    label: int  # 0, 1 or 2: index of the correct reason


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _paths(subtask: str, split: str, data_dir: Path) -> tuple[Path, Path]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}. Use one of {SPLITS}.")
    data_file, answer_file = _FILES[subtask][split]
    split_dir = data_dir / split
    return split_dir / data_file, split_dir / answer_file


def _read_questions(path: Path) -> list[dict[str, str]]:
    """Read a question file (has a header row) into a list of dicts."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_answers(path: Path) -> dict[str, str]:
    """Read an answer file (no header row: id, label) into an id -> label map."""
    with open(path, encoding="utf-8", newline="") as f:
        return {row[0].strip(): row[1].strip() for row in csv.reader(f) if row}


def _check_join(rows: list[dict[str, str]], answers: dict[str, str], path: Path) -> None:
    """Fail if any question lacks an answer or any answer lacks a question."""
    question_ids = {row["id"].strip() for row in rows}
    missing = question_ids - answers.keys()
    extra = answers.keys() - question_ids
    if missing or extra:
        raise ValueError(
            f"{path.name}: {len(missing)} questions without answers, "
            f"{len(extra)} answers without questions."
        )


def _check_count(examples: Sequence, split: str, subtask: str) -> None:
    expected = EXPECTED_COUNTS[split]
    if len(examples) != expected:
        raise ValueError(
            f"Subtask {subtask} {split}: expected {expected} examples, "
            f"found {len(examples)}."
        )


def load_subtask_a(split: str, data_dir: Path = DATA_DIR) -> list[SubtaskAExample]:
    """Load Subtask A (validation) for one split: 'train', 'dev' or 'test'."""
    data_path, answer_path = _paths("A", split, data_dir)
    rows = _read_questions(data_path)
    answers = _read_answers(answer_path)
    _check_join(rows, answers, data_path)

    examples = [
        SubtaskAExample(
            id=row["id"].strip(),
            sent0=row["sent0"].strip(),
            sent1=row["sent1"].strip(),
            label=int(answers[row["id"].strip()]),
        )
        for row in rows
    ]
    _check_count(examples, split, "A")
    return examples


def load_subtask_b(split: str, data_dir: Path = DATA_DIR) -> list[SubtaskBExample]:
    """Load Subtask B (explanation) for one split: 'train', 'dev' or 'test'."""
    data_path, answer_path = _paths("B", split, data_dir)
    rows = _read_questions(data_path)
    answers = _read_answers(answer_path)
    _check_join(rows, answers, data_path)

    examples = [
        SubtaskBExample(
            id=row["id"].strip(),
            false_sent=row["FalseSent"].strip(),
            options=(
                row["OptionA"].strip(),
                row["OptionB"].strip(),
                row["OptionC"].strip(),
            ),
            label=LETTERS.index(answers[row["id"].strip()]),
        )
        for row in rows
    ]
    _check_count(examples, split, "B")
    return examples


def load_subtask(subtask: str, split: str, data_dir: Path = DATA_DIR) -> list:
    """Load subtask ``'A'`` or ``'B'`` for one split."""
    if subtask == "A":
        return load_subtask_a(split, data_dir)
    if subtask == "B":
        return load_subtask_b(split, data_dir)
    raise ValueError(f"Unknown subtask {subtask!r}. Use 'A' or 'B'.")


# Number of answer options per subtask.
N_OPTIONS = {"A": 2, "B": 3}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
T = TypeVar("T")


def downsample(examples: Sequence[T], n: int | None, seed: int = SEED) -> list[T]:
    """Return ``n`` examples picked at random with a fixed seed.

    The same seed always gives the same subset. ``n=None`` or an ``n`` at
    least as large as the input returns every example, in original order.
    """
    if n is None or n >= len(examples):
        return list(examples)
    return random.Random(seed).sample(list(examples), n)


# ---------------------------------------------------------------------------
# Quick check: python -m src.data
# ---------------------------------------------------------------------------
def _summary() -> None:
    for split in SPLITS:
        a = load_subtask_a(split)
        b = load_subtask_b(split)
        a_labels = Counter(ex.label for ex in a)
        b_labels = Counter(LETTERS[ex.label] for ex in b)
        print(
            f"{split:<5}  A: {len(a):>6} examples, labels {dict(sorted(a_labels.items()))}"
            f"   B: {len(b):>6} examples, labels {dict(sorted(b_labels.items()))}"
        )
    print("\nSubtask A example:", load_subtask_a("dev")[0])
    print("Subtask B example:", load_subtask_b("dev")[0])


if __name__ == "__main__":
    _summary()
