"""Small helpers shared by every stage: seeding, timing, metrics, saving."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Timer:
    """Context manager that measures wall-clock time.

    Example:
        with Timer("zero-shot A") as t:
            ...
        print(t.seconds)
    """

    def __init__(self, name: str = "", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.seconds = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._start
        if self.verbose:
            label = f"[{self.name}] " if self.name else ""
            print(f"{label}{self.seconds:.1f}s")


def accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    """Fraction of predictions that match the gold labels."""
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels have different lengths")
    if not labels:
        return 0.0
    return sum(p == y for p, y in zip(predictions, labels)) / len(labels)


def save_json(obj: Any, path: Path) -> None:
    """Write ``obj`` as pretty-printed JSON, creating folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
