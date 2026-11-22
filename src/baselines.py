"""Random baseline: the lowest score a method must beat to show it learned anything."""

from __future__ import annotations

import random


def random_predictions(n_examples: int, n_options: int, seed: int) -> list[int]:
    """Pick an answer index uniformly at random for each example.

    The fixed seed makes the predictions identical on every run. Expected
    accuracy is 1 / n_options: 50% for Subtask A and 33% for Subtask B.
    """
    rng = random.Random(seed)
    return [rng.randrange(n_options) for _ in range(n_examples)]
