"""Low-Rank Adaptation (LoRA), written from scratch in PyTorch.

LoRA (Hu et al., 2021) fine-tunes a large model by freezing all of its
weights and adding a small trainable update beside chosen linear layers.
A frozen layer computes ``W x``. With LoRA it computes

    W x + (alpha / r) * B A x

where ``A`` has shape (r, in_features) and ``B`` has shape (out_features, r).
The rank ``r`` is small (for example 8), so ``A`` and ``B`` hold far fewer
numbers than ``W``. ``B`` starts at zero, so before training the model
behaves exactly like the original.

The project's virtual environment does not include the ``peft`` library,
so this module implements the method directly.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        weight = base.weight
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features, dtype=weight.dtype, device=weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, dtype=weight.dtype, device=weight.device)
        )
        # Same initialisation as the LoRA paper: random A, zero B.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return self.base(x) + update * self.scaling

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling:g}"


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    """Freeze ``model`` and wrap every linear layer named in ``target_modules``.

    ``target_modules`` holds layer names such as ``"q_proj"``. Returns the
    full names of the wrapped layers. Raises an error if none match.
    """
    targets = set(target_modules)
    for param in model.parameters():
        param.requires_grad = False

    # Collect first, then replace, so the module tree is not changed while
    # we walk over it.
    to_wrap = [
        (parent, child_name, f"{parent_name}.{child_name}" if parent_name else child_name)
        for parent_name, parent in model.named_modules()
        for child_name, child in parent.named_children()
        if child_name in targets and isinstance(child, nn.Linear)
    ]
    if not to_wrap:
        raise ValueError(f"No linear layers named {sorted(targets)} found in the model.")

    for parent, child_name, _ in to_wrap:
        setattr(parent, child_name, LoRALinear(getattr(parent, child_name), rank, alpha, dropout))
    return [full_name for _, _, full_name in to_wrap]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the LoRA weights: a few MB instead of the whole model."""
    return {name: tensor for name, tensor in model.state_dict().items() if "lora_" in name}
