"""Load the base language model and its tokenizer."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import DEVICE, MODEL_NAME


def load_model_and_tokenizer(model_name: str = MODEL_NAME, device: str = DEVICE):
    """Load a causal LM in float32 on ``device`` and prepare its tokenizer.

    Padding goes on the left so that, in a batch of prompts with different
    lengths, the last position of every row is the real end of its prompt.
    Letter scoring reads the prediction at that last position.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.to(device)
    model.eval()
    return model, tokenizer
