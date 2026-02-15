"""Stage 03: Tensor Data & Train/Val Split

Your task: implement the functions below so that all tests in test_stage03.py pass.

Run tests with:
    pytest stages/test_stage03.py -v

Concepts:
- Neural networks work with numbers (tensors), not strings
- We encode the entire text into one long tensor of integers
- We split this into training data (to learn from) and validation data (to check progress)
- 90/10 is a common split ratio

Hints:
- torch.tensor() converts a list of ints to a tensor
- Use dtype=torch.long for integer tensors
- Simple slicing works for the split: data[:n] and data[n:]
"""
import torch

from stages.stage02 import Tokenizer


def encode_text(text: str, tokenizer: Tokenizer) -> torch.Tensor:
    """Encode the full text as a 1D tensor of token IDs.

    Args:
        text: The full text string
        tokenizer: A Tokenizer instance to encode with

    Returns:
        A 1D torch.long tensor of token IDs
    """
    # TODO: Implement this
    raise NotImplementedError("Implement encode_text()")


def train_val_split(data: torch.Tensor, train_frac: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    """Split data into training and validation sets.

    Args:
        data: The full encoded data tensor
        train_frac: Fraction of data for training (default 0.9)

    Returns:
        A tuple of (train_data, val_data) tensors
    """
    # TODO: Implement this
    raise NotImplementedError("Implement train_val_split()")
