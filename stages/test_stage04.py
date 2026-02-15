"""Stage 04: Batching (Inputs and Targets)

Learn to:
- Understand block_size (context length)
- Build get_batch() to create random training batches
- See how targets are inputs shifted by one position
"""
import torch
import pytest

from stages.stage01 import load_text, get_vocab
from stages.stage02 import Tokenizer
from stages.stage03 import encode_text, train_val_split
from stages.stage04 import get_batch


@pytest.fixture(scope="module")
def train_data():
    text = load_text()
    tokenizer = Tokenizer(get_vocab(text))
    data = encode_text(text, tokenizer)
    train, _ = train_val_split(data)
    return train


def test_batch_shapes(train_data):
    """get_batch() should return x and y with shape (batch_size, block_size)."""
    block_size = 8
    batch_size = 4
    x, y = get_batch(train_data, block_size=block_size, batch_size=batch_size)
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)


def test_batch_dtypes(train_data):
    """Both x and y should be long tensors."""
    x, y = get_batch(train_data, block_size=8, batch_size=4)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_targets_are_shifted_inputs(train_data):
    """For each sequence, y[i] should be x[i] shifted right by 1.

    That is: if x[i] = data[pos:pos+block_size],
    then y[i] = data[pos+1:pos+block_size+1].
    """
    torch.manual_seed(42)
    block_size = 8
    x, y = get_batch(train_data, block_size=block_size, batch_size=4)

    # For each batch element, y should be x shifted by 1
    # This means x[b, 1:] == y[b, :-1] for each batch b
    for b in range(4):
        assert torch.equal(x[b, 1:], y[b, :-1])


def test_batch_values_in_range(train_data):
    """All token values should be within valid range."""
    x, y = get_batch(train_data, block_size=8, batch_size=32)
    assert x.min() >= 0
    assert y.min() >= 0


def test_different_batches_are_random(train_data):
    """Two calls to get_batch() should (almost certainly) produce different data."""
    x1, _ = get_batch(train_data, block_size=8, batch_size=4)
    x2, _ = get_batch(train_data, block_size=8, batch_size=4)
    # It's astronomically unlikely that two random batches are identical
    assert not torch.equal(x1, x2)
