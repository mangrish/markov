"""Stage 03: Tensor Data & Train/Val Split

Learn to:
- Encode the full text as a PyTorch tensor
- Split data into training and validation sets
"""
import torch
import pytest

from stages.stage01 import load_text, get_vocab
from stages.stage02 import Tokenizer
from stages.stage03 import encode_text, train_val_split


@pytest.fixture(scope="module")
def text():
    return load_text()


@pytest.fixture(scope="module")
def tokenizer(text):
    return Tokenizer(get_vocab(text))


@pytest.fixture(scope="module")
def data(text, tokenizer):
    return encode_text(text, tokenizer)


def test_encode_text_returns_tensor(data):
    """encode_text() should return a PyTorch tensor."""
    assert isinstance(data, torch.Tensor)


def test_tensor_dtype_is_long(data):
    """The tensor should have dtype torch.long (int64)."""
    assert data.dtype == torch.long


def test_tensor_is_1d(data):
    """The tensor should be 1-dimensional."""
    assert data.dim() == 1


def test_tensor_length_matches_text(text, data):
    """The tensor length should match the text length."""
    assert len(data) == len(text)


def test_tensor_values_in_range(tokenizer, data):
    """All values should be valid token IDs (0 to vocab_size-1)."""
    assert data.min() >= 0
    assert data.max() < tokenizer.vocab_size


def test_train_val_split_sizes(data):
    """90/10 split: train gets first 90%, val gets remaining 10%."""
    train, val = train_val_split(data)
    n = len(data)
    expected_train = int(0.9 * n)
    assert len(train) == expected_train
    assert len(val) == n - expected_train


def test_train_val_split_types(data):
    """Both splits should be tensors."""
    train, val = train_val_split(data)
    assert isinstance(train, torch.Tensor)
    assert isinstance(val, torch.Tensor)


def test_train_val_no_overlap(data):
    """Train and val together should reconstruct the original data."""
    train, val = train_val_split(data)
    reconstructed = torch.cat([train, val])
    assert torch.equal(reconstructed, data)
