"""Stage 01: Data Loading & Exploration

Learn to:
- Download the tiny Shakespeare dataset
- Read the text file
- Explore its contents (length, unique characters)
"""
import os
import pytest

from stages.stage01 import load_text, get_vocab


@pytest.fixture(scope="module")
def text():
    return load_text()


def test_load_text_returns_string(text):
    """load_text() should return the full Shakespeare text as a string."""
    assert isinstance(text, str)


def test_text_length(text):
    """The tiny Shakespeare dataset is 1,115,394 characters long."""
    assert len(text) == 1115394


def test_text_starts_with_first_citizen(text):
    """The text starts with 'First Citizen'."""
    assert text.startswith("First Citizen")


def test_get_vocab_returns_sorted_list(text):
    """get_vocab() should return a sorted list of unique characters."""
    vocab = get_vocab(text)
    assert isinstance(vocab, list)
    assert vocab == sorted(vocab)


def test_vocab_size(text):
    """There are 65 unique characters in the dataset."""
    vocab = get_vocab(text)
    assert len(vocab) == 65


def test_vocab_contains_newline(text):
    """The vocabulary includes the newline character."""
    vocab = get_vocab(text)
    assert "\n" in vocab


def test_vocab_contains_letters(text):
    """The vocabulary includes uppercase and lowercase letters."""
    vocab = get_vocab(text)
    assert "a" in vocab
    assert "Z" in vocab
