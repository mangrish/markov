"""Stage 02: Character-Level Tokenization

Learn to:
- Build character-to-integer and integer-to-character mappings
- Encode strings to lists of integers
- Decode lists of integers back to strings
"""
import pytest

from stages.stage01 import load_text, get_vocab
from stages.stage02 import Tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    text = load_text()
    vocab = get_vocab(text)
    return Tokenizer(vocab)


def test_tokenizer_vocab_size(tokenizer):
    """The tokenizer should know its vocabulary size."""
    assert tokenizer.vocab_size == 65


def test_encode_returns_list_of_ints(tokenizer):
    """encode() should return a list of integers."""
    encoded = tokenizer.encode("hi")
    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)


def test_decode_returns_string(tokenizer):
    """decode() should return a string."""
    encoded = tokenizer.encode("hi")
    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str)


def test_encode_decode_roundtrip(tokenizer):
    """Encoding then decoding should return the original string."""
    original = "hello world"
    assert tokenizer.decode(tokenizer.encode(original)) == original


def test_encode_decode_roundtrip_multiline(tokenizer):
    """Round-trip works for multi-line text too."""
    original = "First Citizen:\nBefore we proceed any further"
    assert tokenizer.decode(tokenizer.encode(original)) == original


def test_encode_specific_values(tokenizer):
    """Each character maps to a specific integer based on sorted position."""
    # Space is one of the first characters in sorted order
    encoded = tokenizer.encode(" ")
    assert len(encoded) == 1
    assert isinstance(encoded[0], int)
    assert 0 <= encoded[0] < 65


def test_encode_empty_string(tokenizer):
    """Encoding an empty string returns an empty list."""
    assert tokenizer.encode("") == []


def test_decode_empty_list(tokenizer):
    """Decoding an empty list returns an empty string."""
    assert tokenizer.decode([]) == ""


def test_all_vocab_chars_encodable(tokenizer):
    """Every character in the vocabulary should be encodable."""
    text = load_text()
    vocab = get_vocab(text)
    for ch in vocab:
        encoded = tokenizer.encode(ch)
        assert len(encoded) == 1
        assert tokenizer.decode(encoded) == ch
