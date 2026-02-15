"""Stage 07: Text Generation

Learn to:
- Implement autoregressive text generation
- Sample from the model's predicted distribution
- Generate Shakespeare-like text
"""
import torch
import pytest

from stages.stage01 import load_text, get_vocab
from stages.stage02 import Tokenizer
from stages.stage03 import encode_text, train_val_split
from stages.stage06 import train
from stages.stage07 import generate


@pytest.fixture(scope="module")
def trained_setup():
    """Train a model and return (model, tokenizer) for generation tests."""
    text = load_text()
    vocab = get_vocab(text)
    tokenizer = Tokenizer(vocab)
    data = encode_text(text, tokenizer)
    train_data, _ = train_val_split(data)
    torch.manual_seed(42)
    model, _ = train(train_data, vocab_size=len(vocab), steps=500, lr=1e-2)
    return model, tokenizer


def test_generate_returns_tensor(trained_setup):
    """generate() should return a tensor of token IDs."""
    model, _ = trained_setup
    # Start with a single newline token (index 0)
    context = torch.zeros((1, 1), dtype=torch.long)
    output = generate(model, context, max_new_tokens=10)
    assert isinstance(output, torch.Tensor)


def test_generate_output_length(trained_setup):
    """Output should have context length + max_new_tokens columns."""
    model, _ = trained_setup
    context = torch.zeros((1, 1), dtype=torch.long)
    max_new = 20
    output = generate(model, context, max_new_tokens=max_new)
    assert output.shape == (1, 1 + max_new)


def test_generate_preserves_context(trained_setup):
    """The output should start with the original context."""
    model, _ = trained_setup
    context = torch.zeros((1, 1), dtype=torch.long)
    output = generate(model, context, max_new_tokens=10)
    assert output[0, 0].item() == 0  # original context preserved


def test_generate_valid_tokens(trained_setup):
    """All generated tokens should be valid (within vocab range)."""
    model, tokenizer = trained_setup
    context = torch.zeros((1, 1), dtype=torch.long)
    output = generate(model, context, max_new_tokens=100)
    assert output.min() >= 0
    assert output.max() < tokenizer.vocab_size


def test_generate_decodable_text(trained_setup):
    """Generated tokens should decode to a valid string."""
    model, tokenizer = trained_setup
    context = torch.zeros((1, 1), dtype=torch.long)
    output = generate(model, context, max_new_tokens=50)
    tokens = output[0].tolist()
    text = tokenizer.decode(tokens)
    assert isinstance(text, str)
    assert len(text) == 51  # 1 context + 50 new


def test_generate_is_stochastic(trained_setup):
    """Two generations should (almost certainly) differ."""
    model, _ = trained_setup
    context = torch.zeros((1, 1), dtype=torch.long)
    out1 = generate(model, context, max_new_tokens=50)
    out2 = generate(model, context, max_new_tokens=50)
    assert not torch.equal(out1, out2)
