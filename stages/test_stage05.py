"""Stage 05: Bigram Language Model

Learn to:
- Build a simple neural network for language modeling
- Use an embedding table as the model's only layer
- Compute cross-entropy loss
"""
import torch
import torch.nn as nn
import pytest

from stages.stage05 import BigramLanguageModel


@pytest.fixture(scope="module")
def model():
    return BigramLanguageModel(vocab_size=65)


def test_model_is_nn_module(model):
    """BigramLanguageModel should be a PyTorch nn.Module."""
    assert isinstance(model, nn.Module)


def test_model_has_embedding(model):
    """The model should have an embedding table."""
    has_embedding = any(
        isinstance(m, nn.Embedding) for m in model.modules()
    )
    assert has_embedding, "Model should contain an nn.Embedding layer"


def test_forward_logits_shape(model):
    """Forward pass should return logits with shape (B, T, vocab_size)."""
    x = torch.zeros(4, 8, dtype=torch.long)  # batch=4, time=8
    logits, loss = model(x)
    assert logits.shape == (4, 8, 65)


def test_forward_without_targets_no_loss(model):
    """Without targets, loss should be None."""
    x = torch.zeros(4, 8, dtype=torch.long)
    logits, loss = model(x)
    assert loss is None


def test_forward_with_targets_has_loss(model):
    """With targets, loss should be a scalar tensor."""
    x = torch.zeros(4, 8, dtype=torch.long)
    y = torch.zeros(4, 8, dtype=torch.long)
    logits, loss = model(x, y)
    assert loss is not None
    assert loss.dim() == 0  # scalar


def test_loss_is_reasonable_at_init(model):
    """At random init, loss should be near -ln(1/65) ≈ 4.17."""
    torch.manual_seed(42)
    x = torch.randint(0, 65, (32, 8))
    y = torch.randint(0, 65, (32, 8))
    _, loss = model(x, y)
    # Should be roughly 4.17, allow some variance
    assert 3.5 < loss.item() < 5.0, f"Loss {loss.item()} seems unreasonable for random init"


def test_logits_require_grad(model):
    """Logits should be differentiable (part of computation graph)."""
    x = torch.zeros(4, 8, dtype=torch.long)
    y = torch.zeros(4, 8, dtype=torch.long)
    _, loss = model(x, y)
    loss.backward()
    # If we got here without error, gradients flow correctly
