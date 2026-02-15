"""Stage 06: Training Loop

Learn to:
- Set up an optimizer (AdamW)
- Run training steps and watch loss decrease
- Estimate loss on train and val sets
"""
import torch
import pytest

from stages.stage01 import load_text, get_vocab
from stages.stage02 import Tokenizer
from stages.stage03 import encode_text, train_val_split
from stages.stage04 import get_batch
from stages.stage05 import BigramLanguageModel
from stages.stage06 import train, estimate_loss


@pytest.fixture(scope="module")
def data():
    text = load_text()
    tokenizer = Tokenizer(get_vocab(text))
    encoded = encode_text(text, tokenizer)
    train_data, val_data = train_val_split(encoded)
    return train_data, val_data


@pytest.fixture(scope="module")
def vocab_size():
    return 65


def test_train_returns_model_and_losses(data, vocab_size):
    """train() should return a trained model and a list of losses."""
    train_data, _ = data
    model, losses = train(train_data, vocab_size, steps=10, lr=1e-2)
    assert isinstance(model, BigramLanguageModel)
    assert isinstance(losses, list)
    assert len(losses) > 0


def test_loss_decreases(data, vocab_size):
    """Loss should decrease over training steps."""
    train_data, _ = data
    torch.manual_seed(42)
    model, losses = train(train_data, vocab_size, steps=200, lr=1e-2)
    # Compare average of first 10 losses vs last 10 losses
    early_loss = sum(losses[:10]) / 10
    late_loss = sum(losses[-10:]) / 10
    assert late_loss < early_loss, (
        f"Loss did not decrease: early={early_loss:.3f}, late={late_loss:.3f}"
    )


def test_estimate_loss(data, vocab_size):
    """estimate_loss() should return average losses for train and val."""
    train_data, val_data = data
    torch.manual_seed(42)
    model, _ = train(train_data, vocab_size, steps=100, lr=1e-2)
    losses = estimate_loss(model, train_data, val_data, eval_iters=10)
    assert "train" in losses
    assert "val" in losses
    assert isinstance(losses["train"], float)
    assert isinstance(losses["val"], float)
    # Losses should be positive
    assert losses["train"] > 0
    assert losses["val"] > 0
