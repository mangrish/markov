"""Stage 06: Training Loop

Your task: implement train() and estimate_loss() so that all tests in test_stage06.py pass.

Run tests with:
    pytest stages/test_stage06.py -v

Concepts:
- The optimizer (AdamW) updates model weights to minimize loss
- Each training step: get a batch, compute loss, backpropagate, update weights
- We track loss over time to verify the model is learning
- estimate_loss() averages loss over many batches for a stable estimate
  (a single batch's loss is noisy)

Hints:
- Training loop pattern:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        xb, yb = get_batch(data, ...)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
- For estimate_loss(), use torch.no_grad() context manager
  and model.eval() / model.train() to switch modes
"""
import torch

from stages.stage04 import get_batch
from stages.stage05 import BigramLanguageModel

# Hyperparameters — feel free to adjust
BLOCK_SIZE = 8
BATCH_SIZE = 32


def train(
    train_data: torch.Tensor,
    vocab_size: int,
    steps: int = 1000,
    lr: float = 1e-2,
) -> tuple[BigramLanguageModel, list[float]]:
    """Train a bigram model on the given data.

    Args:
        train_data: 1D tensor of training token IDs
        vocab_size: Size of the vocabulary
        steps: Number of training steps
        lr: Learning rate

    Returns:
        Tuple of (trained model, list of loss values per step)
    """
    # TODO: Implement training loop
    # 1. Create model
    # 2. Create AdamW optimizer
    # 3. For each step:
    #    a. Get a batch
    #    b. Forward pass (get loss)
    #    c. Zero gradients
    #    d. Backward pass
    #    e. Optimizer step
    #    f. Record loss
    # 4. Return (model, losses)
    raise NotImplementedError("Implement train()")


@torch.no_grad()
def estimate_loss(
    model: BigramLanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    eval_iters: int = 200,
) -> dict[str, float]:
    """Estimate average loss on train and val sets.

    Args:
        model: The trained model
        train_data: Training data tensor
        val_data: Validation data tensor
        eval_iters: Number of batches to average over

    Returns:
        Dict with 'train' and 'val' average losses
    """
    # TODO: Implement loss estimation
    # 1. Set model to eval mode
    # 2. For each split (train, val):
    #    a. Run eval_iters batches
    #    b. Average the losses
    # 3. Set model back to train mode
    # 4. Return {'train': avg_train_loss, 'val': avg_val_loss}
    raise NotImplementedError("Implement estimate_loss()")
