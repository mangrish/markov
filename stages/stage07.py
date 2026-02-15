"""Stage 07: Text Generation

Your task: implement generate() so that all tests in test_stage07.py pass.

Run tests with:
    pytest stages/test_stage07.py -v

Concepts:
- Autoregressive generation: predict next token, append it, repeat
- The model outputs logits → we convert to probabilities → we sample
- For a bigram model, only the LAST token matters for prediction
  (but we keep the full sequence for the output)

Hints:
- The generation loop:
    for _ in range(max_new_tokens):
        logits, _ = model(idx)          # get predictions
        logits = logits[:, -1, :]       # focus on last time step → (B, C)
        probs = F.softmax(logits, dim=-1)  # convert to probabilities
        next_token = torch.multinomial(probs, num_samples=1)  # sample
        idx = torch.cat([idx, next_token], dim=1)  # append
- torch.multinomial samples from a probability distribution
- F.softmax converts logits to probabilities (sum to 1)
"""
import torch
from torch.nn import functional as F

from stages.stage05 import BigramLanguageModel


def generate(
    model: BigramLanguageModel,
    idx: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Generate new tokens autoregressively.

    Args:
        model: A trained BigramLanguageModel
        idx: Starting context, shape (B, T) of token IDs
        max_new_tokens: Number of new tokens to generate

    Returns:
        Tensor of shape (B, T + max_new_tokens) with generated tokens appended
    """
    # TODO: Implement autoregressive generation
    # For each new token:
    # 1. Get logits from model (forward pass, no targets)
    # 2. Take logits for the last position only
    # 3. Apply softmax to get probabilities
    # 4. Sample next token from the distribution
    # 5. Append to the sequence
    raise NotImplementedError("Implement generate()")
