"""Stage 05: Bigram Language Model

Your task: implement BigramLanguageModel so that all tests in test_stage05.py pass.

Run tests with:
    pytest stages/test_stage05.py -v

Concepts:
- A bigram model predicts the next character using ONLY the current character
- The entire model is just an embedding table: token → logits for next token
- nn.Embedding(vocab_size, vocab_size) gives us a (vocab_size x vocab_size) table
- Row i contains the logits (unnormalized log-probabilities) for what comes after token i
- Cross-entropy loss measures how well our predictions match the actual next tokens

Hints:
- The model takes input idx of shape (B, T) — batch of sequences of token IDs
- Look up each token in the embedding table to get logits of shape (B, T, C)
  where C = vocab_size
- For loss computation, PyTorch's F.cross_entropy expects:
    input:  (B*T, C) — predicted logits
    target: (B*T,)   — actual token IDs
  So you'll need to reshape before computing loss
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """A simple bigram language model.

    Uses an embedding table where each token directly predicts
    the distribution over the next token.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        # TODO: Create the embedding table
        # self.token_embedding_table = nn.Embedding(???, ???)
        raise NotImplementedError("Implement __init__()")

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the bigram model.

        Args:
            idx: Input token IDs, shape (B, T)
            targets: Target token IDs, shape (B, T), or None

        Returns:
            Tuple of (logits, loss):
            - logits: shape (B, T, vocab_size)
            - loss: scalar cross-entropy loss, or None if no targets
        """
        # TODO: Implement the forward pass
        # 1. Look up token embeddings → logits (B, T, C)
        # 2. If targets provided, compute cross-entropy loss
        #    (reshape logits to (B*T, C) and targets to (B*T,))
        # 3. Return (logits, loss)
        raise NotImplementedError("Implement forward()")
