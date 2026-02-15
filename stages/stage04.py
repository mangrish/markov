"""Stage 04: Batching (Inputs and Targets)

Your task: implement get_batch() so that all tests in test_stage04.py pass.

Run tests with:
    pytest stages/test_stage04.py -v

Concepts:
- block_size is the maximum context length the model sees at once
- A batch is multiple independent sequences processed in parallel
- For language modeling, the target is always the next character:
    Input:  [H, e, l, l, o]
    Target: [e, l, l, o, !]
- We randomly sample starting positions in the data

Hints:
- Use torch.randint() to generate random starting positions
- For each position i, the input is data[i:i+block_size]
  and the target is data[i+1:i+block_size+1]
- torch.stack() combines a list of tensors into a batch
- Make sure starting positions don't go past the end of data
"""
import torch


def get_batch(
    data: torch.Tensor,
    block_size: int = 8,
    batch_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random batch of input/target pairs.

    Args:
        data: 1D tensor of token IDs
        block_size: Length of each sequence (context window)
        batch_size: Number of sequences in the batch

    Returns:
        Tuple of (inputs, targets), each with shape (batch_size, block_size)
    """
    # TODO: Implement this
    # 1. Generate `batch_size` random starting indices
    #    (each between 0 and len(data) - block_size - 1)
    # 2. For each index, slice out input and target sequences
    # 3. Stack into batches
    raise NotImplementedError("Implement get_batch()")
