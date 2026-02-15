# Build a Shakespeare Bigram Model from Scratch

A hands-on, test-driven walkthrough of building a character-level language model. Follow along with Andrej Karpathy's ["Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video, one concept at a time.

You'll start with loading text data and finish with a model that generates Shakespeare-like text — all guided by pytest tests that tell you exactly what to implement next.

## Prerequisites

- Python 3.10+
- Basic familiarity with Python and PyTorch (or willingness to learn)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How It Works

Each stage introduces one concept. You run the tests, read the failing output, then write code to make them pass.

```bash
# Run the tests for stage 01
pytest stages/test_stage01.py -v

# Open the implementation file and fill in the TODOs
# Re-run until all tests pass, then move to the next stage
```

## Stages

| # | Topic | What You'll Learn |
|---|-------|-------------------|
| 01 | Data Loading | Download tiny Shakespeare, explore the text, extract vocabulary |
| 02 | Tokenization | Build encode/decode functions (characters to integers and back) |
| 03 | Tensors & Splitting | Convert text to a PyTorch tensor, split into train/val sets |
| 04 | Batching | Understand block size, build random batch sampling |
| 05 | Bigram Model | Create a simple neural net with an embedding table |
| 06 | Training Loop | Set up AdamW optimizer, train the model, watch loss decrease |
| 07 | Text Generation | Implement autoregressive sampling to generate new text |

## What You'll Build

A **bigram language model** — the simplest possible neural language model. It predicts the next character using only the current character. It's not GPT, but it introduces the same core ideas:

- Tokenization (text → numbers)
- Embedding tables
- Cross-entropy loss
- Autoregressive generation

These are the building blocks that scale up into modern LLMs.

## Acknowledgments

Based on Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) and [nanoGPT](https://github.com/karpathy/nanoGPT) projects.
