# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Description

A small language model for learning about how LLMs work. Following Karpathy's "Let's build GPT" video, building a Shakespeare bigram model step by step.

## How to Work Through the Stages

Each stage in `stages/` introduces one concept with tests to guide your implementation.

### Setup

```bash
pip install -r requirements.txt
```

### Workflow

1. Run the tests for the current stage:
   ```bash
   pytest stages/test_stage01.py -v
   ```
2. Open the corresponding implementation file (`stages/stage01.py`) and fill in the TODOs
3. Re-run tests until they all pass
4. Move to the next stage

### Stages

| Stage | Topic | Files |
|-------|-------|-------|
| 01 | Data Loading & Exploration | `stage01.py`, `test_stage01.py` |
| 02 | Character-Level Tokenization | `stage02.py`, `test_stage02.py` |
| 03 | Tensor Data & Train/Val Split | `stage03.py`, `test_stage03.py` |
| 04 | Batching (Inputs & Targets) | `stage04.py`, `test_stage04.py` |
| 05 | Bigram Language Model | `stage05.py`, `test_stage05.py` |
| 06 | Training Loop | `stage06.py`, `test_stage06.py` |
| 07 | Text Generation | `stage07.py`, `test_stage07.py` |

### Running All Tests

```bash
pytest stages/ -v
```
