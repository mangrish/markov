"""Stage 01: Data Loading & Exploration

Your task: implement the functions below so that all tests in test_stage01.py pass.

Run tests with:
    pytest stages/test_stage01.py -v

Concepts:
- The tiny Shakespeare dataset is a single text file (~1.1MB)
- We'll work with it at the character level
- The "vocabulary" is the set of all unique characters in the text

Hints:
- The dataset URL: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- Save it locally to data/input.txt so you only download once
- Python's sorted() and set() are your friends for vocabulary
"""
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_PATH = os.path.join(DATA_DIR, "input.txt")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text() -> str:
    """Load the tiny Shakespeare dataset.

    If the file doesn't exist locally at data/input.txt, download it first.
    Return the full text as a string.
    """
    # TODO: Implement this function
    # 1. Check if DATA_PATH exists
    # 2. If not, create the data/ directory and download from DATA_URL
    # 3. Read and return the file contents
    raise NotImplementedError("Implement load_text()")


def get_vocab(text: str) -> list[str]:
    """Extract the vocabulary (sorted list of unique characters) from the text.

    Args:
        text: The full text string

    Returns:
        A sorted list of unique characters
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement get_vocab()")
