"""Stage 02: Character-Level Tokenization

Your task: implement the Tokenizer class so that all tests in test_stage02.py pass.

Run tests with:
    pytest stages/test_stage02.py -v

Concepts:
- A tokenizer converts between text and numbers
- At the character level, each unique character gets a unique integer
- We use the sorted vocabulary to assign indices (a=0, b=1, etc.)
- This is the simplest possible tokenizer — real LLMs use subword tokenization
  (like BPE), but the principle is the same

Hints:
- You need two lookup dictionaries: char→int and int→char
- Python dict comprehensions with enumerate() work well here
"""


class Tokenizer:
    """A character-level tokenizer.

    Given a vocabulary (sorted list of unique characters), builds
    mappings to convert between characters and integers.
    """

    def __init__(self, vocab: list[str]):
        """Initialize the tokenizer with a vocabulary.

        Args:
            vocab: A sorted list of unique characters
        """
        self.vocab_size = len(vocab)
        # TODO: Build your mappings
        # self.stoi = ???  # string to integer: {'a': 0, 'b': 1, ...}
        # self.itos = ???  # integer to string: {0: 'a', 1: 'b', ...}
        raise NotImplementedError("Implement __init__()")

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of integers.

        Args:
            text: The input string

        Returns:
            A list of integers, one per character
        """
        # TODO: Implement this
        raise NotImplementedError("Implement encode()")

    def decode(self, tokens: list[int]) -> str:
        """Convert a list of integers back to a string.

        Args:
            tokens: A list of integer token IDs

        Returns:
            The decoded string
        """
        # TODO: Implement this
        raise NotImplementedError("Implement decode()")
