"""
Tokenizer wrapper for mlxchat

Reuses nanochat's trained tokenizer (RustBPE + tiktoken).
Returns MLX arrays for compatibility with the rest of the codebase.
"""

import os
import pickle
from functools import lru_cache

import mlx.core as mx


# Special tokens from nanochat
SPECIAL_TOKENS = [
    "<|bos|>",  # Beginning of Sequence - delimits documents
    "<|user_start|>",  # User messages
    "<|user_end|>",
    "<|assistant_start|>",  # Assistant messages
    "<|assistant_end|>",
    "<|python_start|>",  # Python REPL tool calls
    "<|python_end|>",
    "<|output_start|>",  # Python REPL outputs
    "<|output_end|>",
]


class Tokenizer:
    """
    Wrapper around nanochat's RustBPE tokenizer for MLX.

    This tokenizer reuses the trained tokenizer from nanochat,
    which uses tiktoken for efficient inference.
    """

    def __init__(self, enc, bos_token="<|bos|>"):
        """
        Initialize tokenizer with a tiktoken encoding object.

        Args:
            enc: tiktoken.Encoding object
            bos_token: Beginning of sequence token (default: "<|bos|>")
        """
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        """
        Load tokenizer from a directory containing tokenizer.pkl.

        Args:
            tokenizer_dir: Path to directory with tokenizer.pkl

        Returns:
            Tokenizer instance
        """
        import tiktoken

        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(
                f"Tokenizer not found at {pickle_path}. "
                f"Please train a tokenizer using nanochat first."
            )

        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)

        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        """
        Load a pretrained tiktoken tokenizer (e.g., "gpt2", "cl100k_base").

        Args:
            tiktoken_name: Name of the tiktoken encoding

        Returns:
            Tokenizer instance
        """
        import tiktoken

        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken uses "<|endoftext|>" for the document delimiter
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        """Get the vocabulary size."""
        return self.enc.n_vocab

    def get_special_tokens(self):
        """Get the set of special tokens."""
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        """Decode a single token ID to its string representation."""
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        """Encode a special token by exact match."""
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        """Get the beginning of sequence token ID."""
        return self.bos_token_id

    def get_eot_token_id(self):
        """Get the end of text token ID (same as BOS for document boundaries)."""
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, return_array=True):
        """
        Encode text to token IDs.

        Args:
            text: String or list of strings to encode
            prepend: Optional token or token ID to prepend
            append: Optional token or token ID to append
            return_array: If True, return MLX array; if False, return list

        Returns:
            Token IDs as MLX array (if return_array=True) or list
        """
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            # Single string
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)

            if return_array:
                return mx.array(ids, dtype=mx.int32)
            return ids

        elif isinstance(text, list):
            # List of strings
            ids_list = self.enc.encode_ordinary_batch(text, num_threads=8)
            if prepend is not None:
                for ids_row in ids_list:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids_list:
                    ids_row.append(append_id)

            if return_array:
                # Convert to list of MLX arrays
                return [mx.array(ids, dtype=mx.int32) for ids in ids_list]
            return ids_list

        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        """Shorthand for encode()."""
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs or MLX array

        Returns:
            Decoded text string
        """
        # Convert MLX array to Python list if needed
        if isinstance(ids, mx.array):
            ids = ids.tolist()

        return self.enc.decode(ids)


def get_tokenizer(tokenizer_dir=None):
    """
    Get the mlxchat tokenizer.

    By default, looks for nanochat's tokenizer in ~/.cache/nanochat/tokenizer.
    If not found, falls back to tiktoken's GPT-2 tokenizer (vocab_size=50257).
    You can override this by providing a custom tokenizer_dir.

    Args:
        tokenizer_dir: Optional custom path to tokenizer directory

    Returns:
        Tokenizer instance
    """
    if tokenizer_dir is None:
        # Use nanochat's tokenizer by default
        home = os.path.expanduser("~")
        tokenizer_dir = os.path.join(home, ".cache", "nanochat", "tokenizer")

    # Try to load nanochat's trained tokenizer
    pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    if os.path.exists(pickle_path):
        return Tokenizer.from_directory(tokenizer_dir)

    # Fallback to tiktoken's GPT-2 tokenizer
    print(f"Warning: Nanochat tokenizer not found at {pickle_path}")
    print(f"Falling back to tiktoken's GPT-2 tokenizer (vocab_size=50257)")
    print(f"Note: This is fine for testing, but for production training you should")
    print(f"train a custom tokenizer with nanochat's tok_train.py script.")
    return Tokenizer.from_pretrained("gpt2")
