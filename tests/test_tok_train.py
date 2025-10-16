"""
Tests for tokenizer training script.
"""

import os
import tempfile
import shutil
import pickle
import pytest
import mlx.core as mx

# Check if rustbpe is available
try:
    import rustbpe

    RUSTBPE_AVAILABLE = True
except ImportError:
    RUSTBPE_AVAILABLE = False


@pytest.mark.skipif(not RUSTBPE_AVAILABLE, reason="rustbpe not installed")
def test_train_small_tokenizer():
    """Test training a small tokenizer on minimal data."""

    # Create a temporary directory for the tokenizer
    tmpdir = tempfile.mkdtemp()

    try:
        # Small training corpus
        training_texts = [
            "Hello world! This is a test.",
            "Machine learning is awesome.",
            "Python is a great programming language.",
            "MLX makes deep learning fast on Apple Silicon.",
            "Tokenizers split text into smaller units.",
            "BPE stands for Byte Pair Encoding.",
            "Special tokens like <|bos|> are important.",
            "Numbers: 123, 456, 789",
            "Contractions: I'm, you're, it's, we're",
            "Unicode: 你好世界 🌍 🚀",
        ] * 100  # Repeat to get enough data

        # Define special tokens
        special_tokens = [
            "<|bos|>",
            "<|user_start|>",
            "<|user_end|>",
            "<|assistant_start|>",
            "<|assistant_end|>",
        ]

        # Train tokenizer
        vocab_size = 1000  # Small vocab for testing
        vocab_size_no_special = vocab_size - len(special_tokens)

        # Pattern from nanochat (GPT-4 style)
        SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

        tokenizer_trainer = rustbpe.Tokenizer()
        tokenizer_trainer.train_from_iterator(iter(training_texts), vocab_size_no_special, pattern=SPLIT_PATTERN)

        # Construct tiktoken encoding for efficient inference
        import tiktoken

        pattern = tokenizer_trainer.get_pattern()
        mergeable_ranks_list = tokenizer_trainer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens_dict = {name: tokens_offset + i for i, name in enumerate(special_tokens)}

        tokenizer = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens_dict,
        )

        # Save tokenizer
        tokenizer_path = os.path.join(tmpdir, "tokenizer.pkl")
        with open(tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)

        # Verify file was created
        assert os.path.exists(tokenizer_path)

        # Load tokenizer back
        with open(tokenizer_path, "rb") as f:
            loaded_tokenizer = pickle.load(f)

        # Test encoding/decoding
        test_text = "Hello world! Testing tokenizer."
        encoded = loaded_tokenizer.encode_ordinary(test_text)
        decoded = loaded_tokenizer.decode(encoded)

        assert decoded == test_text
        assert len(encoded) > 0

        # Test special tokens
        bos_id = loaded_tokenizer.encode_single_token("<|bos|>")
        assert bos_id is not None
        assert isinstance(bos_id, int)

        # Test vocab size (should be close to requested size, but not necessarily exact)
        # rustbpe may return slightly fewer tokens than requested
        assert loaded_tokenizer.n_vocab >= 256  # At least the base 256 bytes
        assert loaded_tokenizer.n_vocab <= vocab_size  # Not more than requested

        print("✓ Tokenizer trained successfully")
        print(f"  Vocab size: {loaded_tokenizer.n_vocab}")
        print(f"  Test encoding length: {len(encoded)}")

    finally:
        # Cleanup
        shutil.rmtree(tmpdir)


@pytest.mark.skipif(not RUSTBPE_AVAILABLE, reason="rustbpe not installed")
def test_tokenizer_special_tokens():
    """Test that special tokens are handled correctly."""
    import tiktoken

    training_texts = ["Hello world!"] * 100
    special_tokens = ["<|test|>", "<|special|>"]

    SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    vocab_size = 500
    vocab_size_no_special = vocab_size - len(special_tokens)

    tokenizer_trainer = rustbpe.Tokenizer()
    tokenizer_trainer.train_from_iterator(iter(training_texts), vocab_size_no_special, pattern=SPLIT_PATTERN)

    mergeable_ranks_list = tokenizer_trainer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
    tokens_offset = len(mergeable_ranks)
    special_tokens_dict = {name: tokens_offset + i for i, name in enumerate(special_tokens)}

    tokenizer = tiktoken.Encoding(
        name="rustbpe",
        pat_str=tokenizer_trainer.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens_dict,
    )

    # Test special token encoding
    test_id = tokenizer.encode_single_token("<|test|>")
    special_id = tokenizer.encode_single_token("<|special|>")

    assert test_id is not None
    assert special_id is not None
    assert test_id != special_id

    # Test special token decoding
    test_decoded = tokenizer.decode([test_id])
    assert test_decoded == "<|test|>"

    print("✓ Special tokens work correctly")


@pytest.mark.skipif(not RUSTBPE_AVAILABLE, reason="rustbpe not installed")
def test_token_bytes_computation():
    """Test computing token bytes for evaluation."""
    import tiktoken

    training_texts = ["Hello world! Testing 123."] * 100
    special_tokens = ["<|bos|>"]

    SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    vocab_size = 500
    vocab_size_no_special = vocab_size - len(special_tokens)

    tokenizer_trainer = rustbpe.Tokenizer()
    tokenizer_trainer.train_from_iterator(iter(training_texts), vocab_size_no_special, pattern=SPLIT_PATTERN)

    mergeable_ranks_list = tokenizer_trainer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
    tokens_offset = len(mergeable_ranks)
    special_tokens_dict = {name: tokens_offset + i for i, name in enumerate(special_tokens)}

    tokenizer = tiktoken.Encoding(
        name="rustbpe",
        pat_str=tokenizer_trainer.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens_dict,
    )

    # Compute token bytes
    vocab_size = tokenizer.n_vocab
    special_set = tokenizer.special_tokens_set

    token_bytes = []
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        if token_str in special_set:
            token_bytes.append(0)
        else:
            id_bytes = len(token_str.encode("utf-8"))
            token_bytes.append(id_bytes)

    # Verify token_bytes
    assert len(token_bytes) == vocab_size
    assert all(b >= 0 for b in token_bytes)

    # Special tokens should have 0 bytes
    bos_id = tokenizer.encode_single_token("<|bos|>")
    assert token_bytes[bos_id] == 0

    # Regular tokens should have > 0 bytes
    nonzero_bytes = [b for b in token_bytes if b > 0]
    assert len(nonzero_bytes) > 0

    print("✓ Token bytes computed correctly")
    print(f"  Tokens with bytes: {len(nonzero_bytes)}/{vocab_size}")
    print(
        f"  Min/max/mean bytes: {min(nonzero_bytes)}/{max(nonzero_bytes)}/{sum(nonzero_bytes)/len(nonzero_bytes):.2f}"
    )


@pytest.mark.skipif(not RUSTBPE_AVAILABLE, reason="rustbpe not installed")
def test_tokenizer_roundtrip():
    """Test that encoding and decoding preserves text."""
    import tiktoken

    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
    ] * 200

    special_tokens = ["<|bos|>"]
    SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    vocab_size = 1000
    vocab_size_no_special = vocab_size - len(special_tokens)

    tokenizer_trainer = rustbpe.Tokenizer()
    tokenizer_trainer.train_from_iterator(iter(training_texts), vocab_size_no_special, pattern=SPLIT_PATTERN)

    mergeable_ranks_list = tokenizer_trainer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
    tokens_offset = len(mergeable_ranks)
    special_tokens_dict = {name: tokens_offset + i for i, name in enumerate(special_tokens)}

    tokenizer = tiktoken.Encoding(
        name="rustbpe",
        pat_str=tokenizer_trainer.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens_dict,
    )

    # Test various texts
    test_cases = [
        "Simple text",
        "Text with numbers: 123, 456",
        "Punctuation!!! Test??? Yes...",
        "Unicode: 你好 🌍",
        "Contractions: I'm, you're, it's",
        "Mixed case: CamelCase snake_case UPPERCASE lowercase",
    ]

    for test_text in test_cases:
        encoded = tokenizer.encode_ordinary(test_text)
        decoded = tokenizer.decode(encoded)
        assert decoded == test_text, f"Roundtrip failed for: {test_text}"

    print("✓ All roundtrip tests passed")


@pytest.mark.skipif(not RUSTBPE_AVAILABLE, reason="rustbpe not installed")
def test_tokenizer_mlx_integration():
    """Test that tokenizer works with MLX arrays (via wrapper)."""
    import tiktoken
    from mlxchat.tokenizer import Tokenizer

    # Train a small tokenizer
    training_texts = ["Hello world!"] * 100
    special_tokens = ["<|bos|>"]
    SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    vocab_size = 500
    vocab_size_no_special = vocab_size - len(special_tokens)

    tokenizer_trainer = rustbpe.Tokenizer()
    tokenizer_trainer.train_from_iterator(iter(training_texts), vocab_size_no_special, pattern=SPLIT_PATTERN)

    mergeable_ranks_list = tokenizer_trainer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
    tokens_offset = len(mergeable_ranks)
    special_tokens_dict = {name: tokens_offset + i for i, name in enumerate(special_tokens)}

    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=tokenizer_trainer.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens_dict,
    )

    # Wrap in MLX Tokenizer
    tokenizer = Tokenizer(enc, bos_token="<|bos|>")

    # Test encoding returns MLX array
    text = "Hello world!"
    ids = tokenizer.encode(text, return_array=True)

    assert isinstance(ids, mx.array)
    assert ids.dtype == mx.int32
    assert len(ids) > 0

    # Test decoding
    decoded = tokenizer.decode(ids)
    assert decoded == text

    print("✓ MLX integration works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
