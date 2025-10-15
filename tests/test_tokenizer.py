"""Tests for tokenizer"""

import pytest
import mlx.core as mx

from mlxchat.tokenizer import Tokenizer, SPECIAL_TOKENS


def test_tokenizer_from_pretrained():
    """Test loading a pretrained tiktoken tokenizer"""
    # Use GPT-2 tokenizer for testing (doesn't require nanochat setup)
    tokenizer = Tokenizer.from_pretrained("gpt2")

    assert tokenizer is not None
    assert tokenizer.get_vocab_size() > 0
    assert tokenizer.get_bos_token_id() is not None


def test_tokenizer_encode_decode_string():
    """Test encoding and decoding a single string"""
    tokenizer = Tokenizer.from_pretrained("gpt2")

    text = "Hello, world!"
    ids = tokenizer.encode(text, return_array=True)

    # Check it's an MLX array
    assert isinstance(ids, mx.array)
    assert ids.dtype == mx.int32
    assert len(ids) > 0

    # Decode back
    decoded = tokenizer.decode(ids)
    assert decoded == text


def test_tokenizer_encode_decode_list():
    """Test encoding and decoding a list of strings"""
    tokenizer = Tokenizer.from_pretrained("gpt2")

    texts = ["Hello", "world"]
    ids_list = tokenizer.encode(texts, return_array=True)

    # Check it's a list of MLX arrays
    assert isinstance(ids_list, list)
    assert len(ids_list) == 2
    assert all(isinstance(ids, mx.array) for ids in ids_list)

    # Decode back
    decoded_texts = [tokenizer.decode(ids) for ids in ids_list]
    assert decoded_texts == texts


def test_tokenizer_encode_with_prepend():
    """Test encoding with prepended token"""
    tokenizer = Tokenizer.from_pretrained("gpt2")

    text = "Hello"
    bos_id = tokenizer.get_bos_token_id()

    # Encode with prepend
    ids = tokenizer.encode(text, prepend=bos_id, return_array=True)

    # First token should be BOS
    assert ids[0].item() == bos_id


def test_tokenizer_encode_with_append():
    """Test encoding with appended token"""
    tokenizer = Tokenizer.from_pretrained("gpt2")

    text = "Hello"
    bos_id = tokenizer.get_bos_token_id()

    # Encode with append
    ids = tokenizer.encode(text, append=bos_id, return_array=True)

    # Last token should be BOS
    assert ids[-1].item() == bos_id


def test_tokenizer_return_list():
    """Test encoding with return_array=False"""
    tokenizer = Tokenizer.from_pretrained("gpt2")

    text = "Hello"
    ids = tokenizer.encode(text, return_array=False)

    # Check it's a Python list
    assert isinstance(ids, list)
    assert all(isinstance(id, int) for id in ids)


def test_tokenizer_call():
    """Test that __call__ works as shorthand for encode"""
    tokenizer = Tokenizer.from_pretrained("gpt2")

    text = "Hello"
    ids1 = tokenizer.encode(text)
    ids2 = tokenizer(text)

    # Should be the same
    assert mx.array_equal(ids1, ids2)


def test_tokenizer_special_tokens():
    """Test accessing special tokens"""
    tokenizer = Tokenizer.from_pretrained("gpt2")

    special_tokens = tokenizer.get_special_tokens()
    assert isinstance(special_tokens, set)
    assert len(special_tokens) > 0


def test_tokenizer_id_to_token():
    """Test converting token ID to string"""
    tokenizer = Tokenizer.from_pretrained("gpt2")

    # Encode a simple word
    text = "Hello"
    ids = tokenizer.encode(text, return_array=False)

    # Decode each token
    for token_id in ids:
        token_str = tokenizer.id_to_token(token_id)
        assert isinstance(token_str, str)


def test_tokenizer_vocab_size():
    """Test getting vocabulary size"""
    tokenizer = Tokenizer.from_pretrained("gpt2")

    vocab_size = tokenizer.get_vocab_size()
    assert isinstance(vocab_size, int)
    assert vocab_size > 0


def test_tokenizer_round_trip():
    """Test that encode -> decode is a perfect round trip"""
    tokenizer = Tokenizer.from_pretrained("gpt2")

    # Test various texts
    texts = [
        "Hello, world!",
        "This is a test.",
        "1234567890",
        "Special characters: !@#$%^&*()",
    ]

    for text in texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text, f"Round trip failed for: {text}"


def test_tokenizer_batch_encode():
    """Test batch encoding efficiency"""
    tokenizer = Tokenizer.from_pretrained("gpt2")

    # Create a batch of texts
    batch = ["Text one", "Text two", "Text three"]

    # Encode as batch
    ids_batch = tokenizer.encode(batch, return_array=True)

    # Encode individually
    ids_individual = [tokenizer.encode(text, return_array=True) for text in batch]

    # Results should be the same
    for i in range(len(batch)):
        assert mx.array_equal(ids_batch[i], ids_individual[i])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
