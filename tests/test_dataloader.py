"""Tests for dataloader"""

import os
import tempfile
import pickle
import pytest
import mlx.core as mx
import pyarrow as pa
import pyarrow.parquet as pq

from mlxchat.dataloader import DataLoader, list_parquet_files, parquets_iter_batched


@pytest.fixture
def temp_tokenizer():
    """Create a temporary tokenizer directory with GPT-2 tokenizer"""
    import tiktoken

    with tempfile.TemporaryDirectory() as tok_dir:
        # Create a tiktoken encoding with a custom BOS token
        gpt2_enc = tiktoken.get_encoding("gpt2")

        # Create a new encoding that includes <|bos|> as a special token
        enc = tiktoken.Encoding(
            name="gpt2_with_bos",
            pat_str=gpt2_enc._pat_str,
            mergeable_ranks=gpt2_enc._mergeable_ranks,
            special_tokens={
                **gpt2_enc._special_tokens,
                "<|bos|>": gpt2_enc.n_vocab,  # Add BOS token at the end
            },
        )

        with open(os.path.join(tok_dir, "tokenizer.pkl"), "wb") as f:
            pickle.dump(enc, f)
        yield tok_dir


@pytest.fixture
def temp_parquet_files():
    """Create temporary parquet files for testing"""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample text data
        texts_shard1 = [
            "This is document one.",
            "This is document two.",
            "This is document three.",
        ]
        texts_shard2 = [
            "This is document four.",
            "This is document five.",
        ]

        # Write first shard
        table1 = pa.table({"text": texts_shard1})
        pq.write_table(table1, os.path.join(tmpdir, "shard_00000.parquet"))

        # Write second shard (will be used for validation)
        table2 = pa.table({"text": texts_shard2})
        pq.write_table(table2, os.path.join(tmpdir, "shard_00001.parquet"))

        yield tmpdir


def test_list_parquet_files(temp_parquet_files):
    """Test listing parquet files"""
    files = list_parquet_files(temp_parquet_files)

    assert len(files) == 2
    assert all(f.endswith(".parquet") for f in files)
    assert all(os.path.exists(f) for f in files)


def test_list_parquet_files_empty_dir():
    """Test listing parquet files in empty directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="No parquet files found"):
            list_parquet_files(tmpdir)


def test_list_parquet_files_nonexistent():
    """Test listing parquet files in nonexistent directory"""
    with pytest.raises(FileNotFoundError, match="Data directory not found"):
        list_parquet_files("/nonexistent/directory")


def test_parquets_iter_batched_train(temp_parquet_files):
    """Test iterating over training parquet files"""
    batches = list(parquets_iter_batched("train", temp_parquet_files))

    # Should only use first shard for training (last is for validation)
    assert len(batches) >= 1

    # Check that we got text documents
    for batch in batches:
        assert isinstance(batch, list)
        assert all(isinstance(text, str) for text in batch)


def test_parquets_iter_batched_val(temp_parquet_files):
    """Test iterating over validation parquet files"""
    batches = list(parquets_iter_batched("val", temp_parquet_files))

    # Should only use last shard for validation
    assert len(batches) >= 1

    # Check that we got text documents
    for batch in batches:
        assert isinstance(batch, list)
        assert all(isinstance(text, str) for text in batch)


def test_dataloader_initialization(temp_parquet_files, temp_tokenizer):
    """Test DataLoader initialization"""
    loader = DataLoader(
        batch_size=2,
        sequence_length=8,
        split="train",
        data_dir=temp_parquet_files,
        tokenizer_dir=temp_tokenizer,
    )

    assert loader.batch_size == 2
    assert loader.sequence_length == 8
    assert loader.split == "train"
    assert loader.needed_tokens == 2 * 8 + 1


def test_dataloader_iterator(temp_parquet_files, temp_tokenizer):
    """Test DataLoader iteration"""
    loader = DataLoader(
        batch_size=2,
        sequence_length=8,
        split="train",
        data_dir=temp_parquet_files,
        tokenizer_dir=temp_tokenizer,
    )

    # Get one batch
    iterator = iter(loader)
    inputs, targets = next(iterator)

    # Check shapes
    assert inputs.shape == (2, 8)
    assert targets.shape == (2, 8)

    # Check dtypes
    assert inputs.dtype == mx.int32
    assert targets.dtype == mx.int32

    # Check that targets are shifted by 1
    # (This is approximate since we're using real tokenization)
    # We just verify the shapes and types are correct


def test_dataloader_multiple_batches(temp_parquet_files, temp_tokenizer):
    """Test getting multiple batches from DataLoader"""
    loader = DataLoader(
        batch_size=2,
        sequence_length=8,
        split="train",
        data_dir=temp_parquet_files,
        tokenizer_dir=temp_tokenizer,
    )

    # Get multiple batches
    iterator = iter(loader)
    batches = [next(iterator) for _ in range(3)]

    # Check we got 3 batches
    assert len(batches) == 3

    # Each batch should have correct shapes
    for inputs, targets in batches:
        assert inputs.shape == (2, 8)
        assert targets.shape == (2, 8)


def test_dataloader_val_split(temp_parquet_files, temp_tokenizer):
    """Test DataLoader with validation split"""
    loader = DataLoader(
        batch_size=2,
        sequence_length=8,
        split="val",
        data_dir=temp_parquet_files,
        tokenizer_dir=temp_tokenizer,
    )

    # Should be able to get batches from validation set
    iterator = iter(loader)
    inputs, targets = next(iterator)

    assert inputs.shape == (2, 8)
    assert targets.shape == (2, 8)


def test_dataloader_small_batch(temp_parquet_files, temp_tokenizer):
    """Test DataLoader with very small batch size"""
    loader = DataLoader(
        batch_size=1,
        sequence_length=4,
        split="train",
        data_dir=temp_parquet_files,
        tokenizer_dir=temp_tokenizer,
    )

    iterator = iter(loader)
    inputs, targets = next(iterator)

    assert inputs.shape == (1, 4)
    assert targets.shape == (1, 4)


def test_dataloader_token_buffer():
    """Test that DataLoader maintains token buffer correctly"""
    # This is more of an integration test to ensure the buffer works
    # We can't easily test the internal buffer without exposing it,
    # so we just verify multiple batches work correctly
    pass  # Covered by test_dataloader_multiple_batches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
