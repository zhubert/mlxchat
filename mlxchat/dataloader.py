"""
Dataloader for mlxchat

Reads FineWeb parquet shards and yields batches of tokenized sequences.
Simplified version without distributed training support.
"""

import os
from collections import deque

import mlx.core as mx
import pyarrow.parquet as pq

from mlxchat.tokenizer import get_tokenizer


def list_parquet_files(data_dir):
    """
    List all parquet files in a directory.

    Args:
        data_dir: Path to directory containing parquet files

    Returns:
        List of full paths to parquet files
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            f"Please download FineWeb shards using nanochat first."
        )

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {data_dir}. "
            f"Please download FineWeb shards using nanochat first."
        )

    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(split, data_dir):
    """
    Iterate through parquet files, yielding batches of text documents.

    Args:
        split: Either "train" or "val"
        data_dir: Path to directory containing parquet files

    Yields:
        List of text strings (one batch from a row group)
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    parquet_paths = list_parquet_files(data_dir)

    # Use last file for validation, all others for training
    if split == "train":
        parquet_paths = parquet_paths[:-1]
    else:
        parquet_paths = parquet_paths[-1:]

    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts


class DataLoader:
    """
    Dataloader that streams tokenized batches from parquet files.

    This loader:
    1. Reads parquet files containing text documents
    2. Tokenizes documents on-the-fly
    3. Yields batches of shape (B, T) for training
    """

    def __init__(
        self,
        batch_size,
        sequence_length,
        split="train",
        data_dir=None,
        tokenizer_dir=None,
        tokenizer_batch_size=128,
    ):
        """
        Initialize dataloader.

        Args:
            batch_size: Number of sequences per batch (B)
            sequence_length: Length of each sequence (T)
            split: Either "train" or "val"
            data_dir: Path to directory with parquet files
                     (default: ~/.cache/nanochat/base_data)
            tokenizer_dir: Path to tokenizer directory
                          (default: ~/.cache/nanochat/tokenizer)
            tokenizer_batch_size: Number of documents to tokenize at once
        """
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.split = split
        self.tokenizer_batch_size = tokenizer_batch_size

        # Set default data directory
        if data_dir is None:
            home = os.path.expanduser("~")
            data_dir = os.path.join(home, ".cache", "nanochat", "base_data")
        self.data_dir = data_dir

        # Load tokenizer
        self.tokenizer = get_tokenizer(tokenizer_dir)
        self.bos_token_id = self.tokenizer.get_bos_token_id()

        # Token buffer for streaming
        self.token_buffer = deque()

        # Number of tokens needed per batch (+1 for target)
        self.needed_tokens = batch_size * sequence_length + 1

    def _document_batches(self):
        """
        Infinite iterator over document batches.

        Yields:
            List of text strings
        """
        while True:
            for batch in parquets_iter_batched(self.split, self.data_dir):
                # Yield in smaller batches for tokenizer
                for i in range(0, len(batch), self.tokenizer_batch_size):
                    yield batch[i:i+self.tokenizer_batch_size]

    def __iter__(self):
        """
        Iterate over batches of tokenized sequences.

        Yields:
            Tuple of (inputs, targets) where:
            - inputs: MLX array of shape (B, T) with dtype int32
            - targets: MLX array of shape (B, T) with dtype int32
        """
        batches = self._document_batches()

        while True:
            # Accumulate enough tokens for one batch
            while len(self.token_buffer) < self.needed_tokens:
                doc_batch = next(batches)

                # Tokenize the batch with BOS token prepended
                token_lists = self.tokenizer.encode(
                    doc_batch,
                    prepend=self.bos_token_id,
                    return_array=False,
                )

                # Add all tokens to buffer
                for tokens in token_lists:
                    self.token_buffer.extend(tokens)

            # Extract needed tokens from buffer
            tokens = []
            for _ in range(self.needed_tokens):
                tokens.append(self.token_buffer.popleft())

            # Convert to MLX array
            tokens_array = mx.array(tokens, dtype=mx.int32)

            # Split into inputs and targets
            inputs = tokens_array[:-1]  # All but last
            targets = tokens_array[1:]  # All but first

            # Reshape to (B, T)
            inputs = inputs.reshape(self.batch_size, self.sequence_length)
            targets = targets.reshape(self.batch_size, self.sequence_length)

            yield inputs, targets


def get_dataloader(
    batch_size,
    sequence_length,
    split="train",
    data_dir=None,
    tokenizer_dir=None,
):
    """
    Convenience function to create a dataloader.

    Args:
        batch_size: Number of sequences per batch
        sequence_length: Length of each sequence
        split: Either "train" or "val"
        data_dir: Path to parquet files (default: ~/.cache/nanochat/base_data)
        tokenizer_dir: Path to tokenizer (default: ~/.cache/nanochat/tokenizer)

    Returns:
        DataLoader instance
    """
    return DataLoader(
        batch_size=batch_size,
        sequence_length=sequence_length,
        split=split,
        data_dir=data_dir,
        tokenizer_dir=tokenizer_dir,
    )
