"""
Dataloader for mlxchat

Reads FineWeb parquet shards and yields batches of tokenized sequences.
Supports on-demand shard downloading for limited storage.
"""

import os
import logging
from collections import deque

import mlx.core as mx
import pyarrow.parquet as pq

from mlxchat.tokenizer import get_tokenizer
from mlxchat.dataset import ShardCache

logger = logging.getLogger(__name__)


def list_parquet_files(data_dir, require_files=True):
    """
    List all parquet files in a directory.

    Args:
        data_dir: Path to directory containing parquet files
        require_files: If True, raise error if no files found

    Returns:
        List of full paths to parquet files
    """
    if not os.path.exists(data_dir):
        if require_files:
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}. " f"Please download FineWeb shards or enable streaming mode."
            )
        return []

    parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet") and not f.endswith(".tmp")])

    if not parquet_files and require_files:
        raise FileNotFoundError(
            f"No parquet files found in {data_dir}. " f"Please download FineWeb shards or enable streaming mode."
        )

    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(split, data_dir, shard_cache=None):
    """
    Iterate through parquet files, yielding batches of text documents.

    Supports two modes:
    1. Static: Iterate over existing files in data_dir
    2. Streaming: Download shards on-demand using shard_cache

    Args:
        split: Either "train" or "val"
        data_dir: Path to directory containing parquet files
        shard_cache: Optional ShardCache for on-demand downloads

    Yields:
        List of text strings (one batch from a row group)
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    if shard_cache is None:
        # Static mode: use existing files
        parquet_paths = list_parquet_files(data_dir, require_files=True)

        # Use last file for validation, all others for training
        if split == "train":
            parquet_paths = parquet_paths[:-1]
        else:
            parquet_paths = parquet_paths[-1:]

        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                texts = rg.column("text").to_pylist()
                yield texts
    else:
        # Streaming mode: download shards on-demand
        from mlxchat.dataset import MAX_SHARD

        # Determine shard range based on split
        if split == "train":
            shard_indices = range(0, MAX_SHARD)  # All but last
        else:
            shard_indices = [MAX_SHARD]  # Last shard only
            logger.info(f"Validation mode: accessing shard {MAX_SHARD}")

        for shard_idx in shard_indices:
            logger.info(f"Accessing shard {shard_idx} for {split} split...")
            filepath = shard_cache.get_shard_path(shard_idx)
            if filepath is None:
                logger.warning(f"Failed to get shard {shard_idx}, skipping")
                continue

            logger.info(f"Opening parquet file: {filepath}")
            pf = pq.ParquetFile(filepath)
            logger.info(f"Shard {shard_idx} has {pf.num_row_groups} row groups")

            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                texts = rg.column("text").to_pylist()
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
        streaming=False,
        max_cached_shards=20,
    ):
        """
        Initialize dataloader.

        Args:
            batch_size: Number of sequences per batch (B)
            sequence_length: Length of each sequence (T)
            split: Either "train" or "val"
            data_dir: Path to directory with parquet files
                     (default: ~/.cache/mlxchat/base_data)
            tokenizer_dir: Path to tokenizer directory
                          (default: ~/.cache/mlxchat/tokenizer)
            tokenizer_batch_size: Number of documents to tokenize at once
            streaming: If True, download shards on-demand
            max_cached_shards: Max shards to keep on disk (streaming mode only)
        """
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.split = split
        self.tokenizer_batch_size = tokenizer_batch_size
        self.streaming = streaming

        # Set default data directory
        if data_dir is None:
            home = os.path.expanduser("~")
            data_dir = os.path.join(home, ".cache", "mlxchat", "base_data")
        self.data_dir = data_dir

        # Setup shard cache for streaming mode
        if streaming:
            self.shard_cache = ShardCache(
                data_dir=data_dir,
                max_shards=max_cached_shards,
                enable_cache_cleanup=True,
            )
        else:
            self.shard_cache = None

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
            for batch in parquets_iter_batched(self.split, self.data_dir, self.shard_cache):
                # Yield in smaller batches for tokenizer
                for i in range(0, len(batch), self.tokenizer_batch_size):
                    yield batch[i : i + self.tokenizer_batch_size]

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

            # Evaluate arrays immediately to prevent lazy graph buildup
            mx.eval(inputs, targets)

            yield inputs, targets


def get_dataloader(
    batch_size,
    sequence_length,
    split="train",
    data_dir=None,
    tokenizer_dir=None,
    streaming=False,
    max_cached_shards=20,
):
    """
    Convenience function to create a dataloader.

    Args:
        batch_size: Number of sequences per batch
        sequence_length: Length of each sequence
        split: Either "train" or "val"
        data_dir: Path to parquet files
        tokenizer_dir: Path to tokenizer (default: ~/.cache/mlxchat/tokenizer)
        streaming: If True, download shards on-demand
        max_cached_shards: Max shards to keep on disk (streaming mode only)

    Returns:
        DataLoader instance
    """
    return DataLoader(
        batch_size=batch_size,
        sequence_length=sequence_length,
        split=split,
        data_dir=data_dir,
        tokenizer_dir=tokenizer_dir,
        streaming=streaming,
        max_cached_shards=max_cached_shards,
    )
