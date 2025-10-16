"""
Train a tokenizer using rustbpe (MLX version).
Compatible with nanochat's tokenizer format.
"""

import os
import pickle
import time
import argparse
import logging
import mlx.core as mx
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Import rustbpe directly for training
try:
    import rustbpe
except ImportError:
    logger.error("rustbpe is not installed.")
    logger.error("Please run: make build-tokenizer")
    exit(1)

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
parser.add_argument(
    "--max_chars", type=int, default=10_000_000_000, help="Maximum characters to train on (default: 10B)"
)
parser.add_argument("--doc_cap", type=int, default=10_000, help="Maximum characters per document (default: 10,000)")
parser.add_argument("--vocab_size", type=int, default=65536, help="Vocabulary size (default: 65536 = 2^16)")
parser.add_argument(
    "--output_dir", type=str, default=None, help="Output directory (default: ~/.cache/mlxchat/tokenizer)"
)
args = parser.parse_args()
logger.info(f"max_chars: {args.max_chars:,}")
logger.info(f"doc_cap: {args.doc_cap:,}")
logger.info(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator from FineWeb shards


def text_iterator():
    """
    1) Load text from FineWeb shards
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    from mlxchat.dataset import download_shard
    import pyarrow.parquet as pq

    logger.info("Streaming text from FineWeb shards for tokenizer training...")
    nchars = 0
    shard_idx = 0

    while nchars < args.max_chars:
        # Download shard if needed
        shard_path = download_shard(shard_idx)
        if shard_path is None:
            logger.info(f"Reached end of available shards at shard {shard_idx}")
            break

        logger.info(f"Processing shard {shard_idx:04d} ({nchars:,} / {args.max_chars:,} chars)")

        # Read shard (parquet file)
        table = pq.read_table(shard_path)
        texts = table["text"].to_pylist()

        for doc_text in texts:
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[: args.doc_cap]

            nchars += len(doc_text)
            yield doc_text

            if nchars >= args.max_chars:
                logger.info(f"Reached max_chars limit at {nchars:,} characters")
                return

        shard_idx += 1


text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer

logger.info("Training tokenizer...")
t0 = time.time()

# Define special tokens (same as nanochat)
special_tokens = [
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

# Train tokenizer using rustbpe
tokenizer_trainer = rustbpe.Tokenizer()
vocab_size_no_special = args.vocab_size - len(special_tokens)
assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"

# Pattern from nanochat (GPT-4 style)
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

tokenizer_trainer.train_from_iterator(text_iter, vocab_size_no_special, pattern=SPLIT_PATTERN)

# Construct tiktoken encoding for efficient inference
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

t1 = time.time()
train_time = t1 - t0
logger.info(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk

if args.output_dir is None:
    # Save to mlxchat's tokenizer directory
    home = os.path.expanduser("~")
    tokenizer_dir = os.path.join(home, ".cache", "mlxchat", "tokenizer")
else:
    tokenizer_dir = args.output_dir

os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.pkl")

# Save using pickle (compatible with nanochat)
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)

logger.info(f"Saved tokenizer to {tokenizer_path}")

# -----------------------------------------------------------------------------
# Quick inline sanity check

test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""

encoded = tokenizer.encode_ordinary(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text, f"Sanity check failed!\nOriginal: {test_text}\nDecoded: {decoded}"
logger.info("✓ Sanity check passed")

# -----------------------------------------------------------------------------
# Cache token_bytes for bits-per-byte evaluation

logger.info("Computing token_bytes for evaluation...")
vocab_size = tokenizer.n_vocab
special_set = tokenizer.special_tokens_set

token_bytes = []
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    if token_str in special_set:
        token_bytes.append(0)  # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8"))
        token_bytes.append(id_bytes)

# Save as MLX array (instead of torch.save)
token_bytes_mx = mx.array(token_bytes, dtype=mx.int32)
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.npy")
mx.save(token_bytes_path, token_bytes_mx)
logger.info(f"Saved token_bytes to {token_bytes_path}")

# Print summary statistics
token_bytes_nonzero = [b for b in token_bytes if b > 0]
logger.info("Tokenizer statistics:")
logger.info(f"  Vocab size: {vocab_size:,}")
logger.info(f"  Special tokens: {len(special_set)}")
logger.info(
    f"  Token bytes (min/max/mean): {min(token_bytes_nonzero)}/{max(token_bytes_nonzero)}/{sum(token_bytes_nonzero)/len(token_bytes_nonzero):.2f}"
)
logger.info("✓ Tokenizer training complete!")
logger.info("  You can now train models with this tokenizer.")
logger.info("  Run: python -m scripts.base_train --streaming")
