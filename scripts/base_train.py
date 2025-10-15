"""
Train GPT model with MLX on Apple Silicon

Usage:
    python -m scripts.base_train --depth=12 --device_batch_size=4

Or with streaming:
    python -m scripts.base_train --depth=12 --streaming --max-cached-shards=20
"""

import os
import time
import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlxchat.gpt import GPT, GPTConfig
from mlxchat.muon import Muon
from mlxchat.dataloader import DataLoader
from mlxchat.tokenizer import get_tokenizer


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GPT model with MLX")

    # Model architecture
    parser.add_argument("--depth", type=int, default=12, help="Transformer depth (12/16/20/26)")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")

    # Training horizon
    parser.add_argument("--num-iterations", type=int, default=-1, help="Number of training steps (-1 = use param-data ratio)")
    parser.add_argument("--target-param-data-ratio", type=float, default=20.0, help="Chinchilla data:param ratio")

    # Optimization
    parser.add_argument("--device-batch-size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--total-batch-size", type=int, default=524288, help="Total batch size in tokens")
    parser.add_argument("--embedding-lr", type=float, default=0.2, help="Adam LR for embeddings")
    parser.add_argument("--unembedding-lr", type=float, default=0.004, help="Adam LR for LM head")
    parser.add_argument("--matrix-lr", type=float, default=0.02, help="Muon LR for matrices")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping (0.0 = disabled)")

    # Data
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode for data")
    parser.add_argument("--max-cached-shards", type=int, default=20, help="Max shards to cache (streaming mode)")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--tokenizer-dir", type=str, default=None, help="Tokenizer directory")

    # Evaluation
    parser.add_argument("--eval-every", type=int, default=250, help="Evaluate every N steps")
    parser.add_argument("--eval-tokens", type=int, default=20*524288, help="Tokens for validation")

    # Output
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--model-tag", type=str, default="", help="Model tag for checkpoint directory")

    return parser.parse_args()


def get_model_config(depth, max_seq_len, vocab_size):
    """
    Derive model configuration from depth.

    Architecture scaling rules from nanochat:
    - num_layers = depth
    - model_dim = depth * 64 (aspect ratio 64)
    - num_heads = ceil(model_dim / 128) (head_dim 128)
    - num_kv_heads = num_heads (1:1 MQA ratio)
    """
    num_layers = depth
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)  # ceil division
    num_kv_heads = num_heads

    return GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim,
    )


def setup_optimizers(model, config, args):
    """
    Setup dual optimizer system: Muon for matrices, Adam for embeddings/lm_head.

    Returns:
        tuple: (adam_optimizer, muon_optimizer)
    """
    model_dim = config.n_embd

    # Scale learning rates by 1/sqrt(dmodel/768)
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    print(f"LR scaling factor (∝1/√(d/{768})): {dmodel_lr_scale:.6f}")

    # Separate parameters into groups
    matrix_params = []
    embedding_params = []
    lm_head_params = []

    for name, param in model.named_parameters():
        if "wte" in name:
            embedding_params.append(param)
        elif "lm_head" in name:
            lm_head_params.append(param)
        else:
            # All transformer block parameters
            matrix_params.append(param)

    print(f"Matrix params: {len(matrix_params)}")
    print(f"Embedding params: {len(embedding_params)}")
    print(f"LM head params: {len(lm_head_params)}")

    # Create Adam optimizer for embeddings and LM head
    adam_optimizer = optim.Adam(
        learning_rate=args.unembedding_lr * dmodel_lr_scale,
    )
    adam_optimizer.init({
        **{f"wte_{i}": p for i, p in enumerate(embedding_params)},
        **{f"lm_head_{i}": p for i, p in enumerate(lm_head_params)},
    })

    # Create Muon optimizer for transformer blocks
    muon_optimizer = Muon(
        learning_rate=args.matrix_lr,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    )
    muon_optimizer.init({f"matrix_{i}": p for i, p in enumerate(matrix_params)})

    return adam_optimizer, muon_optimizer, (embedding_params, lm_head_params, matrix_params)


def get_lr_multiplier(step, num_iterations, warmup_ratio=0.0, warmdown_ratio=0.2, final_lr_frac=0.0):
    """Learning rate schedule with warmup and warmdown."""
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)

    if step < warmup_iters:
        return (step + 1) / warmup_iters
    elif step <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - step) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac


def get_muon_momentum(step):
    """Momentum schedule for Muon optimizer."""
    frac = min(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


def loss_fn(model, inputs, targets):
    """Compute loss for a batch."""
    return model(inputs, targets=targets)


def main():
    args = get_args()

    print("=" * 80)
    print("MLXChat Training")
    print("=" * 80)

    # Load tokenizer
    tokenizer = get_tokenizer(args.tokenizer_dir)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    # Get model configuration
    config = get_model_config(args.depth, args.max_seq_len, vocab_size)
    print(f"\nModel Configuration:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Model dim: {config.n_embd}")
    print(f"  Num heads: {config.n_head}")
    print(f"  Num KV heads: {config.n_kv_head}")

    # Create model
    print(f"\nInitializing model...")
    model = GPT(config)
    model.init_weights()

    # Count parameters
    num_params = sum(p.size for p in model.parameters().values())
    print(f"Number of parameters: {num_params:,}")

    # Calculate training horizon
    tokens_per_batch = args.device_batch_size * args.max_seq_len
    grad_accum_steps = args.total_batch_size // tokens_per_batch
    print(f"\nBatch Configuration:")
    print(f"  Device batch size: {args.device_batch_size}")
    print(f"  Tokens per batch: {tokens_per_batch:,}")
    print(f"  Total batch size: {args.total_batch_size:,}")
    print(f"  Gradient accumulation steps: {grad_accum_steps}")

    if args.num_iterations > 0:
        num_iterations = args.num_iterations
        print(f"\nUsing specified number of iterations: {num_iterations:,}")
    else:
        target_tokens = args.target_param_data_ratio * num_params
        num_iterations = int(target_tokens // args.total_batch_size)
        print(f"\nCalculating iterations from data:param ratio {args.target_param_data_ratio}:")
        print(f"  Target tokens: {target_tokens:,}")
        print(f"  Number of iterations: {num_iterations:,}")

    total_tokens = args.total_batch_size * num_iterations
    print(f"  Total training tokens: {total_tokens:,}")
    print(f"  Tokens:Params ratio: {total_tokens / num_params:.2f}")

    # Setup optimizers
    print(f"\nSetting up optimizers...")
    adam_opt, muon_opt, param_groups = setup_optimizers(model, config, args)
    embedding_params, lm_head_params, matrix_params = param_groups

    # Setup dataloaders
    print(f"\nSetting up dataloaders...")
    if args.streaming:
        print(f"  Streaming mode enabled (max {args.max_cached_shards} shards)")

    train_loader = DataLoader(
        batch_size=args.device_batch_size,
        sequence_length=args.max_seq_len,
        split="train",
        data_dir=args.data_dir,
        tokenizer_dir=args.tokenizer_dir,
        streaming=args.streaming,
        max_cached_shards=args.max_cached_shards,
    )

    # Training state
    smooth_train_loss = 0.0
    ema_beta = 0.9
    min_val_loss = float("inf")
    total_training_time = 0.0

    print(f"\n{'='*80}")
    print("Starting Training")
    print(f"{'='*80}\n")

    # Training loop
    train_iter = iter(train_loader)

    for step in range(num_iterations + 1):
        last_step = (step == num_iterations)

        # TODO: Add validation evaluation
        # TODO: Add checkpoint saving

        if last_step:
            break

        # Training step
        mx.eval(model.parameters())  # Ensure all params are evaluated
        t0 = time.time()

        # Gradient accumulation
        total_loss = 0.0
        for micro_step in range(grad_accum_steps):
            inputs, targets = next(train_iter)

            # Forward and backward
            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad_fn(model, inputs, targets)

            # Accumulate loss
            total_loss += loss.item()

            # TODO: Accumulate gradients properly
            # For now, just do one micro-step
            break

        # Average loss over accumulation steps
        train_loss = total_loss / max(1, micro_step + 1)

        # TODO: Apply gradients with optimizers
        # TODO: Gradient clipping
        # TODO: LR scheduling

        t1 = time.time()
        dt = t1 - t0

        if step > 10:
            total_training_time += dt

        # EMA smoothing
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss
        debiased_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))

        # Logging
        if step % 10 == 0:
            pct_done = 100 * step / num_iterations
            tok_per_sec = int(tokens_per_batch / dt)
            print(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
                  f"loss: {debiased_loss:.6f} | dt: {dt*1000:.2f}ms | "
                  f"tok/sec: {tok_per_sec:,} | total time: {total_training_time/60:.2f}m")

    print(f"\n{'='*80}")
    print("Training Complete")
    print(f"  Total time: {total_training_time/60:.2f} minutes")
    print(f"  Min validation loss: {min_val_loss:.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
