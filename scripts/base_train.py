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
from mlxchat.checkpoint_manager import save_checkpoint, get_base_dir


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
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")

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


def flatten_params(tree, prefix=""):
    """Flatten nested parameter dict into list of (name, param) tuples."""
    items = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            items.extend(flatten_params(v, new_prefix))
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            new_prefix = f"{prefix}.{i}" if prefix else str(i)
            items.extend(flatten_params(v, new_prefix))
    elif isinstance(tree, mx.array):
        items.append((prefix, tree))
    return items


def setup_optimizers(model, config, args):
    """
    Setup dual optimizer system: Muon for matrices, Adam for embeddings/lm_head.

    Returns:
        tuple: (adam_optimizer, muon_optimizer, param_groups)
    """
    model_dim = config.n_embd

    # Scale learning rates by 1/sqrt(dmodel/768)
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    print(f"  LR scaling factor (∝1/√(d/{768})): {dmodel_lr_scale:.6f}")

    # Flatten parameters with names
    all_params = flatten_params(model.parameters())

    # Separate parameters into groups
    matrix_params = {}
    embedding_params = {}
    lm_head_params = {}

    for name, param in all_params:
        if name.startswith("wte"):
            embedding_params[name] = param
        elif name.startswith("lm_head"):
            lm_head_params[name] = param
        else:
            # All transformer block parameters
            matrix_params[name] = param

    print(f"  Matrix params: {len(matrix_params)}")
    print(f"  Embedding params: {len(embedding_params)}")
    print(f"  LM head params: {len(lm_head_params)}")

    # Create Adam optimizer for embeddings and LM head
    adam_params = {**embedding_params, **lm_head_params}
    adam_optimizer = optim.Adam(
        learning_rate=args.unembedding_lr * dmodel_lr_scale,
    )

    # Create Muon optimizer for transformer blocks
    muon_optimizer = Muon(
        learning_rate=args.matrix_lr,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    )

    # Return param groups for applying gradients later
    param_groups = {
        "adam": adam_params,
        "muon": matrix_params,
    }

    return adam_optimizer, muon_optimizer, param_groups


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


def evaluate(model, val_loader, eval_steps):
    """
    Evaluate the model on validation data.

    Args:
        model: GPT model
        val_loader: Validation data loader
        eval_steps: Number of batches to evaluate

    Returns:
        Average validation loss
    """
    model.eval()  # Set to eval mode (though MLX doesn't have dropout/batchnorm)

    total_loss = 0.0
    num_batches = 0

    val_iter = iter(val_loader)

    for _ in range(eval_steps):
        try:
            inputs, targets = next(val_iter)

            # Forward pass only (no gradients)
            loss = model(inputs, targets=targets)
            total_loss += loss.item()
            num_batches += 1

        except StopIteration:
            break

    model.train()  # Set back to train mode

    if num_batches == 0:
        return float('inf')

    return total_loss / num_batches


def clip_gradients(grads, max_norm):
    """
    Clip gradients by global norm.

    Args:
        grads: Dictionary of gradients
        max_norm: Maximum gradient norm

    Returns:
        Clipped gradients and global norm
    """
    from mlx.utils import tree_flatten, tree_map

    # Compute global norm
    # tree_flatten returns list of (path, value) tuples
    grad_list = tree_flatten(grads)
    global_norm = mx.sqrt(sum(mx.sum(mx.square(v)) for _, v in grad_list))

    # Compute scale factor
    scale = mx.minimum(max_norm / (global_norm + 1e-6), 1.0)

    # Scale all gradients
    clipped_grads = tree_map(lambda g: g * scale, grads)

    return clipped_grads, global_norm


def print_banner():
    """Print ASCII art banner for MLXChat."""
    banner = r"""
    ███╗   ███╗██╗     ██╗  ██╗ ██████╗██╗  ██╗ █████╗ ████████╗
    ████╗ ████║██║     ╚██╗██╔╝██╔════╝██║  ██║██╔══██╗╚══██╔══╝
    ██╔████╔██║██║      ╚███╔╝ ██║     ███████║███████║   ██║
    ██║╚██╔╝██║██║      ██╔██╗ ██║     ██╔══██║██╔══██║   ██║
    ██║ ╚═╝ ██║███████╗██╔╝ ██╗╚██████╗██║  ██║██║  ██║   ██║
    ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝
    """
    print(banner)
    print("=" * 80)


def main():
    args = get_args()

    print_banner()

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

    # Count parameters (flatten nested dict structure)
    def count_params(tree):
        total = 0
        if isinstance(tree, dict):
            for v in tree.values():
                total += count_params(v)
        elif isinstance(tree, list):
            for v in tree:
                total += count_params(v)
        elif isinstance(tree, mx.array):
            total += tree.size
        return total

    num_params = count_params(model.parameters())
    print(f"  Number of parameters: {num_params:,}")

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

    # Setup checkpoint directory
    if args.output_dir is None:
        base_dir = get_base_dir()
        args.output_dir = os.path.join(base_dir, "base_checkpoints")

    # Determine model tag from depth if not provided
    if not args.model_tag:
        args.model_tag = f"d{args.depth}"

    checkpoint_dir = os.path.join(args.output_dir, args.model_tag)
    print(f"\nCheckpoint directory: {checkpoint_dir}")
    print(f"  Model tag: {args.model_tag}")
    print(f"  Save every: {args.save_every} steps")

    # Setup optimizers
    print(f"\nSetting up optimizers...")
    adam_opt, muon_opt, param_groups = setup_optimizers(model, config, args)
    # param_groups is a dict with "adam" and "muon" keys
    adam_params = param_groups["adam"]
    muon_params = param_groups["muon"]

    # Setup dataloaders
    print(f"\nSetting up dataloaders...")
    if args.streaming:
        print(f"  Streaming mode enabled: max {args.max_cached_shards} shards cached")

    train_loader = DataLoader(
        batch_size=args.device_batch_size,
        sequence_length=args.max_seq_len,
        split="train",
        data_dir=args.data_dir,
        tokenizer_dir=args.tokenizer_dir,
        streaming=args.streaming,
        max_cached_shards=args.max_cached_shards,
    )

    # Setup validation dataloader
    val_loader = DataLoader(
        batch_size=args.device_batch_size,
        sequence_length=args.max_seq_len,
        split="val",
        data_dir=args.data_dir,
        tokenizer_dir=args.tokenizer_dir,
        streaming=args.streaming,
        max_cached_shards=args.max_cached_shards,
    )

    # Calculate validation steps from eval_tokens
    eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len))
    print(f"  Validation steps per eval: {eval_steps}")

    # Training state
    smooth_train_loss = 0.0
    ema_beta = 0.9
    min_val_loss = float("inf")
    total_training_time = 0.0
    training_start_time = time.time()
    avg_step_time = None  # Running average of step time

    print(f"\n{'='*80}")
    print("Starting Training")
    print(f"{'='*80}\n")

    # Training loop
    train_iter = iter(train_loader)

    for step in range(num_iterations + 1):
        last_step = (step == num_iterations)

        # Training step (do this FIRST, before checkpoint/eval)
        if not last_step:
            mx.eval(model.parameters())  # Ensure all params are evaluated
            t0 = time.time()

            # Gradient accumulation
            from mlx.utils import tree_map, tree_flatten

            accum_grads = None
            total_loss = 0.0

            for micro_step in range(grad_accum_steps):
                inputs, targets = next(train_iter)

                # Forward and backward
                loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
                loss, grads = loss_and_grad_fn(model, inputs, targets)

                # Accumulate loss
                total_loss += loss.item()

                # Accumulate gradients
                if accum_grads is not None:
                    accum_grads = tree_map(mx.add, grads, accum_grads)
                else:
                    accum_grads = grads

                # Clean up
                del grads
                mx.eval(accum_grads)

            # Average loss and gradients over accumulation steps
            train_loss = total_loss / grad_accum_steps
            accum_grads = tree_map(lambda g: g / grad_accum_steps, accum_grads)

            # Gradient clipping
            if args.grad_clip > 0.0:
                accum_grads, grad_norm = clip_gradients(accum_grads, args.grad_clip)

            # Learning rate scheduling
            lr_mult = get_lr_multiplier(step, num_iterations)
            muon_momentum = get_muon_momentum(step)

            # Update Adam learning rate
            adam_opt.learning_rate = args.unembedding_lr * lr_mult * ((config.n_embd / 768) ** -0.5)

            # Update Muon learning rate and momentum
            muon_opt.learning_rate = args.matrix_lr * lr_mult
            muon_opt.momentum = muon_momentum

            # Split gradients by parameter group
            # accum_grads is a nested dict matching model structure
            # We need to apply different optimizers to different parts

            # For simplicity, we'll update all parameters with their respective optimizers
            # by filtering the gradient tree
            def filter_grads(grad_tree, param_tree, filter_fn):
                """Filter gradients based on parameter names."""
                filtered = {}
                for key in grad_tree:
                    if isinstance(grad_tree[key], dict):
                        filtered[key] = filter_grads(grad_tree[key], param_tree[key], filter_fn)
                    else:
                        if filter_fn(key):
                            filtered[key] = grad_tree[key]
                return filtered

            # Split parameters and gradients
            # Adam: wte (embeddings) and lm_head
            # Muon: everything else (transformer blocks)

            # Create filtered gradient dictionaries
            adam_grad_keys = {"wte", "lm_head"}
            muon_grad_keys = {"h"}  # transformer blocks

            # Filter gradients for Adam (embeddings + lm_head)
            adam_grads = {}
            if "wte" in accum_grads:
                adam_grads["wte"] = accum_grads["wte"]
            if "lm_head" in accum_grads:
                adam_grads["lm_head"] = accum_grads["lm_head"]

            # Filter gradients for Muon (transformer blocks)
            muon_grads = {}
            if "h" in accum_grads:
                muon_grads["h"] = accum_grads["h"]

            # Update parameters with each optimizer
            # Note: optimizer.update expects (params_dict, grads_dict)
            if adam_grads:
                adam_params = {}
                if "wte" in adam_grads:
                    adam_params["wte"] = model.wte
                if "lm_head" in adam_grads:
                    adam_params["lm_head"] = model.lm_head
                adam_opt.update(adam_params, adam_grads)

            if muon_grads:
                muon_params = {"h": model.h}
                muon_opt.update(muon_params, muon_grads)

            # Evaluate model and optimizer states
            mx.eval(model.parameters(), adam_opt.state, muon_opt.state)

            t1 = time.time()
            dt = t1 - t0
            total_training_time = time.time() - training_start_time

            # Update running average of step time (after warmup)
            if step >= 10:
                if avg_step_time is None:
                    avg_step_time = dt
                else:
                    # Exponential moving average with alpha=0.1
                    avg_step_time = 0.9 * avg_step_time + 0.1 * dt

            # EMA smoothing
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss
            debiased_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))

            # Calculate time estimates
            steps_remaining = num_iterations - step
            if avg_step_time is not None and steps_remaining > 0:
                time_remaining_sec = avg_step_time * steps_remaining
                time_remaining_min = time_remaining_sec / 60
                time_remaining_hrs = time_remaining_min / 60

                # Format time remaining nicely
                if time_remaining_hrs >= 1:
                    eta_str = f"{time_remaining_hrs:.1f}h"
                elif time_remaining_min >= 1:
                    eta_str = f"{time_remaining_min:.1f}m"
                else:
                    eta_str = f"{time_remaining_sec:.0f}s"
            else:
                eta_str = "calculating..."

            # Logging
            if step % 10 == 0:
                pct_done = 100 * step / num_iterations
                tok_per_sec = int(tokens_per_batch / dt)
                elapsed_min = total_training_time / 60
                print(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
                      f"loss: {debiased_loss:.6f} | dt: {dt*1000:.2f}ms | "
                      f"tok/sec: {tok_per_sec:,} | elapsed: {elapsed_min:.2f}m | eta: {eta_str}")

        # Checkpoint saving (after training step)
        if step > 0 and (step % args.save_every == 0 or last_step):
            print(f"\nSaving checkpoint at step {step}...")

            # Prepare metadata
            meta_data = {
                "step": step,
                "loss": smooth_train_loss / (1 - ema_beta ** step) if step > 0 else 0.0,
                "model_config": {
                    "sequence_len": config.sequence_len,
                    "vocab_size": config.vocab_size,
                    "n_layer": config.n_layer,
                    "n_head": config.n_head,
                    "n_kv_head": config.n_kv_head,
                    "n_embd": config.n_embd,
                },
                "training_config": {
                    "depth": args.depth,
                    "device_batch_size": args.device_batch_size,
                    "total_batch_size": args.total_batch_size,
                    "grad_accum_steps": grad_accum_steps,
                    "num_iterations": num_iterations,
                },
            }

            # Prepare optimizer data
            optimizer_data = {
                "adam": adam_opt.state,
                "muon": muon_opt.state,
            }

            # Save checkpoint
            save_checkpoint(checkpoint_dir, step, model, optimizer_data, meta_data)
            print(f"Checkpoint saved successfully")

        # Validation evaluation (after training step)
        if step > 0 and step % args.eval_every == 0:
            print(f"\nEvaluating at step {step}...")
            val_loss = evaluate(model, val_loader, eval_steps)
            print(f"Validation loss: {val_loss:.6f}")

            # Update min validation loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                print(f"New best validation loss!")

    print(f"\n{'='*80}")
    print("Training Complete")
    print(f"  Total time: {total_training_time/60:.2f} minutes")
    print(f"  Min validation loss: {min_val_loss:.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
