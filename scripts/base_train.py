"""
Train GPT model with MLX on Apple Silicon

Usage:
    python -m scripts.base_train --depth=12 --device_batch_size=4

Or with streaming:
    python -m scripts.base_train --depth=12 --streaming --max-cached-shards=20
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlxchat.gpt import GPT, GPTConfig
from mlxchat.muon import Muon
from mlxchat.dataloader import DataLoader
from mlxchat.tokenizer import get_tokenizer
from mlxchat.checkpoint_manager import save_checkpoint, get_base_dir


def setup_logging(log_dir, model_tag):
    """
    Setup logging to both file and console with timestamps.

    Args:
        log_dir: Directory to save log files
        model_tag: Model tag for log filename

    Returns:
        logger: Configured logger instance
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{model_tag}_{timestamp}.log")

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers = []

    # Create formatter with timestamp
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging to: {log_file}")

    return logger


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GPT model with MLX")

    # Model architecture
    parser.add_argument("--depth", type=int, default=12, help="Transformer depth (12/16/20/26)")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")

    # Training horizon
    parser.add_argument(
        "--num-iterations", type=int, default=-1, help="Number of training steps (-1 = use param-data ratio)"
    )
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
    parser.add_argument("--eval-tokens", type=int, default=20 * 524288, help="Tokens for validation")

    # Output
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for log files")
    parser.add_argument("--model-tag", type=str, default="", help="Model tag for checkpoint directory")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N steps")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")

    # Profiling
    parser.add_argument("--profile", action="store_true", help="Enable detailed profiling of training step")

    # Memory optimization
    parser.add_argument("--low-memory", action="store_true", help="Enable aggressive memory optimization mode")

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


def setup_optimizers(model, config, args, logger):
    """
    Setup dual optimizer system: Muon for matrices, Adam for embeddings/lm_head.

    Args:
        model: GPT model
        config: GPTConfig
        args: Command line arguments
        logger: Logger instance

    Returns:
        tuple: (adam_optimizer, muon_optimizer, param_groups)
    """
    model_dim = config.n_embd

    # Scale learning rates by 1/sqrt(dmodel/768)
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    logger.info(f"  LR scaling factor (∝1/√(d/{768})): {dmodel_lr_scale:.6f}")

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

    logger.info(f"  Matrix params: {len(matrix_params)}")
    logger.info(f"  Embedding params: {len(embedding_params)}")
    logger.info(f"  LM head params: {len(lm_head_params)}")

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
    import gc

    model.eval()  # Set to eval mode (though MLX doesn't have dropout/batchnorm)

    total_loss = 0.0
    num_batches = 0

    val_iter = iter(val_loader)

    for _ in range(eval_steps):
        try:
            inputs, targets = next(val_iter)

            # Forward pass only (no gradients)
            loss = model(inputs, targets=targets)

            # Evaluate immediately to prevent graph buildup
            mx.eval(loss)
            total_loss += loss.item()
            num_batches += 1

            # Explicit cleanup of inputs/targets
            del inputs, targets, loss

        except StopIteration:
            break

    model.train()  # Set back to train mode

    # Force garbage collection after eval
    gc.collect()

    if num_batches == 0:
        return float("inf")

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


def main():
    args = get_args()

    # Apply low-memory mode settings
    if args.low_memory:
        print("=" * 80)
        print("LOW MEMORY MODE ENABLED")
        print("Applying aggressive memory optimization settings...")
        print("=" * 80)

        # Force minimal batch sizes
        if args.device_batch_size > 2:
            print(f"  Reducing device_batch_size: {args.device_batch_size} → 2")
            args.device_batch_size = 2

        if args.total_batch_size > 262144:
            print(f"  Reducing total_batch_size: {args.total_batch_size} → 262144")
            args.total_batch_size = 262144

        # Reduce cached shards
        if args.streaming and args.max_cached_shards > 5:
            print(f"  Reducing max_cached_shards: {args.max_cached_shards} → 5")
            args.max_cached_shards = 5

        # Less frequent evaluation
        if args.eval_every < 500:
            print(f"  Increasing eval_every: {args.eval_every} → 500")
            args.eval_every = 500

        # Reduce evaluation tokens
        if args.eval_tokens > 524288:
            print(f"  Reducing eval_tokens: {args.eval_tokens} → 524288")
            args.eval_tokens = 524288

        print("=" * 80)
        print()

    # Determine model tag early for logging setup
    if not args.model_tag:
        args.model_tag = f"d{args.depth}"

    # Setup logging directory
    if args.log_dir is None:
        base_dir = get_base_dir()
        args.log_dir = os.path.join(base_dir, "logs")

    # Initialize logging
    logger = setup_logging(args.log_dir, args.model_tag)

    logger.info("=" * 80)
    logger.info("MLXChat Training Starting")
    logger.info("=" * 80)

    # Load tokenizer
    tokenizer = get_tokenizer(args.tokenizer_dir)
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Vocab size: {vocab_size:,}")

    # Get model configuration
    config = get_model_config(args.depth, args.max_seq_len, vocab_size)
    logger.info("\nModel Configuration:")
    logger.info(f"  Layers: {config.n_layer}")
    logger.info(f"  Model dim: {config.n_embd}")
    logger.info(f"  Num heads: {config.n_head}")
    logger.info(f"  Num KV heads: {config.n_kv_head}")

    # Create model
    logger.info("\nInitializing model...")
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
    logger.info(f"  Number of parameters: {num_params:,}")

    # Calculate training horizon
    tokens_per_batch = args.device_batch_size * args.max_seq_len
    grad_accum_steps = args.total_batch_size // tokens_per_batch
    logger.info("\nBatch Configuration:")
    logger.info(f"  Device batch size: {args.device_batch_size}")
    logger.info(f"  Tokens per batch: {tokens_per_batch:,}")
    logger.info(f"  Total batch size: {args.total_batch_size:,}")
    logger.info(f"  Gradient accumulation steps: {grad_accum_steps}")

    if args.num_iterations > 0:
        num_iterations = args.num_iterations
        logger.info(f"\nUsing specified number of iterations: {num_iterations:,}")
    else:
        target_tokens = args.target_param_data_ratio * num_params
        num_iterations = int(target_tokens // args.total_batch_size)
        logger.info(f"\nCalculating iterations from data:param ratio {args.target_param_data_ratio}:")
        logger.info(f"  Target tokens: {target_tokens:,}")
        logger.info(f"  Number of iterations: {num_iterations:,}")

    total_tokens = args.total_batch_size * num_iterations
    logger.info(f"  Total training tokens: {total_tokens:,}")
    logger.info(f"  Tokens:Params ratio: {total_tokens / num_params:.2f}")

    # Setup checkpoint directory
    if args.output_dir is None:
        base_dir = get_base_dir()
        args.output_dir = os.path.join(base_dir, "base_checkpoints")

    # Determine model tag from depth if not provided
    if not args.model_tag:
        args.model_tag = f"d{args.depth}"

    checkpoint_dir = os.path.join(args.output_dir, args.model_tag)
    logger.info(f"\nCheckpoint directory: {checkpoint_dir}")
    logger.info(f"  Model tag: {args.model_tag}")
    logger.info(f"  Save every: {args.save_every} steps")

    # Setup optimizers
    logger.info("\nSetting up optimizers...")
    adam_opt, muon_opt, param_groups = setup_optimizers(model, config, args, logger)
    # param_groups is a dict with "adam" and "muon" keys
    adam_params = param_groups["adam"]
    muon_params = param_groups["muon"]

    # Setup dataloaders
    logger.info("\nSetting up dataloaders...")
    if args.streaming:
        logger.info(f"  Streaming mode enabled: max {args.max_cached_shards} shards cached")

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
    logger.info(f"  Validation steps per eval: {eval_steps}")

    # Training state
    smooth_train_loss = 0.0
    ema_beta = 0.9
    min_val_loss = float("inf")
    total_training_time = 0.0
    training_start_time = time.time()
    avg_step_time = None  # Running average of step time
    start_step = 0  # Starting step (0 for fresh training, or loaded from checkpoint)

    # Check for checkpoint resumption
    if args.resume and os.path.exists(checkpoint_dir):
        from mlxchat.checkpoint_manager import load_checkpoint, find_last_step

        try:
            last_step = find_last_step(checkpoint_dir)
            logger.info(f"\n{'='*80}")
            logger.info(f"Resuming from checkpoint at step {last_step}")
            logger.info(f"{'='*80}\n")

            # Load checkpoint
            model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, last_step, load_optimizer=True)

            # Restore model weights
            model.update(model_data)
            logger.info(f"  Loaded model weights from step {last_step}")

            # Restore optimizer states
            if optimizer_data:
                if "adam" in optimizer_data:
                    adam_opt.state = optimizer_data["adam"]
                    logger.info("  Loaded Adam optimizer state")
                if "muon" in optimizer_data:
                    muon_opt.state = optimizer_data["muon"]
                    logger.info("  Loaded Muon optimizer state")

            # Restore training state
            if "loss" in meta_data:
                smooth_train_loss = meta_data["loss"]

            # Start from next step
            start_step = last_step + 1
            logger.info(f"  Resuming training from step {start_step}\n")

        except Exception as e:
            logger.info(f"Warning: Failed to load checkpoint: {e}")
            logger.info("Starting fresh training from step 0\n")
            start_step = 0
    else:
        if args.resume:
            logger.info(f"No checkpoint found in {checkpoint_dir}, starting fresh training\n")

    logger.info(f"\n{'='*80}")
    logger.info("Starting Training")
    logger.info(f"{'='*80}\n")

    # Training loop
    train_iter = iter(train_loader)

    for step in range(start_step, num_iterations + 1):
        last_step = step == num_iterations

        # Training step (do this FIRST, before checkpoint/eval)
        if not last_step:
            t0 = time.time()

            # Profiling timers
            if args.profile:
                profile_times = {}
                import psutil

                process = psutil.Process()

            # Gradient accumulation
            from mlx.utils import tree_map

            # Create loss_and_grad function once (not per micro-step)
            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

            accum_grads = None
            total_loss = 0.0

            for micro_step in range(grad_accum_steps):
                if args.profile:
                    t_data_start = time.time()

                inputs, targets = next(train_iter)

                if args.profile:
                    t_data_end = time.time()
                    profile_times.setdefault("data_loading", []).append((t_data_end - t_data_start) * 1000)

                    t_fwd_bwd_start = time.time()

                # Forward and backward
                loss, grads = loss_and_grad_fn(model, inputs, targets)

                if args.profile:
                    t_fwd_bwd_end = time.time()
                    profile_times.setdefault("forward_backward", []).append((t_fwd_bwd_end - t_fwd_bwd_start) * 1000)

                    t_loss_eval_start = time.time()

                # Accumulate loss (forces evaluation of loss)
                total_loss += loss.item()

                if args.profile:
                    t_loss_eval_end = time.time()
                    profile_times.setdefault("loss_eval", []).append((t_loss_eval_end - t_loss_eval_start) * 1000)

                    t_grad_accum_start = time.time()

                # Accumulate gradients
                if accum_grads is not None:
                    accum_grads = tree_map(mx.add, grads, accum_grads)
                else:
                    accum_grads = grads

                if args.profile:
                    t_grad_accum_end = time.time()
                    profile_times.setdefault("grad_accumulation", []).append(
                        (t_grad_accum_end - t_grad_accum_start) * 1000
                    )

                # Cleanup micro-step intermediates including input tensors
                del loss, grads, inputs, targets

            # Evaluate accumulated gradients once after all micro-steps
            if args.profile:
                t_grad_eval_start = time.time()

            mx.eval(accum_grads)

            if args.profile:
                t_grad_eval_end = time.time()
                profile_times["grad_eval"] = (t_grad_eval_end - t_grad_eval_start) * 1000

            # Average loss and gradients over accumulation steps
            if args.profile:
                t_grad_avg_start = time.time()

            train_loss = total_loss / grad_accum_steps
            accum_grads = tree_map(lambda g: g / grad_accum_steps, accum_grads)

            if args.profile:
                t_grad_avg_end = time.time()
                profile_times["grad_averaging"] = (t_grad_avg_end - t_grad_avg_start) * 1000

            # Gradient clipping
            if args.grad_clip > 0.0:
                if args.profile:
                    t_clip_start = time.time()

                accum_grads, grad_norm = clip_gradients(accum_grads, args.grad_clip)

                if args.profile:
                    t_clip_end = time.time()
                    profile_times["grad_clipping"] = (t_clip_end - t_clip_start) * 1000

            # Learning rate scheduling
            if args.profile:
                t_lr_start = time.time()

            lr_mult = get_lr_multiplier(step, num_iterations)
            muon_momentum = get_muon_momentum(step)

            # Update Adam learning rate
            adam_opt.learning_rate = args.unembedding_lr * lr_mult * ((config.n_embd / 768) ** -0.5)

            # Update Muon learning rate and momentum
            muon_opt.learning_rate = args.matrix_lr * lr_mult
            muon_opt.momentum = muon_momentum

            if args.profile:
                t_lr_end = time.time()
                profile_times["lr_scheduling"] = (t_lr_end - t_lr_start) * 1000

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

            if args.profile:
                t_grad_split_start = time.time()

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

            if args.profile:
                t_grad_split_end = time.time()
                profile_times["grad_splitting"] = (t_grad_split_end - t_grad_split_start) * 1000

            # Update parameters with each optimizer
            # Note: optimizer.update expects (params_dict, grads_dict)
            if adam_grads:
                if args.profile:
                    t_adam_start = time.time()

                adam_params = {}
                if "wte" in adam_grads:
                    adam_params["wte"] = model.wte
                if "lm_head" in adam_grads:
                    adam_params["lm_head"] = model.lm_head
                adam_opt.update(adam_params, adam_grads)
                # Evaluate immediately for better incremental computation
                mx.eval(adam_params, adam_opt.state)

                if args.profile:
                    t_adam_end = time.time()
                    profile_times["adam_update"] = (t_adam_end - t_adam_start) * 1000

            if muon_grads:
                if args.profile:
                    t_muon_start = time.time()

                muon_params = {"h": model.h}
                muon_opt.update(muon_params, muon_grads)
                # Evaluate immediately for better incremental computation
                mx.eval(muon_params, muon_opt.state)

                if args.profile:
                    t_muon_end = time.time()
                    profile_times["muon_update"] = (t_muon_end - t_muon_start) * 1000

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

            # Explicit cleanup of training step intermediates
            del accum_grads, adam_grads, muon_grads

            # Force garbage collection to prevent memory buildup
            # More frequent in low-memory mode
            gc_interval = 5 if args.low_memory else 10
            if step % gc_interval == 0:
                import gc

                gc.collect()

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
            pct_done = 100 * step / num_iterations
            tok_per_sec = int(tokens_per_batch / dt)
            elapsed_min = total_training_time / 60
            logger.info(f"\n{'='*80}")
            logger.info(
                f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
                f"loss: {debiased_loss:.6f} | dt: {dt*1000:.2f}ms | "
                f"tok/sec: {tok_per_sec:,} | elapsed: {elapsed_min:.2f}m | eta: {eta_str}"
            )
            logger.info(f"{'='*80}")

            # Profiling output
            if args.profile:
                logger.info("\n--- PROFILING BREAKDOWN ---")

                # Get memory usage
                mem_info = process.memory_info()
                mem_gb = mem_info.rss / (1024**3)
                logger.info(f"Memory usage: {mem_gb:.2f} GB")

                # Compute and display timing breakdown
                total_accounted = 0.0

                # Display per-microstep timings (averaged)
                logger.info("\nPer-microstep timings (averaged over {} steps):".format(grad_accum_steps))
                for key in ["data_loading", "forward_backward", "loss_eval", "grad_accumulation"]:
                    if key in profile_times:
                        times = profile_times[key]
                        avg_time = sum(times) / len(times)
                        total_time = sum(times)
                        pct = (total_time / (dt * 1000)) * 100
                        logger.info(f"  {key:20s}: {avg_time:8.2f}ms avg | {total_time:8.2f}ms total ({pct:5.1f}%)")
                        total_accounted += total_time

                # Display single-time operations
                logger.info("\nSingle operations:")
                for key in [
                    "grad_eval",
                    "grad_averaging",
                    "grad_clipping",
                    "lr_scheduling",
                    "grad_splitting",
                    "adam_update",
                    "muon_update",
                ]:
                    if key in profile_times:
                        op_time = profile_times[key]
                        pct = (op_time / (dt * 1000)) * 100
                        logger.info(f"  {key:20s}: {op_time:8.2f}ms ({pct:5.1f}%)")
                        total_accounted += op_time

                # Compute unaccounted time
                unaccounted = (dt * 1000) - total_accounted
                pct_unaccounted = (unaccounted / (dt * 1000)) * 100
                logger.info(f"\n  {'Total accounted':20s}: {total_accounted:8.2f}ms ({100 - pct_unaccounted:5.1f}%)")
                logger.info(f"  {'Unaccounted':20s}: {unaccounted:8.2f}ms ({pct_unaccounted:5.1f}%)")
                logger.info(f"  {'Total step time':20s}: {dt*1000:8.2f}ms (100.0%)")
                logger.info("--- END PROFILING ---\n")

        # Checkpoint saving (after training step)
        if step > 0 and (step % args.save_every == 0 or last_step):
            # Prepare metadata
            meta_data = {
                "step": step,
                "loss": smooth_train_loss / (1 - ema_beta**step) if step > 0 else 0.0,
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
            logger.info(f"Checkpoint saved at step {step}")

            # Cleanup checkpoint data
            del optimizer_data, meta_data

        # Validation evaluation (after training step)
        if step > 0 and step % args.eval_every == 0:
            val_loss = evaluate(model, val_loader, eval_steps)
            logger.info(f"Validation loss: {val_loss:.6f}")

            # Update min validation loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                logger.info(f"New best validation loss: {val_loss:.6f}")

            # Force garbage collection after evaluation
            import gc

            gc.collect()

    logger.info(f"\n{'='*80}")
    logger.info("Training Complete")
    logger.info(f"  Total time: {total_training_time/60:.2f} minutes")
    logger.info(f"  Min validation loss: {min_val_loss:.4f}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
