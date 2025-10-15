"""
Test training loop with minimal setup

This script tests the training loop mechanics without requiring
the full nanochat tokenizer and data setup.
"""

import os
import sys
import time
import tempfile
import pickle
import shutil

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import tiktoken

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlxchat.gpt import GPT, GPTConfig
from mlxchat.muon import Muon
from mlxchat.tokenizer import Tokenizer
from mlxchat.checkpoint_manager import save_checkpoint, load_checkpoint


def create_temp_tokenizer():
    """Create a temporary GPT-2 tokenizer with BOS token."""
    # Get GPT-2 encoding
    gpt2_enc = tiktoken.get_encoding("gpt2")

    # Add custom BOS token
    enc = tiktoken.Encoding(
        name="gpt2_with_bos",
        pat_str=gpt2_enc._pat_str,
        mergeable_ranks=gpt2_enc._mergeable_ranks,
        special_tokens={
            **gpt2_enc._special_tokens,
            "<|bos|>": gpt2_enc.n_vocab,
        },
    )

    # Save to temp directory
    temp_dir = tempfile.mkdtemp()
    tok_path = os.path.join(temp_dir, "tokenizer.pkl")

    with open(tok_path, "wb") as f:
        pickle.dump(enc, f)

    return Tokenizer.from_directory(temp_dir), temp_dir


def generate_dummy_data(batch_size, seq_len, vocab_size, num_batches=10):
    """Generate random training data."""
    for _ in range(num_batches):
        inputs = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        targets = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        yield inputs, targets


def clip_gradients(grads, max_norm):
    """Clip gradients by global norm."""
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


def loss_fn(model, inputs, targets):
    """Compute loss for a batch."""
    return model(inputs, targets=targets)


def main():
    print("=" * 80)
    print("MLXChat Training Loop Test")
    print("=" * 80)

    # Create temporary tokenizer
    print("\nCreating temporary tokenizer...")
    tokenizer, temp_dir = create_temp_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    # Small model config
    config = GPTConfig(
        sequence_len=128,
        vocab_size=vocab_size,
        n_layer=4,  # Very small for testing
        n_head=4,
        n_kv_head=4,
        n_embd=256,
    )

    print(f"\nModel Configuration:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Model dim: {config.n_embd}")
    print(f"  Num heads: {config.n_head}")

    # Create model
    print(f"\nInitializing model...")
    model = GPT(config)
    model.init_weights()

    # Count parameters
    from mlx.utils import tree_flatten
    param_list = tree_flatten(model.parameters())
    # tree_flatten returns list of (path, value) tuples
    num_params = sum(v.size for _, v in param_list)
    print(f"Number of parameters: {num_params:,}")

    # Setup optimizers
    print(f"\nSetting up optimizers...")

    # Create Adam optimizer for embeddings and LM head
    adam_optimizer = optim.Adam(learning_rate=0.001)

    # Create Muon optimizer for transformer blocks
    muon_optimizer = Muon(
        learning_rate=0.01,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    )

    print("Optimizers created")

    # Training parameters
    batch_size = 2
    seq_len = 128
    grad_accum_steps = 2
    num_iterations = 5
    grad_clip = 1.0

    print(f"\nTraining Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Gradient accumulation steps: {grad_accum_steps}")
    print(f"  Number of iterations: {num_iterations}")
    print(f"  Gradient clipping: {grad_clip}")

    # Generate dummy data
    data_iter = generate_dummy_data(batch_size, seq_len, vocab_size, num_batches=num_iterations * grad_accum_steps)

    print(f"\n{'='*80}")
    print("Starting Training")
    print(f"{'='*80}\n")

    # Training loop
    from mlx.utils import tree_map

    for step in range(num_iterations):
        t0 = time.time()

        # Gradient accumulation
        accum_grads = None
        total_loss = 0.0

        for micro_step in range(grad_accum_steps):
            inputs, targets = next(data_iter)

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
        if grad_clip > 0.0:
            accum_grads, grad_norm = clip_gradients(accum_grads, grad_clip)
            grad_norm_val = grad_norm.item()
        else:
            grad_norm_val = 0.0

        # Split gradients for each optimizer
        adam_grads = {}
        if "wte" in accum_grads:
            adam_grads["wte"] = accum_grads["wte"]
        if "lm_head" in accum_grads:
            adam_grads["lm_head"] = accum_grads["lm_head"]

        muon_grads = {}
        if "h" in accum_grads:
            muon_grads["h"] = accum_grads["h"]

        # Update parameters with each optimizer
        if adam_grads:
            adam_params = {}
            if "wte" in adam_grads:
                adam_params["wte"] = model.wte
            if "lm_head" in adam_grads:
                adam_params["lm_head"] = model.lm_head
            adam_optimizer.update(adam_params, adam_grads)

        if muon_grads:
            muon_params = {"h": model.h}
            muon_optimizer.update(muon_params, muon_grads)

        # Evaluate model and optimizer states
        mx.eval(model.parameters(), adam_optimizer.state, muon_optimizer.state)

        t1 = time.time()
        dt = t1 - t0

        # Logging
        tokens_per_batch = batch_size * seq_len * grad_accum_steps
        tok_per_sec = int(tokens_per_batch / dt)

        print(f"step {step:02d}/{num_iterations:02d} | "
              f"loss: {train_loss:.6f} | "
              f"grad_norm: {grad_norm_val:.4f} | "
              f"dt: {dt*1000:.2f}ms | "
              f"tok/sec: {tok_per_sec:,}")

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")

    # Test checkpoint saving
    print(f"\n{'='*80}")
    print("Testing Checkpoint Saving")
    print(f"{'='*80}\n")

    checkpoint_dir = tempfile.mkdtemp()
    try:
        # Save checkpoint
        print(f"Saving checkpoint...")
        meta_data = {
            "step": num_iterations,
            "loss": train_loss,
            "model_config": {
                "sequence_len": config.sequence_len,
                "vocab_size": config.vocab_size,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_kv_head": config.n_kv_head,
                "n_embd": config.n_embd,
            },
        }

        optimizer_data = {
            "adam": adam_optimizer.state,
            "muon": muon_optimizer.state,
        }

        save_checkpoint(checkpoint_dir, num_iterations, model, optimizer_data, meta_data)
        print(f"✓ Checkpoint saved to {checkpoint_dir}")

        # Load checkpoint
        print(f"\nLoading checkpoint...")
        loaded_model_data, loaded_optim_data, loaded_meta = load_checkpoint(
            checkpoint_dir, num_iterations, load_optimizer=True
        )
        print(f"✓ Checkpoint loaded successfully")

        # Verify metadata
        assert loaded_meta["step"] == num_iterations
        assert "model_config" in loaded_meta
        print(f"✓ Metadata verified (step={loaded_meta['step']})")

        # Verify model parameters
        assert "wte" in loaded_model_data
        assert "h" in loaded_model_data
        assert "lm_head" in loaded_model_data
        print(f"✓ Model data verified")

        # Verify optimizer states
        assert loaded_optim_data is not None
        assert "adam" in loaded_optim_data
        assert "muon" in loaded_optim_data
        print(f"✓ Optimizer data verified")

        print(f"\n{'='*80}")
        print("All Checkpoint Tests Passed!")
        print(f"{'='*80}")

    finally:
        # Cleanup checkpoint dir
        shutil.rmtree(checkpoint_dir)

    # Cleanup tokenizer
    shutil.rmtree(temp_dir)
    print(f"\nCleaned up temporary files")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
