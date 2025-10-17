"""
Diagnostic script to identify training issues:
1. Check if model parameters are being updated
2. Profile each part of the training step
3. Check for gradient issues

IMPORTANT: This script uses the FIXED optimizer.update() pattern from commit 179f30f.
The correct pattern is: optimizer.update(model, filtered_grads)
NOT: optimizer.update(params_dict, grads)
"""

import time
import mlx.core as mx
import mlx.nn as nn
from mlxchat.gpt import GPT, GPTConfig
from mlxchat.muon import Muon
import mlx.optimizers as optim
from mlxchat.dataloader import DataLoader

# Create a TINY test model for faster diagnosis
config = GPTConfig(
    sequence_len=512,  # Smaller sequence
    vocab_size=65536,
    n_layer=2,  # Much smaller model
    n_head=2,
    n_kv_head=2,
    n_embd=128,
)

print("Initializing model...")
model = GPT(config)
model.init_weights()

# Freeze cos/sin since they're not trainable (rotary embedding cache)
print("Freezing non-trainable parameters (cos, sin)...")
model.freeze(keys=["cos", "sin"])

# Verify what's trainable
trainable_params = model.trainable_parameters()
print(f"Trainable parameter keys: {list(trainable_params.keys())}")


# Count params
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
print(f"Parameters: {num_params:,}")

# Create optimizers
adam_opt = optim.Adam(learning_rate=0.004)
muon_opt = Muon(learning_rate=0.02, momentum=0.95, nesterov=True, ns_steps=5)


# Loss function (same as base_train.py)
def loss_fn(model, inputs, targets):
    return model(inputs, targets=targets)


# Create dummy data
batch_size = 2
seq_len = 512  # Match model config
inputs = mx.random.randint(0, 65536, (batch_size, seq_len))
targets = mx.random.randint(0, 65536, (batch_size, seq_len))

print("\nTest configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {seq_len}")
print(f"  Input shape: {inputs.shape}")
print(f"  Target shape: {targets.shape}")

# Get initial parameter values
print("\n" + "=" * 80)
print("TEST 1: Check if parameters update")
print("=" * 80)

# Sample a few parameters to track
wte_initial = model.wte["weight"][0, 0].item()
h0_attn_q_initial = model.h[0]["attn"]["c_q"]["weight"][0, 0].item()

print(f"Initial wte[0,0]: {wte_initial}")
print(f"Initial h[0].attn.c_q[0,0]: {h0_attn_q_initial}")

# Run training step
print("\nRunning training step...")
t0 = time.time()

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
loss, grads = loss_and_grad_fn(model, inputs, targets)

t1 = time.time()
print(f"Forward + backward: {(t1-t0)*1000:.2f}ms")
print(f"Loss: {loss.item()}")

# Evaluate gradients
t0 = time.time()
mx.eval(grads)
t1 = time.time()
print(f"Gradient evaluation: {(t1-t0)*1000:.2f}ms")

# Check gradient values
print("\nChecking gradients...")
print(f"Gradient keys: {list(grads.keys())}")

# Check if grads contain the expected keys
if "wte" in grads:
    wte_grad_norm = mx.sqrt(mx.sum(mx.square(grads["wte"]["weight"]))).item()
    print(f"wte gradient norm: {wte_grad_norm}")
    print(f"wte gradient shape: {grads['wte']['weight'].shape}")
    print(f"wte gradient mean: {mx.mean(mx.abs(grads['wte']['weight'])).item()}")
else:
    print("WARNING: 'wte' not in gradients!")

if "h" in grads:
    h0_q_grad_norm = mx.sqrt(mx.sum(mx.square(grads["h"][0]["attn"]["c_q"]["weight"]))).item()
    print(f"h[0].attn.c_q gradient norm: {h0_q_grad_norm}")
else:
    print("WARNING: 'h' not in gradients!")


# Check if gradients have NaN or Inf
def check_grads(grad_tree, prefix=""):
    """Check for NaN/Inf in gradients."""
    issues = []
    if isinstance(grad_tree, dict):
        for k, v in grad_tree.items():
            issues.extend(check_grads(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(grad_tree, list):
        for i, v in enumerate(grad_tree):
            issues.extend(check_grads(v, f"{prefix}[{i}]"))
    elif isinstance(grad_tree, mx.array):
        if mx.any(mx.isnan(grad_tree)).item():
            issues.append(f"{prefix}: contains NaN")
        if mx.any(mx.isinf(grad_tree)).item():
            issues.append(f"{prefix}: contains Inf")
        grad_mean = mx.mean(mx.abs(grad_tree)).item()
        if grad_mean == 0:
            issues.append(f"{prefix}: all zeros (mean={grad_mean})")
    return issues


issues = check_grads(grads)
if issues:
    print("\nGradient issues found:")
    for issue in issues[:10]:  # Show first 10
        print(f"  {issue}")
else:
    print("Gradients look OK (no NaN, Inf, or all-zero)")

# Update with optimizers
print("\nUpdating parameters...")
t0 = time.time()

# Adam - pass model with filtered gradients
adam_grads = {"wte": grads["wte"], "lm_head": grads["lm_head"]}
adam_opt.update(model, adam_grads)
mx.eval(model.parameters(), adam_opt.state)

# Muon - pass model with filtered gradients
muon_grads = {"h": grads["h"]}
muon_opt.update(model, muon_grads)
mx.eval(model.parameters(), muon_opt.state)

t1 = time.time()
print(f"Parameter updates: {(t1-t0)*1000:.2f}ms")

# Check if parameters changed
wte_after = model.wte["weight"][0, 0].item()
h0_attn_q_after = model.h[0]["attn"]["c_q"]["weight"][0, 0].item()

print("\nAfter update:")
print(f"  wte[0,0]: {wte_initial} → {wte_after} (delta: {wte_after - wte_initial})")
print(f"  h[0].attn.c_q[0,0]: {h0_attn_q_initial} → {h0_attn_q_after} (delta: {h0_attn_q_after - h0_attn_q_initial})")

if abs(wte_after - wte_initial) < 1e-10 and abs(h0_attn_q_after - h0_attn_q_initial) < 1e-10:
    print("\n⚠️  WARNING: Parameters did not change!")
else:
    print("\n✓ Parameters updated successfully")

# Test with second iteration to see if loss changes
print("\n" + "=" * 80)
print("TEST 2: Verify loss is SAME with same inputs (since params didn't update)")
print("=" * 80)

# Use SAME inputs - loss should be identical since parameters didn't change
loss2, grads2 = loss_and_grad_fn(model, inputs, targets)
print(f"Loss (iteration 1): {loss.item()}")
print(f"Loss (iteration 2, same inputs): {loss2.item()}")
print(f"Loss change: {loss2.item() - loss.item()}")

if abs(loss2.item() - loss.item()) < 1e-6:
    print("✓ Loss is identical (expected since params didn't update)")
else:
    print(f"⚠️  WARNING: Loss changed by {abs(loss2.item() - loss.item()):.6f} despite no param updates!")
    print("This suggests either:")
    print("  1. Parameters ARE updating (but we're not tracking the right ones)")
    print("  2. There's randomness in the forward pass")
    print("  3. MLX is doing something unexpected with lazy evaluation")

# Test data loading
print("\n" + "=" * 80)
print("TEST 3: Profile data loading")
print("=" * 80)

loader = DataLoader(
    batch_size=4,
    sequence_length=2048,
    split="train",
    streaming=True,
    max_cached_shards=5,
)

print("Loading first batch...")
t0 = time.time()
batch1 = next(iter(loader))
t1 = time.time()
print(f"First batch load time: {(t1-t0)*1000:.2f}ms")

print("\nLoading second batch...")
t0 = time.time()
batch2 = next(iter(loader))
t1 = time.time()
print(f"Second batch load time: {(t1-t0)*1000:.2f}ms")

# Check if batches are the same
if mx.array_equal(batch1[0], batch2[0]).item():
    print("⚠️  WARNING: Batches are identical! Data loader may be broken")
else:
    print("✓ Batches are different")

print("\n" + "=" * 80)
print("Diagnosis complete")
print("=" * 80)
