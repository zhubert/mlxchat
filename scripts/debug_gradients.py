"""Debug why gradients are all zeros."""

import mlx.core as mx
import mlx.nn as nn
from mlxchat.gpt import GPT, GPTConfig

# Tiny model
config = GPTConfig(
    sequence_len=512,
    vocab_size=65536,
    n_layer=1,
    n_head=2,
    n_kv_head=2,
    n_embd=128,
)

print("Creating model...")
model = GPT(config)
model.init_weights()

print(f"Trainable params: {list(model.trainable_parameters().keys())}")

# Tiny batch
inputs = mx.random.randint(0, 65536, (1, 8))
targets = mx.random.randint(0, 65536, (1, 8))

print(f"\nInputs shape: {inputs.shape}")
print(f"Targets shape: {targets.shape}")


# Loss function
def loss_fn(model, inputs, targets):
    return model(inputs, targets=targets)


print("\n" + "=" * 80)
print("Computing gradients...")
print("=" * 80)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
loss, grads = loss_and_grad_fn(model, inputs, targets)

# Force evaluation
mx.eval(loss, grads)

print(f"\nLoss: {loss.item()}")
print(f"Gradient keys: {list(grads.keys())}")

# Check lm_head gradients
if "lm_head" in grads:
    lm_head_grad = grads["lm_head"]["weight"]
    print("\nlm_head.weight gradient:")
    print(f"  Shape: {lm_head_grad.shape}")
    print(f"  Mean: {mx.mean(mx.abs(lm_head_grad)).item()}")
    print(f"  Max: {mx.max(mx.abs(lm_head_grad)).item()}")

if "wte" in grads:
    wte_grad = grads["wte"]["weight"]
    print("\nwte.weight gradient:")
    print(f"  Shape: {wte_grad.shape}")
    print(f"  Mean: {mx.mean(mx.abs(wte_grad)).item()}")
    print(f"  Max: {mx.max(mx.abs(wte_grad)).item()}")

print("\n" + "=" * 80)
print("Testing if loss changes with same inputs...")
print("=" * 80)

# Compute loss again
loss2, grads2 = loss_and_grad_fn(model, inputs, targets)
mx.eval(loss2, grads2)

print(f"Loss 1: {loss.item()}")
print(f"Loss 2: {loss2.item()}")
print(f"Difference: {abs(loss.item() - loss2.item())}")

if abs(loss.item() - loss2.item()) > 1e-6:
    print("\n⚠️  BUG: Loss changed despite same inputs!")
else:
    print("\n✓ Loss is identical")
