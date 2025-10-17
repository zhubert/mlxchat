"""Minimal test to verify gradients work in MLX."""

import mlx.core as mx
import mlx.nn as nn


# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def __call__(self, x, targets=None):
        out = self.linear(x)
        if targets is not None:
            loss = mx.mean((out - targets) ** 2)
            return loss
        return out


# Create model
model = SimpleModel()

# Create data
x = mx.random.normal((4, 10))
targets = mx.random.normal((4, 1))


# Loss function
def loss_fn(model, x, targets):
    return model(x, targets=targets)


# Compute gradients
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
loss, grads = loss_and_grad_fn(model, x, targets)

print(f"Loss: {loss.item()}")
print(f"Gradient keys: {list(grads.keys())}")
print(f"Linear weight grad shape: {grads['linear']['weight'].shape}")
print(f"Linear weight grad norm: {mx.sqrt(mx.sum(mx.square(grads['linear']['weight']))).item()}")

# Update model
opt = mx.optimizers.Adam(learning_rate=0.01)
opt.update(model, grads)
mx.eval(model.parameters())

# Check if parameters changed
print(f"\nParameter updated: {opt.state}")
