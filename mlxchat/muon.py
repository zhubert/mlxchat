"""
Muon optimizer from Keller et al.
MLX port from nanochat (non-distributed version only)

https://kellerjordan.github.io/posts/muon/
"""

import mlx.core as mx
import mlx.optimizers


def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    We use a quintic iteration whose coefficients are selected to maximize the slope at zero.
    This iteration produces something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5),
    which does not hurt model performance relative to UV^T, where USV^T = G is the SVD.

    Args:
        G: Input tensor (must be at least 2D)
        steps: Number of Newton-Schulz iterations (default: 5)

    Returns:
        Orthogonalized matrix
    """
    assert G.ndim >= 2, "G must be at least 2D"

    # Quintic iteration coefficients
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G

    # If tall matrix, transpose it (we want wide or square)
    transposed = False
    if G.shape[-2] > G.shape[-1]:
        X = mx.transpose(X, list(range(X.ndim - 2)) + [-1, -2])
        transposed = True

    # Ensure spectral norm is at most 1
    norm = mx.sqrt(mx.sum(mx.square(X), axis=(-2, -1), keepdims=True))
    X = X / (norm + 1e-7)

    # Perform Newton-Schulz iterations
    for _ in range(steps):
        # A = X @ X^T
        A = X @ mx.transpose(X, list(range(X.ndim - 2)) + [-1, -2])
        # Quintic computation: B = b*A + c*A^2
        B = b * A + c * (A @ A)
        # Update: X = a*X + B @ X
        X = a * X + B @ X

    # Transpose back if needed
    if transposed:
        X = mx.transpose(X, list(range(X.ndim - 2)) + [-1, -2])

    return X


class Muon(mlx.optimizers.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration.

    Warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
      or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, flatten their last 3 dimensions.

    Args:
        learning_rate: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD (default: 0.95).
        nesterov: Whether to use Nesterov-style momentum (default: True, recommended).
        ns_steps: The number of Newton-Schulz iteration steps to use (default: 5).
    """

    def __init__(self, learning_rate: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        super().__init__()
        self._learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state for a single parameter"""
        state["momentum_buffer"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict) -> mx.array:
        """
        Apply Muon update to a single parameter.

        Args:
            gradient: The gradient for this parameter
            parameter: The parameter to update
            state: The optimizer state for this parameter

        Returns:
            The updated parameter
        """
        # Get or initialize momentum buffer
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = mx.zeros_like(gradient)

        buf = state["momentum_buffer"]

        # Update momentum buffer: buf = momentum * buf + (1 - momentum) * grad
        # Using lerp: buf.lerp_(g, 1 - momentum) means buf = buf + (g - buf) * (1 - momentum)
        buf = buf + (gradient - buf) * (1 - self.momentum)
        state["momentum_buffer"] = buf

        # Choose gradient for update (Nesterov or standard)
        if self.nesterov:
            # Nesterov: g = grad + momentum * buf  (equivalent to grad.lerp_(buf, momentum))
            g = gradient + (buf - gradient) * self.momentum
        else:
            g = buf

        # Orthogonalize the gradient
        g = zeropower_via_newtonschulz5(g, steps=self.ns_steps)

        # Apply aspect ratio scaling: scale by sqrt(max(1, rows/cols))
        if parameter.ndim >= 2:
            aspect_ratio_scale = mx.sqrt(max(1.0, parameter.shape[-2] / parameter.shape[-1]))
        else:
            aspect_ratio_scale = 1.0

        # Update parameter
        lr = self.learning_rate.item() if isinstance(self.learning_rate, mx.array) else self.learning_rate
        return parameter - lr * aspect_ratio_scale * g

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate
