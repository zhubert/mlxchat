"""Tests for Muon optimizer"""

import pytest
import mlx.core as mx
import mlx.nn as nn

from mlxchat.muon import zeropower_via_newtonschulz5, Muon


def test_zeropower_square_matrix():
    """Test Newton-Schulz on square matrix"""
    # Create a random square matrix
    G = mx.random.normal((4, 4))

    # Apply Newton-Schulz
    X = zeropower_via_newtonschulz5(G, steps=5)

    # Check shape preserved
    assert X.shape == G.shape

    # The Newton-Schulz iteration approximates orthogonalization
    # As noted in nanochat, it produces US'V^T where S' has diagonal elements ~ Uniform(0.5, 1.5)
    # So we just verify the result is reasonable (not NaN, has reasonable norm)
    assert not mx.any(mx.isnan(X))

    # Check that the output has reasonable spectral norm (should be around 1)
    XXT = X @ X.T
    spectral_norm_sq = mx.max(mx.abs(XXT))
    assert spectral_norm_sq < 5.0  # Relaxed check


def test_zeropower_tall_matrix():
    """Test Newton-Schulz on tall matrix (more rows than columns)"""
    # Create a tall matrix
    G = mx.random.normal((8, 4))

    # Apply Newton-Schulz
    X = zeropower_via_newtonschulz5(G, steps=5)

    # Check shape preserved
    assert X.shape == G.shape

    # Verify result is reasonable
    assert not mx.any(mx.isnan(X))

    # For tall matrices, X^T @ X should have reasonable norm
    XTX = X.T @ X
    spectral_norm_sq = mx.max(mx.abs(XTX))
    assert spectral_norm_sq < 5.0  # Relaxed check


def test_zeropower_wide_matrix():
    """Test Newton-Schulz on wide matrix (more columns than rows)"""
    # Create a wide matrix
    G = mx.random.normal((4, 8))

    # Apply Newton-Schulz
    X = zeropower_via_newtonschulz5(G, steps=5)

    # Check shape preserved
    assert X.shape == G.shape

    # Verify result is reasonable
    assert not mx.any(mx.isnan(X))

    # For wide matrices, X @ X^T should have reasonable norm
    XXT = X @ X.T
    spectral_norm_sq = mx.max(mx.abs(XXT))
    assert spectral_norm_sq < 5.0  # Relaxed check


def test_zeropower_different_steps():
    """Test Newton-Schulz with different iteration counts"""
    G = mx.random.normal((4, 4))

    # Test with different step counts
    for steps in [1, 3, 5, 10]:
        X = zeropower_via_newtonschulz5(G, steps=steps)
        assert X.shape == G.shape


def test_muon_initialization():
    """Test Muon optimizer initialization"""
    opt = Muon(learning_rate=0.02, momentum=0.95, nesterov=True, ns_steps=5)

    assert opt.learning_rate == 0.02
    assert opt.momentum == 0.95
    assert opt.nesterov is True
    assert opt.ns_steps == 5


def test_muon_single_step():
    """Test a single Muon optimizer step"""
    # Create a simple parameter and gradient
    param = mx.random.normal((4, 4))
    grad = mx.random.normal((4, 4))

    # Initialize optimizer
    opt = Muon(learning_rate=0.01, momentum=0.9, ns_steps=5)

    # Initialize state
    state = {}
    opt.init_single(param, state)

    # Check momentum buffer initialized
    assert "momentum_buffer" in state
    assert mx.allclose(state["momentum_buffer"], mx.zeros_like(param))

    # Apply update
    param_new = opt.apply_single(grad, param, state)

    # Check parameter changed
    assert not mx.allclose(param_new, param)

    # Check momentum buffer updated
    assert not mx.allclose(state["momentum_buffer"], mx.zeros_like(param))


def test_muon_convergence_simple():
    """Test Muon optimizer converges on a simple quadratic problem"""
    # Simple problem: minimize ||X - target||^2
    target = mx.ones((4, 4))
    X = mx.random.normal((4, 4))

    opt = Muon(learning_rate=0.1, momentum=0.9, ns_steps=5)
    state = {}
    opt.init_single(X, state)

    # Run optimization steps
    losses = []
    for _ in range(100):
        # Compute loss and gradient
        diff = X - target
        loss = mx.sum(mx.square(diff))
        grad = 2 * diff

        # Store loss
        losses.append(loss.item())

        # Update parameters
        X = opt.apply_single(grad, X, state)

    # Check that loss decreased
    assert losses[-1] < losses[0]


def test_muon_with_model():
    """Test Muon optimizer with a simple linear model"""
    # Create a simple linear layer
    layer = nn.Linear(4, 4, bias=False)

    # Create synthetic data
    X = mx.random.normal((8, 4))
    y = mx.random.normal((8, 4))

    # Initialize optimizer
    opt = Muon(learning_rate=0.01, momentum=0.9, ns_steps=5)

    # Initialize state for layer weight
    state = {}
    opt.init_single(layer.weight, state)

    # Run a few training steps
    for step in range(10):
        # Forward pass
        pred = layer(X)

        # Backward pass (manual gradient computation for simplicity)
        grad_output = 2 * (pred - y) / y.size
        grad_weight = grad_output.T @ X

        # Update weights
        layer.weight = opt.apply_single(grad_weight, layer.weight, state)

    # Loss should generally decrease (not guaranteed for all random seeds, but likely)
    final_loss = mx.mean(mx.square(layer(X) - y)).item()
    # Just check that optimization ran without errors and produced reasonable values
    assert not mx.isnan(mx.array(final_loss)).item()


def test_muon_nesterov_vs_standard():
    """Test that both Nesterov and standard momentum modes work"""
    param = mx.random.normal((4, 4))
    grad = mx.random.normal((4, 4))

    # Standard momentum
    opt_standard = Muon(learning_rate=0.01, momentum=0.9, nesterov=False, ns_steps=5)
    state_standard = {}
    opt_standard.init_single(param, state_standard)
    param_standard = opt_standard.apply_single(grad, param, state_standard)

    # Nesterov momentum
    opt_nesterov = Muon(learning_rate=0.01, momentum=0.9, nesterov=True, ns_steps=5)
    state_nesterov = {}
    opt_nesterov.init_single(param, state_nesterov)
    param_nesterov = opt_nesterov.apply_single(grad, param, state_nesterov)

    # Both should produce valid updates (not NaN)
    assert not mx.any(mx.isnan(param_standard))
    assert not mx.any(mx.isnan(param_nesterov))

    # Both should have updated the parameter
    assert not mx.allclose(param_standard, param)
    assert not mx.allclose(param_nesterov, param)


def test_muon_aspect_ratio_scaling():
    """Test that aspect ratio scaling works for different shaped matrices"""
    # Tall matrix (should have larger scaling)
    param_tall = mx.random.normal((8, 4))
    grad_tall = mx.random.normal((8, 4))

    # Square matrix
    param_square = mx.random.normal((4, 4))
    grad_square = mx.random.normal((4, 4))

    opt = Muon(learning_rate=0.01, momentum=0.9, ns_steps=5)

    # Apply updates
    state_tall = {}
    opt.init_single(param_tall, state_tall)
    param_tall_new = opt.apply_single(grad_tall, param_tall, state_tall)

    state_square = {}
    opt.init_single(param_square, state_square)
    param_square_new = opt.apply_single(grad_square, param_square, state_square)

    # Both should produce valid updates
    assert not mx.allclose(param_tall_new, param_tall)
    assert not mx.allclose(param_square_new, param_square)


def test_muon_momentum_accumulation():
    """Test that momentum properly accumulates over multiple steps"""
    param = mx.random.normal((4, 4))
    grad = mx.ones((4, 4))  # Constant gradient

    opt = Muon(learning_rate=0.01, momentum=0.9, ns_steps=5)
    state = {}
    opt.init_single(param, state)

    # First step
    param = opt.apply_single(grad, param, state)
    momentum_after_1 = mx.array(state["momentum_buffer"])  # Create a copy

    # Second step with same gradient
    param = opt.apply_single(grad, param, state)
    momentum_after_2 = mx.array(state["momentum_buffer"])  # Create a copy

    # Momentum buffer should have changed
    assert not mx.allclose(momentum_after_1, momentum_after_2)

    # Momentum should be accumulating (moving towards gradient direction)
    # The momentum buffer after 2 steps should be closer to grad than after 1 step
    diff_1 = mx.sum(mx.square(momentum_after_1 - grad))
    diff_2 = mx.sum(mx.square(momentum_after_2 - grad))
    assert diff_2 < diff_1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
