"""Tests for GPT model components"""

import pytest
import mlx.core as mx

from mlxchat.gpt import (
    GPTConfig,
    norm,
    apply_rotary_emb,
    repeat_kv,
    CausalSelfAttention,
    MLP,
    Block,
    GPT,
)


def test_gpt_config():
    """Test GPTConfig dataclass"""
    config = GPTConfig()
    assert config.sequence_len == 1024
    assert config.vocab_size == 50304
    assert config.n_layer == 12
    assert config.n_head == 6
    assert config.n_kv_head == 6
    assert config.n_embd == 768

    # Test custom config
    config = GPTConfig(n_layer=16, n_embd=1024, n_head=8)
    assert config.n_layer == 16
    assert config.n_embd == 1024
    assert config.n_head == 8


def test_norm():
    """Test RMSNorm function"""
    # Test with simple input
    x = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = norm(x)

    # Check shape preserved
    assert y.shape == x.shape

    # Check that RMS is approximately 1 along last dimension
    rms = mx.sqrt(mx.mean(mx.square(y), axis=-1))
    assert mx.allclose(rms, mx.ones_like(rms), atol=1e-5)


def test_apply_rotary_emb():
    """Test rotary position embeddings"""
    # Create dummy input (B, T, H, D)
    B, T, H, D = 2, 4, 6, 64
    x = mx.random.normal((B, T, H, D))

    # Create cos/sin (1, T, 1, D//2)
    cos = mx.random.normal((1, T, 1, D // 2))
    sin = mx.random.normal((1, T, 1, D // 2))

    # Apply rotary embeddings
    y = apply_rotary_emb(x, cos, sin)

    # Check shape preserved
    assert y.shape == x.shape


def test_repeat_kv():
    """Test key/value head replication for MQA"""
    # Test with no repetition
    x = mx.random.normal((2, 2, 8, 64))  # (B, n_kv_head, T, D)
    y = repeat_kv(x, n_rep=1)
    assert y.shape == x.shape

    # Test with repetition
    y = repeat_kv(x, n_rep=3)
    expected_shape = (2, 6, 8, 64)  # n_kv_head * n_rep = 2 * 3 = 6
    assert y.shape == expected_shape


def test_causal_self_attention():
    """Test CausalSelfAttention module"""
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=2, n_embd=128)
    attn = CausalSelfAttention(config, layer_idx=0)

    # Create input
    B, T = 2, 8
    x = mx.random.normal((B, T, config.n_embd))

    # Create rotary embeddings
    head_dim = config.n_embd // config.n_head
    cos = mx.random.normal((1, T, 1, head_dim // 2))
    sin = mx.random.normal((1, T, 1, head_dim // 2))
    cos_sin = (cos, sin)

    # Forward pass
    y = attn(x, cos_sin, kv_cache=None)

    # Check output shape
    assert y.shape == (B, T, config.n_embd)


def test_mlp():
    """Test MLP module"""
    config = GPTConfig(n_embd=128)
    mlp = MLP(config)

    # Create input
    B, T = 2, 8
    x = mx.random.normal((B, T, config.n_embd))

    # Forward pass
    y = mlp(x)

    # Check output shape
    assert y.shape == x.shape


def test_block():
    """Test transformer Block"""
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=2, n_embd=128)
    block = Block(config, layer_idx=0)

    # Create input
    B, T = 2, 8
    x = mx.random.normal((B, T, config.n_embd))

    # Create rotary embeddings
    head_dim = config.n_embd // config.n_head
    cos = mx.random.normal((1, T, 1, head_dim // 2))
    sin = mx.random.normal((1, T, 1, head_dim // 2))
    cos_sin = (cos, sin)

    # Forward pass
    y = block(x, cos_sin, kv_cache=None)

    # Check output shape
    assert y.shape == x.shape


def test_gpt_model():
    """Test full GPT model"""
    # Small config for testing
    config = GPTConfig(
        sequence_len=64,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
    )
    model = GPT(config)

    # Check model components exist
    assert model.wte is not None
    assert len(model.h) == config.n_layer
    assert model.lm_head is not None
    assert model.cos is not None
    assert model.sin is not None


def test_gpt_forward_inference():
    """Test GPT forward pass in inference mode"""
    config = GPTConfig(
        sequence_len=64,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
    )
    model = GPT(config)

    # Create input tokens
    B, T = 2, 8
    idx = mx.random.randint(0, config.vocab_size, (B, T))

    # Forward pass (inference mode)
    logits = model(idx, targets=None, kv_cache=None)

    # Check output shape
    expected_shape = (B, T, config.vocab_size)
    assert logits.shape == expected_shape


def test_gpt_forward_training():
    """Test GPT forward pass in training mode"""
    config = GPTConfig(
        sequence_len=64,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
    )
    model = GPT(config)

    # Create input tokens and targets
    B, T = 2, 8
    idx = mx.random.randint(0, config.vocab_size, (B, T))
    targets = mx.random.randint(0, config.vocab_size, (B, T))

    # Forward pass (training mode)
    loss = model(idx, targets=targets, kv_cache=None)

    # Check loss is a scalar
    assert loss.shape == ()
    # Check loss is positive
    assert loss.item() > 0


def test_gpt_init_weights():
    """Test GPT weight initialization"""
    config = GPTConfig(
        sequence_len=64,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
    )
    model = GPT(config)
    model.init_weights()

    # Check that lm_head is zeroed
    assert mx.allclose(model.lm_head.weight, mx.zeros_like(model.lm_head.weight))

    # Check that c_proj weights are zeroed in blocks
    for block in model.h:
        assert mx.allclose(block.mlp.c_proj.weight, mx.zeros_like(block.mlp.c_proj.weight))
        assert mx.allclose(block.attn.c_proj.weight, mx.zeros_like(block.attn.c_proj.weight))


def test_rotary_embeddings():
    """Test rotary embedding precomputation"""
    config = GPTConfig(n_embd=128, n_head=4, n_kv_head=4)
    model = GPT(config)

    # Check rotary embeddings shape
    head_dim = config.n_embd // config.n_head
    expected_shape = (1, model.rotary_seq_len, 1, head_dim // 2)
    assert model.cos.shape == expected_shape
    assert model.sin.shape == expected_shape


def test_different_sequence_lengths():
    """Test model with different sequence lengths"""
    config = GPTConfig(
        sequence_len=64,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
    )
    model = GPT(config)

    # Test with various sequence lengths
    for T in [1, 4, 16, 32, 64]:
        B = 2
        idx = mx.random.randint(0, config.vocab_size, (B, T))
        logits = model(idx, targets=None, kv_cache=None)
        assert logits.shape == (B, T, config.vocab_size)


def test_mqa_configuration():
    """Test Multi-Query Attention with different n_kv_head"""
    # Standard MHA (n_kv_head == n_head)
    config = GPTConfig(n_head=8, n_kv_head=8, n_embd=256, n_layer=2)
    model = GPT(config)
    B, T = 2, 8
    idx = mx.random.randint(0, config.vocab_size, (B, T))
    logits = model(idx)
    assert logits.shape == (B, T, config.vocab_size)

    # MQA (n_kv_head < n_head)
    config = GPTConfig(n_head=8, n_kv_head=2, n_embd=256, n_layer=2)
    model = GPT(config)
    logits = model(idx)
    assert logits.shape == (B, T, config.vocab_size)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
