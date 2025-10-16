"""
GPT model (MLX port from nanochat)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Multi-Query Attention (MQA) support for more efficient inference
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (MQA)
    n_embd: int = 768


def norm(x):
    """Purely functional rmsnorm with no learnable params"""
    # RMSNorm: x / sqrt(mean(x^2) + eps)
    # MLX doesn't have F.rms_norm, so we implement it manually
    eps = 1e-5
    rms = mx.sqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)
    return x / rms


def apply_rotary_emb(x, cos, sin):
    """Apply rotary position embeddings to x"""
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = mx.concatenate([y1, y2], axis=3)  # re-assemble
    return out


def repeat_kv(x, n_rep):
    """Repeat key/value heads for multi-query attention
    Equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep)
    """
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    # Expand and reshape to replicate each KV head n_rep times
    x = mx.expand_dims(x, axis=2)  # (bs, n_kv_heads, 1, slen, head_dim)
    x = mx.broadcast_to(x, (bs, n_kv_heads, n_rep, slen, head_dim))
    x = mx.reshape(x, (bs, n_kv_heads * n_rep, slen, head_dim))
    return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def __call__(self, x, cos_sin, kv_cache=None):
        B, T, C = x.shape

        # Project to queries, keys, and values
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK norm
        q = norm(q)
        k = norm(k)

        # Transpose to (B, H, T, D) format for attention
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Apply KV cache if provided
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        Tq = q.shape[2]  # number of queries
        Tk = k.shape[2]  # number of keys/values

        # Apply MQA: replicate key/value heads for each query head
        nrep = self.n_head // self.n_kv_head
        k = repeat_kv(k, nrep)
        v = repeat_kv(v, nrep)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * scale  # (B, H, Tq, Tk)

        # Apply causal mask
        if kv_cache is None or Tq == Tk:
            # Training mode: causal mask
            mask = mx.tril(mx.ones((Tq, Tk)))
            scores = mx.where(mask, scores, -float("inf"))
        elif Tq == 1:
            # Inference with single query: attend to all keys
            pass  # no mask needed
        else:
            # Inference with multiple queries: prefix + causal
            mask = mx.zeros((Tq, Tk))
            prefix_len = Tk - Tq
            if prefix_len > 0:
                mask[:, :prefix_len] = 1
            # Causal within chunk
            causal_mask = mx.tril(mx.ones((Tq, Tq)))
            mask[:, prefix_len:] = causal_mask
            scores = mx.where(mask, scores, -float("inf"))

        # Apply softmax and compute output
        attn = mx.softmax(scores, axis=-1)
        y = attn @ v  # (B, H, Tq, D)

        # Transpose back to (B, Tq, H, D) and reshape
        y = mx.transpose(y, (0, 2, 1, 3))
        y = mx.reshape(y, (B, Tq, self.n_embd))

        # Project back to residual stream
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.square(nn.relu(x))  # ReLU^2 activation
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, cos_sin, kv_cache=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings and transformer blocks
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Precompute rotary embeddings (over-allocate for safety)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos = cos
        self.sin = sin

    def init_weights(self):
        """Initialize model weights"""
        # Initialize all linear layers and embeddings
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Custom initialization from nanochat
                fan_out, fan_in = module.weight.shape
                std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
                module.weight = mx.random.normal(module.weight.shape, scale=std)
            elif isinstance(module, nn.Embedding):
                module.weight = mx.random.normal(module.weight.shape, scale=1.0)

        # Zero out classifier weights
        self.lm_head.weight = mx.zeros_like(self.lm_head.weight)

        # Zero out c_proj weights in all blocks
        for block in self.h:
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight)

        # Recompute rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos = cos
        self.sin = sin

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        """Precompute rotary position embeddings"""
        # Stride the channels
        channel_range = mx.arange(0, head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))

        # Stride the time steps
        t = mx.arange(seq_len, dtype=mx.float32)

        # Calculate rotation frequencies at each (time, channel) pair
        freqs = mx.expand_dims(t, axis=1) * mx.expand_dims(inv_freq, axis=0)  # outer product
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)

        # Add batch and head dims for broadcasting: (1, seq_len, 1, head_dim//2)
        cos = mx.expand_dims(mx.expand_dims(cos, axis=0), axis=2)
        sin = mx.expand_dims(mx.expand_dims(sin, axis=0), axis=2)

        return cos, sin

    def __call__(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = idx.shape

        # Get rotary embeddings for current sequence
        assert T <= self.cos.shape[1], f"Sequence length {T} exceeds rotary cache {self.cos.shape[1]}"

        # Offset rotary embeddings if using KV cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0 : T0 + T], self.sin[:, T0 : T0 + T]

        # Forward through transformer
        x = self.wte(idx)
        x = norm(x)
        for block in self.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward through lm_head
        softcap = 15
        if targets is not None:
            # Training mode: compute loss
            logits = self.lm_head(x)
            logits = softcap * mx.tanh(logits / softcap)  # logits softcap

            # Compute cross-entropy loss
            logits_flat = mx.reshape(logits, (-1, logits.shape[-1]))
            targets_flat = mx.reshape(targets, (-1,))

            # MLX cross-entropy
            loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction=loss_reduction)
            return loss
        else:
            # Inference mode: return logits
            logits = self.lm_head(x)
            logits = softcap * mx.tanh(logits / softcap)  # logits softcap
            return logits
