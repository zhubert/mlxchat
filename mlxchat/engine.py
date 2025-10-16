"""
Engine for efficient inference with MLX.

Implements KV caching for fast autoregressive generation.
"""

import mlx.core as mx


class KVCache:
    """
    KV cache for efficient autoregressive generation.
    Stores cached key/value tensors from each transformer layer.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        """
        Initialize KV cache.

        Args:
            batch_size: Number of sequences to generate in parallel
            num_heads: Number of KV heads (for multi-query attention)
            seq_len: Maximum sequence length
            head_dim: Dimension per attention head
            num_layers: Number of transformer layers
        """
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.pos = 0  # Current position in the cache

        # Cache shape: (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        # The "2" is for keys and values
        self.kv_cache = None

    def reset(self):
        """Reset cache position to 0."""
        self.pos = 0

    def get_pos(self):
        """Get current position in cache."""
        return self.pos

    def insert_kv(self, layer_idx, k, v):
        """
        Insert key/value tensors into cache for a specific layer.

        Args:
            layer_idx: Which transformer layer (0-indexed)
            k: Keys tensor of shape (B, H, T_add, D)
            v: Values tensor of shape (B, H, T_add, D)

        Returns:
            Tuple of (cached_keys, cached_values) up to current position
        """
        # Lazy initialize cache on first use
        if self.kv_cache is None:
            cache_shape = (
                self.num_layers,
                2,  # keys and values
                self.batch_size,
                self.num_heads,
                self.seq_len,
                self.head_dim,
            )
            self.kv_cache = mx.zeros(cache_shape, dtype=k.dtype)

        B, H, T_add, D = k.shape
        t0, t1 = self.pos, self.pos + T_add

        # Grow cache if needed (shouldn't happen if seq_len is set correctly)
        if t1 > self.kv_cache.shape[4]:
            raise ValueError(
                f"Cache overflow: trying to insert at position {t1} but cache size is {self.kv_cache.shape[4]}"
            )

        # Insert keys and values
        # Note: MLX doesn't support item assignment, so we need to use array slicing
        # We'll create a new array with the updated values
        updated_k_cache = self.kv_cache[layer_idx, 0]
        updated_v_cache = self.kv_cache[layer_idx, 1]

        # Update the cache slices
        # For MLX, we need to build the cache by concatenating
        if self.pos == 0:
            # First insertion - just use the new k,v
            updated_k_cache = mx.pad(k, [(0, 0), (0, 0), (0, self.seq_len - T_add), (0, 0)])
            updated_v_cache = mx.pad(v, [(0, 0), (0, 0), (0, self.seq_len - T_add), (0, 0)])
        else:
            # Subsequent insertions - concatenate with existing cache
            k_before = updated_k_cache[:, :, :t0, :]
            v_before = updated_v_cache[:, :, :t0, :]
            k_after_pad = mx.zeros((B, H, self.seq_len - t1, D), dtype=k.dtype)
            v_after_pad = mx.zeros((B, H, self.seq_len - t1, D), dtype=v.dtype)

            updated_k_cache = mx.concatenate([k_before, k, k_after_pad], axis=2)
            updated_v_cache = mx.concatenate([v_before, v, v_after_pad], axis=2)

        # Update the main cache
        # Build a new cache array with the updated layer
        kv_list = []
        for li in range(self.num_layers):
            if li == layer_idx:
                kv_list.append(mx.stack([updated_k_cache, updated_v_cache], axis=0))
            else:
                kv_list.append(self.kv_cache[li])
        self.kv_cache = mx.stack(kv_list, axis=0)

        # Increment position after last layer
        if layer_idx == self.num_layers - 1:
            self.pos = t1

        # Return views of cached k,v up to current position
        cached_k = updated_k_cache[:, :, :t1, :]
        cached_v = updated_v_cache[:, :, :t1, :]

        return cached_k, cached_v


def sample_next_token(logits, temperature=1.0, top_k=None):
    """
    Sample next token from logits.

    Args:
        logits: Logits tensor of shape (B, vocab_size)
        temperature: Sampling temperature (0.0 = greedy)
        top_k: If set, only sample from top k tokens

    Returns:
        Token indices of shape (B,)
    """
    assert temperature >= 0.0, "temperature must be non-negative"

    # Greedy sampling
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1)

    # Top-k filtering
    if top_k is not None:
        k = min(top_k, logits.shape[-1])
        # Get top k values and indices
        top_logits = mx.topk(logits, k, axis=-1)
        # Filter logits
        logits = mx.where(logits >= top_logits[:, -1:], logits, float("-inf"))

    # Temperature scaling
    logits = logits / temperature

    # Sample from categorical distribution
    probs = mx.softmax(logits, axis=-1)
    return mx.random.categorical(probs, axis=-1)


class Engine:
    """
    Inference engine for efficient text generation with KV caching.
    """

    def __init__(self, model, tokenizer):
        """
        Initialize inference engine.

        Args:
            model: GPT model
            tokenizer: Tokenizer instance
        """
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, tokens, num_samples=1, max_tokens=256, temperature=1.0, top_k=None, seed=42):
        """
        Generate tokens autoregressively with streaming output.

        Args:
            tokens: List of input token ids
            num_samples: Number of sequences to generate in parallel
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            seed: Random seed

        Yields:
            Token ids for each generated token (one at a time)
        """
        mx.random.seed(seed)

        # Get special tokens
        eot_token = self.tokenizer.get_eot_token_id()

        # Prepare input
        # Shape: (1, T)
        ids = mx.array([tokens], dtype=mx.int32)

        # Initialize KV cache for prefill
        config = self.model.config
        kv_cache = KVCache(
            batch_size=1,
            num_heads=config.n_kv_head,
            seq_len=len(tokens) + max_tokens,
            head_dim=config.n_embd // config.n_head,
            num_layers=config.n_layer,
        )

        # Prefill: process all input tokens at once
        logits = self.model(ids, kv_cache=kv_cache)
        # Get logits for last position
        logits = logits[:, -1, :]  # (1, vocab_size)

        # Generate tokens one at a time
        for _ in range(max_tokens):
            # Sample next token
            next_token = sample_next_token(logits, temperature, top_k)
            next_token_id = next_token[0].item()

            # Check for end of text
            if next_token_id == eot_token:
                break

            # Yield the token
            yield next_token_id

            # Prepare next input (single token)
            ids = mx.array([[next_token_id]], dtype=mx.int32)

            # Forward pass with cache
            logits = self.model(ids, kv_cache=kv_cache)
            logits = logits[:, -1, :]  # (1, vocab_size)

            # Ensure computation happens
            mx.eval(logits)

    def generate_text(self, prompt, max_tokens=256, temperature=1.0, top_k=None, seed=42):
        """
        Generate text from a string prompt (convenience wrapper).

        Args:
            prompt: Input text string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            seed: Random seed

        Returns:
            Generated text string
        """
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)

        # Generate tokens
        generated = []
        for token_id in self.generate(
            tokens=tokens, max_tokens=max_tokens, temperature=temperature, top_k=top_k, seed=seed
        ):
            generated.append(token_id)

        # Decode and return
        return self.tokenizer.decode(generated)
