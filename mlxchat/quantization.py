"""
Quantization utilities for MLX models

Provides 4-bit and 8-bit quantization for inference to reduce memory usage.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple


def quantize_weight_4bit(weight: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Quantize a weight matrix to 4-bit representation.

    Uses a simple per-row quantization scheme:
    - For each row, compute scale = max(abs(row)) / 7
    - Quantize values to range [-7, 7] (4-bit signed integers)
    - Store quantized values, scales, and zeros

    Args:
        weight: Original weight matrix (float)

    Returns:
        Tuple of (quantized_weight, scales, zeros)
        - quantized_weight: int8 array with 4-bit values
        - scales: per-row scaling factors
        - zeros: per-row zero points
    """
    # Get shape
    original_shape = weight.shape

    # Flatten if needed, treating as 2D (out_features, in_features)
    if len(original_shape) > 2:
        weight = weight.reshape(original_shape[0], -1)

    # Compute per-row statistics
    row_max = mx.max(mx.abs(weight), axis=-1, keepdims=True)
    scales = row_max / 7.0  # 4-bit signed range is -7 to 7
    scales = mx.maximum(scales, 1e-8)  # avoid division by zero

    # Quantize to 4-bit range [-7, 7]
    quantized = mx.round(weight / scales)
    quantized = mx.clip(quantized, -7, 7)

    # Store as int8 (MLX doesn't have int4, so we use int8)
    quantized = quantized.astype(mx.int8)

    return quantized, scales, original_shape


def dequantize_weight_4bit(quantized: mx.array, scales: mx.array, original_shape: tuple) -> mx.array:
    """
    Dequantize a 4-bit weight matrix back to float.

    Args:
        quantized: int8 array with 4-bit values
        scales: per-row scaling factors
        original_shape: original shape to restore

    Returns:
        Dequantized weight matrix (float)
    """
    # Dequantize
    weight = quantized.astype(mx.float32) * scales

    # Reshape to original shape
    if len(original_shape) > 2:
        weight = weight.reshape(original_shape)

    return weight


def quantize_weight_8bit(weight: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Quantize a weight matrix to 8-bit representation.

    Uses a simple per-row quantization scheme:
    - For each row, compute scale = max(abs(row)) / 127
    - Quantize values to range [-127, 127] (8-bit signed integers)

    Args:
        weight: Original weight matrix (float)

    Returns:
        Tuple of (quantized_weight, scales, original_shape)
    """
    # Get shape
    original_shape = weight.shape

    # Flatten if needed, treating as 2D (out_features, in_features)
    if len(original_shape) > 2:
        weight = weight.reshape(original_shape[0], -1)

    # Compute per-row statistics
    row_max = mx.max(mx.abs(weight), axis=-1, keepdims=True)
    scales = row_max / 127.0  # 8-bit signed range is -127 to 127
    scales = mx.maximum(scales, 1e-8)  # avoid division by zero

    # Quantize to 8-bit range [-127, 127]
    quantized = mx.round(weight / scales)
    quantized = mx.clip(quantized, -127, 127)

    # Store as int8
    quantized = quantized.astype(mx.int8)

    return quantized, scales, original_shape


def dequantize_weight_8bit(quantized: mx.array, scales: mx.array, original_shape: tuple) -> mx.array:
    """
    Dequantize an 8-bit weight matrix back to float.

    Args:
        quantized: int8 array with quantized values
        scales: per-row scaling factors
        original_shape: original shape to restore

    Returns:
        Dequantized weight matrix (float)
    """
    # Dequantize
    weight = quantized.astype(mx.float32) * scales

    # Reshape to original shape
    if len(original_shape) > 2:
        weight = weight.reshape(original_shape)

    return weight


class QuantizedLinear(nn.Module):
    """
    Quantized linear layer for inference.

    Stores weights in quantized format (4-bit or 8-bit) and dequantizes
    on-the-fly during forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.has_bias = bias

        # These will be set by quantize_from_linear()
        self.quantized_weight = None
        self.scales = None
        self.original_shape = None
        self.bias = None

    def quantize_from_linear(self, linear: nn.Linear):
        """
        Quantize weights from a standard linear layer.

        Args:
            linear: nn.Linear layer to quantize
        """
        weight = linear.weight

        if self.bits == 4:
            self.quantized_weight, self.scales, self.original_shape = quantize_weight_4bit(weight)
        elif self.bits == 8:
            self.quantized_weight, self.scales, self.original_shape = quantize_weight_8bit(weight)
        else:
            raise ValueError(f"Unsupported bits: {self.bits}, must be 4 or 8")

        # Copy bias if present
        if self.has_bias and hasattr(linear, "bias") and linear.bias is not None:
            self.bias = linear.bias

    def __call__(self, x: mx.array) -> mx.array:
        # Dequantize weights on-the-fly
        if self.bits == 4:
            weight = dequantize_weight_4bit(self.quantized_weight, self.scales, self.original_shape)
        elif self.bits == 8:
            weight = dequantize_weight_8bit(self.quantized_weight, self.scales, self.original_shape)
        else:
            raise ValueError(f"Unsupported bits: {self.bits}")

        # Perform linear transformation
        out = x @ weight.T

        if self.has_bias and self.bias is not None:
            out = out + self.bias

        return out


def quantize_model(model, bits: int = 4, quantize_embeddings: bool = False):
    """
    Quantize a GPT model for inference.

    Replaces nn.Linear layers with QuantizedLinear layers.
    By default, does not quantize embeddings and output layers.

    Args:
        model: GPT model to quantize
        bits: Number of bits for quantization (4 or 8)
        quantize_embeddings: Whether to quantize embedding layers

    Returns:
        Quantized model
    """
    from mlxchat.gpt import GPT

    if not isinstance(model, GPT):
        raise ValueError("Only GPT models are supported")

    # Quantize transformer blocks
    for block_idx, block in enumerate(model.h):
        # Quantize attention layers
        attn = block.attn

        # Replace query, key, value projections
        attn.c_q = _quantize_linear(attn.c_q, bits)
        attn.c_k = _quantize_linear(attn.c_k, bits)
        attn.c_v = _quantize_linear(attn.c_v, bits)
        attn.c_proj = _quantize_linear(attn.c_proj, bits)

        # Quantize MLP layers
        mlp = block.mlp
        mlp.c_fc = _quantize_linear(mlp.c_fc, bits)
        mlp.c_proj = _quantize_linear(mlp.c_proj, bits)

    # Optionally quantize output layer
    if not quantize_embeddings:
        # Don't quantize lm_head for better quality
        pass
    else:
        # Quantize lm_head
        if hasattr(model, "lm_head"):
            model.lm_head = _quantize_linear(model.lm_head, bits)

    return model


def _quantize_linear(linear: nn.Linear, bits: int) -> QuantizedLinear:
    """Helper to quantize a single linear layer."""
    has_bias = hasattr(linear, "bias") and linear.bias is not None
    quantized = QuantizedLinear(
        in_features=linear.weight.shape[1],
        out_features=linear.weight.shape[0],
        bits=bits,
        bias=has_bias,
    )
    quantized.quantize_from_linear(linear)
    return quantized


def estimate_memory_reduction(model, bits: int = 4) -> dict:
    """
    Estimate memory reduction from quantization.

    Args:
        model: GPT model
        bits: Number of bits for quantization

    Returns:
        Dictionary with memory statistics
    """
    from mlxchat.gpt import GPT

    if not isinstance(model, GPT):
        raise ValueError("Only GPT models are supported")

    # Count parameters in transformer blocks
    def count_linear_params(module):
        total = 0
        if isinstance(module, nn.Linear):
            total += module.weight.size
            if hasattr(module, "bias") and module.bias is not None:
                total += module.bias.size
        return total

    transformer_params = 0
    for block in model.h:
        # Attention
        transformer_params += count_linear_params(block.attn.c_q)
        transformer_params += count_linear_params(block.attn.c_k)
        transformer_params += count_linear_params(block.attn.c_v)
        transformer_params += count_linear_params(block.attn.c_proj)
        # MLP
        transformer_params += count_linear_params(block.mlp.c_fc)
        transformer_params += count_linear_params(block.mlp.c_proj)

    # Original memory (assuming bfloat16 = 2 bytes per param)
    original_bytes = transformer_params * 2

    # Quantized memory
    if bits == 4:
        # 4-bit = 0.5 bytes per param + scales (float32 = 4 bytes per row)
        # Approximate scales as ~1% of params
        quantized_bytes = transformer_params * 0.5 + transformer_params * 0.01 * 4
    elif bits == 8:
        # 8-bit = 1 byte per param + scales
        quantized_bytes = transformer_params * 1.0 + transformer_params * 0.01 * 4
    else:
        quantized_bytes = original_bytes

    reduction_ratio = original_bytes / quantized_bytes

    return {
        "transformer_params": transformer_params,
        "original_memory_mb": original_bytes / (1024**2),
        "quantized_memory_mb": quantized_bytes / (1024**2),
        "reduction_ratio": reduction_ratio,
        "memory_saved_mb": (original_bytes - quantized_bytes) / (1024**2),
    }
