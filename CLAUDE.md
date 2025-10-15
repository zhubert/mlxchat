# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mlxchat is an MLX port of [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy, optimized for training GPT models on Apple Silicon (M1/M2/M3/M4 MacBooks). The goal is to leverage unified memory architecture (up to 36GB) for single-machine training without distributed computing complexity.

**Key Differences from nanochat:**
- Uses MLX instead of PyTorch for Apple Silicon native Metal acceleration
- Single-machine training only (no DDP/distributed training)
- Reuses nanochat's tokenizer and data shards
- Same GPT architecture: RoPE positional embeddings, QK normalization, Multi-Query Attention, ReLU² activation

## Development Commands

### Installation
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Or just core dependencies
pip install mlx numpy tiktoken regex fastapi uvicorn datasets psutil requests
```

### Data Management

**Streaming Mode (Recommended for MacBooks with limited storage):**
```bash
# Train with automatic shard downloads (keeps only 20 shards cached)
python -m scripts.base_train --streaming --max-cached-shards 20
```
- Storage: ~3-8 GB depending on cache size
- Coverage: Full dataset (1823 shards, ~300 GB total)
- Shards downloaded on-demand during training
- Old shards automatically removed when cache is full

**Pre-download Mode:**
```bash
# Download specific number of shards
python -m mlxchat.dataset --num-shards 50  # ~8 GB

# Then train without streaming
python -m scripts.base_train --depth=12
```

**Storage Requirements:**
- Streaming (20 shards): ~3.2 GB → 100% dataset coverage
- Streaming (50 shards): ~8 GB → 100% dataset coverage
- Pre-download 50: ~8 GB → ~2.7% coverage
- Pre-download 100: ~16 GB → ~5.5% coverage
- Full download: ~300 GB → 100% coverage

### Training
```bash
# Train a small model (d12, 186M params)
python -m scripts.base_train --depth=12 --device_batch_size=4

# Train larger models (adjust depth: 12/16/20/26)
python -m scripts.base_train --depth=20 --device_batch_size=2
```

### Inference
```bash
# CLI chat interface
python -m scripts.chat_cli

# Web UI
python -m scripts.chat_web
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_gpt.py

# Run tests with verbose output
pytest -v
```

### Code Quality
```bash
# Format code (120 char line length)
black .

# Lint code
ruff check .
```

## Architecture

### Core Components

**mlxchat/gpt.py** - GPT model implementation
- `GPTConfig`: Model configuration (n_layer, n_embd, n_head derived from depth parameter)
- `norm()`: RMSNorm without learnable parameters
- `apply_rotary_emb()`: Rotary Position Embeddings (RoPE)
- `repeat_kv()`: Key/value expansion for Multi-Query Attention
- `CausalSelfAttention`: Attention layer with QK normalization
- `MLP`: Feed-forward network with ReLU² activation and 4x expansion
- `Block`: Transformer block (attention + MLP with residuals)
- `GPT`: Main model class with token embeddings, transformer blocks, and LM head (untied weights)
- Forward pass includes logit soft-capping with tanh

**mlxchat/muon.py** - Muon optimizer
- `zeropower_via_newtonschulz5()`: Newton-Schulz orthogonalization (5 iterations)
- `Muon`: SGD with momentum + orthogonalization post-processing
- Used for transformer blocks (embeddings/LM head use Adam)
- Aspect ratio scaling for tall/wide weight matrices

**mlxchat/engine.py** - Inference engine with KV cache
- `KVCache`: Manages key-value cache for fast autoregressive generation
- `sample_next_token()`: Temperature and top-k sampling
- `Engine`: Handles prefill phase (process prompt) and decode phase (generation)
- Includes tool use state machine for calculator functionality

**mlxchat/dataloader.py** - Data loading
- Reads nanochat FineWeb shards from `~/.cache/nanochat/data/`
- Yields batches of tokenized sequences (fixed length 2048)
- No distributed sampling (sequential iteration only)
- Returns MLX arrays instead of PyTorch tensors

**mlxchat/tokenizer.py** - Tokenizer wrapper
- Wraps nanochat's RustBPE tokenizer
- Points to shared tokenizer directory: `~/.cache/nanochat/tokenizer`
- Returns MLX arrays for encode/decode operations
- Special tokens: `<|endoftext|>`, `<|user|>`, `<|assistant|>`, `<|calculator|>`, etc.

**mlxchat/common.py** - Common utilities
- `get_base_dir()`: Returns `~/.cache/mlxchat`
- `print0()`: Printing utility
- `compute_init()`: Seeding and device setup (no DDP logic)
- MLX device is automatic (unified memory architecture)

### Training Scripts

**scripts/base_train.py** - Pretraining script
- Configuration: depth parameter (12/16/20/26) → determines n_layer, n_embd, n_head
- Dual optimizer setup: Muon for transformer, Adam for embeddings/lm_head
- Learning rate scaling based on model size
- Gradient accumulation (no distributed sync)
- Checkpoint saving with model/optimizer/metadata
- Periodic validation evaluation (bits-per-byte)
- Optional wandb integration

**scripts/chat_cli.py** - CLI chat interface
- Loads model checkpoint and tokenizer
- Multi-turn conversation loop with streaming token generation
- Command-line arguments: temperature, top-k, max tokens, checkpoint path

**scripts/chat_web.py** - FastAPI web server
- POST `/chat/completions`: OpenAI-compatible chat endpoint with SSE streaming
- GET `/`: Serves HTML UI
- GET `/health`: Health check endpoint
- Includes embedded `ui.html` for browser-based chat

### Model Sizes

Recommended for M3 Pro 36GB:

| Depth | Params | Memory | Training Time | Status |
|-------|--------|--------|---------------|--------|
| d12   | 186M   | ~4GB   | ~8 hours      | Start here |
| d16   | 336M   | ~7GB   | ~14 hours     | Good balance |
| d20   | 561M   | ~12GB  | ~24 hours     | Max recommended |
| d26   | 1.1B   | ~24GB  | ~48 hours     | Experimental |

## MLX-Specific Patterns

### Key MLX API Differences from PyTorch

1. **Layers**: Use `mx.nn.Linear` instead of `torch.nn.Linear`
2. **Concatenation**: Use `mx.concatenate()` instead of `torch.cat()`
3. **Gradient computation**: Use `mx.grad()` instead of `.backward()`
4. **Array creation**: Use `mx.array()` instead of `torch.tensor()`
5. **Device management**: MLX uses unified memory, no explicit `.to(device)` needed
6. **Attention**: No built-in `scaled_dot_product_attention`, implement manually with causal mask

### Attention Implementation

The causal self-attention must be implemented manually in MLX:
```python
# Compute attention scores
scores = (Q @ K.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)

# Apply causal mask (lower triangular)
mask = mx.tril(mx.ones((seq_len, seq_len)))
scores = mx.where(mask, scores, float('-inf'))

# Softmax and matmul with values
attn = mx.softmax(scores, axis=-1)
out = attn @ V
```

### Newton-Schulz Iteration

The Muon optimizer uses Newton-Schulz iteration for orthogonalization:
```python
# 5 iterations of Newton-Schulz
for _ in range(5):
    A = (3 * I - Z @ Z) @ Z / 2
    Z = A
```

Handle tall vs wide matrices by transposing before iteration.

## Project Status

**~70% Complete** - Core infrastructure is implemented and tested. See TODO.md for detailed status.

**Completed:**
- ✅ Phase 1.1: GPT Model (mlxchat/gpt.py, 14 tests passing)
- ✅ Phase 1.2: Muon Optimizer (mlxchat/muon.py, 11 tests passing)
- ✅ Phase 2.1: Tokenizer (mlxchat/tokenizer.py, 12 tests passing)
- ✅ Phase 2.2: Dataloader with Streaming (mlxchat/dataloader.py + dataset.py, 11 tests passing)

**In Progress:**
- 🚧 Phase 2.3: Training Script (scripts/base_train.py) - needs MLX gradient patterns
  - Critical blocker: Gradient accumulation across micro-batches
  - Critical blocker: Multi-optimizer coordination (Adam + Muon)
  - Critical blocker: Gradient clipping implementation

**Not Started:**
- ⏳ Phase 2.4: Checkpoint Manager
- ⏳ Phase 3: Inference & chat UI
- ⏳ Phase 4: Evaluation & fine-tuning

**Next Milestone:** Complete training script to successfully train d12 for 100+ iterations

## Implementation Guidelines

### When Porting from nanochat

1. **Remove all DDP/distributed code**: No `torchrun`, `dist.init_process_group()`, or distributed samplers
2. **Convert tensor operations**: Replace PyTorch operations with MLX equivalents
3. **Preserve architecture**: Keep same model architecture (RoPE, QK norm, MQA, ReLU²)
4. **Reuse tokenizer**: Point to nanochat's tokenizer directory, don't rebuild
5. **Test incrementally**: Each component should have unit tests before integration

### Code Style

- Line length: 120 characters (configured in pyproject.toml)
- Formatter: black
- Linter: ruff
- Type hints encouraged but not required

### Testing Strategy

1. **Unit tests**: Test each component in isolation (model, optimizer, dataloader, engine)
2. **Integration tests**: Train for 100 steps, verify loss decreases
3. **Comparison tests**: Compare MLX outputs to PyTorch nanochat with same initialization
4. **Generation tests**: Verify model generates coherent text after training

## Common Pitfalls

1. **Memory**: MLX uses unified memory - monitor total system RAM, not just GPU
2. **Batch size**: Adjust `device_batch_size` based on available RAM (start small)
3. **Gradient accumulation**: Use this instead of distributed training for effective large batch sizes
4. **Tokenizer paths**: Ensure `~/.cache/nanochat/tokenizer` exists or download from nanochat
5. **Data shards**: Use streaming mode (`--streaming`) or pre-download shards (`python -m mlxchat.dataset`)
6. **Causal masking**: Must implement manually, no MLX equivalent to PyTorch's built-in attention

## MLX Training Patterns (Research Needed)

The training script (scripts/base_train.py) is incomplete due to gaps in MLX knowledge. These patterns need research:

### Gradient Accumulation
PyTorch accumulates gradients automatically across multiple `.backward()` calls:
```python
# PyTorch pattern
for micro_step in range(grad_accum_steps):
    loss = model(x, y)
    loss.backward()  # Gradients accumulate in .grad
optimizer.step()
optimizer.zero_grad()
```

MLX uses a functional API with `mx.grad()`. How to accumulate? Options:
1. Call `mx.grad()` multiple times and sum gradient dictionaries manually?
2. Use `mx.compile()` with gradient accumulation built into the function?
3. Average losses first, then take single gradient?

### Multi-Optimizer Coordination
Need to apply two optimizers to different parameter groups:
- Adam optimizer → embeddings + lm_head
- Muon optimizer → transformer blocks

Questions:
1. Do we compute separate gradient dictionaries for each optimizer?
2. How to update the model with results from two optimizers?
3. Can we use a single `mx.grad()` call and split the gradients?

### Gradient Clipping
PyTorch has `torch.nn.utils.clip_grad_norm_()` but MLX doesn't. Need to:
1. Compute global norm: `sqrt(sum(||g||^2))` across all gradients
2. Scale all gradients by `min(1, clip_value / global_norm)`
3. Apply scaled gradients with optimizer

### Recommended Approach
1. Study MLX examples repo for training patterns
2. Look at mlx-examples/llms for transformer training
3. Search MLX docs for gradient accumulation examples
4. Test with small model (d12) and few iterations before full training

## Resources

- nanochat source: https://github.com/karpathy/nanochat
- MLX documentation: https://ml-explore.github.io/mlx/
- MLX examples: https://github.com/ml-explore/mlx-examples
