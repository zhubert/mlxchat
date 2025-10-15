# mlxchat Progress Summary

## Overview

This document summarizes the development progress on mlxchat, an MLX port of nanochat for Apple Silicon.

**Status**: ~80% complete (14 hours invested)
**Tests**: 48/48 passing
**Training**: Core loop working with gradient accumulation and dual optimizers!
**Next Milestone**: Add checkpoint save/load functionality

---

## Accomplishments

### Phase 1.1: GPT Model ✅ (14 tests passing)

**File**: `mlxchat/gpt.py`

Fully implemented GPT model in MLX with all architectural features:
- GPTConfig dataclass with validation
- RMSNorm without learnable parameters (`norm()`)
- Rotary Position Embeddings (`apply_rotary_emb()`)
- Multi-Query Attention support (`repeat_kv()`)
- CausalSelfAttention with QK normalization
- MLP with ReLU² activation and 4x expansion
- Transformer blocks with residual connections
- Token embeddings and untied LM head
- Precomputed rotary embeddings cache
- Forward pass for training (with loss) and inference (logits only)
- Logit soft-capping with tanh
- Proper weight initialization

**Key MLX Challenge**: Implemented causal attention manually (no built-in `scaled_dot_product_attention`)

**Tests**: 14 comprehensive tests covering all components, initialization, and forward passes

---

### Phase 1.2: Muon Optimizer ✅ (11 tests passing)

**File**: `mlxchat/muon.py`

Implemented Muon optimizer for training transformer blocks:
- Newton-Schulz iteration for gradient orthogonalization (5 steps, quintic coefficients)
- Handles tall/wide/square matrices correctly with transposition
- SGD with momentum (standard and Nesterov variants)
- Aspect ratio scaling for non-square weight matrices
- Extends MLX optimizer base class properly
- Per-parameter state management for momentum buffers

**Key Challenge**: Newton-Schulz produces approximate orthogonalization (not perfect identity), required relaxed test tolerances

**Tests**: 11 tests covering Newton-Schulz on different matrix shapes, optimizer initialization, convergence, neural network integration

---

### Phase 2.1: Tokenizer ✅ (12 tests passing)

**File**: `mlxchat/tokenizer.py`

Created wrapper around nanochat's tokenizer returning MLX arrays:
- Wraps RustBPE + tiktoken encoding
- Returns MLX arrays for compatibility with mlxchat
- Supports single string and batch encoding
- Supports prepend/append special tokens
- `get_tokenizer()` helper function
- Reuses nanochat's trained tokenizer (no retraining needed)

**Tests**: 12 tests covering encode/decode round trips, special tokens, vocabulary, batch encoding

---

### Phase 2.2: Dataloader ✅ (11 tests passing)

**Files**: `mlxchat/dataloader.py`, `mlxchat/dataset.py`

Implemented data loading with critical streaming innovation:
- DataLoader class for streaming tokenized batches
- Reads FineWeb parquet shards (nanochat format)
- Token buffer for efficient batch construction
- Support for train/val splits
- Utility functions: `list_parquet_files()`, `parquets_iter_batched()`

**Streaming Innovation**: Solved MacBook storage constraints (300GB dataset → 3-8GB)
- `ShardCache` class with LRU eviction policy
- On-demand shard downloads from HuggingFace
- Rolling cache keeps only N shards on disk (configurable)
- Automatic cleanup of old shards when cache is full
- `download_shards()` utility for pre-downloading subsets
- CLI tool: `python -m mlxchat.dataset`

**Storage Impact**:
- Streaming (20 shards): ~3.2 GB → 100% dataset coverage ⭐
- Streaming (50 shards): ~8 GB → 100% dataset coverage
- Full download: ~300 GB → 100% dataset coverage

**Tests**: 11 tests covering file listing, iteration, initialization, different batch sizes and splits

---

### Phase 2.3: Training Script ✅ (Core complete!)

**Files**: `scripts/base_train.py`, `scripts/test_train_loop.py`

Successfully implemented core training loop with all critical MLX patterns:

**Implemented**:
- ✅ Command-line argument parsing
- ✅ Model configuration from depth parameter
- ✅ Dual optimizer setup (Muon + Adam)
- ✅ **Gradient accumulation** using `tree_map(mx.add, grads, accum_grads)`
- ✅ **Multi-optimizer coordination** by splitting gradients into separate dicts
- ✅ **Gradient clipping** by global norm (manual implementation)
- ✅ Learning rate scheduling (warmup + constant + warmdown)
- ✅ Momentum scheduling for Muon optimizer
- ✅ Proper MLX evaluation pattern (`mx.eval()` on params and optimizer states)
- ✅ Progress logging and EMA smoothing
- ✅ Data loading with streaming support

**Test Results** (test_train_loop.py):
- ✅ Successfully trains small model (4 layers, 256 dim, 29M params)
- ✅ Gradient accumulation works across 2 micro-batches
- ✅ Dual optimizers (Adam + Muon) update correctly
- ✅ Gradient clipping maintains grad_norm ~0.707
- ✅ Achieves ~7,000 tokens/sec on M3 Pro
- ✅ Training loop completes 5 iterations without errors

**Remaining (Lower Priority)**:
- ⏳ Checkpoint saving/loading
- ⏳ Validation evaluation loop
- ⏳ Sample generation during training

**Key MLX Patterns Discovered**:
1. **Gradient Accumulation**: Use `tree_map(mx.add, grads, accum_grads)` to accumulate, then average with `tree_map(lambda g: g / n, accum_grads)`
2. **Multi-Optimizer**: Split gradients by key ("wte", "lm_head", "h"), call `optimizer.update()` separately for each group
3. **Gradient Clipping**: Use `tree_flatten(grads)` to get (path, value) tuples, compute global norm, scale with `tree_map()`

---

## Development Process

### Methodology
- **Test-Driven Development**: Built each component with comprehensive tests before moving forward
- **Incremental Commits**: Committed after each completed phase
- **UV Virtual Environment**: Used uv for dependency management

### Commits
1. Initial commit: Project structure (README, TODO, CLAUDE, pyproject.toml)
2. Phase 1.1: GPT model with 14 tests passing
3. Phase 1.2: Muon optimizer with 11 tests passing
4. Phase 2.1: Tokenizer with 12 tests passing
5. Phase 2.2: Dataloader with 11 tests passing
6. Phase 2.2.1: Streaming data support
7. Phase 2.3 (partial): Training script skeleton
8. Documentation: Updated TODO, README, CLAUDE with current status
9. **Phase 2.3 (complete): Training script with gradient accumulation and dual optimizers** ⭐
10. Documentation: Updated with training script completion

### Errors Fixed
1. **Test failure**: n_kv_head > n_head validation issue → Fixed by setting n_kv_head=4 in test
2. **Newton-Schulz tests too strict** → Relaxed tolerance checks (iteration is approximate)
3. **Missing tokenizer in tests** → Created temp tokenizer fixture with GPT-2 + custom BOS token
4. **Parameter counting errors** → Fixed tree_flatten usage (returns (path, value) tuples, not just values)
5. **Gradient clipping errors** → Fixed to iterate over tuples from tree_flatten

---

## Next Steps

### Priority 1: Complete Training Script ✅ DONE (Actual: 3 hours)

**Research Phase**: ✅
1. ✅ Studied MLX gradient accumulation issue #929
2. ✅ Understood gradient accumulation patterns (tree_map)
3. ✅ Learned multi-optimizer coordination (split gradients)
4. ✅ Figured out gradient clipping (tree_flatten + scale)

**Implementation Phase**: ✅
1. ✅ Fixed gradient accumulation using tree_map
2. ✅ Fixed optimizer updates (Adam + Muon split by key)
3. ✅ Added gradient clipping by global norm
4. ✅ Tested with test_train_loop.py (5 iterations successful)

**Remaining**:
- ⏳ Add validation loop
- ⏳ Add checkpoint save/load

### Priority 2: Checkpoint Manager (Estimated: 2-4 hours)

**File**: `mlxchat/checkpoint_manager.py` (new)

- Implement `save_checkpoint()` - save model weights (.npz), optimizer state, metadata
- Implement `load_checkpoint()` - load weights into model, optimizer state, return metadata
- Support multiple checkpoint types (base, mid-training)
- Test save → load → resume training

### Priority 3: Inference Engine (Estimated: 1 day)

**File**: `mlxchat/engine.py`

- Port KVCache class for fast autoregressive generation
- Implement prefill phase (process prompt)
- Implement decode phase (generation)
- Add temperature and top-k sampling
- Test text generation from trained model

### Priority 4: Chat Interfaces (Estimated: 1 day)

**Files**: `scripts/chat_cli.py`, `scripts/chat_web.py`

- CLI chat interface with multi-turn conversations
- Web UI with FastAPI + Server-Sent Events
- Test chatting with d12 model

---

## Technical Decisions

### Architecture Choices
- **MLX over PyTorch**: Native Apple Silicon acceleration, unified memory
- **Single machine**: No distributed training complexity (DDP/distributed samplers removed)
- **Streaming data**: LRU cache with on-demand downloads for storage constraints
- **Same tokenizer**: Reuse nanochat's trained tokenizer (no retraining)
- **Dual optimizer**: Muon for transformer blocks, Adam for embeddings/LM head

### MLX-Specific Patterns
- Manual causal attention implementation (no built-in function)
- Functional gradient API (`mx.grad()`) vs PyTorch imperative (`.backward()`)
- Newton-Schulz orthogonalization with transposition for tall/wide matrices
- MLX arrays instead of PyTorch tensors throughout
- Unified memory → no explicit device management

---

## Repository Structure

```
mlxchat/
├── mlxchat/              # Core library (complete)
│   ├── gpt.py            # ✅ GPT model (14 tests)
│   ├── muon.py           # ✅ Muon optimizer (11 tests)
│   ├── tokenizer.py      # ✅ Tokenizer wrapper (12 tests)
│   ├── dataloader.py     # ✅ Dataloader (11 tests)
│   ├── dataset.py        # ✅ Streaming support
│   └── __main__.py       # ✅ CLI for dataset downloads
├── scripts/              # Training/inference scripts (partial)
│   └── base_train.py     # 🚧 Training script (50% complete)
├── tests/                # Test suite (48 tests passing)
│   ├── test_gpt.py       # ✅ 14 tests
│   ├── test_muon.py      # ✅ 11 tests
│   ├── test_tokenizer.py # ✅ 12 tests
│   └── test_dataloader.py# ✅ 11 tests
├── TODO.md               # ✅ Detailed implementation plan
├── CLAUDE.md             # ✅ Development guidance
├── README.md             # ✅ Project overview
├── PROGRESS.md           # ✅ This file
└── pyproject.toml        # ✅ Dependencies and config
```

---

## Success Criteria

### MVP (1-2 more sessions, 4-8 hours)
- [x] Complete training script with gradient accumulation ✅
- [ ] Add checkpoint save/load functionality
- [ ] Train d12 (186M) for 100+ iterations with real data
- [ ] Verify loss decreases over training
- [ ] Generate coherent text from trained model (requires inference engine)

### Full Success (1-2 weeks)
- [ ] Train d12 to completion (~20 Chinchilla tokens)
- [ ] Train d20 (561M) without OOM
- [ ] Match nanochat's validation loss
- [ ] Working chat CLI and web UI
- [ ] All evaluation tasks implemented

---

## Key Resources

- **nanochat source**: https://github.com/karpathy/nanochat
- **MLX docs**: https://ml-explore.github.io/mlx/
- **MLX examples**: https://github.com/ml-explore/mlx-examples
- **FineWeb dataset**: HuggingFace `HuggingFaceFW/fineweb-edu` (100B tokens, 1823 shards)

---

## Notes

- **Time invested**: ~14 hours across 10 commits
- **Code quality**: 48 tests passing, comprehensive test coverage
- **Innovation**: Streaming data system reduces storage from 300GB to 3-8GB
- **Breakthrough**: Successfully implemented MLX gradient accumulation, multi-optimizer coordination, and gradient clipping ⭐
- **Performance**: Achieving ~7,000 tok/sec on small models (M3 Pro)
- **Hardware target**: M3 Pro 36GB (can train d12-d20 models)
