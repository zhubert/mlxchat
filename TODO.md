# mlxchat TODO - MLX Port of nanochat

## Overview

MLX port of nanochat for single-machine training on Apple Silicon.

**Status:** Core functionality complete! Training, inference, and CLI working.

**Target:** Train d12 (186M params) → d20 (561M params) models on MacBook

---

## ✅ COMPLETED PHASES

### Phase 1.1: GPT Model ✅
- [x] Complete GPT model implementation with MLX
- [x] RMSNorm, RoPE, Multi-Query Attention
- [x] Causal attention implementation
- [x] Forward pass for training and inference
- [x] 14 tests passing

### Phase 1.2: Muon Optimizer ✅
- [x] Newton-Schulz orthogonalization
- [x] SGD with momentum
- [x] Aspect ratio scaling
- [x] 11 tests passing

### Phase 1.3: Common Utilities ✅
- [x] Cache directory management
- [x] Printing utilities and ASCII banner
- [x] Seeding and initialization

---

## Phase 2: Training Infrastructure ✅

### 2.1: Tokenizer Integration ✅
- [x] RustBPE wrapper returning MLX arrays
- [x] Special tokens support
- [x] 12 tests passing

### 2.2: Data Loading ✅
- [x] Dataloader for FineWeb shards
- [x] Streaming mode with LRU cache
- [x] On-demand shard downloads
- [x] 11 tests passing

---

### 2.3: Training Script ✅
- [x] Gradient accumulation with tree_map
- [x] Multi-optimizer coordination (Adam + Muon)
- [x] Gradient clipping by global norm
- [x] Learning rate and momentum scheduling
- [x] Validation evaluation loop
- [x] Checkpoint saving/loading with automatic resumption
- [x] Progress logging with accurate timing

### 2.4: Checkpoint Manager ✅
- [x] Save/load model weights (.npz format)
- [x] Save/load optimizer state
- [x] Metadata handling (step, loss, config)
- [x] Support for multiple checkpoint types
- [x] 4 tests passing

---

## Phase 3: Inference & Chat ✅

### 3.1: Inference Engine ✅
- [x] KVCache implementation for fast generation
- [x] Temperature and top-k sampling
- [x] Prefill and decode phases
- [x] Streaming token generation

### 3.2: Chat CLI ✅
- [x] Multi-turn conversation support
- [x] Command-line interface
- [x] Streaming output
- [x] Special token handling

---

## 🔮 FUTURE WORK (Optional)

### Phase 4: Web UI

**File**: `scripts/chat_web.py`

- [ ] FastAPI server setup
- [ ] POST `/chat/completions` endpoint with streaming
- [ ] Serve HTML UI
- [ ] Server-Sent Events for streaming
- [ ] Health and status endpoints

**Estimated effort:** 1-2 days

### Phase 5: Evaluation Tasks

**Files**: `scripts/base_eval.py`, `tasks/*.py`

- [ ] Base evaluation (validation loss, perplexity)
- [ ] ARC-Easy / ARC-Challenge
- [ ] GSM8K (math problems)
- [ ] HumanEval (code generation)
- [ ] MMLU (multiple choice)
- [ ] SmolTalk (chat quality)

**Estimated effort:** 2-3 days

### Phase 6: Fine-tuning Scripts

**Files**: `scripts/mid_train.py`, `scripts/chat_sft.py`, `scripts/chat_rl.py`

- [ ] Mid-training for conversation format
- [ ] Supervised fine-tuning (SFT)
- [ ] Reinforcement learning with GRPO

**Estimated effort:** 3-5 days

---

## ✅ Testing & Validation

### Unit Tests (52 tests passing)
- [x] `tests/test_gpt.py` - Model forward/backward passes (14 tests)
- [x] `tests/test_muon.py` - Optimizer convergence (11 tests)
- [x] `tests/test_tokenizer.py` - Encode/decode (12 tests)
- [x] `tests/test_dataloader.py` - Batch loading (11 tests)
- [x] `tests/test_checkpoint.py` - Save/load checkpoints (4 tests)

### Integration Testing ✅
- [x] Training loop with gradient accumulation verified
- [x] Checkpoint save/load and resumption working
- [x] Text generation from trained models
- [x] Multi-turn conversations via CLI

---

## Additional Enhancements (Optional)

### Performance Optimization
- [ ] Profile memory usage during training
- [ ] Tune batch sizes for different hardware
- [ ] Implement gradient checkpointing for larger models
- [ ] Benchmark MLX performance improvements
- [ ] Use MLX compiled functions where beneficial

### Documentation
- [x] Installation guide (README)
- [x] Training tutorial with Makefile
- [x] Architecture overview (CLAUDE.md)
- [ ] API reference from docstrings
- [ ] Troubleshooting guide

### Stretch Goals
- [ ] LoRA fine-tuning support
- [ ] Quantization (4-bit, 8-bit inference)
- [ ] Model export to GGUF
- [ ] Gradio UI alternative
- [ ] MLX distributed training

---

## 🎯 Success Criteria

**Core MVP ✅ ACHIEVED:**
1. [x] Train models (d12-d20) with full training system
2. [x] Verify loss decreases over training
3. [x] Save and load checkpoints with automatic resumption
4. [x] Generate text from trained models
5. [x] Chat via CLI with streaming generation
6. [x] Streaming data support for limited storage

**Future Enhancements (Optional):**
1. [ ] Train d12 to full convergence
2. [ ] Train d20 (561M) and d26 (1.1B) models
3. [ ] Match nanochat's validation metrics
4. [ ] Web UI with FastAPI
5. [ ] Evaluation benchmarks (ARC, GSM8K, MMLU, etc.)
6. [ ] Fine-tuning scripts (SFT, RL)

---

## ⏱️ Project Summary

**Status:** Core functionality complete! ✅

| Phase | Status | Priority |
|-------|--------|----------|
| Phase 1: Model & Optimizer | ✅ Complete | **P0** |
| Phase 2: Training Infrastructure | ✅ Complete | **P0** |
| Phase 3: Inference & Chat CLI | ✅ Complete | **P0** |
| Phase 4: Web UI | ⏳ Future work | **P2** |
| Phase 5: Evaluation | ⏳ Future work | **P2** |
| Phase 6: Fine-tuning | ⏳ Future work | **P3** |

**Current Status:** Full training-to-inference pipeline working! Can train GPT models on MacBooks and chat with them via CLI.

---

## Notes & Key Decisions

1. **Architecture:** Identical to nanochat (RoPE, QK norm, MQA, ReLU²)
2. **Tokenizer:** Reuses nanochat's trained RustBPE tokenizer
3. **Data:** Compatible with nanochat's FineWeb shards
4. **Hardware:** Optimized for M3 Pro 36GB (works on other M-series)
5. **Requirements:** MLX ≥0.20.0, Python 3.10+
6. **Approach:** Single-machine training, no distributed complexity

## Key Technical Solutions ✅

### MLX Gradient Accumulation
- Use `tree_map(mx.add, grads, accum_grads)` to accumulate
- Average with `tree_map(lambda g: g / n_steps, accum_grads)`
- Call `mx.eval(accum_grads)` after each accumulation

### MLX Multi-Optimizer
- Split gradients into separate dicts by parameter name
- Call `optimizer.update()` separately for each group
- Evaluate both states with `mx.eval(adam_opt.state, muon_opt.state)`

### MLX Gradient Clipping
- Use `tree_flatten(grads)` to get all gradient values
- Compute global norm across all gradients
- Scale with `tree_map(lambda g: g * scale, grads)`

### Performance Characteristics
- ~7,000 tok/sec training throughput on M3 Pro
- MLX uses bfloat16 by default for matmuls
- Unified memory architecture eliminates device transfers

---

## References

- nanochat: https://github.com/karpathy/nanochat
- MLX docs: https://ml-explore.github.io/mlx/
- MLX examples: https://github.com/ml-explore/mlx-examples
