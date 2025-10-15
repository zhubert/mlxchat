# mlxchat TODO - MLX Port of nanochat

## Overview

Port nanochat to MLX for single-machine training on Apple Silicon (M3 Pro 36GB).

**Target:** Train d12 (186M params) → d20 (561M params) models on MacBook

---

## ✅ COMPLETED PHASES

### Phase 1.1: GPT Model ✅ (14 tests passing)
- [x] Create `GPTConfig` dataclass
- [ ] Implement `norm()` function (RMSNorm without learnable params)
- [ ] Implement `apply_rotary_emb()` for RoPE
- [ ] Implement `repeat_kv()` for Multi-Query Attention
- [ ] Port `CausalSelfAttention` class
  - [ ] Q/K/V projections (no bias)
  - [ ] RoPE application
  - [ ] QK norm
  - [ ] Causal attention (replace PyTorch's scaled_dot_product_attention)
  - [ ] Output projection
- [ ] Port `MLP` class
  - [ ] 4x expansion
  - [ ] ReLU² activation
  - [ ] Projection back
- [ ] Port `Block` class (attention + MLP with residuals)
- [ ] Port `GPT` class
  - [ ] Token embedding
  - [ ] Transformer blocks
  - [ ] LM head (untied weights)
  - [ ] Rotary embedding precomputation
  - [ ] Weight initialization
  - [ ] Logit softcap (tanh)
- [ ] Add forward pass for training (with loss)
- [ ] Add forward pass for inference (logits only)
- [ ] Test: Forward pass with random inputs

**Key MLX differences:**
- Use `mx.nn.Linear` instead of `torch.nn.Linear`
- Implement causal attention manually (no built-in scaled_dot_product_attention)
- Use `mx.concatenate` instead of `torch.cat`

---

### 1.2 Muon Optimizer (`mlxchat/muon.py`)
**Source:** `nanochat/muon.py` (84 lines, non-distributed only)

- [ ] Port `zeropower_via_newtonschulz5()` function
  - [ ] Matrix operations in MLX
  - [ ] Newton-Schulz iteration (5 steps)
  - [ ] Handle tall/wide matrices
- [ ] Port `Muon` optimizer class
  - [ ] Inherit from MLX optimizer base
  - [ ] SGD with momentum
  - [ ] Orthogonalization post-processing
  - [ ] Aspect ratio scaling
- [ ] Test: Optimizer step on dummy parameters
- [ ] Verify convergence on small problem

**Skip:** `DistMuon` (distributed version not needed)

---

### 1.3 Common Utilities (`mlxchat/common.py`)
**Source:** `nanochat/common.py` (137 lines)

- [ ] Port `get_base_dir()` (cache directory)
- [ ] Port `print0()` (printing utility)
- [ ] Port `print_banner()` (ASCII art)
- [ ] Create simplified `compute_init()`
  - [ ] Remove DDP logic
  - [ ] Keep: seeding, device setup
  - [ ] MLX device is automatic (unified memory)
- [ ] Port `DummyWandb` class

**Remove:** All DDP/distributed code (`get_dist_info`, `is_ddp`, `dist.init_process_group`)

---

## Phase 2: Training Infrastructure (Week 2) 🏗️

### 2.1 Tokenizer Integration (`mlxchat/tokenizer.py`)
**Source:** `nanochat/tokenizer.py` (396 lines)

**Strategy:** Reuse nanochat's tokenizer, just wrap for MLX compatibility

- [ ] Copy special tokens and patterns from nanochat
- [ ] Create wrapper for `RustBPETokenizer`
  - [ ] Point to nanochat's tokenizer directory
  - [ ] Decode/encode methods return MLX arrays
- [ ] Port `get_tokenizer()` helper
- [ ] Port `get_token_bytes()` helper (for evaluation)
- [ ] Test: Encode/decode round-trip

**Note:** Can share the same trained tokenizer from nanochat in `~/.cache/nanochat/tokenizer`

---

### 2.2 Data Loading (`mlxchat/dataloader.py`)
**Source:** `nanochat/dataloader.py`

- [ ] Read nanochat data shards from `~/.cache/nanochat/data/`
- [ ] Implement data loading without DDP
  - [ ] Sequential shard loading
  - [ ] Tokenization on-the-fly or cached
  - [ ] Return MLX arrays instead of PyTorch tensors
- [ ] Implement batching
  - [ ] Fixed sequence length (2048)
  - [ ] Adjustable batch size
- [ ] Test: Load one batch and verify shape

**Key change:** No distributed sampler, just sequential iteration

---

### 2.3 Training Script ✅ (COMPLETE - all features working)
**Source:** `nanochat/scripts/base_train.py` (~300 lines)

- [x] **CRITICAL:** Implement proper gradient accumulation in MLX
  - ✅ Use tree_map(mx.add, ...) to accumulate gradients across micro-batches
  - ✅ Average accumulated gradients before optimizer update
  - ✅ Proper MLX evaluation pattern with mx.eval()

- [x] **CRITICAL:** Implement optimizer updates correctly
  - ✅ Apply Adam optimizer to embedding + lm_head params
  - ✅ Apply Muon optimizer to transformer block params
  - ✅ Split gradients into separate dicts for each optimizer
  - ✅ Handle learning rate scheduling (warmup + constant + warmdown)
  - ✅ Handle momentum scheduling for Muon

- [x] **CRITICAL:** Add gradient clipping
  - ✅ Implemented manual gradient clipping by global norm
  - ✅ Use tree_flatten to get all gradient values
  - ✅ Compute scale factor and apply with tree_map

- [x] Add validation evaluation loop
  - ✅ Periodic eval on held-out data every N steps (--eval-every flag)
  - ✅ Compute validation loss
  - ✅ Track best validation loss
  - ✅ Evaluation runs in eval mode (no gradients)

- [x] Add checkpoint saving/loading
  - ✅ Save model weights (mx.savez to .npz)
  - ✅ Save optimizer state (Adam + Muon)
  - ✅ Save training metadata (step, loss, config)
  - ✅ Periodic checkpoint saving (--save-every flag)
  - ✅ Automatic checkpoint directory setup

- [ ] Add sample generation during training (Optional)
  - Use trained model to generate text
  - Useful for qualitative evaluation

- [ ] Optional: wandb integration for experiment tracking

**Test Results:** ✅
- Created test_train_loop.py to validate training mechanics
- Successfully completes 5 iterations with gradient accumulation
- Multi-optimizer coordination works (Adam + Muon)
- Gradient clipping works correctly
- Checkpoint save/load verified
- ~7,000 tokens/sec on M3 Pro (small 4-layer model)

**Current Status:** COMPLETE - Full training system with validation and checkpointing!

**Location:** `scripts/base_train.py`, `scripts/test_train_loop.py`

---

### 2.4 Checkpoint Manager ✅ (COMPLETE - 4 tests passing)
**Source:** `nanochat/checkpoint_manager.py`

- [x] Implement `save_checkpoint()`
  - ✅ Save MLX model weights (`.npz` format with flattened parameters)
  - ✅ Save optimizer state (supports multiple optimizers)
  - ✅ Save metadata (step, loss, config as JSON)
- [x] Implement `load_checkpoint()`
  - ✅ Load weights into model
  - ✅ Load optimizer state
  - ✅ Return metadata
  - ✅ Unflatten nested structures (dicts + lists)
- [x] Support multiple checkpoints (base, mid, sft, rl)
- [x] Test: Save and load round-trip (4 comprehensive tests)
- [x] Utilities: flatten_dict(), unflatten_dict(), find_last_step(), find_largest_model()

**Test Results:** ✅
- test_flatten_unflatten_dict: Handles nested dicts and lists
- test_save_and_load_checkpoint: Full save/load with optimizers
- test_find_last_step: Finds most recent checkpoint
- test_save_checkpoint_without_optimizer: Saves model only

**Current Status:** COMPLETE - Integrated into training script!

**Location:** `mlxchat/checkpoint_manager.py`, `tests/test_checkpoint.py`

---

## 📋 NEXT PRIORITIES

### Priority 1: Complete Training Script (Phase 2.3) ✅ DONE
**Status:** Core training loop complete and tested!

**Completed:**
1. ✅ Gradient accumulation using tree_map
2. ✅ Multi-optimizer coordination (Adam + Muon)
3. ✅ Gradient clipping by global norm
4. ✅ Learning rate and momentum scheduling
5. ✅ Test script validates all mechanics work

**Remaining (Lower Priority):**
- [ ] Add validation loop
- [ ] Add checkpoint save/load

**Actual Effort:** ~3 hours (research + implementation + testing)

### Priority 2: Checkpoint Manager (Phase 2.4) - NOW UNBLOCKED
**Status:** Training script complete, can now implement checkpointing

- [ ] Implement `save_checkpoint()` function
  - Save MLX model weights (`.npz` or `.safetensors`)
  - Save optimizer state (both Adam and Muon)
  - Save metadata (step, loss, config, etc.)

- [ ] Implement `load_checkpoint()` function
  - Load weights into model
  - Load optimizer state
  - Return metadata for resuming training

- [ ] Support multiple checkpoint types
  - Base model checkpoints
  - Mid-training checkpoints (if doing RL later)

- [ ] Test: Save → Load → Resume training

**Location:** `mlxchat/checkpoint_manager.py` (new file)

**Estimated Effort:** 2-4 hours

---

## 🔮 FUTURE PHASES

### Phase 3: Inference & Chat (Week 3-4) 💬

### 3.1 KV Cache (`mlxchat/engine.py`)
**Source:** `nanochat/engine.py` (344 lines)

- [ ] Port `KVCache` class
  - [ ] Initialize cache tensors
  - [ ] `insert_kv()` method
  - [ ] `prefill()` for batch replication
  - [ ] Dynamic cache growth
- [ ] Port `sample_next_token()`
  - [ ] Temperature sampling
  - [ ] Top-k sampling
- [ ] Port `Engine` class
  - [ ] Prefill phase (process prompt)
  - [ ] Decode phase (autoregressive generation)
  - [ ] Tool use state machine (calculator)
- [ ] Test: Generate text from trained model

**Note:** KV cache is crucial for fast inference

---

### 3.2 Chat CLI (`scripts/chat_cli.py`)
**Source:** `nanochat/scripts/chat_cli.py`

- [ ] Load model and tokenizer
- [ ] Implement conversation loop
  - [ ] Read user input
  - [ ] Format with special tokens
  - [ ] Generate assistant response
  - [ ] Stream tokens as they're generated
- [ ] Support multi-turn conversations
- [ ] Add command-line arguments
  - [ ] Temperature, top-k, max tokens
  - [ ] Model checkpoint selection
- [ ] Test: Chat with d12 model

---

### 3.3 Web UI (`scripts/chat_web.py`)
**Source:** `nanochat/scripts/chat_web.py` (199 lines)

- [ ] FastAPI server setup
- [ ] POST `/chat/completions` endpoint
  - [ ] Parse conversation messages
  - [ ] Tokenize with special tokens
  - [ ] Stream responses (Server-Sent Events)
- [ ] GET `/` endpoint (serve HTML UI)
- [ ] GET `/logo.svg` endpoint
- [ ] GET `/health` endpoint
- [ ] Copy `ui.html` from nanochat
- [ ] Test: Chat via web browser

**Note:** FastAPI code should be mostly unchanged (just model loading differs)

---

## Phase 4: Evaluation & Fine-tuning (Week 4) 📊

### 4.1 Base Evaluation (`scripts/base_eval.py`)
**Source:** `nanochat/scripts/base_eval.py`

- [ ] Port CORE metric evaluation
- [ ] Download eval_bundle from S3
- [ ] Implement loss evaluation on validation set
- [ ] Test: Evaluate d12 model

---

### 4.2 Task Evaluation (`tasks/`)
**Source:** `nanochat/tasks/*.py`

- [ ] Port common evaluation utilities (`tasks/common.py`)
- [ ] Port individual tasks:
  - [ ] ARC-Easy / ARC-Challenge
  - [ ] GSM8K (math problems)
  - [ ] HumanEval (code generation)
  - [ ] MMLU (multiple choice)
  - [ ] SmolTalk (chat quality)
- [ ] Test: Run all evals on trained model

---

### 4.3 Fine-tuning Scripts (Optional, stretch goal)
**Source:** `nanochat/scripts/mid_train.py`, `chat_sft.py`, `chat_rl.py`

- [ ] Mid-training (conversation format)
- [ ] Supervised fine-tuning
- [ ] Reinforcement learning (GRPO on GSM8K)

**Note:** These are lower priority, focus on pretraining first

---

## ✅ Testing & Validation

### Unit Tests (48 tests passing)
- [x] `tests/test_gpt.py` - Model forward/backward passes (14 tests)
- [x] `tests/test_muon.py` - Optimizer convergence (11 tests)
- [x] `tests/test_tokenizer.py` - Encode/decode (12 tests)
- [x] `tests/test_dataloader.py` - Batch loading (11 tests)
- [ ] `tests/test_engine.py` - KV cache and generation (Phase 3)

### Integration Tests (TODO)
- [ ] Train d12 for 10 iterations, verify loss decreases
- [ ] Train d12 for 100 iterations, save/load checkpoint
- [ ] Generate text from trained model
- [ ] Compare MLX vs PyTorch outputs (optional, for validation)

---

## Performance Optimization 🚀

### Memory Optimization
- [ ] Profile memory usage during training
- [ ] Tune batch size for 36GB RAM
- [ ] Implement gradient checkpointing if needed
- [ ] Test max model size (d20? d26?)

### Speed Optimization
- [ ] Benchmark tokens/sec vs PyTorch
- [ ] Profile hotspots with MLX profiler
- [ ] Optimize data loading pipeline
- [ ] Use MLX's compiled functions where possible

---

## Documentation 📚

- [ ] Installation guide
- [ ] Training tutorial (d12 from scratch)
- [ ] Inference tutorial (chat with model)
- [ ] Architecture comparison (MLX vs PyTorch)
- [ ] Troubleshooting guide (OOM, slow training)
- [ ] API reference (generated from docstrings)

---

## Stretch Goals 🌟

- [ ] Support for LoRA fine-tuning
- [ ] Quantization (4-bit, 8-bit inference)
- [ ] Model export to GGUF for llama.cpp
- [ ] Gradio UI (alternative to FastAPI)
- [ ] Distributed training across multiple Macs (MLX distributed)
- [ ] Vision-language model support

---

## 🎯 Success Criteria

**Minimum Viable Product (MVP):**
1. [ ] Train d12 (186M) model for 100+ iterations ← **NEXT MILESTONE**
2. [ ] Verify loss decreases over training
3. [ ] Save and load checkpoints
4. [ ] Generate coherent text from trained model
5. [ ] Chat via CLI

**Full Success (Long-term):**
1. [ ] Train d12 to completion (~20 Chinchilla tokens)
2. [ ] Train d20 (561M) model without OOM
3. [ ] Match nanochat's validation loss
4. [ ] Web UI working
5. [ ] All evaluation tasks passing

---

## ⏱️ Timeline & Progress

**Completed So Far:** ~90% of core infrastructure ✅

| Phase | Status | Time Spent | Priority |
|-------|--------|------------|----------|
| Phase 1.1: GPT Model | ✅ Complete | ~2 hours | **P0** |
| Phase 1.2: Muon Optimizer | ✅ Complete | ~2 hours | **P0** |
| Phase 2.1: Tokenizer | ✅ Complete | ~1 hour | **P0** |
| Phase 2.2: Dataloader | ✅ Complete | ~2 hours | **P0** |
| Phase 2.2.1: Streaming | ✅ Complete | ~2 hours | **P0** |
| Phase 2.3: Training Script | ✅ Complete | ~6 hours | **P0** |
| Phase 2.4: Checkpoints | ✅ Complete | ~3 hours | **P0** |
| Phase 3: Inference | ⏳ Not started | Est: 1 day | **P1** |
| Phase 4: Evaluation | ⏳ Not started | Est: 2 days | **P2** |

**Current Milestone:** Full training system complete! Ready for real training runs ✅

**Total Time to MVP:** Ready NOW! Can train models end-to-end
**Total Time to Full Port:** ~3-4 days for inference + chat + evaluation

---

## Notes & Assumptions

1. **Tokenizer:** Reuse nanochat's trained tokenizer (no need to retrain)
2. **Data:** Use nanochat's downloaded FineWeb shards
3. **Hardware:** Optimized for M3 Pro 36GB (adjust batch size for other configs)
4. **MLX Version:** Requires MLX ≥0.20.0
5. **Python:** Python 3.10+ required

---

## ❓ Open Questions & Research Needed

### Critical Questions - RESOLVED ✅
- [x] **MLX Gradient Accumulation:** How to properly accumulate gradients across micro-batches?
  - ✅ SOLVED: Use `tree_map(mx.add, grads, accum_grads)` to accumulate
  - ✅ Average with `tree_map(lambda g: g / n_steps, accum_grads)` before update
  - ✅ Call `mx.eval(accum_grads)` after each accumulation

- [x] **MLX Multi-Optimizer:** How to coordinate Adam + Muon on different param groups?
  - ✅ SOLVED: Split gradients into separate dicts by key ("wte", "lm_head", "h")
  - ✅ Call optimizer.update() separately for each param/grad pair
  - ✅ Evaluate both optimizer states with `mx.eval(adam_opt.state, muon_opt.state)`

- [x] **MLX Gradient Clipping:** No built-in `clip_grad_norm_`, how to implement?
  - ✅ SOLVED: Use `tree_flatten(grads)` to get (path, value) tuples
  - ✅ Compute global norm: `sqrt(sum(sum(v^2) for _, v in flattened))`
  - ✅ Scale with `tree_map(lambda g: g * scale, grads)`

### Performance Questions
- [x] What's the actual tokens/sec on M3 Pro vs 8xH100?
  - ✅ Measured: ~7,000 tok/sec for small model (4 layers, 256 dim) on M3 Pro
  - Need to benchmark with full d12 model (12 layers, 768 dim)
  - Expect: ~2,000-4,000 tok/sec for d12 on M3 Pro

- [ ] Do we need mixed precision in MLX?
  - MLX uses bfloat16 by default for matmuls
  - Is explicit dtype management needed?

### Future Optimization
- [ ] Can we share optimizer state between nanochat/mlxchat checkpoints?
  - Likely not directly compatible (PyTorch vs MLX)
  - Could write conversion utility

- [ ] Should we use MLX's graph compilation?
  - `mx.compile()` for performance
  - Need to test if it works with our model

---

## References

- nanochat: https://github.com/karpathy/nanochat
- MLX docs: https://ml-explore.github.io/mlx/
- MLX examples: https://github.com/ml-explore/mlx-examples
