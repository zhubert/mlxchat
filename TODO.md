# mlxchat TODO - MLX Port of nanochat

## Overview

Port nanochat to MLX for single-machine training on Apple Silicon (M3 Pro 36GB).

**Target:** Train d12 (186M params) → d20 (561M params) models on MacBook

---

## Phase 1: Core Model (Week 1) 🎯

### 1.1 GPT Model (`mlxchat/gpt.py`)
**Source:** `nanochat/gpt.py` (323 lines)

- [ ] Create `GPTConfig` dataclass
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

### 2.3 Training Script (`scripts/base_train.py`)
**Source:** `nanochat/scripts/base_train.py` (~300 lines)

- [ ] Port configuration system
  - [ ] Model architecture (depth → n_layer, n_embd, n_head)
  - [ ] Training hyperparameters
  - [ ] Batch size and gradient accumulation
- [ ] Initialize model
  - [ ] Create GPT with config
  - [ ] Initialize weights
- [ ] Setup optimizers
  - [ ] Muon for transformer blocks
  - [ ] Adam for embeddings/lm_head
  - [ ] Learning rates with scaling
- [ ] Training loop
  - [ ] Forward pass (compute loss)
  - [ ] Backward pass (MLX auto-differentiation)
  - [ ] Gradient accumulation
  - [ ] Optimizer step
  - [ ] Logging (loss, tokens/sec)
- [ ] Checkpoint saving
  - [ ] Save model weights
  - [ ] Save optimizer state
  - [ ] Save training metadata
- [ ] Validation evaluation
  - [ ] Periodic eval on held-out data
  - [ ] Compute bits-per-byte
- [ ] Integration with wandb (optional)
- [ ] Test: Train for 10 steps on dummy data

**Key differences:**
- Remove `torchrun` wrapper
- Remove DDP sync points
- Use MLX's `mx.grad()` for backprop
- Single device (no device_id management)

---

### 2.4 Checkpoint Manager (`mlxchat/checkpoint_manager.py`)
**Source:** `nanochat/checkpoint_manager.py`

- [ ] Implement `save_checkpoint()`
  - [ ] Save MLX model weights (`.npz` format)
  - [ ] Save optimizer state
  - [ ] Save metadata (step, loss, config)
- [ ] Implement `load_checkpoint()`
  - [ ] Load weights into model
  - [ ] Load optimizer state
  - [ ] Return metadata
- [ ] Support multiple checkpoints (base, mid, sft, rl)
- [ ] Test: Save and load round-trip

---

## Phase 3: Inference & Chat (Week 3) 💬

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

## Testing & Validation ✅

### Unit Tests
- [ ] `tests/test_gpt.py` - Model forward/backward passes
- [ ] `tests/test_muon.py` - Optimizer convergence
- [ ] `tests/test_tokenizer.py` - Encode/decode
- [ ] `tests/test_dataloader.py` - Batch loading
- [ ] `tests/test_engine.py` - KV cache and generation

### Integration Tests
- [ ] Train d12 for 100 steps, verify loss decreases
- [ ] Generate coherent text from trained model
- [ ] Compare MLX outputs to PyTorch nanochat (same init)

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

## Success Criteria 🎯

**Minimum Viable Product (MVP):**
1. ✅ Train d12 (186M) model on MacBook M3 Pro
2. ✅ Achieve similar loss curve to nanochat
3. ✅ Generate coherent text
4. ✅ Chat via CLI

**Full Success:**
1. ✅ Train d20 (561M) model without OOM
2. ✅ Match nanochat's CORE metric within 5%
3. ✅ Web UI working
4. ✅ All evaluation tasks passing

---

## Timeline Estimate

| Phase | Duration | Priority |
|-------|----------|----------|
| Phase 1: Core Model | 1 week | **P0** (Critical) |
| Phase 2: Training | 1 week | **P0** (Critical) |
| Phase 3: Inference | 3 days | **P1** (High) |
| Phase 4: Evaluation | 4 days | **P2** (Medium) |
| Optimization | Ongoing | **P2** (Medium) |
| Fine-tuning | 1 week | **P3** (Low) |

**Total MVP:** ~2.5 weeks
**Total Full Port:** ~4 weeks

---

## Notes & Assumptions

1. **Tokenizer:** Reuse nanochat's trained tokenizer (no need to retrain)
2. **Data:** Use nanochat's downloaded FineWeb shards
3. **Hardware:** Optimized for M3 Pro 36GB (adjust batch size for other configs)
4. **MLX Version:** Requires MLX ≥0.20.0
5. **Python:** Python 3.10+ required

---

## Questions to Resolve

- [ ] Can we share optimizer state between nanochat/mlxchat checkpoints?
- [ ] What's the actual tokens/sec on M3 Pro vs 8xH100?
- [ ] Do we need mixed precision training in MLX? (bf16 by default?)
- [ ] Can we reuse nanochat's Rust tokenizer binary or rebuild?

---

## References

- nanochat: https://github.com/karpathy/nanochat
- MLX docs: https://ml-explore.github.io/mlx/
- MLX examples: https://github.com/ml-explore/mlx-examples
