```
    ███╗   ███╗██╗     ██╗  ██╗ ██████╗██╗  ██╗ █████╗ ████████╗
    ████╗ ████║██║     ╚██╗██╔╝██╔════╝██║  ██║██╔══██╗╚══██╔══╝
    ██╔████╔██║██║      ╚███╔╝ ██║     ███████║███████║   ██║
    ██║╚██╔╝██║██║      ██╔██╗ ██║     ██╔══██║██╔══██║   ██║
    ██║ ╚═╝ ██║███████╗██╔╝ ██╗╚██████╗██║  ██║██║  ██║   ██║
    ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝
```

# mlxchat

> nanochat ported to Apple MLX - Train GPT models on your MacBook

An MLX port of [nanochat](https://github.com/karpathy/nanochat), optimized for Apple Silicon (M1/M2/M3/M4).

## Why MLX?

- **Unified Memory**: Leverage the full 36GB of your MacBook's RAM for training
- **Apple Silicon Optimized**: Native Metal acceleration for M-series chips
- **Single Machine**: No distributed training complexity - just one computer
- **Same Architecture**: Identical GPT model to nanochat (RoPE, QK norm, MQA, ReLU²)

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. It works seamlessly with this project:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -e .

# Or install with dev dependencies
uv pip install -e ".[dev]"

# Run commands using uv (creates virtual env automatically)
uv run python -m scripts.base_train --depth=12 --streaming
```

**Why uv?**
- 10-100x faster than pip for dependency resolution
- Built-in virtual environment management
- Drop-in replacement for pip (works with existing `pyproject.toml`)
- No changes needed to the project structure

### Using pip

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

### Using Make (Easiest)

```bash
# See all available commands
make help

# Test training (2 steps, ~1 minute)
make train-test

# Quick training run (100 steps, ~5-10 minutes)
make train-quick

# Full training run (d12 model, ~3225 steps)
make train

# Train larger models
make train-d16      # 336M params
make train-d20      # 561M params

# Custom training
make train DEPTH=12 STEPS=500 BATCH_SIZE=8

# After training, chat with your model
make chat-cli       # Terminal interface

# Other utilities
make test           # Run all tests
make format         # Format code
make disk-usage     # Check cache sizes
```

### Using Python Directly

```bash
# Test training loop (2 steps)
uv run python -u -m scripts.base_train \
  --depth=12 \
  --streaming \
  --num-iterations 2

# Full training with streaming (recommended)
uv run python -u -m scripts.base_train \
  --depth=12 \
  --device-batch-size=8 \
  --streaming \
  --max-cached-shards 20

# Train larger model with custom batch size
uv run python -u -m scripts.base_train \
  --depth=16 \
  --device-batch-size=6 \
  --streaming

# Download shards manually (optional)
uv run python -m mlxchat.dataset --num-shards 50

# Chat with trained model
uv run python -m scripts.chat_cli --checkpoint ~/.cache/mlxchat/base_checkpoints/d12/model_latest.npz
```

## Common Workflows

### First Time Setup
```bash
# 1. Install dependencies
make install

# 2. Run tests to verify installation
make test

# 3. Quick training test (2 steps, verifies everything works)
make train-test
```

### Development Workflow
```bash
# Run quick training to test changes (100 steps)
make train-quick

# Monitor performance
make disk-usage          # Check cache usage
make watch-checkpoints   # Watch checkpoint saves in real-time

# Code quality
make format              # Format before committing
make lint                # Check for issues
```

### Production Training
```bash
# Train d12 model (start here)
make train               # ~8 hours, 186M params

# Train larger models (more capable but slower)
make train-d16          # ~14 hours, 336M params
make train-d20          # ~24 hours, 561M params

# Resume from checkpoint (automatic)
# Training will resume from latest checkpoint if interrupted
```

### After Training
```bash
# Chat with your model
make chat-cli           # Terminal interface
```

## Model Sizes

Recommended for M3 Pro 36GB:

| Model | Params | Memory | Training Time | Recommendation |
|-------|--------|--------|---------------|----------------|
| d12   | 186M   | ~4GB   | ~8 hours      | ✅ Start here  |
| d16   | 336M   | ~7GB   | ~14 hours     | ✅ Good balance |
| d20   | 561M   | ~12GB  | ~24 hours     | ✅ Max recommended |
| d26   | 1.1B   | ~24GB  | ~48 hours     | ⚠️ Experimental |

## Project Status

**Complete** - Full training-to-inference pipeline working!

- [x] Phase 1.1: GPT Model (14 tests passing)
- [x] Phase 1.2: Muon Optimizer (11 tests passing)
- [x] Phase 2.1: Tokenizer (12 tests passing)
- [x] Phase 2.2: Dataloader with Streaming (11 tests passing)
- [x] Phase 2.3: Training Script (gradient accumulation, multi-optimizer, clipping, validation, ETA tracking)
- [x] Phase 2.4: Checkpoint Manager (4 tests passing)
- [x] Phase 3: Inference Engine & Chat CLI (KV cache, streaming generation, temperature/top-k sampling)

**Latest**: Complete training system! Train GPT models on your MacBook with streaming data support, automatic checkpoint management, and chat with trained models via CLI. Training runs with accurate time estimates, periodic validation, and automatic resumption from checkpoints.

**Features**:
- Streaming data mode (train on 300GB dataset with only 3-8GB storage)
- Low-memory training mode (10-15 GB for d12, 6-10 GB ultra-low mode)
- Efficient training (~7K tok/sec on M3 Pro)
- Fast inference with KV cache
- Model quantization (4-bit/8-bit for 4-8x memory reduction)
- Automatic checkpoint save/load
- Multi-turn chat interface
- Memory monitoring and profiling tools
- Pre-commit hooks and code formatting

## Future Work (Optional)

Core functionality is complete! These are optional enhancements:

### Web UI
- FastAPI server with `/chat/completions` endpoint
- Browser-based chat interface
- Server-Sent Events for streaming

### Evaluation Tasks
- Base evaluation (validation loss, perplexity)
- ARC-Easy / ARC-Challenge
- GSM8K (math problems)
- HumanEval (code generation)
- MMLU (multiple choice)

### Fine-tuning
- Mid-training for conversation format
- Supervised fine-tuning (SFT)
- Reinforcement learning with GRPO

### Stretch Goals
- LoRA fine-tuning support
- Model export to GGUF format
- Gradio UI alternative
- Gradient checkpointing for larger models

## Memory Optimization

For detailed memory optimization strategies, see [MEMORY.md](MEMORY.md).

**Quick reference:**
- 16-32 GB RAM: `make train-low-memory` (10-15 GB usage)
- 8-16 GB RAM: `make train-ultra-low` (6-10 GB usage)
- Monitor memory: `make monitor-memory`
- Quantize models: `make quantize` (4-8x reduction)

## Data Management for Limited Storage

mlxchat supports **streaming mode** to handle the ~300GB FineWeb dataset on MacBooks with limited storage:

### Streaming Mode (Recommended) 🌟
```bash
python -m scripts.base_train --streaming --max-cached-shards 20
```
- **Storage**: ~3-8 GB (only keeps 20-50 shards on disk)
- **Coverage**: Full 100B token dataset
- Downloads shards on-demand as training progresses
- Automatically removes old shards when cache is full

### Pre-download Mode
```bash
# Download specific number of shards first
python -m mlxchat.dataset --num-shards 50

# Then train without streaming
python -m scripts.base_train --depth=12
```

### Storage Requirements by Strategy

| Strategy | Storage | Dataset Coverage |
|----------|---------|------------------|
| Streaming (20 shards) | ~3.2 GB | 100% (1823 shards) |
| Streaming (50 shards) | ~8 GB | 100% (1823 shards) |
| Pre-download 50 | ~8 GB | ~2.7% (50 shards) |
| Pre-download 100 | ~16 GB | ~5.5% (100 shards) |
| Full download | ~300 GB | 100% (1823 shards) |

## Differences from nanochat

1. **No Distributed Training**: Single machine, gradient accumulation only
2. **MLX instead of PyTorch**: Native Apple Silicon acceleration
3. **Simplified Optimizers**: No DistMuon/DistAdamW
4. **Same Tokenizer**: Reuses rustbpe + tiktoken from nanochat
5. **Streaming Data**: On-demand shard downloads for limited storage

## Architecture

```
mlxchat/
├── mlxchat/           # Core library
│   ├── gpt.py         # GPT model (MLX)
│   ├── muon.py        # Muon optimizer (MLX)
│   ├── engine.py      # Inference engine with KV cache
│   ├── dataloader.py  # Data loading utilities
│   └── tokenizer.py   # Tokenizer wrapper (reuses nanochat's)
├── scripts/           # Training and evaluation scripts
│   ├── base_train.py  # Pretraining script
│   ├── chat_cli.py    # CLI chat interface
│   └── chat_web.py    # Web UI (FastAPI)
└── tests/             # Unit tests

```

## License

MIT (same as nanochat)

## Acknowledgements

- Original nanochat by [Andrej Karpathy](https://github.com/karpathy)
- MLX by Apple
