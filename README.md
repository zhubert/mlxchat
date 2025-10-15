# mlxchat

> nanochat ported to Apple MLX - Train GPT models on your MacBook

An MLX port of [nanochat](https://github.com/karpathy/nanochat), optimized for Apple Silicon (M1/M2/M3/M4).

## Why MLX?

- **Unified Memory**: Leverage the full 36GB of your MacBook's RAM for training
- **Apple Silicon Optimized**: Native Metal acceleration for M-series chips
- **Single Machine**: No distributed training complexity - just one computer
- **Same Architecture**: Identical GPT model to nanochat (RoPE, QK norm, MQA, ReLU²)

## Quick Start

```bash
# Install dependencies
pip install -e .

# Option 1: Streaming mode (recommended for limited storage)
# Downloads shards on-demand, keeps only 20 shards cached (~3GB)
python -m scripts.base_train --depth=12 --streaming

# Option 2: Pre-download a subset (for faster iteration)
# Download first 50 shards (~8GB)
python -m mlxchat.dataset --num-shards 50

# Train a small model (d12, 186M params)
# Core training loop works! Validation and checkpointing coming soon.
python -m scripts.base_train --depth=12 --device_batch_size=4

# Test training loop with dummy data
python scripts/test_train_loop.py

# Chat with your model (coming soon)
# python -m scripts.chat_cli
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

**~80% Complete** - Core training infrastructure complete and tested!

- [x] Phase 1.1: GPT Model (14 tests passing)
- [x] Phase 1.2: Muon Optimizer (11 tests passing)
- [x] Phase 2.1: Tokenizer (12 tests passing)
- [x] Phase 2.2: Dataloader with Streaming (11 tests passing)
- [x] Phase 2.3: Training Script (core complete - gradient accumulation, multi-optimizer, clipping)
- [ ] Phase 2.4: Checkpoint Manager
- [ ] Phase 3: Inference & UI (chat CLI, web interface)

**Latest**: Training loop working! Successfully tested with gradient accumulation, dual optimizers (Adam + Muon), and gradient clipping. Achieving ~7K tok/sec on small models.

**Next Milestone**: Add checkpoint save/load functionality

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
