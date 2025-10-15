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
pip install mlx numpy tiktoken

# Train a small model (d12, 186M params)
python -m scripts.base_train --depth=12 --device_batch_size=4

# Chat with your model
python -m scripts.chat_cli
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

- [ ] Phase 1: Core Model (GPT, Muon optimizer)
- [ ] Phase 2: Training Loop (data loading, training script)
- [ ] Phase 3: Inference & UI (chat CLI, web interface)

## Differences from nanochat

1. **No Distributed Training**: Single machine, gradient accumulation only
2. **MLX instead of PyTorch**: Native Apple Silicon acceleration
3. **Simplified Optimizers**: No DistMuon/DistAdamW
4. **Same Tokenizer**: Reuses rustbpe + tiktoken from nanochat

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
