# Memory Optimization Guide

Complete guide for reducing memory usage during mlxchat training and inference.

---

## Quick Start

### Check Your Memory
```bash
make memory-info
```

### Choose Training Mode Based on RAM

**16-32 GB RAM (Recommended):**
```bash
make train-low-memory
```
Uses 10-15 GB, trains d12 (186M params)

**8-16 GB RAM:**
```bash
make train-ultra-low
```
Uses 6-10 GB, trains d12 (186M params)

**32+ GB RAM:**
```bash
make train
```
Standard training (~90 MB with batch_size=8, optimal performance)

### Monitor Memory (Optional)
Open a second terminal:
```bash
make monitor-memory
```

---

## Training Commands Reference

| Command | Memory Usage | Description |
|---------|--------------|-------------|
| `make train-low-memory` | 10-15 GB | Recommended for most users |
| `make train-ultra-low` | 6-10 GB | Minimal memory usage |
| `make train-profile` | 10-15 GB | With detailed profiling |
| `make monitor-memory` | N/A | Real-time memory monitor |

### Custom Training Options

```bash
# Train with specific depth
make train-low-memory DEPTH=16

# Train for specific number of steps
make train-low-memory STEPS=1000

# Adjust save frequency
make train-low-memory SAVE_EVERY=100
```

---

## Model Size Guidelines

Choose model depth based on available memory:

| Model | Params | Training Memory | Inference Memory | Recommended RAM |
|-------|--------|-----------------|------------------|-----------------|
| d12   | 186M   | 10-15 GB        | 2-4 GB           | 16+ GB          |
| d16   | 336M   | 18-25 GB        | 4-7 GB           | 32+ GB          |
| d20   | 561M   | 30-40 GB        | 7-12 GB          | 64+ GB          |
| d26   | 1.1B   | 60-80 GB        | 14-24 GB         | 128+ GB         |

**Memory Formula:**
- Training: ~3-4x model size (includes optimizer states, gradients, activations)
- Inference: ~2x model size (includes KV cache)

---

## Low-Memory Training Mode

### Automatic Configuration

The `--low-memory` flag automatically optimizes:
- ✅ Reduces batch size (8 → 2)
- ✅ Reduces total batch tokens (524K → 262K)
- ✅ Reduces cached shards (20 → 5)
- ✅ Evaluates less frequently (250 → 500 steps)
- ✅ Reduces eval tokens (10M → 524K)
- ✅ Aggressive garbage collection (every 5 steps)

**Expected memory usage:** 10-15 GB for d12 model

### Using Low-Memory Mode

```bash
# Via Makefile (recommended)
make train-low-memory

# Or directly with Python
python -m scripts.base_train \
  --depth=12 \
  --streaming \
  --low-memory \
  --profile
```

---

## Ultra-Low Memory Mode

For extreme memory savings (6-10 GB for d12):

```bash
# Via Makefile
make train-ultra-low

# Or manually configured
python -m scripts.base_train \
  --depth=12 \
  --device-batch-size=1 \
  --total-batch-size=65536 \
  --streaming \
  --max-cached-shards=3 \
  --eval-every=1000 \
  --eval-tokens=262144 \
  --profile
```

---

## Quantization for Inference

Reduce model size by 4-8x for inference with minimal quality loss.

### 4-bit Quantization (8x reduction)

```bash
# Via Makefile
make quantize

# Or manually
python -m scripts.quantize_checkpoint \
  --checkpoint ~/.cache/mlxchat/base_checkpoints/d12/model_latest.npz \
  --bits 4 \
  --output ~/.cache/mlxchat/base_checkpoints/d12/model_latest_q4.npz
```

### 8-bit Quantization (4x reduction)

```bash
# Via Makefile
make quantize-8bit

# Or manually
python -m scripts.quantize_checkpoint \
  --checkpoint ~/.cache/mlxchat/base_checkpoints/d12/model_latest.npz \
  --bits 8 \
  --output ~/.cache/mlxchat/base_checkpoints/d12/model_latest_q8.npz
```

### Use Quantized Model

```bash
# Via Makefile
make chat-quantized

# Or manually
python -m scripts.chat_cli \
  --checkpoint ~/.cache/mlxchat/base_checkpoints/d12/model_latest_q4.npz
```

### Memory Savings

| Model | Full Size | 4-bit | 8-bit |
|-------|-----------|-------|-------|
| d12 (186M) | 1.4 GB | 350 MB | 700 MB |
| d16 (336M) | 2.5 GB | 625 MB | 1.25 GB |
| d20 (561M) | 4.2 GB | 1.0 GB | 2.1 GB |
| d26 (1.1B) | 8.3 GB | 2.1 GB | 4.2 GB |

---

## Advanced Optimization Techniques

### 1. Reduce Gradient Accumulation

Lower `--total-batch-size` to reduce gradient accumulation steps:

```bash
# Default: 524K tokens = 64 accumulation steps (batch_size=8)
--total-batch-size=131072  # 16 steps

# Ultra-low:
--total-batch-size=65536   # 8 steps
```

### 2. Streaming Mode

Always use streaming mode to avoid downloading entire dataset:

```bash
--streaming --max-cached-shards=5
```

**Storage requirements:**
- Streaming (5 shards): ~800 MB
- Streaming (20 shards): ~3.2 GB
- Full download: ~300 GB

### 3. Reduce Evaluation Frequency

Evaluation requires extra memory for validation passes:

```bash
--eval-every=1000      # Less frequent (default: 250)
--eval-tokens=262144   # Fewer tokens (default: 10M)
```

### 4. Profile Memory Usage

Use `--profile` flag to see detailed memory breakdown:

```bash
python -m scripts.base_train --depth=12 --profile
```

Shows:
- Memory usage per training step
- Time breakdown for each operation
- Helps identify memory bottlenecks

---

## Understanding Memory Usage

### Memory Breakdown for d12 Model

1. **Model weights** (~1.5 GB in bfloat16)
   - Token embeddings: 400 MB
   - Transformer layers: 900 MB
   - Output layer: 200 MB

2. **Optimizer states** (~3 GB)
   - Adam: 2x model size (momentum + variance)
   - Muon: 1x model size (momentum only)
   - Total: ~3x model size

3. **Gradients** (~1.5 GB)
   - Same size as model weights

4. **Activations** (varies by batch size)
   - Batch size 8: ~2-4 GB
   - Batch size 2: ~0.5-1 GB
   - Batch size 1: ~0.25-0.5 GB

5. **Data shards** (varies by streaming config)
   - 5 shards: ~800 MB
   - 20 shards: ~3.2 GB

**Total for d12:**
- With batch_size=8: ~10-15 GB
- With batch_size=1: ~6-10 GB

---

## Troubleshooting

### Problem: Training still uses too much memory

**Solutions:**
1. Use `--low-memory` flag or reduce to d12: `--depth=12`
2. Reduce batch size to 1: `--device-batch-size=1`
3. Disable evaluation temporarily: `--eval-every=100000`
4. Check for other processes using memory
5. Restart machine to clear lingering processes

### Problem: Out of Memory during evaluation

**Solutions:**
1. Reduce eval tokens: `--eval-tokens=131072`
2. Reduce batch size: `--device-batch-size=1`
3. Evaluate less often: `--eval-every=1000`

### Problem: Memory keeps growing during training

**Solutions:**
1. Use `--low-memory` flag (enables aggressive GC)
2. Monitor with: `python -m scripts.monitor_memory`
3. Reduce checkpoint frequency: `--save-every=50`
4. Close other applications

### Problem: Still crashing with low-memory mode

Try manual ultra-low configuration:

```bash
uv run python -m scripts.base_train \
  --depth=12 \
  --device-batch-size=1 \
  --total-batch-size=32768 \
  --streaming \
  --max-cached-shards=2 \
  --eval-every=2000 \
  --eval-tokens=131072 \
  --low-memory
```

---

## Memory Optimization Checklist

Before starting training, verify:

- [ ] Using `--low-memory` flag OR manually set batch_size=1-2
- [ ] Using `--streaming` mode
- [ ] Model depth appropriate for your RAM (d12 for 16-32GB)
- [ ] Running memory monitor in separate terminal
- [ ] Closed unnecessary applications
- [ ] Using `--profile` to identify bottlenecks
- [ ] Set reasonable `--eval-every` interval (500-1000)

---

## Monitoring Tools

### Real-time Memory Monitor

Run in a separate terminal while training:

```bash
make monitor-memory

# Or directly
python -m scripts.monitor_memory
```

Displays:
- Total system memory and usage
- Python process memory consumption
- Warnings when usage exceeds 80%
- Updates every 2 seconds

### Memory Info

Check system capabilities and recommendations:

```bash
make memory-info
```

Shows:
- Total system RAM
- Available memory
- Recommended training mode
- Disk space usage

---

## Best Practices

### 1. Start Small
- Begin with d12 (186M params)
- Test with `--low-memory` flag
- Monitor memory usage
- Scale up only if comfortable

### 2. Always Use Streaming
- Enables training on full dataset
- Uses only 1-3 GB disk space
- No need to download 300 GB

### 3. Monitor During Training
- Run `make monitor-memory` in separate terminal
- Watch for memory pressure warnings
- Stop training if memory exceeds 90%

### 4. Quantize for Production
- 4-bit models are 8x smaller
- Minimal quality loss (~1%)
- Much faster inference
- Run `make quantize` after training

### 5. Profile First Run
- Use `--profile` on first training run
- Identify memory bottlenecks
- Adjust settings accordingly
- Remove `--profile` for production runs

---

## Summary Commands

**Training:**
```bash
# Low memory (10-15 GB)
make train-low-memory

# Ultra-low memory (6-10 GB)
make train-ultra-low

# With profiling
make train-profile
```

**Monitoring:**
```bash
# Real-time memory tracking
make monitor-memory

# System info and recommendations
make memory-info
```

**Quantization:**
```bash
# 4-bit quantization (8x smaller)
make quantize

# 8-bit quantization (4x smaller)
make quantize-8bit

# Chat with quantized model
make chat-quantized
```

---

## Additional Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Memory Profiling Guide](https://ml-explore.github.io/mlx/build/html/usage/memory.html)
- [Apple Silicon Unified Memory](https://support.apple.com/guide/mac-help/mh25842/mac)
- Project README: `README.md`
- Developer Guide: `CLAUDE.md`
