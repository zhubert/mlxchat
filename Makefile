.PHONY: help install install-rust build-tokenizer train-tokenizer test train train-quick train-test train-low-memory train-ultra-low train-profile chat-cli chat-web chat-quantized download-data format lint clean monitor-memory quantize quantize-4bit quantize-8bit memory-info disk-usage

# Default target
help:
	@echo "mlxchat - MLX port of nanochat for Apple Silicon"
	@echo ""
	@echo "Available targets:"
	@echo "  make install       - Install package with dev dependencies"
	@echo "  make install-rust  - Install Rust toolchain (required for tokenizer)"
	@echo "  make build-tokenizer - Build rustbpe tokenizer (2-3x faster than tiktoken)"
	@echo "  make train-tokenizer - Train a custom tokenizer (recommended before training)"
	@echo "  make test          - Run all tests"
	@echo ""
	@echo "Training:"
	@echo "  make train         - Train d12 model (full run, resumes automatically)"
	@echo "  make train-quick   - Quick training test (100 steps, resumes automatically)"
	@echo "  make train-test    - Minimal training test (2 steps)"
	@echo "  make train-low-memory - Train with aggressive memory optimization (10-15 GB)"
	@echo "  make train-ultra-low  - Train with ultra-low memory settings (6-10 GB)"
	@echo "  make train-profile    - Train with detailed profiling and memory tracking"
	@echo ""
	@echo "Memory Optimization:"
	@echo "  make memory-info      - Show memory usage and optimization recommendations"
	@echo "  make monitor-memory   - Monitor system memory usage in real-time"
	@echo "  make quantize         - Quantize trained model to 4-bit (8x smaller)"
	@echo "  make quantize-8bit    - Quantize trained model to 8-bit (4x smaller)"
	@echo "  make chat-quantized   - Chat with quantized model"
	@echo ""
	@echo "Inference:"
	@echo "  make chat-cli      - Run CLI chat interface"
	@echo "  make chat-web      - Run web UI chat interface"
	@echo ""
	@echo "Data Management:"
	@echo "  make download-data - Download first 50 shards (~8GB)"
	@echo ""
	@echo "Monitoring:"
	@echo "  make logs          - List recent training logs"
	@echo "  make tail-log      - Follow latest training log in real-time"
	@echo "  make view-log      - View latest training log"
	@echo "  make disk-usage    - Show disk usage for cache/checkpoints"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make clean         - Clean cache and build artifacts"
	@echo ""
	@echo "Training with custom depth:"
	@echo "  make train DEPTH=16 STEPS=1000 SAVE_EVERY=100"
	@echo ""
	@echo "Features:"
	@echo "  - Streaming mode enabled by default (20 cached shards)"
	@echo "  - Automatic checkpoint resumption (--resume)"
	@echo "  - Timestamped logs in ~/.cache/mlxchat/logs/"
	@echo "  - Low-memory mode for training on limited RAM"
	@echo "  - 4-bit/8-bit quantization for inference"

# Installation
install:
	uv pip install -e ".[dev]"

install-rust:
	@echo "Installing Rust toolchain..."
	@command -v rustc >/dev/null 2>&1 || { \
		echo "Rust not found. Installing via rustup..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		echo ""; \
		echo "Rust installed! Please run:"; \
		echo "  source \"$$HOME/.cargo/env\""; \
		echo "  make build-tokenizer"; \
	}
	@command -v rustc >/dev/null 2>&1 && echo "Rust is already installed: $$(rustc --version)"

build-tokenizer:
	@echo "Building rustbpe tokenizer..."
	@command -v rustc >/dev/null 2>&1 || { \
		echo "Error: Rust is not installed. Run 'make install-rust' first."; \
		exit 1; \
	}
	cd rustbpe && uv run maturin develop --release
	@echo ""
	@echo "✓ rustbpe tokenizer built successfully!"
	@echo "  This tokenizer is 2-3x faster than tiktoken."
	@echo "  You can now train tokenizers with: make train-tokenizer"

train-tokenizer:
	@echo "Training tokenizer (this will take a few minutes)..."
	@echo "Training on 10M characters from FineWeb..."
	uv run python -m scripts.tok_train --max_chars 10000000 --vocab_size 65536
	@echo ""
	@echo "✓ Tokenizer training complete!"
	@echo "  Location: ~/.cache/mlxchat/tokenizer/tokenizer.pkl"

# Testing
test:
	uv run pytest -v

test-model:
	uv run pytest tests/test_gpt.py -v

test-dataloader:
	uv run pytest tests/test_dataloader.py -v

# Training targets
DEPTH ?= 12
BATCH_SIZE ?= 4
TOTAL_BATCH_SIZE ?= 16384
STEPS ?= -1
CACHED_SHARDS ?= 20
EVAL_EVERY ?= 25
SAVE_EVERY ?= 50

train:
	uv run python -u -m scripts.base_train \
		--depth=$(DEPTH) \
		--streaming \
		--max-cached-shards $(CACHED_SHARDS) \
		--device-batch-size $(BATCH_SIZE) \
		--total-batch-size $(TOTAL_BATCH_SIZE) \
		--num-iterations $(STEPS) \
		--eval-every $(EVAL_EVERY) \
		--save-every $(SAVE_EVERY) \
		--resume

train-quick:
	uv run python -u -m scripts.base_train \
		--depth=12 \
		--streaming \
		--max-cached-shards 20 \
		--device-batch-size 8 \
		--num-iterations 100 \
		--save-every 25 \
		--resume

train-test:
	uv run python -u -m scripts.base_train \
		--depth=12 \
		--streaming \
		--max-cached-shards 5 \
		--device-batch-size 8 \
		--num-iterations 2 \
		--save-every 1 \
		--resume

# Low-memory training targets
train-low-memory:
	@echo "Starting low-memory training mode (10-15 GB expected)"
	@echo "Tip: Run 'make monitor-memory' in another terminal to track usage"
	@echo ""
	uv run python -u -m scripts.base_train \
		--depth=12 \
		--streaming \
		--low-memory \
		--resume

train-ultra-low:
	@echo "Starting ultra-low memory training mode (6-10 GB expected)"
	@echo "Tip: Run 'make monitor-memory' in another terminal to track usage"
	@echo ""
	uv run python -u -m scripts.base_train \
		--depth=12 \
		--device-batch-size 1 \
		--total-batch-size 65536 \
		--streaming \
		--max-cached-shards 3 \
		--eval-every 1000 \
		--eval-tokens 262144 \
		--resume

train-profile:
	@echo "Starting training with detailed profiling and memory tracking"
	@echo "This will show memory usage and timing breakdown for each step"
	@echo ""
	uv run python -u -m scripts.base_train \
		--depth=12 \
		--streaming \
		--low-memory \
		--profile \
		--resume

# Training different model sizes
train-d12:
	$(MAKE) train DEPTH=12 BATCH_SIZE=8

train-d16:
	$(MAKE) train DEPTH=16 BATCH_SIZE=6

train-d20:
	$(MAKE) train DEPTH=20 BATCH_SIZE=4

train-d26:
	$(MAKE) train DEPTH=26 BATCH_SIZE=2

# Inference
CHECKPOINT ?= ~/.cache/mlxchat/base_checkpoints
TEMP ?= 1.0
TOP_K ?= 50

chat-cli:
	uv run python -m scripts.chat_cli \
		--checkpoint $(CHECKPOINT) \
		--temperature $(TEMP) \
		--top-k $(TOP_K)

chat-web:
	uv run python -m scripts.chat_web \
		--checkpoint $(CHECKPOINT) \
		--port 8000

# Memory optimization
monitor-memory:
	@echo "Monitoring system memory usage..."
	@echo "Press Ctrl+C to stop"
	@echo ""
	uv run python -m scripts.monitor_memory

# Quantization targets
CHECKPOINT_FILE ?= ~/.cache/mlxchat/base_checkpoints/d12/model_latest.npz
BITS ?= 4

quantize:
	@echo "Quantizing model to $(BITS)-bit..."
	@echo "Checkpoint: $(CHECKPOINT_FILE)"
	@echo ""
	uv run python -m scripts.quantize_checkpoint \
		--checkpoint $(CHECKPOINT_FILE) \
		--bits $(BITS)
	@echo ""
	@echo "✓ Quantization complete!"
	@echo "  Use 'make chat-quantized' to test the quantized model"

quantize-8bit:
	$(MAKE) quantize BITS=8

quantize-4bit:
	$(MAKE) quantize BITS=4

# Chat with quantized model
CHECKPOINT_BASE ?= ~/.cache/mlxchat/base_checkpoints/d12
QUANTIZED_CHECKPOINT ?= $(CHECKPOINT_BASE)/model_latest_q4.npz

chat-quantized:
	@if [ ! -f "$(QUANTIZED_CHECKPOINT)" ]; then \
		echo "Error: Quantized checkpoint not found at $(QUANTIZED_CHECKPOINT)"; \
		echo "Run 'make quantize' first to create a quantized model"; \
		exit 1; \
	fi
	@echo "Starting chat with quantized model..."
	@echo "Checkpoint: $(QUANTIZED_CHECKPOINT)"
	@echo ""
	uv run python -m scripts.chat_cli \
		--checkpoint $(QUANTIZED_CHECKPOINT) \
		--temperature $(TEMP) \
		--top-k $(TOP_K)

# Data management
download-data:
	uv run python -m mlxchat.dataset --num-shards 50

download-data-minimal:
	uv run python -m mlxchat.dataset --num-shards 10

download-data-full:
	@echo "Warning: This will download ~300GB of data"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		uv run python -m mlxchat.dataset --num-shards 1823; \
	fi

# Code quality
format:
	uv run black . --line-length 120

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf ~/.cache/mlxchat/base_checkpoints
	rm -rf ~/.cache/mlxchat/logs
	@echo "Cleaned build artifacts, checkpoints, and logs"

clean-cache:
	@echo "Warning: This will delete all cached data, checkpoints, and tokenizer"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf ~/.cache/mlxchat; \
	fi

# Monitoring
watch-checkpoints:
	watch -n 5 'ls -lh ~/.cache/mlxchat/base_checkpoints/d12/ | tail -20'

disk-usage:
	@echo "Disk Usage:"
	@echo "==========="
	@echo ""
	@echo "Cached shards:"
	@du -sh ~/.cache/mlxchat/base_data 2>/dev/null || echo "  No shards cached"
	@echo ""
	@echo "Checkpoints:"
	@du -sh ~/.cache/mlxchat/base_checkpoints 2>/dev/null || echo "  No checkpoints saved"
	@echo ""
	@echo "Tokenizer:"
	@du -sh ~/.cache/mlxchat/tokenizer 2>/dev/null || echo "  No tokenizer trained"
	@echo ""
	@echo "Logs:"
	@du -sh ~/.cache/mlxchat/logs 2>/dev/null || echo "  No logs"

memory-info:
	@echo "================================================================================"
	@echo "MEMORY OPTIMIZATION GUIDE"
	@echo "================================================================================"
	@echo ""
	@echo "Current System Memory:"
	@uv run python -c "import psutil; mem = psutil.virtual_memory(); print(f'  Total: {mem.total/(1024**3):.1f} GB'); print(f'  Available: {mem.available/(1024**3):.1f} GB'); print(f'  Used: {mem.used/(1024**3):.1f} GB ({mem.percent:.1f}%)')" 2>/dev/null || echo "  (Run 'make install' to see memory stats)"
	@echo ""
	@echo "Recommended Commands by Available Memory:"
	@echo "==========================================="
	@echo ""
	@echo "If you have 16-32 GB RAM:"
	@echo "  make train-low-memory     # Uses 10-15 GB, trains d12 (186M params)"
	@echo ""
	@echo "If you have 8-16 GB RAM:"
	@echo "  make train-ultra-low      # Uses 6-10 GB, trains d12 (186M params)"
	@echo ""
	@echo "If you have 32+ GB RAM:"
	@echo "  make train                # Standard training (15-25 GB)"
	@echo ""
	@echo "To monitor memory during training:"
	@echo "  make monitor-memory       # Run this in a separate terminal"
	@echo ""
	@echo "To quantize trained models for inference:"
	@echo "  make quantize             # 4-bit quantization (8x smaller)"
	@echo "  make quantize-8bit        # 8-bit quantization (4x smaller)"
	@echo ""
	@echo "For more details, see: MEMORY_OPTIMIZATION.md"
	@echo "================================================================================"

# Logging
logs:
	@echo "Training logs:"
	@ls -lht ~/.cache/mlxchat/logs/ 2>/dev/null | head -10 || echo "No logs found"

tail-log:
	@LOG_FILE=$$(ls -t ~/.cache/mlxchat/logs/train_*.log 2>/dev/null | head -1); \
	if [ -n "$$LOG_FILE" ]; then \
		echo "Tailing: $$LOG_FILE"; \
		echo ""; \
		tail -f "$$LOG_FILE"; \
	else \
		echo "No training logs found in ~/.cache/mlxchat/logs/"; \
	fi

view-log:
	@LOG_FILE=$$(ls -t ~/.cache/mlxchat/logs/train_*.log 2>/dev/null | head -1); \
	if [ -n "$$LOG_FILE" ]; then \
		echo "Viewing: $$LOG_FILE"; \
		echo ""; \
		less +G "$$LOG_FILE"; \
	else \
		echo "No training logs found in ~/.cache/mlxchat/logs/"; \
	fi

clean-logs:
	@echo "Warning: This will delete all training logs"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf ~/.cache/mlxchat/logs; \
		echo "Logs deleted"; \
	fi
