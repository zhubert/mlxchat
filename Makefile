.PHONY: help install install-rust build-tokenizer test train train-quick train-test chat-cli chat-web download-data format lint clean

# Default target
help:
	@echo "mlxchat - MLX port of nanochat for Apple Silicon"
	@echo ""
	@echo "Available targets:"
	@echo "  make install       - Install package with dev dependencies"
	@echo "  make install-rust  - Install Rust toolchain (required for tokenizer)"
	@echo "  make build-tokenizer - Build rustbpe tokenizer (2-3x faster than tiktoken)"
	@echo "  make test          - Run all tests"
	@echo "  make train         - Train d12 model (full run, ~3225 steps)"
	@echo "  make train-quick   - Quick training test (100 steps)"
	@echo "  make train-test    - Minimal training test (2 steps)"
	@echo "  make chat-cli      - Run CLI chat interface"
	@echo "  make chat-web      - Run web UI chat interface"
	@echo "  make download-data - Download first 50 shards (~8GB)"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make clean         - Clean cache and build artifacts"
	@echo ""
	@echo "Training with custom depth:"
	@echo "  make train DEPTH=16 STEPS=1000"
	@echo ""
	@echo "Streaming mode is enabled by default (20 cached shards)"

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
	@echo "  You can now train tokenizers with: python -m scripts.tok_train"

# Testing
test:
	uv run pytest -v

test-model:
	uv run pytest tests/test_gpt.py -v

test-dataloader:
	uv run pytest tests/test_dataloader.py -v

# Training targets
DEPTH ?= 12
BATCH_SIZE ?= 8
STEPS ?= -1
CACHED_SHARDS ?= 20

train:
	uv run python -u -m scripts.base_train \
		--depth=$(DEPTH) \
		--streaming \
		--max-cached-shards $(CACHED_SHARDS) \
		--device-batch-size $(BATCH_SIZE) \
		--num-iterations $(STEPS)

train-quick:
	uv run python -u -m scripts.base_train \
		--depth=12 \
		--streaming \
		--max-cached-shards 20 \
		--device-batch-size 8 \
		--num-iterations 100

train-test:
	uv run python -u -m scripts.base_train \
		--depth=12 \
		--streaming \
		--max-cached-shards 5 \
		--device-batch-size 8 \
		--num-iterations 2

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
CHECKPOINT ?= ~/.cache/mlxchat/base_checkpoints/d12/model_latest.npz
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

clean-cache:
	@echo "Warning: This will delete all cached shards and checkpoints"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf ~/.cache/mlxchat; \
		rm -rf ~/.cache/nanochat; \
	fi

# Monitoring
watch-checkpoints:
	watch -n 5 'ls -lh ~/.cache/mlxchat/base_checkpoints/d12/ | tail -20'

disk-usage:
	@echo "Cached shards:"
	@du -sh ~/.cache/nanochat/data 2>/dev/null || echo "No shards cached"
	@echo ""
	@echo "Checkpoints:"
	@du -sh ~/.cache/mlxchat/base_checkpoints 2>/dev/null || echo "No checkpoints saved"
