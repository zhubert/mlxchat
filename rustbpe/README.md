# rustbpe

> The missing tiktoken training code

A very lightweight Rust library for training a GPT tokenizer. The issue is that the inference library [tiktoken](https://github.com/openai/tiktoken) is great, but only does inference. Separately, the huggingface [tokenizers](https://github.com/huggingface/tokenizers) library does training, but it is rather bloated and really hard to navigate because it has to support all the different historical baggage of how people dealt with tokenizers over the years. More recently, I also wrote the [minbpe](https://github.com/karpathy/minbpe) library which does both training and inference, but only in inefficient Python. Basically what I really want is a non-fancy, super simple, but still relatively efficient training code for GPT tokenizer (more efficient than minbpe, much cleaner/simpler than tokenizers), and then export the trained vocab for inference with tiktoken. Does that make sense? So here we are. There are more opportunities for optimization here, I just stopped a bit early because unlike minbpe before it, rustbpe is now simple and fast enough, and not a significant bottleneck for nanochat.

## About this copy

This is a local copy of rustbpe from [nanochat](https://github.com/karpathy/nanochat) for use in mlxchat. The original rustbpe had an invalid Rust edition (2024) which has been fixed to 2021 for compatibility with current Rust toolchains.

## Building

```bash
# From mlxchat root directory
make build-tokenizer

# Or manually:
uv run maturin develop --manifest-path rustbpe/Cargo.toml
```

## Usage

See mlxchat/tokenizer.py for Python usage examples.
