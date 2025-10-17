"""
Utilities for saving and loading model/optim/state checkpoints for MLX.
"""

import os
import re
import glob
import json
import logging

import mlx.core as mx

from mlxchat.gpt import GPT, GPTConfig
from mlxchat.tokenizer import get_tokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_base_dir():
    """Get the base directory for mlxchat data."""
    home = os.path.expanduser("~")
    return os.path.join(home, ".cache", "mlxchat")


def save_checkpoint(checkpoint_dir, step, model, optimizer_data, meta_data):
    """
    Save a checkpoint with model, optimizer(s), and metadata.

    Args:
        checkpoint_dir: Directory to save checkpoint files
        step: Training step number
        model: GPT model instance
        optimizer_data: Dictionary with optimizer states (e.g., {"adam": adam_opt.state, "muon": muon_opt.state})
        meta_data: Dictionary with training metadata
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the model state (parameters)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.npz")
    model_params = model.parameters()
    # Flatten and convert to dict for mx.savez
    flat_params = dict(flatten_dict(model_params))
    mx.savez(model_path, **flat_params)
    logger.info(f"Saved model file to: {model_path}")

    # Save the optimizer state (useful for resuming training or fine-tuning)
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.npz")
        # Flatten optimizer states for saving
        flat_optim = {}
        for opt_name, opt_state in optimizer_data.items():
            for key, value in flatten_dict(opt_state):
                flat_optim[f"{opt_name}.{key}"] = value
        mx.savez(optimizer_path, **flat_optim)
        logger.info(f"Saved optimizer file to: {optimizer_path}")

    # Save the metadata dict as json
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)
    logger.info(f"Saved metadata file to: {meta_path}")


def load_checkpoint(checkpoint_dir, step, load_optimizer=False):
    """
    Load a checkpoint with model, optimizer(s), and metadata.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        step: Training step number
        load_optimizer: Whether to load optimizer state

    Returns:
        model_data: Nested dictionary of model parameters
        optimizer_data: Dictionary of optimizer states (or None)
        meta_data: Dictionary with training metadata
    """
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.npz")
    model_arrays = mx.load(model_path)
    model_data = unflatten_dict(model_arrays)

    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.npz")
        if os.path.exists(optimizer_path):
            optim_arrays = mx.load(optimizer_path)
            # Unflatten optimizer states properly
            # First, group arrays by optimizer name
            optimizer_data = {}
            for flat_key, value in optim_arrays.items():
                # Split "adam.key.subkey" -> ("adam", "key.subkey")
                opt_name, *rest = flat_key.split(".", 1)
                if opt_name not in optimizer_data:
                    optimizer_data[opt_name] = {}
                if rest:
                    # Use the remaining key as-is for unflattening
                    key = rest[0]
                    if key not in optimizer_data[opt_name]:
                        optimizer_data[opt_name][key] = value

            # Now unflatten each optimizer's state using unflatten_dict
            # which properly converts numeric keys to lists
            for opt_name in optimizer_data:
                optimizer_data[opt_name] = unflatten_dict(optimizer_data[opt_name])

    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step):
    """
    Build a model from a given checkpoint.

    Returns:
        model: GPT model with loaded weights
        tokenizer: Tokenizer instance
        meta_data: Metadata saved during training
    """
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, load_optimizer=False)

    # Build model from config
    model_config_kwargs = meta_data["model_config"]
    logger.info(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)

    # Initialize weights (needed for rotary embeddings)
    model.init_weights()

    # Load the model state
    model.update(model_data)

    # Load the Tokenizer
    tokenizer = get_tokenizer()

    # Sanity check: compatibility between model and tokenizer
    assert (
        tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    ), f"Tokenizer vocab size {tokenizer.get_vocab_size()} != model vocab size {model_config_kwargs['vocab_size']}"

    return model, tokenizer, meta_data


def find_largest_model(checkpoint_dir):
    """Find the largest model (by depth) in the checkpoint directory."""
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # Try to find model tags of the form d<number>
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # If that failed, take the most recently updated model
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    """Find the last checkpoint step in the directory."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.npz"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step


def load_model_from_dir(checkpoints_dir, model_tag=None, step=None):
    """
    Load a model from a checkpoint directory.

    Args:
        checkpoints_dir: Base directory containing model checkpoints
        model_tag: Model tag (e.g., "d12", "d20"), or None to auto-detect
        step: Training step, or None to load the last checkpoint

    Returns:
        model: GPT model with loaded weights
        tokenizer: Tokenizer instance
        meta_data: Metadata saved during training
    """
    if model_tag is None:
        # Guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        logger.info(f"No model tag provided, guessing model tag: {model_tag}")

    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)

    if step is None:
        # Guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)

    assert step is not None, f"No checkpoints found in {checkpoint_dir}"

    # Build the model
    logger.info(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step)

    return model, tokenizer, meta_data


def load_model(source, model_tag=None, step=None):
    """
    Load a model from a named checkpoint source.

    Args:
        source: Checkpoint source ("base", "mid", "sft", "rl")
        model_tag: Model tag (e.g., "d12", "d20"), or None to auto-detect
        step: Training step, or None to load the last checkpoint

    Returns:
        model: GPT model with loaded weights
        tokenizer: Tokenizer instance
        meta_data: Metadata saved during training
    """
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]

    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)

    return load_model_from_dir(checkpoints_dir, model_tag, step)


# -----------------------------------------------------------------------------
# Utility functions for flattening/unflattening nested dictionaries


def flatten_dict(d, parent_key="", sep="."):
    """
    Flatten a nested dictionary or list structure.

    Example:
        {"a": {"b": 1, "c": 2}, "d": 3} -> [("a.b", 1), ("a.c", 2), ("d", 3)]
        {"a": [{"b": 1}, {"b": 2}]} -> [("a.0.b", 1), ("a.1.b", 2)]
    """
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, (dict, list)):
                items.extend(flatten_dict(v, new_key, sep=sep))
            else:
                items.append((new_key, v))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            if isinstance(v, (dict, list)):
                items.extend(flatten_dict(v, new_key, sep=sep))
            else:
                items.append((new_key, v))
    else:
        # Base case: just a value
        items.append((parent_key, d))
    return items


def unflatten_dict(flat_dict, sep="."):
    """
    Unflatten a flat dictionary back into a nested structure (with lists).

    Example:
        {"a.b": 1, "a.c": 2, "d": 3} -> {"a": {"b": 1, "c": 2}, "d": 3}
        {"a.0.b": 1, "a.1.b": 2} -> {"a": [{"b": 1}, {"b": 2}]}
    """
    result = {}

    # First pass: build structure with dicts only
    for flat_key, value in flat_dict.items():
        parts = flat_key.split(sep)
        current = result

        # Navigate/create nested structure
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the final value
        last_part = parts[-1]
        current[last_part] = value

    # Second pass: convert dicts with all-numeric keys to lists
    def convert_numeric_dicts(obj):
        if isinstance(obj, dict):
            # Check if all keys are numeric
            if obj and all(k.isdigit() for k in obj.keys()):
                # Convert to list
                max_idx = max(int(k) for k in obj.keys())
                result_list = [None] * (max_idx + 1)
                for k, v in obj.items():
                    result_list[int(k)] = convert_numeric_dicts(v)
                return result_list
            else:
                return {k: convert_numeric_dicts(v) for k, v in obj.items()}
        else:
            return obj

    return convert_numeric_dicts(result)
