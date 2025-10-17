"""
Quantize a trained checkpoint to 4-bit or 8-bit for inference

Usage:
    python -m scripts.quantize_checkpoint \
        --checkpoint ~/.cache/mlxchat/base_checkpoints/d12/model_latest.npz \
        --bits 4 \
        --output ~/.cache/mlxchat/base_checkpoints/d12/model_latest_q4.npz
"""

import argparse
import os

import mlx.core as mx

from mlxchat.gpt import GPT, GPTConfig
from mlxchat.checkpoint_manager import load_checkpoint
from mlxchat.quantization import quantize_model, estimate_memory_reduction


def get_args():
    parser = argparse.ArgumentParser(description="Quantize a trained checkpoint")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.npz)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Number of bits for quantization (4 or 8)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for quantized checkpoint (default: add _qN suffix)",
    )
    parser.add_argument(
        "--quantize-embeddings",
        action="store_true",
        help="Also quantize embedding and output layers (may reduce quality)",
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Determine checkpoint directory and step
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint metadata to get config
    checkpoint_dir = os.path.dirname(checkpoint_path)
    filename = os.path.basename(checkpoint_path)

    # Extract step from filename (e.g., model_00100.npz or model_latest.npz)
    if "latest" in filename:
        # Find the latest checkpoint
        from mlxchat.checkpoint_manager import find_last_step

        step = find_last_step(checkpoint_dir)
    else:
        # Extract step from filename
        step_str = filename.replace("model_", "").replace(".npz", "")
        step = int(step_str)

    print(f"Loading checkpoint from step {step}...")

    # Load model weights and metadata
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, load_optimizer=False)

    # Get model config from metadata
    if "model_config" not in meta_data:
        raise ValueError("Checkpoint does not contain model_config metadata")

    config_dict = meta_data["model_config"]
    config = GPTConfig(
        sequence_len=config_dict["sequence_len"],
        vocab_size=config_dict["vocab_size"],
        n_layer=config_dict["n_layer"],
        n_head=config_dict["n_head"],
        n_kv_head=config_dict["n_kv_head"],
        n_embd=config_dict["n_embd"],
    )

    print("\nModel Configuration:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Model dim: {config.n_embd}")
    print(f"  Num heads: {config.n_head}")
    print(f"  Vocab size: {config.vocab_size}")

    # Create model and load weights
    print("\nCreating model...")
    model = GPT(config)
    model.update(model_data)

    # Estimate memory reduction
    print(f"\nEstimating memory reduction for {args.bits}-bit quantization...")
    stats = estimate_memory_reduction(model, bits=args.bits)
    print(f"  Transformer parameters: {stats['transformer_params']:,}")
    print(f"  Original memory: {stats['original_memory_mb']:.2f} MB")
    print(f"  Quantized memory: {stats['quantized_memory_mb']:.2f} MB")
    print(f"  Reduction ratio: {stats['reduction_ratio']:.2f}x")
    print(f"  Memory saved: {stats['memory_saved_mb']:.2f} MB")

    # Quantize model
    print(f"\nQuantizing model to {args.bits}-bit...")
    quantized_model = quantize_model(
        model,
        bits=args.bits,
        quantize_embeddings=args.quantize_embeddings,
    )

    # Determine output path
    if args.output is None:
        # Add _qN suffix before .npz extension
        base_path = checkpoint_path.replace(".npz", "")
        output_path = f"{base_path}_q{args.bits}.npz"
    else:
        output_path = args.output

    # Save quantized model
    print(f"\nSaving quantized model to: {output_path}")

    # Update metadata to indicate quantization
    meta_data["quantization"] = {
        "bits": args.bits,
        "quantize_embeddings": args.quantize_embeddings,
    }

    # Get quantized model state
    quantized_state = quantized_model.parameters()

    # Save to file
    mx.savez(output_path, **quantized_state)

    # Also save metadata separately
    meta_path = output_path.replace("model_", "meta_").replace(".npz", ".json")
    import json

    with open(meta_path, "w") as f:
        # Convert any non-serializable types
        serializable_meta = {}
        for k, v in meta_data.items():
            if isinstance(v, dict):
                serializable_meta[k] = v
            else:
                serializable_meta[k] = str(v)
        json.dump(serializable_meta, f, indent=2)

    print(f"Metadata saved to: {meta_path}")
    print("\nQuantization complete!")
    print("\nTo use the quantized model:")
    print(f"  python -m scripts.chat_cli --checkpoint {output_path}")


if __name__ == "__main__":
    main()
