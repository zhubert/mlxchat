"""
Interactive chat CLI for MLXChat

Usage:
    python -m scripts.chat_cli --checkpoint checkpoints/d12/step_01000
    python -m scripts.chat_cli -i mid --temperature 0.8
"""

import argparse
import sys
from pathlib import Path

import mlx.core as mx

from mlxchat.engine import Engine
from mlxchat.checkpoint_manager import load_model, load_model_from_dir
from mlxchat.tokenizer import get_tokenizer


def print_banner():
    """Print ASCII art banner for MLXChat."""
    banner = r"""
    ███╗   ███╗██╗     ██╗  ██╗ ██████╗██╗  ██╗ █████╗ ████████╗
    ████╗ ████║██║     ╚██╗██╔╝██╔════╝██║  ██║██╔══██╗╚══██╔══╝
    ██╔████╔██║██║      ╚███╔╝ ██║     ███████║███████║   ██║
    ██║╚██╔╝██║██║      ██╔██╗ ██║     ██╔══██║██╔══██║   ██║
    ██║ ╚═╝ ██║███████╗██╔╝ ██╗╚██████╗██║  ██║██║  ██║   ██║
    ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝
    """
    print(banner)
    print("=" * 80)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Chat with MLXChat model')

    # Model loading
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                        help='Path to checkpoint directory')
    parser.add_argument('-i', '--source', type=str, default="base",
                        help='Source of the model: base|sft|mid|rl')
    parser.add_argument('-g', '--model-tag', type=str, default=None,
                        help='Model tag to load (e.g., "d12")')
    parser.add_argument('-s', '--step', type=int, default=None,
                        help='Specific checkpoint step to load')

    # Generation parameters
    parser.add_argument('-t', '--temperature', type=float, default=0.6,
                        help='Temperature for generation (0.0 = greedy)')
    parser.add_argument('-k', '--top-k', type=int, default=50,
                        help='Top-k sampling parameter')
    parser.add_argument('-m', '--max-tokens', type=int, default=256,
                        help='Maximum tokens to generate per response')

    # Single-shot mode
    parser.add_argument('-p', '--prompt', type=str, default='',
                        help='Prompt the model once and exit')

    # Tokenizer
    parser.add_argument('--tokenizer-dir', type=str, default=None,
                        help='Custom tokenizer directory')

    return parser.parse_args()


def main():
    args = get_args()

    # Print banner
    print_banner()
    print("MLXChat Interactive Mode")
    print("-" * 80)

    # Load model (which also loads the tokenizer)
    print(f"Loading model...")
    if args.checkpoint:
        # Load from specific checkpoint directory path
        model, tokenizer, metadata = load_model_from_dir(
            args.checkpoint,
            model_tag=args.model_tag,
            step=args.step
        )
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        # Load from source (base/sft/mid/rl)
        try:
            model, tokenizer, metadata = load_model(
                source=args.source,
                model_tag=args.model_tag,
                step=args.step
            )
            print(f"Loaded model from source: {args.source}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Please train a model first or specify a valid checkpoint path.")
            sys.exit(1)

    print(f"Vocab size: {tokenizer.get_vocab_size():,}")

    # Print model info
    if metadata:
        print(f"Model configuration:")
        if 'model_config' in metadata:
            cfg = metadata['model_config']
            print(f"  Layers: {cfg.get('n_layer', 'N/A')}")
            print(f"  Model dim: {cfg.get('n_embd', 'N/A')}")
            print(f"  Num heads: {cfg.get('n_head', 'N/A')}")
        if 'step' in metadata:
            print(f"  Training step: {metadata['step']}")
        if 'loss' in metadata:
            print(f"  Training loss: {metadata['loss']:.4f}")

    # Set model to eval mode
    model.eval()

    # Create inference engine
    engine = Engine(model, tokenizer)

    # Get special tokens
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    print("-" * 80)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("-" * 80)

    # Initialize conversation with BOS token
    conversation_tokens = [bos]

    while True:
        # Get user input
        if args.prompt:
            # Single-shot mode: use prompt from command line
            user_input = args.prompt
        else:
            # Interactive mode: read from stdin
            try:
                user_input = input("\nUser: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

        # Handle special commands
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if user_input.lower() == 'clear':
            conversation_tokens = [bos]
            print("Conversation cleared.")
            continue

        if not user_input:
            continue

        # Add user message to conversation
        conversation_tokens.append(user_start)
        user_tokens = tokenizer.encode(user_input, return_array=False)
        conversation_tokens.extend(user_tokens)
        conversation_tokens.append(user_end)

        # Start assistant response
        conversation_tokens.append(assistant_start)

        # Generate response
        response_tokens = []
        print("\nAssistant: ", end="", flush=True)

        try:
            for token_id in engine.generate(
                tokens=conversation_tokens,
                num_samples=1,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            ):
                response_tokens.append(token_id)

                # Decode and print token
                token_text = tokenizer.decode([token_id])
                print(token_text, end="", flush=True)

                # Check if we hit assistant_end token
                if token_id == assistant_end:
                    break

            print()  # Newline after response

        except KeyboardInterrupt:
            print("\n[Generation interrupted]")

        # Ensure assistant_end is at the end
        if not response_tokens or response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)

        # Add response to conversation
        conversation_tokens.extend(response_tokens)

        # In single-shot mode, exit after first response
        if args.prompt:
            break


if __name__ == "__main__":
    main()
