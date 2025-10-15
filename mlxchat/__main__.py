"""
CLI for mlxchat dataset management

Usage:
    python -m mlxchat.dataset --num-shards 50
"""

import argparse
from mlxchat.dataset import download_shards, get_data_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu-100B shards for mlxchat"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=10,
        help="Number of shards to download (default: 10)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to save shards (default: ~/.cache/mlxchat/base_data)",
    )

    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir else get_data_dir()

    print(f"mlxchat Dataset Downloader")
    print(f"=" * 50)
    print(f"Downloading {args.num_shards} shards...")
    print(f"Target directory: {data_dir}")
    print(f"Estimated size: ~{args.num_shards * 0.165:.1f} GB")
    print()

    successful = download_shards(args.num_shards, data_dir)

    print()
    print(f"=" * 50)
    print(f"Download complete: {successful}/{args.num_shards} shards")

    if successful < args.num_shards:
        print(f"Warning: Failed to download {args.num_shards - successful} shards")
        return 1

    print(f"Ready to train! Use: python -m scripts.base_train")
    return 0


if __name__ == "__main__":
    exit(main())
