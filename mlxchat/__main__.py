"""
CLI for mlxchat dataset management

Usage:
    python -m mlxchat.dataset --num-shards 50
"""

import argparse
import logging
from mlxchat.dataset import download_shards, get_data_dir

# Configure logging for CLI
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu-100B shards for mlxchat")
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

    logger.info("mlxchat Dataset Downloader")
    logger.info("=" * 50)
    logger.info(f"Downloading {args.num_shards} shards...")
    logger.info(f"Target directory: {data_dir}")
    logger.info(f"Estimated size: ~{args.num_shards * 0.165:.1f} GB")

    successful = download_shards(args.num_shards, data_dir)

    logger.info("=" * 50)
    logger.info(f"Download complete: {successful}/{args.num_shards} shards")

    if successful < args.num_shards:
        logger.warning(f"Failed to download {args.num_shards - successful} shards")
        return 1

    logger.info("Ready to train! Use: python -m scripts.base_train")
    return 0


if __name__ == "__main__":
    exit(main())
