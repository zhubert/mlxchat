"""
Dataset utilities for mlxchat

Handles downloading FineWeb shards on-demand with cache management.
Inspired by nanochat but optimized for single-machine use with limited storage.
"""

import os
import time
import requests


# FineWeb-Edu-100B dataset configuration
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822  # Last shard is shard_01822.parquet
SHARD_FILENAME_FORMAT = "shard_{index:05d}.parquet"


def get_data_dir():
    """Get the data directory for FineWeb shards."""
    home = os.path.expanduser("~")
    data_dir = os.path.join(home, ".cache", "mlxchat", "base_data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def shard_index_to_filename(index):
    """Convert shard index to filename."""
    return SHARD_FILENAME_FORMAT.format(index=index)


def download_shard(shard_index, data_dir=None, max_attempts=5):
    """
    Download a single FineWeb shard.

    Args:
        shard_index: Index of the shard to download (0 to MAX_SHARD)
        data_dir: Directory to save the shard (default: ~/.cache/mlxchat/base_data)
        max_attempts: Maximum number of download attempts

    Returns:
        Path to the shard file if successful, None otherwise
    """
    if data_dir is None:
        data_dir = get_data_dir()
    else:
        # Ensure the directory exists even if a custom path was provided
        os.makedirs(data_dir, exist_ok=True)

    # Construct filename and path
    filename = shard_index_to_filename(shard_index)
    filepath = os.path.join(data_dir, filename)

    # Skip if already exists
    if os.path.exists(filepath):
        return filepath

    # Construct remote URL
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            # Write to temporary file first
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)

            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"✓ Downloaded {filename}")
            return filepath

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")

            # Clean up any partial files
            for path in [temp_path, filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

            # Exponential backoff
            if attempt < max_attempts:
                wait_time = 2**attempt
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"✗ Failed to download {filename}")
                return None

    return None


class ShardCache:
    """
    Manages a rolling cache of downloaded shards.

    Automatically downloads shards on-demand and deletes old shards
    to keep storage usage under control.
    """

    def __init__(self, data_dir=None, max_shards=20, enable_cache_cleanup=True):
        """
        Initialize shard cache.

        Args:
            data_dir: Directory for shards (default: ~/.cache/mlxchat/base_data)
            max_shards: Maximum number of shards to keep on disk
            enable_cache_cleanup: If True, delete old shards when cache is full
        """
        self.data_dir = data_dir if data_dir else get_data_dir()
        self.max_shards = max_shards
        self.enable_cache_cleanup = enable_cache_cleanup
        self.access_order = []  # Track access order for LRU eviction

    def get_shard_path(self, shard_index):
        """
        Get path to a shard, downloading if necessary.

        Args:
            shard_index: Index of the shard to get

        Returns:
            Path to the shard file, or None if download failed
        """
        filename = shard_index_to_filename(shard_index)
        filepath = os.path.join(self.data_dir, filename)

        # Download if doesn't exist
        if not os.path.exists(filepath):
            print(f"Shard {shard_index} not cached, downloading...")
            result = download_shard(shard_index, self.data_dir)
            if result is None:
                return None

        # Update access order for LRU
        if shard_index in self.access_order:
            self.access_order.remove(shard_index)
        self.access_order.append(shard_index)

        # Clean up old shards if needed
        if self.enable_cache_cleanup:
            self._cleanup_if_needed()

        return filepath

    def _cleanup_if_needed(self):
        """Remove least recently used shards if cache is too large."""
        # Count current shards
        current_shards = [f for f in os.listdir(self.data_dir) if f.endswith(".parquet") and not f.endswith(".tmp")]

        # If over limit, remove least recently used
        if len(current_shards) > self.max_shards:
            num_to_remove = len(current_shards) - self.max_shards

            # Remove oldest accessed shards
            for _ in range(num_to_remove):
                if not self.access_order:
                    break

                # Find the oldest shard that still exists
                oldest_index = self.access_order.pop(0)
                oldest_filename = shard_index_to_filename(oldest_index)
                oldest_path = os.path.join(self.data_dir, oldest_filename)

                if os.path.exists(oldest_path):
                    try:
                        os.remove(oldest_path)
                        print(f"Removed old shard: {oldest_filename}")
                    except OSError as e:
                        print(f"Failed to remove {oldest_filename}: {e}")

    def get_cache_info(self):
        """Get information about current cache status."""
        current_shards = [f for f in os.listdir(self.data_dir) if f.endswith(".parquet") and not f.endswith(".tmp")]

        total_size = sum(os.path.getsize(os.path.join(self.data_dir, f)) for f in current_shards)

        return {
            "num_shards": len(current_shards),
            "max_shards": self.max_shards,
            "total_size_mb": total_size / (1024 * 1024),
            "avg_shard_size_mb": (total_size / len(current_shards) / (1024 * 1024)) if current_shards else 0,
        }


def download_shards(num_shards, data_dir=None, num_workers=1):
    """
    Download a specific number of shards sequentially.

    Useful for initial setup or downloading a small subset.

    Args:
        num_shards: Number of shards to download (starting from 0)
        data_dir: Directory to save shards
        num_workers: Currently unused (for compatibility), downloads sequentially

    Returns:
        Number of successfully downloaded shards
    """
    if data_dir is None:
        data_dir = get_data_dir()

    num_shards = min(num_shards, MAX_SHARD + 1)
    print(f"Downloading {num_shards} shards to {data_dir}...")

    successful = 0
    for i in range(num_shards):
        if download_shard(i, data_dir) is not None:
            successful += 1

    print(f"Downloaded {successful}/{num_shards} shards")
    return successful
