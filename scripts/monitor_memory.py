"""
Monitor memory usage during training

Usage:
    python -m scripts.monitor_memory

This script monitors system memory usage and displays it in real-time.
Run this in a separate terminal while training to track memory consumption.
"""

import time
import psutil
import os


def format_bytes(bytes_val):
    """Format bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def get_process_memory(process_name="python"):
    """Get memory usage of processes matching name."""
    total_memory = 0
    matching_processes = []

    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            if process_name.lower() in proc.info["name"].lower():
                mem = proc.info["memory_info"].rss
                total_memory += mem
                matching_processes.append(
                    {
                        "pid": proc.info["pid"],
                        "name": proc.info["name"],
                        "memory": mem,
                    }
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    return total_memory, matching_processes


def monitor_memory(interval=1.0):
    """Monitor memory usage continuously."""
    print("=" * 80)
    print("MEMORY MONITOR")
    print("=" * 80)
    print("Press Ctrl+C to stop\n")

    try:
        max_memory = 0
        max_python_memory = 0

        while True:
            # Get system memory stats
            mem = psutil.virtual_memory()

            # Get Python process memory
            python_memory, python_procs = get_process_memory("python")

            # Track maximums
            max_memory = max(max_memory, mem.used)
            max_python_memory = max(max_python_memory, python_memory)

            # Clear screen (works on Unix-like systems)
            os.system("clear" if os.name == "posix" else "cls")

            print("=" * 80)
            print("MEMORY MONITOR")
            print("=" * 80)

            print("\nSYSTEM MEMORY:")
            print(f"  Total:     {format_bytes(mem.total)}")
            print(f"  Available: {format_bytes(mem.available)}")
            print(f"  Used:      {format_bytes(mem.used)} ({mem.percent:.1f}%)")
            print(f"  Free:      {format_bytes(mem.free)}")
            print(f"  Peak used: {format_bytes(max_memory)}")

            if python_memory > 0:
                print("\nPYTHON PROCESSES:")
                print(f"  Total memory: {format_bytes(python_memory)}")
                print(f"  Peak memory:  {format_bytes(max_python_memory)}")
                print(f"  Process count: {len(python_procs)}")

                if len(python_procs) > 0:
                    print("\n  Top processes:")
                    sorted_procs = sorted(python_procs, key=lambda x: x["memory"], reverse=True)
                    for proc in sorted_procs[:5]:  # Show top 5
                        print(f"    PID {proc['pid']:6d}: {format_bytes(proc['memory']):>12s}  ({proc['name']})")

            # Memory pressure warning
            if mem.percent > 90:
                print("\n⚠️  WARNING: Memory usage is very high (>90%)!")
                print("   Consider reducing batch size or enabling --low-memory mode")
            elif mem.percent > 80:
                print("\n⚠️  CAUTION: Memory usage is high (>80%)")

            print(f"\nUpdating every {interval}s... (Ctrl+C to stop)")
            print("=" * 80)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print(f"Peak system memory used: {format_bytes(max_memory)}")
        print(f"Peak Python memory used: {format_bytes(max_python_memory)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Monitor memory usage")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Update interval in seconds (default: 1.0)",
    )

    args = parser.parse_args()
    monitor_memory(interval=args.interval)


if __name__ == "__main__":
    main()
