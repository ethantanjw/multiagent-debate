#!/usr/bin/env python3
"""
Wait for vLLM server to be ready before proceeding.

Usage:
    python wait_for_vllm.py [--host HOST] [--port PORT] [--timeout TIMEOUT]
"""

import argparse
import time
import requests
import sys


def wait_for_server(host="localhost", port=8000, timeout=300):
    """
    Wait for vLLM server to be ready.

    Args:
        host: Server host
        port: Server port
        timeout: Maximum time to wait in seconds

    Returns:
        True if server is ready, False if timeout
    """
    url = f"http://{host}:{port}/v1/models"
    start_time = time.time()

    print(f"Waiting for vLLM server at {host}:{port}...")
    print(f"Timeout: {timeout} seconds")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                models = response.json()
                print(f"\n✓ vLLM server is ready!")
                print(f"Available models: {models}")
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass

        elapsed = int(time.time() - start_time)
        print(f"\r  Waiting... ({elapsed}s elapsed)", end="", flush=True)
        time.sleep(2)

    print(f"\n✗ Timeout waiting for vLLM server after {timeout} seconds")
    return False


def main():
    parser = argparse.ArgumentParser(description="Wait for vLLM server to be ready")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")

    args = parser.parse_args()

    if wait_for_server(args.host, args.port, args.timeout):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
