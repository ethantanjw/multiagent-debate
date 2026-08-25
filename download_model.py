#!/usr/bin/env python
"""
Download and cache Llama-3.1-8B-Instruct model to scratch directory.

Usage:
    python download_model.py

This script should be run once on the cluster to download the model
to ~/scratch/models/ for faster access during experiments.
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_model():
    """Download Llama-3.1-8B-Instruct to scratch directory."""

    # Model to download
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    # Cache directory - use $SCRATCH directly (like your other models)
    scratch = os.environ.get('SCRATCH', str(Path.home() / 'scratch'))
    cache_dir = os.environ.get('MODEL_CACHE_DIR', scratch)

    print("="*80)
    print("MODEL DOWNLOAD SCRIPT")
    print("="*80)
    print(f"\nModel: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print("\nThis will download ~16GB of model weights.")
    print("Make sure you:")
    print("  1. Have logged in to HuggingFace with access to Llama models")
    print("  2. Have enough space in your scratch directory (~20GB free)")
    print("\n" + "="*80)

    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Cache directory created: {cache_dir}")

    # Download tokenizer
    print(f"\n[1/2] Downloading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        print(f"✓ Tokenizer downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading tokenizer: {e}")
        print("\nMake sure you have access to Llama models:")
        print("  1. Visit: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        print("  2. Accept the license agreement")
        print("  3. Run: huggingface-cli login --token YOUR_TOKEN")
        return False

    # Download model
    print(f"\n[2/2] Downloading model weights (~16GB)...")
    print("This may take 10-30 minutes depending on network speed...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            # Don't load to GPU, just download
            device_map=None,
            low_cpu_mem_usage=True
        )
        print(f"✓ Model downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"\nModel cached in: {cache_dir}")
    print("\nTo use this model in experiments, set MODEL_CACHE_DIR:")
    print(f"  export MODEL_CACHE_DIR={cache_dir}")
    print("\nOr add it to your SLURM script:")
    print(f"  export MODEL_CACHE_DIR={cache_dir}")
    print("="*80)

    return True

if __name__ == "__main__":
    success = download_model()
    exit(0 if success else 1)
