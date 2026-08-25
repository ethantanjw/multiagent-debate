#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=download_llama
#SBATCH --output=slurm/output/%j_%x.out

#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ============================================================================
# One-time Model Download Script
# ============================================================================
# This script downloads Llama-3.1-8B-Instruct to ~/scratch/models/
# Run this once before running experiments to avoid downloading during jobs.
#
# Usage:
#   sbatch setup_download_model.sh
# ============================================================================

echo "============================================================================"
echo "MODEL DOWNLOAD JOB"
echo "============================================================================"

# Environment configuration - use scratch for all caching
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export MODEL_CACHE_DIR=$SCRATCH
export HF_HUB_DOWNLOAD_TIMEOUT=300

# Load modules
module load gcc arrow

# Navigate to project directory
cd /project/6105522/junkais/LLM-Conversation

# Activate virtual environment
source .venv/bin/activate

# Install accelerate if not already installed
pip install 'accelerate>=0.26.0' -q

# Login to HuggingFace (token should be in .env file)
if [ -f .env ]; then
    source .env
fi

if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to HuggingFace..."
    huggingface-cli login --token $HF_TOKEN
else
    echo "WARNING: HF_TOKEN not set, relying on cached login"
fi

# Download model
echo "Starting model download..."
python download_model.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "SUCCESS! Model downloaded to ~/scratch/models/"
    echo "You can now run experiments with slurm_llama.sh"
    echo "============================================================================"
else
    echo ""
    echo "============================================================================"
    echo "ERROR: Model download failed. Check the output above for details."
    echo "============================================================================"
    exit 1
fi
