#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=conversation_llama
#SBATCH --output=slurm/output/%j_%x.out

#SBATCH --time=0-18:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G

# ============================================================================
# Environment Configuration
# ============================================================================

# Hugging Face cache configuration - all in scratch
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export HF_HUB_DOWNLOAD_TIMEOUT=120

# Model cache directory (for Llama weights)
export MODEL_CACHE_DIR=$SCRATCH

# Load modules
module load gcc arrow

# Navigate to project directory
cd /project/6105522/junkais/LLM-Conversation

# ============================================================================
# Code Update (pull latest from GitHub)
# ============================================================================

echo "Updating code from GitHub..."
git pull origin main

# ============================================================================
# Virtual Environment Setup
# ============================================================================

echo "Activating virtual environment..."
source .venv/bin/activate

# ============================================================================
# Check for .env file
# ============================================================================

if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please create .env file with OPENAI_API_KEY"
    exit 1
fi

# ============================================================================
# Model Download (if needed)
# ============================================================================

# Check if model is already cached
MODEL_CHECK_PATH="$MODEL_CACHE_DIR/Llama-3.1-8B-Instruct"

if [ ! -d "$MODEL_CHECK_PATH" ]; then
    echo "Model not found in cache. Downloading..."
    echo "This is a one-time download (~16GB, may take 10-30 minutes)"

    # Make sure we're logged in to HuggingFace
    huggingface-cli login --token $HF_TOKEN

    # Download the model
    python download_model.py

    if [ $? -ne 0 ]; then
        echo "ERROR: Model download failed!"
        exit 1
    fi
else
    echo "Model found in cache: $MODEL_CHECK_PATH"
fi

# ============================================================================
# Run Evolution
# ============================================================================

echo "Starting evolution experiment..."
python run_evolution.py

echo "Job completed!"
