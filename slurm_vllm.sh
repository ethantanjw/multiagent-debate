#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/evolution_%j.log
#SBATCH --error=logs/evolution_%j.err
#SBATCH --job-name=llm_evolution_vllm

# ============================================================================
# CONFIGURATION - Change these variables to adjust vLLM settings
# ============================================================================
NUM_GPUS=1                  # Number of GPUs to use (must match SLURM --gres above)
MAX_CONTEXT_LENGTH=32768    # Maximum context length (Qwen3 supports 32K)

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "LLM Conversation - Evolution with vLLM"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load required modules
module load cuda/12.2
module load gcc arrow

# Set up environment
export SCRATCH=${SCRATCH:-$HOME/scratch}
export MODEL_CACHE_DIR=$SCRATCH
export HF_HOME=$SCRATCH
export TRANSFORMERS_CACHE=$SCRATCH

# Load HuggingFace token from .env file for gated model access
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# vLLM configuration
export VLLM_PORT=8000
export VLLM_HOST="localhost"
MODEL_NAME="Qwen/Qwen3-8B"

# HuggingFace stores models in cache with -- instead of /
# Try both possible locations
if [ -d "${SCRATCH}/${MODEL_NAME}" ]; then
    MODEL_PATH="${SCRATCH}/${MODEL_NAME}"
elif [ -d "${SCRATCH}/models/Qwen--Qwen3-8B" ]; then
    MODEL_PATH="${SCRATCH}/models/Qwen--Qwen3-8B"
else
    # Use model name directly - vLLM will find it in HF cache
    MODEL_PATH="${MODEL_NAME}"
fi

echo ""
echo "Environment:"
echo "  SCRATCH: $SCRATCH"
echo "  MODEL_CACHE_DIR: $MODEL_CACHE_DIR"
echo "  vLLM Port: $VLLM_PORT"
echo "  Model: $MODEL_NAME"
echo "=========================================="

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
pip install datasets

# Note: Model check removed - vLLM will auto-discover model in HuggingFace cache
# If model is not found, vLLM will provide a clear error message

echo ""
echo "Step 1: Starting vLLM server in background..."
echo "=========================================="

# Start vLLM server in background
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --dtype auto \
    --tensor-parallel-size ${NUM_GPUS} \
    --trust-remote-code \
    --max-model-len ${MAX_CONTEXT_LENGTH} \
    --gpu-memory-utilization 0.90 \
    --disable-log-requests \
    > logs/vllm_server_${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"
echo "Logs: logs/vllm_server_${SLURM_JOB_ID}.log"

# Function to cleanup vLLM server on exit
cleanup() {
    echo ""
    echo "=========================================="
    echo "Cleaning up..."
    echo "=========================================="
    if [ ! -z "$VLLM_PID" ]; then
        echo "Stopping vLLM server (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null
        wait $VLLM_PID 2>/dev/null
        echo "vLLM server stopped"
    fi
    echo "Cleanup complete"
}

# Register cleanup function to run on script exit
trap cleanup EXIT INT TERM

echo ""
echo "Step 2: Waiting for vLLM server to be ready..."
echo "=========================================="

# Wait for vLLM server to be ready
python wait_for_vllm.py --host $VLLM_HOST --port $VLLM_PORT --timeout 300

if [ $? -ne 0 ]; then
    echo "ERROR: vLLM server failed to start within timeout"
    echo "Check logs/vllm_server_${SLURM_JOB_ID}.log for details"
    exit 1
fi

echo ""
echo "Step 3: Running prompt evolution..."
echo "=========================================="

# Run evolution
python run_evolution.py

EVOLUTION_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Evolution completed with exit code: $EVOLUTION_EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

# Cleanup will be called automatically via trap

exit $EVOLUTION_EXIT_CODE
