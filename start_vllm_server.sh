#!/bin/bash
#
# Start vLLM server for Llama-3.1-8B-Instruct
#
# This script starts a vLLM server that serves the model via OpenAI-compatible API
# on port 8000. The server loads the model from $SCRATCH and enables tensor parallelism
# for multi-GPU support.

set -e

# Configuration
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH="${SCRATCH}/${MODEL_NAME}"
PORT=8000
HOST="0.0.0.0"
TENSOR_PARALLEL_SIZE=1  # Set to number of GPUs if using multiple

echo "=========================================="
echo "Starting vLLM Server"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Model Path: ${MODEL_PATH}"
echo "Port: ${PORT}"
echo "Host: ${HOST}"
echo "Tensor Parallel: ${TENSOR_PARALLEL_SIZE}"
echo "=========================================="

# Check if model exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: Model not found at ${MODEL_PATH}"
    echo "Please download the model first using download_model.py"
    exit 1
fi

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --disable-log-requests \
    2>&1 | tee vllm_server.log
