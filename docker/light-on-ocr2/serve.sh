#!/usr/bin/env bash

# use this script to start a vLLM server on NVIDIA Blackwell
# Basic usage:
#   bash docker/light-on-ocr2/serve.sh

# Note: by default, the script will start a vLLM server with the bbox-soup model on port 8002. 
# bbox-soup is a model that can handle both text and images, and it is suitable for general-purpose OCR tasks.
# If you don't need image processing, use the text-only model instead: lightonai/LightOnOCR-2-1B for better performance.
# You can override the model and port by setting the MODEL and PORT environment variables before running the script.
# Example:
#   MODEL="lightonai/LightOnOCR-2-1B" PORT="8003" bash docker/light-on-ocr2/serve.sh

# Exit immediately if a command exits with a non-zero status
set -e

# ==============================================================================
# CONFIGURATION & CONFIG SELECTION
# ==============================================================================
# Default to the largest dense model if nothing is provided
MODEL="${MODEL:-"lightonai/LightOnOCR-2-1B-bbox-soup"}"
PORT="${PORT:-8002}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.1}"
IMAGE="vllm/vllm-openai:v0.20.0"

echo "========================================================================"
echo "Initializing vLLM Server for on NVIDIA Blackwell"
echo "Model: $MODEL"
echo "Port:  $PORT"
echo "========================================================================"

# HF_TOKEN comes from the repo-root .env via --env-file so it never shows up in `ps`.
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found. Create it with a line: HF_TOKEN=<read-only token>" >&2
    exit 1
fi

# ==============================================================================
# DOCKER EXECUTION
# ==============================================================================
# Running with --ipc=host and matching your system's 128G shm-size for lightning-fast
# intra-node/intra-GPU communication protocols.
docker run --rm -it \
    --gpus all \
    --ipc=host \
    -p "$PORT:$PORT" \
    --shm-size=128G \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    --env-file "$ENV_FILE" \
    $IMAGE \
        --model "$MODEL" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --host 0.0.0.0 \
        --port "$PORT" \
        --limit-mm-per-prompt '{"image": 1}' \
        --mm-processor-cache-gb 0 \
        --no-enable-prefix-caching
