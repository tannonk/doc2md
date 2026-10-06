#!/usr/bin/env bash

# Use this script to start Nemotron 3 Nano Omni with vLLM on DGX Spark.
# Example usage:
#   bash docker/nemotron-3-omni/serve.sh
#
# Optional memory-tuning overrides:
#   MODEL=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8 \
#   PORT=8000 \
#   bash docker/nemotron-3-omni/serve.sh

set -e

MODEL="${MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-210000}"
# 0.35 leaves ~6 GiB for the API-server's own CUDA context (image resizing runs there,
# outside vLLM's budget); the KV cache at 0.35 still holds ~380k tokens. See docs/gpu_memory_budget.md.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
IMAGE="vllm/vllm-openai:v0.20.0"

echo "========================================================================"
echo "Initializing vLLM Server for on NVIDIA Blackwell"
echo "Model:                $MODEL"
echo "Port:                 $PORT"
echo "Max model length:     $MAX_MODEL_LEN"
echo "GPU memory fraction:  $GPU_MEMORY_UTILIZATION"
echo "========================================================================"

# HF_TOKEN comes from the repo-root .env via --env-file so it never shows up in `ps`.
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found. Create it with a line: HF_TOKEN=<read-only token>" >&2
    exit 1
fi

docker run --rm -it \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -p "$PORT:$PORT" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    --env-file "$ENV_FILE" \
    --entrypoint /bin/bash \
    "$IMAGE" -c \
    "python3 -m pip install 'vllm[audio]' && exec vllm serve '$MODEL' \
        --host 0.0.0.0 \
        --port '$PORT' \
        --max-num-seqs 8 \
        --max-model-len '$MAX_MODEL_LEN' \
        --trust-remote-code \
        --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' \
        --limit-mm-per-prompt '{\"video\": 0, \"image\": 5, \"audio\": 0}' \
        --media-io-kwargs '{\"video\": {\"fps\": 2, \"num_frames\": 256}}' \
        --video-pruning-rate 0.5 \
        --allowed-local-media-path / \
        --enable-prefix-caching \
        --max-num-batched-tokens 32768 \
        --reasoning-parser nemotron_v3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder"
