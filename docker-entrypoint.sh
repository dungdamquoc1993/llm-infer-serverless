#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/qwen35-awq}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.95}"
PORT="${VLLM_PORT:-8000}"

cmd=(
  vllm serve "$MODEL_PATH"
  --quantization compressed-tensors
  --trust-remote-code
  --host 0.0.0.0
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTIL"
  --limit-mm-per-prompt '{"image": 0, "video": 0}'
  --enforce-eager
)

if [[ -n "${VLLM_API_KEY:-}" ]]; then
  cmd+=(--api-key "$VLLM_API_KEY")
fi

exec "${cmd[@]}"
