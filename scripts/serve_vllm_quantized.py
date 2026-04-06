"""
Serve a (quantized) HF model folder with vLLM.

Example (AWQ 4-bit):
  export MODEL_PATH=/workspace/lora_output/merged_model_awq
  export VLLM_QUANTIZATION=awq
  python scripts/serve_vllm_quantized.py
"""

from __future__ import annotations

import os
import shlex
from typing import List


def _env(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    return v if v is not None and v != "" else default


def main() -> int:
    model = _env("MODEL_PATH")
    if not model:
        print("[ERROR] Set MODEL_PATH to the quantized model folder (or HF repo id).")
        return 2

    host = _env("VLLM_HOST", "0.0.0.0")
    port = _env("VLLM_PORT", "8000")

    # Required for quantized models
    quant = _env("VLLM_QUANTIZATION")  # e.g. awq, gptq

    # Optional knobs
    max_model_len = _env("VLLM_MAX_MODEL_LEN")
    gpu_mem_util = _env("VLLM_GPU_MEMORY_UTILIZATION")
    dtype = _env("VLLM_DTYPE")  # for some quant formats, dtype may be ignored

    cmd: List[str] = ["vllm", "serve", model, "--host", host, "--port", str(port)]
    if quant:
        cmd += ["--quantization", quant]
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
    if gpu_mem_util:
        cmd += ["--gpu-memory-utilization", str(gpu_mem_util)]
    if dtype:
        cmd += ["--dtype", str(dtype)]

    print("[INFO] Starting vLLM server:")
    print("       " + " ".join(shlex.quote(x) for x in cmd))
    print()
    print("[INFO] Test:")
    print(f"  curl http://localhost:{port}/v1/models")
    print(f'  curl http://localhost:{port}/v1/chat/completions -H "Content-Type: application/json" -d \'{{"model":"{model}","messages":[{{"role":"user","content":"Xin chào"}}],"temperature":0.2,"max_tokens":256}}\'')
    print()

    os.execvp(cmd[0], cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

