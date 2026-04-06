"""
Start a vLLM OpenAI-compatible server with a LoRA adapter.

Designed for RunPod "vLLM template" style pods where vLLM is already installed.

Example (serve base + LoRA, no merge needed):
  export MODEL_PATH=/workspace/work/base_model            # or HF repo id
  export LORA_MODULES="shop=dundq3/qwen3_5_serverless-lora"
  python scripts/serve_vllm_lora.py

Then query:
  curl http://localhost:8000/v1/models
  curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model":"shop",
    "messages":[{"role":"user","content":"Xin chào"}],
    "temperature":0.2,
    "max_tokens":256
  }'

Notes:
  - vLLM serves each LoRA as its own "model id" (the left side of name=path).
  - This script uses static LoRA loading via `--lora-modules` so you do NOT
    need runtime LoRA updating endpoints.
"""

from __future__ import annotations

import os
import shlex
from typing import List


def _env(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    return v if v is not None and v != "" else default


def _split_lora_modules(raw: str) -> List[str]:
    """
    Accept:
      - "name=path" (single)
      - "a=path1 b=path2" (space separated)
      - "a=path1,b=path2" (comma separated)
      - JSON strings are also accepted by vLLM; pass through as one arg.
    """
    s = raw.strip()
    if not s:
        return []
    if s.startswith("{") and s.endswith("}"):
        return [s]
    parts: List[str] = []
    for chunk in s.replace(",", " ").split():
        if chunk:
            parts.append(chunk)
    return parts


def main() -> int:
    model = _env("MODEL_PATH", _env("MODEL_REPO", "Qwen/Qwen3.5-9B"))
    if not model:
        print("[ERROR] Missing MODEL_PATH or MODEL_REPO.")
        return 2

    # LoRA
    lora_modules_raw = _env("LORA_MODULES", _env("ADAPTER_REPO"))
    # Backward compatible: if ADAPTER_REPO is set, default to name "lora"
    if (
        lora_modules_raw
        and "=" not in lora_modules_raw
        and not lora_modules_raw.strip().startswith("{")
    ):
        lora_modules_raw = f"lora={lora_modules_raw}"

    lora_modules = _split_lora_modules(lora_modules_raw or "")
    if not lora_modules:
        print("[ERROR] Missing LoRA modules. Set LORA_MODULES, e.g.:")
        print('  export LORA_MODULES="shop=dundq3/qwen3_5_serverless-lora"')
        return 2

    host = _env("VLLM_HOST", "0.0.0.0")
    port = _env("VLLM_PORT", "8000")

    # Performance/compat knobs (all optional)
    max_model_len = _env("VLLM_MAX_MODEL_LEN")  # e.g. 4096
    gpu_mem_util = _env("VLLM_GPU_MEMORY_UTILIZATION")  # e.g. 0.90
    tensor_parallel = _env("VLLM_TP")  # e.g. 1
    dtype = _env("VLLM_DTYPE")  # auto/bfloat16/float16
    quant = _env("VLLM_QUANTIZATION")  # awq/gptq/bitsandbytes/...

    # LoRA config (match your training rank to avoid wasted memory)
    max_lora_rank = _env("VLLM_MAX_LORA_RANK", _env("LORA_R", "16"))
    max_loras = _env("VLLM_MAX_LORAS", "1")

    cmd: List[str] = ["vllm", "serve", model, "--host", host, "--port", str(port)]
    cmd += [
        "--enable-lora",
        "--max-lora-rank",
        str(max_lora_rank),
        "--max-loras",
        str(max_loras),
    ]
    for m in lora_modules:
        cmd += ["--lora-modules", m]

    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
    if gpu_mem_util:
        cmd += ["--gpu-memory-utilization", str(gpu_mem_util)]
    if tensor_parallel:
        cmd += ["--tensor-parallel-size", str(tensor_parallel)]
    if dtype:
        cmd += ["--dtype", str(dtype)]
    if quant:
        cmd += ["--quantization", str(quant)]

    print("[INFO] Starting vLLM server:")
    print("       " + " ".join(shlex.quote(x) for x in cmd))
    print()
    print("[INFO] After startup, try:")
    print(f"  curl http://localhost:{port}/v1/models")
    # best-effort: pick first lora name
    first_name = "lora"
    first = lora_modules[0]
    if "=" in first:
        first_name = first.split("=", 1)[0]
    print(
        f'  curl http://localhost:{port}/v1/chat/completions -H "Content-Type: application/json" -d \'{{"model":"{first_name}","messages":[{{"role":"user","content":"Xin chào"}}],"temperature":0.2,"max_tokens":256}}\''
    )
    print()

    # Replace current process (so SIGTERM works nicely in pods)
    os.execvp(cmd[0], cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
