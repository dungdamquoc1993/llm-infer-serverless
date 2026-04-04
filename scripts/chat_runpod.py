#!/usr/bin/env python3
"""
Gọi vLLM trên RunPod qua OpenAI SDK. Cấu hình bằng biến môi trường (file .env ở thư mục gốc repo).

Bắt buộc:
  VLLM_BASE_URL   URL RunPod (vd https://xxxxx-8000.proxy.runpod.net) — không cần /v1

Tuỳ chọn:
  VLLM_MODEL              Mặc định: /workspace/qwen35-awq
  VLLM_API_KEY            Nếu pod bật --api-key thì đặt trùng giá trị
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


def base_url_with_v1(raw: str) -> str:
    raw = raw.strip().rstrip("/")
    if raw.endswith("/v1"):
        return raw
    return f"{raw}/v1"


def main() -> int:
    p = argparse.ArgumentParser(description="Chat completion qua vLLM (OpenAI-compatible).")
    p.add_argument(
        "message",
        nargs="*",
        default=None,
        help="Câu hỏi (mặc định: biến VLLM_PROMPT hoặc '2+2=?')",
    )
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("VLLM_MAX_TOKENS", "256")))
    args = p.parse_args()

    base = os.environ.get("VLLM_BASE_URL", "").strip()
    if not base:
        print("Thiếu VLLM_BASE_URL trong .env (URL proxy RunPod, không cần /v1).", file=sys.stderr)
        return 1

    model = os.environ.get("VLLM_MODEL", "/workspace/qwen35-awq").strip()
    api_key = os.environ.get("VLLM_API_KEY", "not-needed").strip() or "not-needed"

    if args.message:
        user_text = " ".join(args.message).strip()
    else:
        user_text = os.environ.get("VLLM_PROMPT", "2+2=?").strip()

    try:
        from openai import OpenAI
    except ImportError:
        print("Cần: pip install openai python-dotenv", file=sys.stderr)
        return 1

    client = OpenAI(base_url=base_url_with_v1(base), api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_text}],
        max_tokens=args.max_tokens,
    )
    text = resp.choices[0].message.content
    print(text or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
