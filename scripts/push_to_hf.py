"""
Push a local Hugging Face model folder to the Hub.

Works for:
  - LoRA adapter folders (adapter_config.json, adapter_model.safetensors, ...)
  - Merged full models (config.json, model*.safetensors, tokenizer files, ...)
  - Quantized models (e.g. AWQ export folder)

Example:
  export HF_TOKEN=...
  python scripts/push_to_hf.py \
    --local_dir /workspace/lora_output/merged_model_16bit \
    --repo_id dundq3/qwen3_5_serverless-merged \
    --private
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", required=True, help="Local folder to upload.")
    ap.add_argument("--repo_id", required=True, help="HF repo id: username/name")
    ap.add_argument("--repo_type", default="model", choices=["model", "dataset", "space"])
    ap.add_argument("--private", action="store_true", help="Create repo as private if it doesn't exist.")
    ap.add_argument("--commit_message", default="Upload model artifacts")
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    local_dir = Path(args.local_dir).expanduser().resolve()
    if not local_dir.exists() or not local_dir.is_dir():
        print(f"[ERROR] local_dir not found: {local_dir}")
        return 2
    if not args.hf_token:
        print("[ERROR] Missing HF token. Set HF_TOKEN or pass --hf_token.")
        return 2

    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError:
        print('[ERROR] Missing dependency: huggingface_hub. Install with:')
        print('  pip install -U "huggingface_hub>=0.23"')
        return 2

    api = HfApi(token=args.hf_token)
    print(f"[INFO] Ensuring repo exists: {args.repo_id} (type={args.repo_type}, private={args.private})")
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)

    print(f"[INFO] Uploading folder: {local_dir}")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(local_dir),
        commit_message=args.commit_message,
    )

    print(f"[DONE] https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

