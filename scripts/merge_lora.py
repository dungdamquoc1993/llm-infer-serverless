"""
Merge a PEFT LoRA adapter into a base model and save a standalone merged model.

This script is designed for the common "trained with QLoRA/LoRA, want a merged
model for deployment" workflow, and it also prints a disk-space estimate before
doing any heavy work to avoid "Disk quota exceeded" mid-merge.

Typical usage (on a box with enough disk):
  python scripts/merge_lora.py \
    --base /workspace/work/base_model \
    --adapter dundq3/qwen3_5_serverless-lora \
    --out /workspace/work/lora_output/merged_model_16bit

Notes:
  - Merge does NOT require GPU; CPU-only works (slower).
  - You need enough free disk to write a full copy of the merged weights
    (roughly the size of the base model weights) plus some overhead.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


def _bytes_to_gb(n: int) -> float:
    return n / (1024**3)


def _dir_weight_bytes(p: Path) -> int:
    """
    Sum model weight files commonly seen in HF repos.
    We only count *potentially large* files to estimate disk use.
    """
    if not p.exists():
        return 0
    total = 0
    patterns = [
        "*.safetensors",
        "*.bin",
        "*.pt",
    ]
    for pat in patterns:
        for f in p.rglob(pat):
            # Skip tiny non-weight artifacts if any
            try:
                size = f.stat().st_size
            except FileNotFoundError:
                continue
            if size >= 10 * 1024 * 1024:  # >= 10MB
                total += size
    return total


def _disk_free_bytes(where: Path) -> int:
    usage = shutil.disk_usage(str(where))
    return usage.free


def _print_space_report(
    base_path: Path, adapter_path: str, out_dir: Path
) -> tuple[int, int, int]:
    # Base path is local dir; adapter might be HF repo id or local dir.
    base_bytes = _dir_weight_bytes(base_path)

    adapter_bytes = 0
    adapter_as_path = Path(adapter_path)
    if adapter_as_path.exists():
        adapter_bytes = _dir_weight_bytes(adapter_as_path)

    out_mount = out_dir if out_dir.exists() else out_dir.parent
    free_bytes = _disk_free_bytes(out_mount)

    print("=" * 70)
    print("DISK ESTIMATE (before merge)")
    print("=" * 70)
    print(
        f"Base weights (approx):    {_bytes_to_gb(base_bytes):6.2f} GB  ({base_path})"
    )
    if adapter_bytes:
        print(
            f"Adapter weights (approx): {_bytes_to_gb(adapter_bytes):6.2f} GB  ({adapter_as_path})"
        )
    else:
        print(
            f"Adapter:                 (HF repo id or not on disk yet)  ({adapter_path})"
        )
    print(f"Output dir:              {out_dir}")
    print(f"Free disk at output:     {_bytes_to_gb(free_bytes):6.2f} GB  ({out_mount})")

    # Conservative estimate:
    # - merged weights roughly ~= base weights (a new full copy)
    # - plus ~2GB overhead for configs, tokenizer, temp files, and safety buffer
    overhead = int(2 * (1024**3))
    # Sometimes the save creates/rewrites shards; add 5% buffer.
    needed = int(base_bytes * 1.05) + overhead

    print(
        f"Estimated additional need:{_bytes_to_gb(needed):6.2f} GB  (base*1.05 + 2GB buffer)"
    )
    if free_bytes < needed:
        short = needed - free_bytes
        print(f"[WARN] Not enough free disk. Short by ~{_bytes_to_gb(short):.2f} GB.")
    else:
        print("[OK] Disk looks sufficient for merge.")
    print("=" * 70)
    print()
    return base_bytes, needed, free_bytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base", required=True, help="Base model path (local dir) or HF repo id."
    )
    ap.add_argument(
        "--adapter", required=True, help="LoRA adapter path (local dir) or HF repo id."
    )
    ap.add_argument("--out", required=True, help="Output directory for merged model.")
    ap.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Compute dtype for merge.",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Where to run the merge.",
    )
    ap.add_argument(
        "--push", default=None, help="Optional: push merged model to this HF repo id."
    )
    ap.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token (or set HF_TOKEN).",
    )
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading base.",
    )
    ap.add_argument(
        "--yes", action="store_true", help="Proceed even if disk estimate warns."
    )
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    base_as_path = Path(args.base)
    if not base_as_path.exists():
        print(
            "[ERROR] --base must be a local directory for disk estimation/merge in this script."
        )
        print("        Tip: use `huggingface-cli download` to download base first.")
        return 2

    _, needed, free = _print_space_report(base_as_path, args.adapter, out_dir)
    if free < needed and not args.yes:
        print(
            "[ABORT] Add disk / change --out to a larger volume, or rerun with --yes to force."
        )
        return 3

    # Heavy imports after the disk check.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available; falling back to CPU.")
        device = "cpu"

    print(f"[INFO] Loading base from: {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model = model.to("cpu")

    print(f"[INFO] Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("[INFO] Merging adapter into base (merge_and_unload)...")
    merged = model.merge_and_unload()

    print(f"[INFO] Saving merged model to: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(out_dir)

    if args.push:
        if not args.hf_token:
            print("[WARN] --push set but no HF token provided; skipping push.")
        else:
            print(f"[INFO] Pushing merged model to: {args.push}")
            merged.push_to_hub(args.push, token=args.hf_token, private=True)
            tokenizer.push_to_hub(args.push, token=args.hf_token, private=True)
            print(f"[DONE] https://huggingface.co/{args.push}")

    print("[DONE] Merge completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
