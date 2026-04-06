"""
Quantize a (merged) HF model folder to AWQ 4-bit for vLLM.

Recommended flow:
  1) Merge LoRA -> merged_model_16bit/   (you already did this)
  2) Quantize -> merged_model_awq/
  3) Serve with vLLM: --quantization awq

Example:
  python scripts/quantize_awq.py \
    --in_dir /workspace/lora_output/merged_model_16bit \
    --out_dir /workspace/lora_output/merged_model_awq \
    --group_size 128
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Input HF model folder (merged FP16/BF16).")
    ap.add_argument("--out_dir", required=True, help="Output folder for AWQ quantized model.")
    ap.add_argument("--w_bit", type=int, default=4, help="Weight bit-width (use 4 for vLLM).")
    ap.add_argument("--group_size", type=int, default=128, help="Group size, commonly 128.")
    ap.add_argument("--zero_point", action="store_true", help="Enable zero_point (default True in many examples).")
    ap.add_argument("--no_zero_point", action="store_true", help="Force zero_point=False (e.g. for some kernels).")
    ap.add_argument("--version", default="GEMM", help='AWQ version, commonly "GEMM".')
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not in_dir.exists():
        print(f"[ERROR] in_dir not found: {in_dir}")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide zero_point value.
    if args.no_zero_point:
        zero_point = False
    else:
        # default to True unless explicitly disabled
        zero_point = True if args.zero_point or not args.no_zero_point else False

    try:
        from awq import AutoAWQForCausalLM
    except ModuleNotFoundError:
        print('[ERROR] Missing dependency: autoawq/awq. Install with one of:')
        print('  pip install -U autoawq')
        print('  # or')
        print('  pip install -U "autoawq[triton]"')
        return 2

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        print('[ERROR] Missing dependency: transformers. Install with:')
        print('  pip install -U "transformers>=5.2.0"')
        return 2

    quant_config = {
        "zero_point": zero_point,
        "q_group_size": int(args.group_size),
        "w_bit": int(args.w_bit),
        "version": str(args.version),
    }

    print("=" * 70)
    print("AWQ QUANT CONFIG")
    print("=" * 70)
    print(f"  in_dir     : {in_dir}")
    print(f"  out_dir    : {out_dir}")
    print(f"  config     : {quant_config}")
    print("=" * 70)
    print()

    print("[INFO] Loading model (this can take a while)...")
    model = AutoAWQForCausalLM.from_pretrained(str(in_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(in_dir), trust_remote_code=args.trust_remote_code)

    print("[INFO] Quantizing (AWQ)...")
    model.quantize(tokenizer, quant_config=quant_config)

    print("[INFO] Saving quantized model...")
    model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print("[DONE] AWQ model saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

