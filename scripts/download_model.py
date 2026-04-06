"""
Tải base model từ HuggingFace Hub về Network Volume trên RunPod.

TẠI SAO CẦN BẢN GỐC (bf16/fp16)?
──────────────────────────────────
QLoRA KHÔNG dùng model đã quantize sẵn (GGUF/GPTQ). Quy trình thực tế:
  1. Download file trọng số gốc (bf16, ~19 GB với Qwen3.5-9B)
  2. Lúc train: bitsandbytes đọc trọng số gốc → quantize NF4 ON-THE-FLY → load vào VRAM
  3. LoRA adapters được khởi tạo và train ở bf16 bên cạnh base model đã frozen

→ Chỉ adapter (~100–200 MB) được lưu khi checkpoint, không cần lưu lại base.
→ Sau train, merge adapter + base → export GGUF nếu cần inference.

VRAM Qwen3.5-9B (theo bảng official):
  BF16 (gốc)  : ~19 GB  → chỉ đủ inference, KHÔNG đủ train trên RTX 4000 Ada 20GB
  4-bit QLoRA : ~6.5 GB → OK; cộng optimizer + activation → tổng ~13–16 GB ✓

Biến môi trường (đọc từ .env):
  MODEL_REPO   - HF repo ID (mặc định: Qwen/Qwen3.5-9B)
  WEIGHTS_DIR  - thư mục lưu trên Network Volume (mặc định: /workspace/base_model)
  HF_TOKEN     - nếu model private

Chạy:
  python scripts/download_model.py
  MODEL_REPO=Qwen/Qwen3.5-9B WEIGHTS_DIR=/workspace/qwen35-9b python scripts/download_model.py
"""

import os
import sys
from pathlib import Path

# ── Load .env từ root repo ────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass  # trên pod dùng env vars trực tiếp

# ── Config ────────────────────────────────────────────────────────────────────
# Qwen3.5-9B: model chính thức tại https://huggingface.co/Qwen/Qwen3.5-9B
# Kiến trúc hybrid (Gated DeltaNet + Attention), 10B params, bf16 ~19 GB
# Các lựa chọn trong dòng Qwen3.5 (VRAM 4-bit / bf16):
#   Qwen/Qwen3.5-0.8B   ~3.5 / 9 GB    (test nhanh)
#   Qwen/Qwen3.5-4B     ~5.5 / 14 GB
#   Qwen/Qwen3.5-9B     ~6.5 / 19 GB   ← recommended với 20 GB VRAM (RTX 4000 Ada)
#   Qwen/Qwen3.5-27B    ~17 / 54 GB    (cần A100 40GB hoặc multi-GPU)
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen3.5-9B")
SAVE_DIR   = os.environ.get("WEIGHTS_DIR", "/workspace/base_model")

# Các file không cần thiết cho training
IGNORE_PATTERNS = [
    "*.md", "*.txt", "*.gitattributes",
    "original/",           # Llama-style original weights
    "*.gguf",              # quantized inference weights
    "*.ggml",
]

def fmt_gb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**3:.2f} GB"

def main():
    hf_token = os.environ.get("HF_TOKEN")

    print("=" * 60)
    print(f"Model repo : {MODEL_REPO}")
    print(f"Lưu vào    : {SAVE_DIR}")
    print(f"HF token   : {'có' if hf_token else 'không (chỉ dùng được model public)'}")
    print("=" * 60)
    print()
    print("Đây là base model bf16 — dùng cho QLoRA/LoRA fine-tuning.")
    print("bitsandbytes sẽ quantize 4-bit on-the-fly khi load để train.")
    print()

    from huggingface_hub import snapshot_download, repo_info

    # Kiểm tra repo tồn tại trước khi download
    try:
        info = repo_info(MODEL_REPO, token=hf_token)
        print(f"Repo xác nhận: {info.id}  (revision: {info.sha[:8] if info.sha else 'unknown'})")
    except Exception as e:
        print(f"[ERROR] Không truy cập được repo '{MODEL_REPO}': {e}")
        print("        Kiểm tra MODEL_REPO và HF_TOKEN.")
        sys.exit(1)

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\nBắt đầu download → {SAVE_DIR} ...")
    print("(Với 8B model ~16 GB, mất 10–20 phút tùy băng thông)\n")

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=SAVE_DIR,
        token=hf_token,
        ignore_patterns=IGNORE_PATTERNS,
    )

    print(f"\n[DONE] Download xong. Nội dung {SAVE_DIR}:")
    total_bytes = 0
    for p in sorted(Path(SAVE_DIR).rglob("*")):
        if p.is_file():
            sz = p.stat().st_size
            total_bytes += sz
            rel = p.relative_to(SAVE_DIR)
            print(f"  {rel}  ({fmt_gb(sz)})")

    print(f"\nTổng dung lượng: {fmt_gb(total_bytes)}")
    print("\nThêm vào .env trên pod:")
    print(f"  MODEL_PATH={SAVE_DIR}")


if __name__ == "__main__":
    main()
