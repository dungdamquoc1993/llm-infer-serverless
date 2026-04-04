"""
Chạy 1 lần trên RunPod Pod (CPU/GPU) có attach Network Volume → lưu weights.

Volume trên Pod gắn tại /workspace → mặc định SAVE_DIR=/workspace/qwen35-awq.

Ghi đè: WEIGHTS_DIR=/đường/khác python download_model.py
"""

import os
from huggingface_hub import snapshot_download

MODEL_REPO = "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit"
SAVE_DIR = os.environ.get("WEIGHTS_DIR", "/workspace/qwen35-awq")

def main():
    hf_token = os.environ.get("HF_TOKEN")  # chỉ cần nếu model private

    print(f"Bắt đầu download: {MODEL_REPO}")
    print(f"Lưu vào: {SAVE_DIR}")
    print("Dung lượng ước tính: ~22GB — chờ vài phút...\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=SAVE_DIR,
        token=hf_token,
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )

    print(f"\nDownload xong. Kiểm tra:")
    files = os.listdir(SAVE_DIR)
    for f in sorted(files):
        path = os.path.join(SAVE_DIR, f)
        size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.isfile(path) else 0
        print(f"  {f}  ({size_mb:.1f} MB)" if size_mb > 0 else f"  {f}/")

if __name__ == "__main__":
    main()
