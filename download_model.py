"""
Chạy script này 1 LẦN DUY NHẤT trên RunPod Pod (có attach Network Volume).

Đường mount của Network Volume:
  - Pod: volume thay thế ổ mặc định, gắn tại /workspace (RunPod không cho đổi trên UI).
  - Serverless worker: /runpod-volume (handler đọc model từ đó).

Cùng một volume: tải vào /workspace/qwen35-awq trên Pod thì Serverless vẫn thấy
cùng dữ liệu dưới /runpod-volume/qwen35-awq.

Ghi đè thư mục (nếu cần):
  WEIGHTS_DIR=/đường/khác python download_model.py
"""

import os
from huggingface_hub import snapshot_download

MODEL_REPO = "cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit"
# Mặc định cho Pod (Secure Cloud). Serverless dùng /runpod-volume — cùng volume, khác mount point.
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
