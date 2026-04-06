"""
Upload dataset/train.jsonl + dataset/val.jsonl lên HuggingFace Hub (private).

Chạy từ bất kỳ thư mục nào:
    python scripts/upload_dataset.py

Biến môi trường (đọc từ .env ở root repo):
    HF_TOKEN       - bắt buộc
    HF_DATASET_REPO - tên repo, vd: "your-username/shop-chat-sft"
                      (mặc định: dùng tên thư mục repo + "-dataset")
"""

import os
import sys
from pathlib import Path

# ── Load .env từ root repo ────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    print("[WARN] python-dotenv không có, đọc env vars từ shell.")

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_REPO = os.environ.get("HF_DATASET_REPO", f"{REPO_ROOT.name}-dataset")
DATASET_DIR = REPO_ROOT / "dataset"
FILES_TO_PUSH = ["train.jsonl", "val.jsonl"]


# ── Validate ──────────────────────────────────────────────────────────────────
def main():
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN chưa được set. Thêm vào .env hoặc export.")
        sys.exit(1)

    missing = [f for f in FILES_TO_PUSH if not (DATASET_DIR / f).exists()]
    if missing:
        print(f"[ERROR] Không tìm thấy file: {missing}")
        print(f"        Kiểm tra thư mục {DATASET_DIR}")
        sys.exit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    # Lấy username từ token
    user_info = api.whoami()
    username = user_info["name"]

    # Đảm bảo repo ID có dạng "username/repo-name"
    repo_id = DATASET_REPO if "/" in DATASET_REPO else f"{username}/{DATASET_REPO}"

    print(f"HuggingFace user : {username}")
    print(f"Dataset repo     : {repo_id}")
    print(f"Nguồn            : {DATASET_DIR}")
    print()

    # Tạo repo nếu chưa có
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    print(f"[OK] Repo sẵn sàng: https://huggingface.co/datasets/{repo_id}")

    # Upload từng file
    for filename in FILES_TO_PUSH:
        local_path = DATASET_DIR / filename
        size_kb = local_path.stat().st_size / 1024
        print(f"Đang upload {filename} ({size_kb:.0f} KB) ...", end="", flush=True)

        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"upload {filename}",
        )
        print(" ✓")

    # Tạo README đơn giản nếu chưa có
    try:
        api.hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="dataset")
    except Exception:
        card_content = f"""---
license: other
tags:
- sft
- chat
- vietnamese
- qlora
---

# {repo_id.split('/')[-1]}

Private SFT dataset cho fine-tune shop chat assistant (tiếng Việt).

## Files
- `train.jsonl` – training samples (ChatML format)
- `val.jsonl`   – validation samples

## Format
```json
{{"messages": [{{"role": "system", "content": "..."}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}
```
"""
        api.upload_file(
            path_or_fileobj=card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="add README",
        )

    print()
    print(f"[DONE] Dataset đã upload: https://huggingface.co/datasets/{repo_id}")
    print()
    print("Thêm vào .env trên pod:")
    print(f"  HF_DATASET_REPO={repo_id}")


if __name__ == "__main__":
    main()
