# qwen3-5_serverless

Repo hỗ trợ **fine-tune** model ngôn ngữ (hướng tới **Qwen3.5**) bằng **LoRA / QLoRA** trên môi trường GPU (ví dụ RunPod), với **dataset** lấy từ hội thoại **Facebook Messenger** đã lưu trong PostgreSQL (ứng dụng El Ripley).

Mục tiêu sản phẩm: model trả lời khách **đúng văn phong shop** (một fanpage cụ thể), không chỉ chung chung.

---

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|-----------|----------|
| **`dataset/`** | `train.jsonl`, `val.jsonl` (ChatML, mỗi dòng một sample) — output của pipeline export + clean. |
| **`data/`** | Postgres qua Docker Compose, SQL schema, script **export inbox → JSONL**, `clean_dataset.py`, `pyproject.toml` riêng (môi trường nhẹ cho data). |
| **`scripts/`** | Script phục vụ train trên GPU / RunPod: upload dataset lên HF, download base model, train QLoRA, `.env.example` cho pod. |
| **`docs/`** | Tài liệu tiến trình dataset, cheat sheet QLoRA/stack train, ghi chú RunPod. |
| **`pyproject.toml` (root)** | Dependencies **train** (torch, transformers, trl, peft, bitsandbytes, unsloth, …). |

---

## Yêu cầu nhanh

- **Python 3.11+**
- **Poetry** (khuyến nghị) — hai môi trường tách: `data/` (export) và root (train).
- **Docker** — chạy PostgreSQL local khi phát triển (`data/docker-compose.yml`).

---

## Bắt đầu

### 1. Dữ liệu & export JSONL

```bash
cd data
# Cần file `.env` (POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_PORT, …) — xem docker-compose
docker compose up -d
poetry install
poetry run python export_dataset.py
```

File ghi ra: `../dataset/train.jsonl`, `../dataset/val.jsonl` (ở root repo). Biến kết nối DB đặt trong **`.env` ở root** và/hoặc **`data/.env`** (xem [docs/finetune-process.md](docs/finetune-process.md)).

### 2. Fine-tune (GPU / RunPod, Qwen3.5-9B QLoRA)

#### 2.1. Chuẩn bị dataset trên Hugging Face Hub

Chạy trên **máy local** (Mac, nơi bạn có `dataset/*.jsonl`):

```bash
python scripts/upload_dataset.py
```

Script sẽ:
- Đọc `.env` ở root để lấy `HF_TOKEN`.
- Tạo/tái sử dụng một dataset repo private, upload `dataset/train.jsonl` & `dataset/val.jsonl`.
- In ra dòng gợi ý `HF_DATASET_REPO=...` để copy vào `.env` trên pod.

#### 2.2. Chuẩn bị pod (RunPod)

Trên pod, bạn cần một container image đã cài đủ deps (build từ `Dockerfile` ở root).

**Gợi ý workflow:**

- Push code lên GitHub → GitHub Actions build/push image lên GHCR (xem `.github/workflows/docker-ghcr.yml`)
- Tạo RunPod pod dùng image `ghcr.io/<github-user>/<repo>:latest`
- SSH vào pod → `git clone` repo → copy `.env` → chạy scripts

```bash
git clone https://github.com/<github-user>/<repo>.git /workspace/qwen3-5_serverless
cd /workspace/qwen3-5_serverless
cp scripts/.env.example .env       # template .env dành cho pod
vim .env                           # điền HF_TOKEN, MODEL_REPO, HF_DATASET_REPO, OUTPUT_DIR, ...
```

Download base model **Qwen/Qwen3.5-9B** về Network Volume:

```bash
python scripts/download_model.py
```

Sau khi tải xong, đảm bảo trong `.env`:

```bash
MODEL_PATH=/workspace/base_model   # hoặc WEIGHTS_DIR bạn đã chọn
```

#### 2.3. Chạy train QLoRA

```bash
python scripts/train.py
```

Script `train.py` sẽ:
- Load model `Qwen3.5-9B` qua **Unsloth FastLanguageModel** với `load_in_4bit=True` (QLoRA 4-bit NF4).
- Tự động **tắt thinking mode** (`ENABLE_THINKING=false`) khi apply chat template (không sinh `<think>...</think>` trong training text).
- Load dataset từ:
  - HF Hub (`HF_DATASET_REPO`) nếu set; hoặc
  - Local `dataset/train.jsonl` + `dataset/val.jsonl` nếu không set.
- Sử dụng `DataCollatorForCompletionOnlyLM` để chỉ tính loss trên phần assistant.
- Áp dụng LoRA với `TARGET_MODULES=all-linear` (Unsloth tự chọn linear layers phù hợp cho kiến trúc hybrid của Qwen3.5).
- Lưu checkpoint vào `OUTPUT_DIR` và adapter cuối vào `OUTPUT_DIR/final_adapter`.
- (Mới) Tuỳ chọn merge adapter vào base model → full fine-tuned model BF16 tại `OUTPUT_DIR/merged_model` (`MERGE_MODEL=true`) và có thể push lên HF Hub qua `MERGED_REPO`.

Hyperparameters (epoch, batch size, gradient accumulation, learning rate, `MAX_SEQ_LEN`, …) cấu hình qua `scripts/.env.example` (copy sang `.env` trên pod rồi chỉnh).

**Preset VRAM 16GB (RTX A4000):**

- Đặt `BATCH_SIZE=1`, `GRAD_ACCUM=16` (effective batch vẫn = 16) để tránh OOM.

---

## Tài liệu

| File | Mô tả |
|------|--------|
| [docs/finetune-process.md](docs/finetune-process.md) | Tiến trình: schema inbox, format sample, export, train/val, Docker port, tách Poetry. |
| [docs/qlora-cheat-sheet.md](docs/qlora-cheat-sheet.md) | Ghi chú VRAM, vòng đời train, bảng dependency (transformers, trl, peft, …). |

---

## Biến môi trường

- Sao chép **`.env.example`** → **`.env`** ở root; không commit file chứa secret.
- `HF_TOKEN`, `RUNPOD_API_KEY`, mật khẩu Postgres — chỉ lưu local hoặc secret manager.
