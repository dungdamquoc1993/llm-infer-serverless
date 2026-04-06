# Tiến trình: dataset inbox → JSONL & chuẩn bị fine-tune

Tài liệu này tóm tắt những việc đã làm trong session làm việc về **dữ liệu hội thoại Facebook Messenger** và **chuẩn bị fine-tune** model (Qwen3.5 9B, mục tiêu văn phong shop qua fanpage).

---

## 1. Mục tiêu

- Fine-tune model để trả lời **giống cách shop trả lời khách** trên Messenger.
- **Chỉ một fanpage**: *Giày Đá Bóng - Yêu Bóng Đá Shop* (`fan_page_id`: `693218564214489`).
- Nguồn dữ liệu: PostgreSQL (schema trong `data/sql/02_schema_facebook.sql`), bảng chính:
  - `facebook_conversation_messages`: một dòng = một thread với một khách (PSID).
  - `messages`: từng tin nhắn; `is_echo = false` = khách, `is_echo = true` = page.

---

## 2. Khảo sát dữ liệu (đã làm trong DB)

- Ban đầu số thread trong DB ít hơn kỳ vọng; sau khi **sync thêm** từ Facebook, fanpage mục tiêu đạt khoảng **2542** cuộc (`facebook_conversation_messages` cho đúng `fan_page_id`).
- Phân phối số tin nhắn có text mỗi thread cho thấy đa số hội thoại **đủ dài** (nhiều lượt), phù hợp SFT multi-turn.
- Kết luận: **đủ để train** (kèm sliding window cho thread rất dài); nên ưu tiên dữ liệu sync đầy đủ thay vì chỉ vài trăm thread.

---

## 3. Thiết kế training (đã thống nhất)

### 3.1. Một sample = một hội thoại (hoặc một cửa sổ trượt)

- Chuẩn **SFT với TRL**: thường **một JSONL line = một sample**, trong đó `messages` gồm `system` + nhiều lượt `user` / `assistant`.
- Loss chỉ tính trên phần assistant (trainer/collator mask phần user/system).
- **Không** bắt buộc tách “mỗi lượt khách → một sample” (per-turn); cách đó ít dùng hơn trừ khi cần augment đặc biệt.

### 3.2. System prompt

- Dùng **một system prompt cố định** mô tả vai trò nhân viên shop, phong cách tiếng Việt (xưng hô, ngắn gọn, tư vấn size/giá/ship…).
- **LoRA** chỉ thêm adapter lên base model → giảm nguy cơ “quên” kiến thức tổng quát; không bắt buộc thêm thẻ XML trong từng user message trừ khi sau này cần **multi-fanpage** trong một model.

### 3.3. Train / validation

- File `dataset/train.jsonl` và `dataset/val.jsonl` cùng format.
- **Chia ngẫu nhiên** (seed cố định), tỷ lệ val ~5%, tối thiểu 1 dòng val.
- **Lưu ý:** nếu một thread bị cắt nhiều cửa sổ (sliding window), các cửa sổ có thể rơi vào train và val khác nhau; với quy mô hiện tại thường chấp nhận được. Nếu cần nghiêm ngặt hơn, sau này có thể chia theo `conversation_id`.

---

## 4. Script export (`data/export_dataset.py`)

Đã triển khai pipeline:

1. Chọn thread có **ít nhất một tin page** (`is_echo = true`) và có text.
2. Sắp xếp tin theo thời gian; **bỏ** tin gửi bởi AI agent (`metadata.sent_by == "ai_agent"`) để không học lại output AI cũ.
3. Gộp các tin liên tiếp cùng role (user/assistant) thành một turn.
4. Thread quá dài: **sliding window** (cấu hình `MAX_MSGS_PER_SAMPLE`, `SLIDE_STRIDE`).
5. Ghi ra **JSONL ChatML**: mỗi dòng một object `{"messages": [...]}`.

**Đường dẫn output:** luôn ghi vào thư mục `dataset/` ở **root repo** (không phụ thuộc thư mục chạy lệnh), nhờ resolve `REPO_ROOT` từ vị trí file.

**Biến môi trường:** load `/.env` ở root trước, sau đó `data/.env` (ghi đè key trùng) — phù hợp Docker Compose trong `data/` và cấu hình app ở root.

---

## 5. PostgreSQL & Docker

- Trên máy dev từng có **Postgres cài local** chiếm port **5432**, khiến kết nối `localhost:5432` không vào đúng container.
- Đã cấu hình lại **`data/docker-compose.yml`**: map host **`5434` → container 5432**.
- File **`data/.env`**: `POSTGRES_PORT=5434` (dùng cho compose).
- File **root `.env`**: `POSTGRES_HOST=localhost`, `POSTGRES_PORT=5434` cho script Python.
- Vào DB bằng `docker exec ... psql` (trong container) **không đổi**; từ host kết nối **`localhost:5434`**.

---

## 6. Tách môi trường Poetry

Hai “subproject” rõ vai trò:

| Thư mục | `pyproject.toml` | Mục đích |
|--------|-------------------|----------|
| **`data/`** | `el-ripley-data` | Chỉ cần `python-dotenv`, `psycopg2-binary` — export JSONL, làm việc với Postgres/Docker. |
| **Root** | `qwen-lora-finetune` | `torch`, `transformers`, `trl`, `peft`, `bitsandbytes`, … — train trên GPU (RunPod hoặc máy có GPU). |

Cài đặt tách biệt:

```bash
cd data && poetry install          # chỉ pipeline dữ liệu
cd ..  && poetry install            # môi trường train (nặng hơn)
```

---

## 7. Kết quả số liệu (lần export đã chạy thành công)

Thông số in ra console (có thể thay đổi sau khi sync thêm hoặc chỉnh script):

- Hàng nghìn **samples** trong `train.jsonl` / `val.jsonl` (mỗi dòng = một sample).
- Trung bình ~7–8 “turn” mỗi sample (không tính system), tùy phân phối thực tế.

---

## 8. Data cleaning sau export (đã làm)

Sau khi rà soát file `dataset/train.jsonl` / `dataset/val.jsonl`, phát hiện noise nằm trong **assistant turns** (nếu giữ nguyên thì model sẽ học cả các chuỗi không mong muốn):

- Facebook notification: `X replied to an ad.`
- Auto greeting: `Hi X! Please let us know how we can help you.` (EN/VI)
- CRM labels: `Auto-label added: ...`, `Lead stage set to ...`
- Spam/system text: `This message was automatically moved to spam.`
- Link thông báo bài viết: `X đã trả lời về một bài viết. Xem bài viết(https://...)`
- Dòng dump đơn hàng nội bộ (tab-separated)
- Dòng trùng lặp liên tiếp trong cùng một turn

Đã thêm script:

- `data/clean_dataset.py`

Script này:

1. Backup file gốc thành `dataset/*.jsonl.bak`
2. Clean theo line-level trong **assistant content**
3. Bỏ turn assistant rỗng sau clean, merge lại turn liên tiếp cùng role
4. Đảm bảo sample kết thúc bằng assistant; sample không còn user/assistant hợp lệ thì drop
5. Ghi đè lại `dataset/train.jsonl` và `dataset/val.jsonl` (bản sạch)

Kiểm tra nhanh đã chạy:

- `grep -c "replied to an ad" dataset/train.jsonl` → `0`
- `grep -c "replied to an ad" dataset/train.jsonl.bak` → `644`
- `dataset/val.jsonl` cũng không còn các pattern noise chính

**Kết luận:** dataset hiện tại ở trạng thái **ready để train**.

---

## 9. Bước tiếp theo (train session)

1. Đưa `dataset/` lên Hugging Face Hub (private) nếu train trên cloud – dùng script `scripts/upload_dataset.py` (đọc `.env` để lấy `HF_TOKEN`, tự tạo dataset repo, upload `train.jsonl` & `val.jsonl`, in ra `HF_DATASET_REPO=...` để copy vào pod).
2. Trên RunPod, tải base model **Qwen/Qwen3.5-9B** (bf16 ~19GB) về Network Volume bằng `scripts/download_model.py`:
   - Dùng `MODEL_REPO` (mặc định `Qwen/Qwen3.5-9B`), `WEIGHTS_DIR` (mặc định `/workspace/base_model`), `HF_TOKEN`.
   - QLoRA sẽ quantize 4-bit NF4 **on-the-fly** khi train, nên luôn tải bản bf16 gốc, không dùng GGUF/GPTQ.
3. Fine-tune bằng `scripts/train.py`:
   - Stack: **Unsloth FastLanguageModel + TRL SFTTrainer + PEFT LoRA + bitsandbytes** trên `Qwen3.5-9B`.
   - Dùng **QLoRA 4-bit** (`USE_QLORA=true`) để tiết kiệm VRAM. Với GPU 16GB (RTX A4000) nên đặt `BATCH_SIZE=1`, `GRAD_ACCUM=16` (effective batch vẫn = 16).
   - Dataset: ưu tiên HF Hub (`HF_DATASET_REPO`), fallback local `dataset/train.jsonl` + `dataset/val.jsonl` nếu không set.
   - Thinking mode của Qwen3.5 (`<think>...</think>`) bị **tắt** khi apply chat template (`ENABLE_THINKING=false`) vì data shop chat không có thinking content.
   - LoRA config: `TARGET_MODULES=all-linear` để Unsloth tự chọn linear layers cho kiến trúc hybrid (Gated DeltaNet + Attention) của Qwen3.5; chỉ train adapter (vài % tham số).
   - Sau khi train xong, script sẽ:
     - Lưu adapter vào `OUTPUT_DIR/final_adapter`
     - (Tuỳ chọn) push adapter lên HF Hub qua `ADAPTER_REPO`
     - Merge adapter vào base model → full fine-tuned model BF16 vào `OUTPUT_DIR/merged_model` (bật/tắt bởi `MERGE_MODEL`)
     - (Tuỳ chọn) push merged model lên HF Hub qua `MERGED_REPO`
4. Theo dõi **train loss vs eval loss** (console hoặc W&B nếu set `WANDB_PROJECT`, `WANDB_API_KEY`); điều chỉnh epoch, learning rate, `max_seq_len` dựa trên VRAM và chất lượng.
5. (Tùy chọn) Chia train/val theo `conversation_id` nếu muốn đánh giá “lạ thread” nghiêm hơn.

---

## 10. Môi trường train & scripts RunPod

- **Dockerfile** (root):
  - Base image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
  - Cài các package import trực tiếp trong `scripts/train.py`: `transformers`, `datasets`, `trl`, `wandb`, `python-dotenv`.
  - Cài thêm: `unsloth[cu124-torch240]` từ GitHub.
    - Unsloth sẽ tự kéo các dependency tương thích (ví dụ `peft`, `bitsandbytes`, `accelerate`, `huggingface_hub`, `sentencepiece`, `flash-attn`) để tránh xung đột version.
- **Scripts train/infra** (root `scripts/`):
  - `upload_dataset.py` – upload `dataset/train.jsonl`, `dataset/val.jsonl` lên HF Hub (private), tạo README đơn giản, gợi ý `HF_DATASET_REPO`.
  - `download_model.py` – tải base model (`MODEL_REPO`, mặc định `Qwen/Qwen3.5-9B`) về Network Volume (`WEIGHTS_DIR`), bỏ qua file không cần như `*.gguf`, `original/`.
  - `train.py` – fine-tune QLoRA/LoRA trên Qwen3.5-9B với Unsloth + TRL; tắt thinking mode (`ENABLE_THINKING=false`), dùng `DataCollatorForCompletionOnlyLM` để chỉ tính loss trên phần assistant.
  - `scripts/.env.example` – template `.env` dành cho pod (khác với `.env` local dùng cho Postgres/vLLM); copy thành `.env` trên pod rồi chỉnh giá trị.

**Biến môi trường quan trọng trên pod (xem chi tiết trong `scripts/.env.example`):**

- `HF_TOKEN` – bắt buộc nếu model/dataset private.
- `MODEL_REPO`, `MODEL_PATH` – repo HF & đường dẫn base model trên Network Volume.
- `HF_DATASET_REPO` – repo dataset trên HF; nếu không set, `train.py` dùng `dataset/*.jsonl` local.
- `OUTPUT_DIR` – thư mục lưu checkpoints & adapter (`/workspace/lora_output`).
- `ADAPTER_REPO` – (tuỳ chọn) repo HF để push adapter LoRA.
- `MERGE_MODEL` – `true/false` để merge adapter vào base model sau khi train.
- `MERGED_REPO` – (tuỳ chọn) repo HF để push merged full model (BF16).
- `USE_QLORA`, `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `TARGET_MODULES` – config LoRA/QLoRA (mặc định `TARGET_MODULES=all-linear` cho Qwen3.5).
- `NUM_EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM`, `LR`, `MAX_SEQ_LEN`, `WARMUP_RATIO` – hyperparameter train.
- `USE_FLASH_ATTN`, `ENABLE_THINKING` – tối ưu kernel & bật/tắt thinking mode của Qwen3.5.
- `WANDB_PROJECT`, `WANDB_API_KEY` – logging lên Weights & Biases (tùy chọn).

---

## 11. Troubleshooting nhanh trên RunPod (các lỗi đã gặp)

### 11.1. `HF_DATASET_REPO` có trong `scripts/.env` nhưng script vẫn báo “local JSONL”

`scripts/train.py` **chỉ load `.env` ở root repo**:

- Đúng: `/workspace/llm-infer-serverless/.env`
- Sai: `/workspace/llm-infer-serverless/scripts/.env`

Fix:

```bash
cd /workspace/llm-infer-serverless/scripts
cp .env ../.env
```

### 11.2. `UnicodeDecodeError` khi đọc `.env`

Ví dụ:

`UnicodeDecodeError: 'utf-8' codec can't decode bytes ...`

Nguyên nhân thường là file `.env` trên pod bị lưu **không phải UTF-8** (hoặc bị copy/paste dính ký tự “lạ” → hiện `�`).

Fix:

- Mở và lưu lại `.env` dạng **UTF-8** (không BOM), tránh ký tự “smart quotes”.
- Nhanh nhất: tạo lại `.env` từ `scripts/.env.example` rồi chỉnh từng dòng.

### 11.3. `KeyError: 'qwen3_5'` / Transformers không nhận kiến trúc Qwen3.5

Thông báo điển hình:

- `Transformers does not recognize this architecture` hoặc
- `Unsloth: Your transformers version ... does not support Qwen3.5. The minimum required version is 5.2.0`

Fix (trên pod):

```bash
pip install --upgrade "transformers>=5.2.0" unsloth
```

Lưu ý: upgrade `transformers` lên 5.x có thể gây **xung đột với `vllm`** (vì `vllm` thường pin `transformers<5`). Việc này **không ảnh hưởng training**, nhưng nếu pod dùng `vllm` cùng env thì nên tách môi trường.

### 11.4. `ImportError: cannot import name 'DataCollatorForCompletionOnlyLM' from 'trl'`

Một số phiên bản TRL đã đổi/loại bỏ export của `DataCollatorForCompletionOnlyLM`. Nếu gặp lỗi import:

- Ưu tiên: cập nhật `scripts/train.py` để có fallback import hoặc custom collator (repo hiện đã có fallback).

### 11.5. `AttributeError: 'Qwen3VLProcessor' object has no attribute 'encode'`

Qwen3.5-9B có thể được load qua Unsloth dưới dạng **VL Processor** (không phải tokenizer thuần text). Khi đó cần dùng `processor.tokenizer`.

Repo hiện đã fix trong `scripts/train.py` bằng cách:

- Nếu object trả về có `tokenizer` attribute → lấy `tokenizer = tokenizer.tokenizer` cho text SFT.

### 11.6. `flash_attn` / Triton kernels warning

Các dòng kiểu:

- `Failed to import Triton kernels...`
- `Flash Attention 2 installation seems to be broken. Using Xformers instead.`

Thường là **warning**; training vẫn chạy, chỉ là chậm hơn một chút.

---

## 12. Chạy train an toàn (không chết khi rớt SSH) với `tmux`

Khuyến nghị luôn chạy train trong `tmux` trên pod:

```bash
tmux new -s train
cd /workspace/llm-infer-serverless/scripts
python train.py
```

Detach (thoát tmux nhưng tiến trình vẫn chạy):

- Nhấn `Ctrl+b`, thả ra, rồi nhấn `d`

Vào lại:

```bash
tmux attach -t train
```

Xem danh sách session:

```bash
tmux ls
```

---

## 11. File liên quan nhanh

- Schema FB: `data/sql/02_schema_facebook.sql`
- Compose: `data/docker-compose.yml`, env: `data/.env`
- Export: `data/export_dataset.py`
- Cleaning: `data/clean_dataset.py`
- Dataset: `dataset/train.jsonl`, `dataset/val.jsonl`
- Backup dataset trước clean: `dataset/train.jsonl.bak`, `dataset/val.jsonl.bak`
- Train deps: `pyproject.toml` (root), data deps: `data/pyproject.toml`
- Scripts train & infra: `scripts/upload_dataset.py`, `scripts/download_model.py`, `scripts/train.py`, `scripts/.env.example`

---

## 12. Tài liệu liên quan

- [qlora-cheat-sheet.md](qlora-cheat-sheet.md) — VRAM, vòng đời train minh họa, bảng dependency (transformers, trl, bitsandbytes, …).
