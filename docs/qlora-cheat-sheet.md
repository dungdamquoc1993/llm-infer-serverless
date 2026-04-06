# Ghi chú: QLoRA / stack train & RunPod

Tài liệu tham khảo nhanh (VRAM, vòng đời một lần train, dependency). Không thay cho hướng dẫn chính thức của TRL/transformers.

---

## Tài nguyên RunPod (ví dụ)

- Container disk ~25 GB (image CUDA, PyTorch, libs)
- Network volume ~40 GB (`/workspace/`: model, dataset, checkpoint LoRA)

---

## Cheat sheet: VRAM đang chứa gì?

```
├── Base model weights (4-bit, frozen)     ~5-6 GB
├── LoRA adapter weights (đang train)      ~vài trăm MB
├── Gradient buffer (cộng dồn qua steps)  ~1-2 GB
├── Optimizer states (Adam)               ~1-2 GB
└── Activation (tính xong step xóa luôn) ~tùy seq_len
```

**Disk:**

```
├── Docker image (CUDA, PyTorch, libs)         → Container disk (~25GB)
└── /workspace/
    ├── base model weights (gốc, không đổi)
    ├── dataset
    └── checkpoint (LoRA adapter lưu định kỳ) → Network Volume (~40GB)
```

---

## Life cycle một lần train (minh họa)

Giả sử: **1000 samples**, `epoch=3`, `batch=2`, `grad_accum=8`, `save_steps=500`.

```
EPOCH 1/3  (mỗi sample được thấy 1 lần)
│
├── Step 1:  feed [s1, s2]    → tính gradient, cộng vào buffer  ──┐
├── Step 2:  feed [s3, s4]    → tính gradient, cộng vào buffer    │ grad_accum=8
├── Step 3:  feed [s5, s6]    → tính gradient, cộng vào buffer    │ chưa update
├── ...                                                            │
├── Step 8:  feed [s15, s16]  → tính gradient, cộng vào buffer  ──┘
│            └─► UPDATE LoRA weight, xóa buffer
│
├── Step 9:  feed [s17, s18]  → cộng dồn tiếp                   ──┐
├── ...                                                            │
├── Step 16: feed [s31, s32]  → cộng vào buffer                 ──┘
│            └─► UPDATE LoRA weight, xóa buffer
│
├── ... (cứ 8 steps update 1 lần)
│
└── Step 500: UPDATE xong → LƯU CHECKPOINT ra disk (/workspace/checkpoint-500)

EPOCH 2/3  (1000 samples đó lại, shuffle lại thứ tự)
│
└── ... (tương tự, weight tiếp tục từ chỗ epoch 1 dừng lại)

EPOCH 3/3
│
└── ... → XONG → lưu checkpoint cuối
```

---

## Finetune stack — dependencies (ý nghĩa)

### Base image (ví dụ RunPod)

**`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`**  
PyTorch + CUDA có sẵn trong image — thường không cần `pip install torch` lại trừ khi đổi version.

### Core

| Dep | Vai trò |
|-----|---------|
| **transformers** | Load model, tokenizer, config. Trung tâm stack HF. |
| **datasets** | Load/stream dataset từ Hub hoặc file local. |
| **peft** | LoRA / QLoRA — adapter nhỏ thay vì full fine-tune. |
| **trl** | `SFTTrainer`, `DPOTrainer`, … — fine-tuning có giám sát / preference. |
| **accelerate** | Multi-GPU, mixed precision — `trl`/`transformers` dùng bên dưới. |

### Quantization & memory

| Dep | Vai trò |
|-----|---------|
| **bitsandbytes** | Quantize 4-bit/8-bit khi load — nền QLoRA (`load_in_4bit=True`). |
| **flash-attn** | Attention tối ưu CUDA — nhanh hơn, ít VRAM khi context dài. |

### Tối ưu thêm (tùy chọn)

| Dep | Vai trò |
|-----|---------|
| **unsloth** | Patch stack (Triton kernel) — thường nhanh hơn, giảm VRAM so với chỉ bitsandbytes. |

### Tooling

| Dep | Vai trò |
|-----|---------|
| **huggingface_hub** | Push/pull model, token, download weights. |
| **wandb** | Log loss, LR, … (checkpoint vẫn do trainer lưu). |
| **sentencepiece** | Tokenizer — nhiều model gọi gián tiếp qua `transformers`. |
| **scipy** | Thường là dependency phụ, ít gọi trực tiếp. |

### Quan hệ tổng thể

```
unsloth (optional)
  └── patch → trl + transformers + LoRA kernel
        trl (SFTTrainer)
          └── dùng → transformers + accelerate + peft
                bitsandbytes  → quantize 4bit khi load
                flash-attn    → attention nhanh hơn
datasets      → feed data vào trainer
wandb         → log metrics ra ngoài
huggingface_hub → upload kết quả lên HF Hub
```

---

## PostgreSQL (dev, trong Docker)

Vào `psql` **trong container** (mật khẩu lấy từ `data/.env` hoặc root `.env`):

```bash
docker exec -it -e PGPASSWORD="$POSTGRES_PASSWORD" el_ripley_postgres \
  psql -U el_ripley_user -d el_ripley
```

Từ máy host, script Python dùng `localhost` và port map (ví dụ **5434**) — xem [finetune-process.md](finetune-process.md) mục PostgreSQL.
