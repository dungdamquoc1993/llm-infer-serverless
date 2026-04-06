"""
Fine-tune Qwen3.5-9B với QLoRA/LoRA bằng Unsloth + TRL SFTTrainer.

Model: Qwen/Qwen3.5-9B  https://huggingface.co/Qwen/Qwen3.5-9B
  - Kiến trúc hybrid: Gated DeltaNet (linear attention) + Gated Attention (1/4 blocks)
  - Thinking mode mặc định: sinh <think>...</think> trước câu trả lời
    → PHẢI tắt khi SFT shop chat (data không có thinking content)
  - VRAM: 4-bit ~6.5 GB + optimizer/activation → tổng ~13–16 GB (OK trên 20 GB)

Stack:
  - Unsloth FastLanguageModel  → load model + patch kernels (2x speedup)
  - bitsandbytes NF4           → quantize base 4-bit on-the-fly (QLoRA)
  - PEFT LoRA                  → chỉ train ~1% tham số
  - TRL SFTTrainer             → training loop, loss chỉ trên assistant turns
  - DataCollatorForCompletionOnlyLM → mask user/system khỏi loss

Chạy trên pod:
  python scripts/train.py

Cấu hình qua .env (xem scripts/.env.example).
"""

import os
import sys
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    print("[INFO] Loaded .env từ root repo")
except ImportError:
    print("[WARN] python-dotenv không có, dùng env vars hiện tại")

# ── Helper: đọc env ────────────────────────────────────────────────────────────
def _env(key: str, default=None, cast=str, required=False):
    val = os.environ.get(key, default)
    if required and val is None:
        print(f"[ERROR] Biến môi trường bắt buộc '{key}' chưa được set.")
        sys.exit(1)
    return cast(val) if val is not None else None

# ── Config từ môi trường ───────────────────────────────────────────────────────
HF_TOKEN         = _env("HF_TOKEN")

# Model: ưu tiên path local (sau khi chạy download_model.py), fallback HF Hub
MODEL_PATH       = _env("MODEL_PATH")       or _env("MODEL_REPO", "Qwen/Qwen3.5-9B")

# Dataset: HF Hub repo ID hoặc đường dẫn local JSONL
HF_DATASET_REPO  = _env("HF_DATASET_REPO")
LOCAL_TRAIN_FILE = str(REPO_ROOT / "dataset" / "train.jsonl")
LOCAL_VAL_FILE   = str(REPO_ROOT / "dataset" / "val.jsonl")

OUTPUT_DIR       = _env("OUTPUT_DIR", "/workspace/lora_output")
ADAPTER_REPO     = _env("ADAPTER_REPO")      # optional: push adapter lên HF Hub
MERGE_MODEL      = _env("MERGE_MODEL", "true").lower() == "true"
MERGED_REPO      = _env("MERGED_REPO")       # optional: push merged full model lên HF Hub

USE_QLORA        = _env("USE_QLORA",   "true").lower() == "true"
LORA_R           = _env("LORA_R",       16,    cast=int)
LORA_ALPHA       = _env("LORA_ALPHA",   32,    cast=int)
LORA_DROPOUT     = _env("LORA_DROPOUT", 0.05,  cast=float)
# Qwen3.5-9B: kiến trúc hybrid (Gated DeltaNet + Gated Attention)
# Dùng "all-linear" để Unsloth tự detect đúng tên layers cho kiến trúc mới
# Hoặc chỉ định thủ công nếu muốn giới hạn: q_proj,k_proj,v_proj,o_proj,...
_default_targets  = _env("TARGET_MODULES", "all-linear")
TARGET_MODULES    = (
    list(_default_targets.split(","))
    if _default_targets != "all-linear"
    else "all-linear"
)

NUM_EPOCHS       = _env("NUM_EPOCHS",   3,     cast=int)
BATCH_SIZE       = _env("BATCH_SIZE",   2,     cast=int)
GRAD_ACCUM       = _env("GRAD_ACCUM",   8,     cast=int)
LR               = _env("LR",           2e-4,  cast=float)
MAX_SEQ_LEN      = _env("MAX_SEQ_LEN",  2048,  cast=int)
WARMUP_RATIO     = _env("WARMUP_RATIO", 0.03,  cast=float)
USE_FLASH_ATTN   = _env("USE_FLASH_ATTN", "true").lower() == "true"
# Qwen3.5-9B có thinking mode mặc định (sinh <think>...</think> trước trả lời)
# Với SFT shop chat, data không có thinking content → phải tắt để format khớp
ENABLE_THINKING  = _env("ENABLE_THINKING", "false").lower() == "true"

WANDB_PROJECT    = _env("WANDB_PROJECT")
WANDB_API_KEY    = _env("WANDB_API_KEY")

# ── Tóm tắt config ────────────────────────────────────────────────────────────
print("=" * 65)
print("CONFIG")
print("=" * 65)
print(f"  Model         : {MODEL_PATH}")
print(f"  Dataset       : {HF_DATASET_REPO or 'local JSONL'}")
print(f"  Output        : {OUTPUT_DIR}")
print(f"  Adapter push  : {ADAPTER_REPO or '(không push)'}")
print(f"  Merge model   : {MERGE_MODEL}  →  {MERGED_REPO or '(chỉ lưu local)'}")
print(f"  QLoRA         : {USE_QLORA}  |  r={LORA_R}  alpha={LORA_ALPHA}")
print(f"  Epochs        : {NUM_EPOCHS}  |  batch={BATCH_SIZE}  accum={GRAD_ACCUM}")
print(f"  LR            : {LR}  |  max_seq={MAX_SEQ_LEN}  warmup={WARMUP_RATIO}")
print(f"  Flash-Attn    : {USE_FLASH_ATTN}")
print(f"  Thinking mode : {ENABLE_THINKING}  (False = tắt <think> → SFT style)")
print(f"  W&B project   : {WANDB_PROJECT or '(disabled)'}")
print("=" * 65)
print()

# ── Setup W&B ─────────────────────────────────────────────────────────────────
if WANDB_PROJECT and WANDB_API_KEY:
    import wandb
    wandb.login(key=WANDB_API_KEY)
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    print(f"[INFO] W&B: project={WANDB_PROJECT}")
elif WANDB_PROJECT:
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    print(f"[INFO] W&B: project={WANDB_PROJECT} (dùng key đã login sẵn)")
else:
    print("[INFO] W&B disabled")

# ── Import heavy libs ─────────────────────────────────────────────────────────
print("\n[INFO] Loading Unsloth + TRL...")
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# DataCollatorForCompletionOnlyLM bị remove khỏi TRL >= 0.13 top-level export
# Thử các path theo thứ tự, fallback về custom implementation
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    try:
        from trl.trainer import DataCollatorForCompletionOnlyLM
    except ImportError:
        try:
            from trl.trainer.utils import DataCollatorForCompletionOnlyLM
        except ImportError:
            from dataclasses import dataclass as _dc
            from typing import Any as _Any, List as _List, Optional as _Opt

            @_dc
            class DataCollatorForCompletionOnlyLM:
                """
                Custom fallback: mask mọi token trừ phần assistant response.
                Tìm từng <|im_start|>assistant\\n, unmask đến <|im_end|> tiếp theo.
                """
                response_template: _List[int]
                tokenizer: _Any
                instruction_template: _Opt[_List[int]] = None
                ignore_index: int = -100

                def __post_init__(self):
                    _im_end = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
                    self._im_end_id = _im_end[-1] if _im_end else None

                def __call__(self, features):
                    import torch
                    batch = self.tokenizer.pad(features, return_tensors="pt", padding=True)
                    labels = batch["input_ids"].clone()
                    tlen = len(self.response_template)

                    for i in range(labels.shape[0]):
                        seq = labels[i].tolist()
                        labels[i] = self.ignore_index  # default: mask toàn bộ

                        j = 0
                        while j <= len(seq) - tlen:
                            if seq[j : j + tlen] == self.response_template:
                                start = j + tlen  # bắt đầu content assistant
                                end = start
                                # Unmask đến <|im_end|> (bao gồm cả token đó)
                                while end < len(seq):
                                    if self._im_end_id is not None and seq[end] == self._im_end_id:
                                        end += 1
                                        break
                                    end += 1
                                labels[i, start:end] = batch["input_ids"][i, start:end]
                                j = end
                            else:
                                j += 1

                    batch["labels"] = labels
                    return batch

            print("[INFO] DataCollatorForCompletionOnlyLM: dùng custom fallback implementation")

from transformers import TrainerCallback

print(f"[INFO] torch={torch.__version__}  CUDA={torch.version.cuda}  "
      f"GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# ── VRAM helpers ───────────────────────────────────────────────────────────────
def _gb(b: int) -> float:
    return b / 1024 ** 3

def _vram(label: str = "") -> None:
    """In VRAM hiện tại và peak kể từ lần reset gần nhất."""
    if not torch.cuda.is_available():
        return
    alloc   = _gb(torch.cuda.memory_allocated())
    reserv  = _gb(torch.cuda.memory_reserved())
    peak_a  = _gb(torch.cuda.max_memory_allocated())
    peak_r  = _gb(torch.cuda.max_memory_reserved())
    tag = f"[VRAM] {label:30s}" if label else "[VRAM]"
    print(f"{tag}  alloc={alloc:.2f}GB  reserved={reserv:.2f}GB  "
          f"peak_alloc={peak_a:.2f}GB  peak_reserved={peak_r:.2f}GB")

# Reset counter ngay từ đầu để đo peak từ điểm xuất phát sạch
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# ── Load model + tokenizer ────────────────────────────────────────────────────
print(f"\n[INFO] Loading model: {MODEL_PATH} ...")
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,               # auto: bf16 trên Ampere+, fp16 trên Turing
    load_in_4bit=USE_QLORA,   # NF4 quantization qua bitsandbytes
    token=HF_TOKEN,
    attn_implementation="flash_attention_2" if USE_FLASH_ATTN else "eager",
)
_vram("sau load model")

# Qwen3.5-9B là VL model — Unsloth trả về Processor thay vì Tokenizer thuần.
# Với text-only SFT, cần extract inner tokenizer (có đủ encode/pad/apply_chat_template).
if hasattr(tokenizer, "tokenizer"):
    tokenizer = tokenizer.tokenizer
    print("[INFO] VL Processor detected → dùng processor.tokenizer cho text SFT")

# ── Thêm LoRA adapter ─────────────────────────────────────────────────────────
print("[INFO] Gắn LoRA adapter...")
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"[INFO] Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
_vram("sau gắn LoRA")

# ── Load dataset ──────────────────────────────────────────────────────────────
print("\n[INFO] Loading dataset...")

if HF_DATASET_REPO:
    ds = load_dataset(HF_DATASET_REPO, token=HF_TOKEN)
    train_ds = ds["train"]
    val_ds   = ds.get("validation") or ds.get("val") or ds.get("test")
    if val_ds is None:
        raise ValueError(
            f"Repo '{HF_DATASET_REPO}' không có split 'validation'/'val'/'test'. "
            "Kiểm tra tên split trên Hub."
        )
else:
    # Local JSONL
    for f in [LOCAL_TRAIN_FILE, LOCAL_VAL_FILE]:
        if not Path(f).exists():
            print(f"[ERROR] Không tìm thấy {f}. Set HF_DATASET_REPO hoặc đặt file tại dataset/")
            sys.exit(1)
    ds = load_dataset("json", data_files={
        "train":      LOCAL_TRAIN_FILE,
        "validation": LOCAL_VAL_FILE,
    })
    train_ds = ds["train"]
    val_ds   = ds["validation"]

print(f"[INFO] Train: {len(train_ds):,} samples  |  Val: {len(val_ds):,} samples")

# ── Kiểm tra format dataset ───────────────────────────────────────────────────
sample = train_ds[0]
if "messages" not in sample:
    print(f"[ERROR] Dataset cần có key 'messages'. Keys hiện tại: {list(sample.keys())}")
    sys.exit(1)

# ── Tokenize với chat template ────────────────────────────────────────────────
# Qwen3.5-9B dùng ChatML: <|im_start|>role\ncontent<|im_end|>
# enable_thinking=False: tắt <think>...</think> trong output
# → bắt buộc với SFT vì data shop chat không có thinking content

def apply_template(examples):
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
            chat_template_kwargs={"enable_thinking": ENABLE_THINKING},
        )
        texts.append(text)
    return {"text": texts}

print(f"[INFO] Áp dụng chat template (enable_thinking={ENABLE_THINKING})...")
train_ds = train_ds.map(apply_template, batched=True, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(apply_template,   batched=True, remove_columns=val_ds.column_names)

# ── DataCollator: chỉ tính loss trên assistant turns ─────────────────────────
# Response template trong ChatML của Qwen: <|im_start|>assistant\n
# DataCollatorForCompletionOnlyLM sẽ mask tất cả trước token này
response_template_ids = tokenizer.encode(
    "<|im_start|>assistant\n",
    add_special_tokens=False,
)
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer,
)

# ── Training args ─────────────────────────────────────────────────────────────
bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=max(1, BATCH_SIZE // 2),
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    bf16=bf16_ok,
    fp16=not bf16_ok,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb" if WANDB_PROJECT else "none",
    run_name="qwen3-qlora-shop" if WANDB_PROJECT else None,
    optim="adamw_8bit",        # bitsandbytes 8-bit Adam → ít VRAM hơn
    seed=42,
    dataloader_num_workers=2,
    remove_unused_columns=False,
)

print(f"[INFO] bf16={bf16_ok}  optim=adamw_8bit  "
      f"effective_batch={BATCH_SIZE * GRAD_ACCUM}")

# ── VRAM callback ─────────────────────────────────────────────────────────────
class VramCallback(TrainerCallback):
    """Log peak VRAM sau mỗi logging step và sau mỗi evaluation."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            peak = _gb(torch.cuda.max_memory_allocated())
            reserved = _gb(torch.cuda.memory_reserved())
            step = state.global_step
            print(f"[VRAM] step={step:>6d}  peak_alloc={peak:.2f}GB  reserved={reserved:.2f}GB")

    def on_evaluate(self, args, state, control, **kwargs):
        _vram(f"eval step={state.global_step}")

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    args=training_args,
    callbacks=[VramCallback()],
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n[INFO] Bắt đầu training...")
print(f"       Output: {OUTPUT_DIR}")
print()

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()   # reset để đo peak riêng cho training loop
_vram("trước khi train")

trainer_output = trainer.train()

print("\n[INFO] Training xong.")
print(f"       Train loss cuối : {trainer_output.training_loss:.4f}")
_vram("sau khi train (peak toàn bộ vòng train)")

# ── Lưu LoRA adapter (~200-500 MB) ───────────────────────────────────────────
adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
print(f"\n[INFO] Lưu LoRA adapter → {adapter_dir}")
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

# ── Push adapter lên HF Hub ──────────────────────────────────────────────────
if ADAPTER_REPO and HF_TOKEN:
    print(f"\n[INFO] Push adapter lên HF Hub: {ADAPTER_REPO}")
    model.push_to_hub(ADAPTER_REPO, token=HF_TOKEN, private=True)
    tokenizer.push_to_hub(ADAPTER_REPO, token=HF_TOKEN, private=True)
    print(f"[DONE] Adapter: https://huggingface.co/{ADAPTER_REPO}")
elif ADAPTER_REPO:
    print("[WARN] ADAPTER_REPO được set nhưng HF_TOKEN thiếu → bỏ qua push.")

# ── Merge adapter vào base model → full fine-tuned model (BF16) ─────────────
# Unsloth merge từng layer → không cần thêm VRAM so với lúc train
# Output: model BF16 đầy đủ (~18 GB), sẵn sàng cho inference hoặc quantize tiếp
if MERGE_MODEL:
    merged_dir = os.path.join(OUTPUT_DIR, "merged_model")
    print(f"\n[INFO] Merge LoRA adapter vào base model (16-bit) → {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"[DONE] Merged model: {merged_dir}")

    if MERGED_REPO and HF_TOKEN:
        print(f"\n[INFO] Push merged model lên HF Hub: {MERGED_REPO}")
        model.push_to_hub_merged(
            MERGED_REPO, tokenizer,
            save_method="merged_16bit",
            token=HF_TOKEN, private=True,
        )
        print(f"[DONE] Merged model: https://huggingface.co/{MERGED_REPO}")
    elif MERGED_REPO:
        print("[WARN] MERGED_REPO được set nhưng HF_TOKEN thiếu → bỏ qua push.")

print("\n[DONE] Hoàn thành! Kiểm tra:")
print(f"  Adapter    : {adapter_dir}")
if MERGE_MODEL:
    print(f"  Merged     : {os.path.join(OUTPUT_DIR, 'merged_model')}")
print(f"  Checkpoints: {OUTPUT_DIR}")
