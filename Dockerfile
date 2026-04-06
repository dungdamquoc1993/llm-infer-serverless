FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Các package được import trực tiếp trong train.py
# peft, bitsandbytes, accelerate, huggingface_hub, sentencepiece
# → KHÔNG cài ở đây, để unsloth tự quản lý version tương thích
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        transformers>=4.51.0 \
        datasets>=3.0.0 \
        trl>=0.17.0 \
        wandb \
        python-dotenv

# Unsloth: kéo theo peft, bitsandbytes, accelerate, flash-attn, sentencepiece
# variant cu124-torch240 khớp với base image (CUDA 12.4, PyTorch 2.4.0)
RUN pip install --no-cache-dir \
    "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git@March-2026"

CMD ["sleep", "infinity"]
