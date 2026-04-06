# Base image: unsloth official — đã bao gồm PyTorch, CUDA, unsloth,
# transformers, TRL, PEFT, bitsandbytes, flash-attn, datasets, sentencepiece
FROM unsloth/unsloth:latest

# Chỉ cài thêm 2 package chưa có trong base image
RUN pip install --no-cache-dir wandb python-dotenv

CMD ["sleep", "infinity"]
