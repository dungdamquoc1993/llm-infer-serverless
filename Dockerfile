# Base image: unsloth official — đã bao gồm PyTorch, CUDA, unsloth,
# TRL, PEFT, bitsandbytes, flash-attn, datasets, sentencepiece
FROM unsloth/unsloth:latest

# Pin transformers >= 5.2.0 (required for Qwen3.5 architecture)
# unsloth:latest có thể ship transformers cũ hơn → upgrade bắt buộc
# Lưu ý: vllm (nếu có trong image) thường pin transformers<5 → sẽ có
# dependency warning nhưng không ảnh hưởng training
RUN pip install --no-cache-dir \
    "transformers>=5.2.0" \
    wandb \
    python-dotenv

CMD ["sleep", "infinity"]
