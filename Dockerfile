FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# vLLM: không dùng URL wheel cố định (nightly đổi tên → 404). Lấy bản mới nhất từ index.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        vllm runpod huggingface_hub compressed-tensors \
        --pre \
        --extra-index-url https://wheels.vllm.ai/nightly

COPY handler.py .
COPY download_model.py .

ENV MODEL_PATH="/runpod-volume/qwen35-awq"
ENV MAX_MODEL_LEN="32768"
ENV GPU_MEMORY_UTIL="0.90"

CMD ["python", "-u", "handler.py"]
