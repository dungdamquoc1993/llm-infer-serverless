FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Cài vLLM nightly (bắt buộc để support Qwen3.5)
RUN pip install --no-cache-dir \
    "vllm @ https://wheels.vllm.ai/nightly/vllm-1.0.0.dev-cp311-cp311-manylinux1_x86_64.whl" \
    runpod \
    huggingface_hub \
    compressed-tensors

COPY handler.py .
COPY download_model.py .

ENV MODEL_PATH="/runpod-volume/qwen35-awq"
ENV MAX_MODEL_LEN="32768"
ENV GPU_MEMORY_UTIL="0.90"

CMD ["python", "-u", "handler.py"]
