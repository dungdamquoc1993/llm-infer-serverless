FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Không dùng wheels nightly: bản đó hay link symbol CUDA mới (vd cuMemcpyBatchAsync)
# mà driver trên một số node RunPod chưa có → ImportError vllm._C.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        vllm huggingface_hub compressed-tensors

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY download_model.py .

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV MODEL_PATH="/workspace/qwen35-awq"
ENV MAX_MODEL_LEN="32768"
ENV GPU_MEMORY_UTIL="0.90"
ENV VLLM_PORT="8000"

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
