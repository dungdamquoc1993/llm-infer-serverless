# Qwen3.5-35B-A3B AWQ 4-bit — RunPod **GPU Pod** + vLLM

Một **container** chạy `vllm serve` (API tương thích OpenAI). Weights nằm trên **Network Volume**; image không đóng gói model.

## Cấu trúc

```
├── Dockerfile              # Image: CUDA + vLLM, entrypoint = vllm serve
├── docker-entrypoint.sh    # Tham số từ biến môi trường
├── download_model.py       # Tải weights lên volume (chạy 1 lần)
├── scripts/chat_runpod.py  # Gọi API từ máy bạn (OpenAI SDK + .env)
├── requirements.txt
└── .github/workflows/      # Build & push GHCR (tuỳ chọn)
```

---

## 1. Network Volume

RunPod → Storage → volume **cùng region** với GPU Pod, size **≥ 30 GB**.

---

## 2. Tải weights (một lần)

Trên pod (CPU rẻ hoặc GPU) có volume gắn tại **`/workspace`**:

```bash
pip install huggingface_hub
cd /workspace
# copy download_model.py hoặc clone repo
python download_model.py
```

Ghi đè thư mục: `WEIGHTS_DIR=/đường/khác python download_model.py`

Model private: `HF_TOKEN=... python download_model.py`

---

## 3. GPU Pod — dùng image có sẵn hoặc build repo

- GPU **≥ 24 GB** VRAM, attach **cùng volume** → weights tại **`/workspace/qwen35-awq`** (mặc định).

### Biến môi trường (container)

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `MODEL_PATH` | `/workspace/qwen35-awq` | Thư mục model trên volume |
| `MAX_MODEL_LEN` | `200000` | Giới hạn token (prompt + sinh); **GPU 24GB** thường phải **hạ** (vd `32768`–`65536`) tránh OOM khi khởi động hoặc context dài |
| `GPU_MEMORY_UTIL` | `0.90` | Tỷ lệ VRAM cho vLLM |
| `VLLM_PORT` | `8000` | Cổng HTTP |
| `VLLM_API_KEY` | (trống) | Nếu set → vLLM bật `--api-key` |

### Image từ GHCR

Deploy pod với image `ghcr.io/<owner>/<repo>:latest`, **không** cần start command tùy chỉnh (entrypoint đã gọi `vllm serve`).

Expose **HTTP port 8000** (hoặc đúng `VLLM_PORT`).

### Build local / tự push

```bash
docker build -t qwen35-vllm .
docker run --gpus all -e MODEL_PATH=/path/to/weights -p 8000:8000 qwen35-vllm
```

---

## 4. Gọi API (OpenAI client)

`model` trong body thường là **đường dẫn local trên server** (giống doc vLLM), ví dụ:

```bash
curl -s "http://YOUR_POD:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/qwen35-awq",
    "messages": [{"role": "user", "content": "2+2=?"}],
    "max_tokens": 128
  }'
```

Python: `openai` SDK với `base_url="http://YOUR_POD:8000/v1"`. Nếu bật `VLLM_API_KEY`, set `api_key` tương ứng.

### Script có sẵn (biến môi trường)

1. Copy `.env.example` → `.env`, điền **`VLLM_BASE_URL`** = URL proxy RunPod (có cổng trong hostname, **không** cần thêm `/v1`).
2. Tuỳ chọn: `VLLM_MODEL`, `VLLM_API_KEY` (nếu pod có bật key), `VLLM_PROMPT`, `VLLM_MAX_TOKENS`.
3. Cài dependency: `pip install -r requirements.txt` hoặc `poetry install`.
4. Chạy:

```bash
python scripts/chat_runpod.py
python scripts/chat_runpod.py "Viết một câu chào bằng tiếng Việt"
```

---

## OOM / pod không lên được sau khi tăng context

`MAX_MODEL_LEN=200000` cấp cho vLLM “trần” rất cao; **KV cache** tính theo đó nên **GPU 24GB** (A5000, 4090, …) thường **không đủ** — hạ trên RunPod → Environment, ví dụ `MAX_MODEL_LEN=32768` hoặc `65536`, rồi restart pod. GPU VRAM lớn hơn mới có thể giữ gần 200k.

## Lỗi `undefined symbol: cuMemcpyBatchAsync` (vLLM `_C.abi3.so`)

Wheel **vLLM nightly** thường yêu cầu **driver NVIDIA mới**; node RunPod cũ hơn sẽ lỗi khi `import vllm`. Image repo này cài **vLLM stable từ PyPI** (không nightly). Nếu vẫn lỗi: thử **region/GPU type khác** trên RunPod hoặc hạ `vllm` xuống bản stable cũ hơn trong `Dockerfile`.

---

## GHCR

Workflow `.github/workflows/docker-ghcr.yml` build khi đổi `Dockerfile`, `docker-entrypoint.sh`, `download_model.py`.
