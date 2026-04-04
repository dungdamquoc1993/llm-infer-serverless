# Qwen3.5-35B-A3B AWQ 4-bit — RunPod Serverless

## Cấu trúc project

```
qwen3-5_serverless/
├── Dockerfile           # Image cho serverless worker
├── handler.py           # RunPod serverless handler (vLLM)
├── download_model.py    # Script download weights vào Network Volume (chạy 1 lần)
├── requirements.txt
└── README.md
```

---

## Bước 1 — Tạo RunPod Network Volume

1. Vào **RunPod → Storage → + Network Volume**
2. Tên: `qwen35-weights`
3. Dung lượng: **30 GB**
4. Region: chọn cùng region bạn sẽ deploy serverless

---

## Bước 2 — Download weights vào Volume (chạy 1 lần)

### Mount point: Pod khác Serverless (RunPod không hiện chữ “mount point” khi tạo Pod)

Theo [Network volumes — RunPod](https://docs.runpod.io/storage/network-volumes):

| Loại | Volume của bạn xuất hiện ở đâu |
|------|----------------------------------|
| **Pod** | Thường gắn tại **`/workspace`** (volume thay thế ổ mặc định của pod). UI chỉ bắt chọn volume, **không** có ô nhập mount path. |
| **Serverless** | Gắn tại **`/runpod-volume`** |

Cùng một Network Volume: file bạn tải vào `/workspace/qwen35-awq` trên pod sẽ nằm đúng chỗ mà worker đọc qua `/runpod-volume/qwen35-awq` (cùng volume, hai đường mount khác nhau).

1. **Tạo pod tạm** trên RunPod (Secure Cloud, vì Network Volume chỉ gắn được với Pod loại này):
   - Template: `RunPod Pytorch 2.4` hoặc Ubuntu — miễn có Python
   - GPU/CPU: tùy (chỉ cần internet ổn để kéo ~22GB)
   - **Attach Network Volume** `qwen35-weights` — chỉ cần chọn volume trong danh sách, không cần gõ mount path

2. Mở **terminal** trong pod, chạy:
   ```bash
   pip install huggingface_hub
   python download_model.py
   ```
   Chờ ~10-15 phút (22GB). Xong thì **terminate pod** (không cần nữa).

---

## Bước 3 — Build và push Docker image

Chạy trên máy local của bạn (cần Docker + tài khoản Docker Hub):

```bash
docker build -t YOUR_DOCKERHUB_USERNAME/qwen35-serverless:latest .
docker push YOUR_DOCKERHUB_USERNAME/qwen35-serverless:latest
```

Thay `YOUR_DOCKERHUB_USERNAME` bằng username Docker Hub của bạn.

---

## Bước 4 — Tạo Serverless Endpoint trên RunPod

1. Vào **RunPod → Serverless → + New Endpoint**
2. Chọn **Custom deployment → Deploy from Docker registry**
3. Điền thông tin:

| Field | Giá trị |
|---|---|
| Container Image | `YOUR_DOCKERHUB_USERNAME/qwen35-serverless:latest` |
| GPU | RTX 4090 (24GB) hoặc A40 (48GB) |
| Min Workers | `0` |
| Max Workers | `3` (tùy budget) |
| Idle Timeout | `5` phút |
| Network Volume | Chọn `qwen35-weights` → `/runpod-volume` |

4. Thêm **Environment Variables** (nếu cần):
   - `MAX_MODEL_LEN` = `32768`
   - `GPU_MEMORY_UTIL` = `0.90`

5. Click **Deploy** → đợi worker khởi động lần đầu (~2-3 phút)

---

## Bước 5 — Gọi API

### Dùng messages (OpenAI format)

```python
import requests

RUNPOD_API_KEY = "your_api_key_here"
ENDPOINT_ID = "your_endpoint_id_here"

response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "input": {
            "messages": [
                {"role": "system", "content": "Bạn là một trợ lý AI hữu ích."},
                {"role": "user", "content": "Giải thích kiến trúc Transformer"},
            ],
            "max_tokens": 2048,
            "temperature": 1.0,
            "enable_thinking": True,   # False để tắt chế độ thinking
        }
    },
    timeout=300,
)

result = response.json()
print(result["output"]["output"])         # final response
print(result["output"]["thinking"])       # thinking content (nếu enable)
print(result["output"]["usage"])          # token usage
print(result["output"]["tokens_per_second"])
```

### Dùng prompt thẳng

```python
response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
    json={
        "input": {
            "prompt": "Viết một bài thơ về Hà Nội",
            "max_tokens": 512,
            "temperature": 0.7,
            "enable_thinking": False,
        }
    },
)
```

### Response format

```json
{
  "output": {
    "output": "Nội dung trả lời chính...",
    "thinking": "Quá trình suy nghĩ của model...",
    "usage": {
      "prompt_tokens": 45,
      "completion_tokens": 312,
      "total_tokens": 357
    },
    "elapsed_seconds": 8.3,
    "tokens_per_second": 37.6
  }
}
```

---

## Lưu ý

- **Cold start** lần đầu mất ~2-3 phút (load 22GB weights vào VRAM)
- Với `min_workers=0`, khi không có request thì RunPod scale về 0, **không tốn tiền**
- Nếu gặp OOM, giảm `MAX_MODEL_LEN` xuống `16384`
- Model mặc định chạy **thinking mode** — tắt bằng `"enable_thinking": false`
