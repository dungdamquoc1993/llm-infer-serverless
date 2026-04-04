"""
RunPod Serverless Handler cho Qwen3.5-35B-A3B AWQ 4-bit
API tương thích OpenAI Chat Completions format
"""

import os
import time
import runpod
from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/qwen35-awq")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTIL", "0.90"))

print(f"[Startup] Loading model từ: {MODEL_PATH}")
print(f"[Startup] Max context length: {MAX_MODEL_LEN}")

llm = LLM(
    model=MODEL_PATH,
    quantization="compressed-tensors",  # AWQ repo dùng compressed-tensors format
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    trust_remote_code=True,
    dtype="auto",
)

print("[Startup] Model loaded, sẵn sàng nhận request.")


def build_messages_prompt(messages: list) -> str:
    """Fallback nếu không dùng chat template từ tokenizer."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if c.get("type") == "text"
            )
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def handler(job: dict) -> dict:
    job_input = job.get("input", {})

    # --- Lấy tham số đầu vào ---
    messages = job_input.get("messages")
    prompt = job_input.get("prompt")

    if not messages and not prompt:
        return {"error": "Cần truyền 'messages' (list) hoặc 'prompt' (string)"}

    # Chuyển messages thành prompt nếu cần
    if messages:
        tokenizer = llm.get_tokenizer()
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = build_messages_prompt(messages)

    # --- Sampling params ---
    enable_thinking = job_input.get("enable_thinking", True)
    temperature = job_input.get("temperature", 1.0 if enable_thinking else 0.7)
    top_p = job_input.get("top_p", 0.95 if enable_thinking else 0.8)
    top_k = job_input.get("top_k", 20)
    max_tokens = job_input.get("max_tokens", 4096)
    presence_penalty = job_input.get("presence_penalty", 1.5)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
    )

    # --- Inference ---
    t0 = time.time()
    outputs = llm.generate([prompt], sampling_params)
    elapsed = time.time() - t0

    generated_text = outputs[0].outputs[0].text
    input_tokens = len(outputs[0].prompt_token_ids)
    output_tokens = len(outputs[0].outputs[0].token_ids)

    # Tách thinking content ra khỏi final response
    thinking_content = None
    final_response = generated_text

    if "<think>" in generated_text and "</think>" in generated_text:
        think_start = generated_text.find("<think>")
        think_end = generated_text.find("</think>") + len("</think>")
        thinking_content = generated_text[think_start + len("<think>"):think_end - len("</think>")].strip()
        final_response = generated_text[think_end:].strip()

    return {
        "output": final_response,
        "thinking": thinking_content,
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "elapsed_seconds": round(elapsed, 2),
        "tokens_per_second": round(output_tokens / elapsed, 1) if elapsed > 0 else 0,
    }


runpod.serverless.start({"handler": handler})
