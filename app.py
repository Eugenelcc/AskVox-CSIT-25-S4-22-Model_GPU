import runpod
from llama_cpp import Llama
import time

llm = Llama(
    model_path="./model.gguf",
    n_ctx=2048,
    n_gpu_layers=30,
    verbose=False,
)

def handler(job):
    input_data = job.get("input", {})
    prompt = input_data.get("prompt", "")
    history = input_data.get("history", [])

    parts = [f"[SYSTEM]\n{SYSTEM_PROMPT.strip()}"]

    for h in history[-4:]:
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            parts.append(f"[USER] {content}")
        elif role == "assistant":
            parts.append(f"[ASSISTANT] {content}")

    parts.append(f"[USER] {prompt}")
    parts.append("[ASSISTANT]")

    full_prompt = "\n".join(parts)



runpod.serverless.start({"handler": handler})
