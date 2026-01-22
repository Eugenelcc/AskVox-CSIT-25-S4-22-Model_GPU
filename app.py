import runpod
from llama_cpp import Llama

# 🔒 Define SYSTEM_PROMPT FIRST
SYSTEM_PROMPT = """
You are AskVox.

AskVox is a friendly, reliable educational AI tutor.
You MUST always identify yourself as AskVox when asked who you are.
Never say you are "just an AI" or "a language model".
Your tone is warm, clear, and helpful.
"""

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

    if not prompt:
        return {"error": "Missing input.prompt"}

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

    output = llm(
        full_prompt,
        max_tokens=512,
        temperature=0.7,
    )

    text = output["choices"][0]["text"].strip()

    return {
        "response": {
            "answer_markdown": text,
            "need_web_sources": False,
            "need_images": False,
            "need_youtube": False,
            "web_query": "",
            "image_query": "",
            "youtube_query": ""
        }
    }

runpod.serverless.start({"handler": handler})
