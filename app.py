import runpod
from llama_cpp import Llama

llm = Llama(
    model_path="./model.gguf",
    n_ctx=8192,
    n_gpu_layers=-1,   # IMPORTANT for 70B
    n_threads=8,
    verbose=False,
)

SYSTEM_PROMPT = (
    "You are AskVox, a friendly and helpful AI assistant. "
    "Answer clearly, accurately, and conversationally."
)

def handler(job):
    inp = job.get("input", {})
    user_prompt = inp.get("prompt")

    if not user_prompt:
        return {"error": "Missing input.prompt"}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    prompt = "<|begin_of_text|>"

    for msg in messages:
        prompt += (
            f"<|start_header_id|>{msg['role']}<|end_header_id|>\n"
            f"{msg['content'].strip()}<|eot_id|>"
        )

    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

    output = llm(
        prompt,
        max_tokens=1024,      # safer for RunPod sync
        temperature=0.7,
        top_p=0.95,
        stop=["<|eot_id|>"],
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }

runpod.serverless.start({"handler": handler})
