import runpod
from llama_cpp import Llama

llm = Llama(
    model_path="./model.gguf",
    n_ctx=4096,
    n_gpu_layers=20,
    n_threads=8,
    verbose=False,
)

SYSTEM_PROMPT = (
    "You are AskVox, a friendly and helpful AI assistant. "
    "Answer clearly and conversationally."
)

def handler(job):
    inp = job.get("input", {})
    user_prompt = inp.get("prompt")
    if not user_prompt:
        return {"error": "Missing input.prompt"}

    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    output = llm(
        prompt,
        max_tokens=1200,
        temperature=0.7,
        stop=["<|eot_id|>"],
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }

runpod.serverless.start({"handler": handler})
