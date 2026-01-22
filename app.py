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
    if "input" not in job or "prompt" not in job["input"]:
        return {"error": "Missing input.prompt"}

    user_prompt = job["input"]["prompt"]

    full_prompt = (
        "[SYSTEM] You are AskVox, a safe educational AI tutor.\n"
        f"[USER] {user_prompt}\n"
        "[ASSISTANT]"
    )

    output = llm(
        full_prompt,
        max_tokens=512,
        temperature=0.7,
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }


runpod.serverless.start({"handler": handler})
