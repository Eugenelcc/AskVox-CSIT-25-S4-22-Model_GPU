import runpod
from llama_cpp import Llama
import os
import time

MODEL_PATH = "./model.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=30,
    verbose=False,
)

def handler(event):
    """
    Expected input:
    {
      "input": {
        "message": "hello",
        "history": [...]
      }
    }
    """

    data = event.get("input", {})
    message = data.get("message", "")

    prompt = f"[SYSTEM] You are AskVox, a safe educational AI tutor.\n[USER] {message}\n[ASSISTANT]"

    start = time.perf_counter()
    output = llm(prompt, max_tokens=512, temperature=0.7)
    elapsed = time.perf_counter() - start

    text = output["choices"][0]["text"].strip()

    return {
        "response": text,
        "latency_sec": round(elapsed, 2),
    }

runpod.serverless.start({"handler": handler})
