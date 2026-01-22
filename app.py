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
    try:
        prompt = job["input"]["prompt"]
    except KeyError:
        return {"error": "Missing input.prompt"}

    full_prompt = f"""[SYSTEM] You are AskVox, a safe educational AI tutor.
[USER] {prompt}
[ASSISTANT]
"""

    start = time.perf_counter()
    output = llm(full_prompt, max_tokens=512, temperature=0.7)
    end = time.perf_counter()

    return {
        "response": output["choices"][0]["text"].strip(),
        "latency_sec": round(end - start, 2)
    }

runpod.serverless.start({"handler": handler})
