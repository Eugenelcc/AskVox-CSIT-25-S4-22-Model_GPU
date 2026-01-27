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

    prompt = job["input"]["prompt"]

    output = llm(
        full_prompt,
        max_tokens=1200,
        temperature=0.7,
        stop=["[ASSISTANT]", "[USER]", "[SYSTEM]"]
    )


    return {
        "response": output["choices"][0]["text"].strip()
    }



runpod.serverless.start({"handler": handler})
