import runpod
from llama_cpp import Llama

llm = Llama(
    model_path="./model.gguf",
    n_ctx=2048,
    n_gpu_layers=30,
    verbose=False,
)

def handler(job):
    inp = job.get("input", {})
    prompt = inp.get("prompt")

    if not prompt:
        return {"error": "Missing input.prompt"}

    output = llm(
        prompt,
        max_tokens=800,
        temperature=0.45,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["\nUser:", "\nAssistant:"],
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }



runpod.serverless.start({"handler": handler})
