import runpod
from llama_cpp import Llama

llm = Llama(
    model_path="./Llama-3.2-3B.Q6_K.gguf",
    n_ctx=2048,
    n_gpu_layers=0,   # set >0 ONLY if CUDA is confirmed
    n_threads=8,
    verbose=False,
)

def handler(job):
    inp = job.get("input", {})
    prompt = inp.get("prompt") or inp.get("message")

    if not prompt:
        return {"error": "Missing input.prompt"}

    output = llm(
        prompt,
        max_tokens=1200,
        temperature=0.6,
        top_p=0.9,
        repeat_penalty=1.15,
        stop=["\nUser:"],
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }

runpod.serverless.start({"handler": handler})
