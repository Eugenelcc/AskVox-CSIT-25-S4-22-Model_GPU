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

    stop = inp.get("stop") or ["[ASSISTANT]", "[USER]", "[SYSTEM]"]

    output = llm(
        prompt,                    # ✅ use the actual prompt
        max_tokens=1200,
        temperature=0.7,
        stop=stop,                 # ✅ STOP TOKENS (this fixes looping)
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }

runpod.serverless.start({"handler": handler})
