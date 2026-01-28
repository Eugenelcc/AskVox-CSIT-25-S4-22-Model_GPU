import runpod
from llama_cpp import Llama

llm = Llama(
    model_path="./model.gguf",
    n_ctx=2048,
    n_gpu_layers=30,
    verbose=False,
)

def handler(job):
    prompt = job.get("input", {}).get("prompt")
    if not prompt:
        return {"error": "Missing input.prompt"}

    full_prompt = (
        "You are AskVox, a friendly, knowledgeable AI assistant. "
        "Respond clearly and helpfully.\n\n"
        f"{prompt}\n"
    )

    output = llm(
        full_prompt,
        max_tokens=800,
        temperature=0.45,
        top_p=0.9,
        repeat_penalty=1.15,
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }

runpod.serverless.start({"handler": handler})
