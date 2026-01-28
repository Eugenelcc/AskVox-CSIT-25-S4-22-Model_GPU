import runpod
from llama_cpp import Llama

# Load model once at cold start
llm = Llama(
    model_path="./model.gguf",   # Llama-3.2-3B base GGUF
    n_ctx=4096,
    n_gpu_layers=20,             # adjust for VRAM
    n_threads=8,
    verbose=False,
)

def handler(job):
    inp = job.get("input", {})
    user_prompt = inp.get("prompt")

    if not user_prompt:
        return {"error": "Missing input.prompt"}

    # BASE model → plain text prompt
    prompt = (
        "You are AskVox, a helpful AI assistant.\n\n"
        f"User: {user_prompt}\n"
        "Assistant:"
    )

    output = llm(
        prompt,
        max_tokens=1200,
        temperature=0.6,
        top_p=0.9,
        stop=["User:", "Assistant:"],
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }

runpod.serverless.start({
    "handler": handler
})
