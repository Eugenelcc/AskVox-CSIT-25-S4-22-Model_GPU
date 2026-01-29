import runpod
from llama_cpp import Llama

# Load model ONCE per worker
llm = Llama(
    model_path="./model.gguf",   # your LLaMA 3.2 base GGUF
    n_ctx=4096,
    n_gpu_layers=20,           
    n_threads=8,
    verbose=False,
)

def handler(job):
    inp = job.get("input", {})
    user_prompt = inp.get("prompt")

    if not user_prompt:
        return {"error": "Missing input.prompt"}

    # Base models want RAW TEXT, not chat templates
    prompt = user_prompt.strip()

    output = llm(
        prompt,
        max_tokens=800,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,        
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }

runpod.serverless.start({
    "handler": handler
})
