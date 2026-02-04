import runpod
from llama_cpp import Llama

# -----------------------
# Model initialization (cold start)
# -----------------------
llm = Llama(
    model_path="./model.gguf",     # Llama-3.3-70B-Instruct Q4 GGUF
    n_ctx=8192,                    # more stable than 4k
    n_gpu_layers=80,               # FULL GPU offload = speed
    n_threads=16,                  # use available vCPUs
    verbose=False,
)

# -----------------------
# System prompt
# -----------------------
SYSTEM_PROMPT = (
    "You are AskVox, a friendly and helpful AI assistant. "
    "Explain topics in a natural, human, tutor-like way. "
    "Prefer clear paragraph-style explanations with context, reasoning, and examples. "
    "Use bullet points or numbered lists only when they genuinely improve clarity "
    "(such as rankings, comparisons, or step-by-step instructions). "
    "When using bullet points, include a short explanation for each item rather than listing names only."
)

# -----------------------
# RunPod handler
# -----------------------
def handler(job):
    inp = job.get("input", {})
    user_prompt = inp.get("prompt")

    if not user_prompt:
        return {"error": "Missing input.prompt"}

    if not isinstance(user_prompt, str):
        return {"error": "input.prompt must be a string"}

    # Build Llama-3.3 Instruct prompt (same behavior as your slow version)
    prompt = "<|begin_of_text|>"

    prompt += (
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
    )

    prompt += (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt.strip()}<|eot_id|>"
    )

    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

    # Generate response
    output = llm(
        prompt,
        max_tokens=1024,                     # serverless-safe
        temperature=0.7,                     # same style as before
        top_p=0.95,
        stop=["<|eot_id|>", "<|start_header_id|>"],
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }

# -----------------------
# Start RunPod serverless
# -----------------------
runpod.serverless.start({"handler": handler})
