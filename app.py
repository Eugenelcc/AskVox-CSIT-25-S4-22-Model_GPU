import os
from llama_cpp import Llama
import runpod

# Updated paths for models and LoRA adapters
BASE_GGUF = "/app/model.gguf"

LORA_GGUF = {
    "cooking":   "/app/Cooking_LoRAadapter.gguf",
    "history":   "/app/History_LoRAadapter.gguf",
    "geography": "/app/Geography_LoRAadapter.gguf",
}

SYSTEM_PROMPT = (
    "You are AskVox, a friendly and helpful AI assistant. "
    "Explain topics in a natural, human, tutor-like way. "
    "Prefer clear paragraph-style explanations with context, reasoning, and examples. "
    "Use bullet points or numbered lists only when they genuinely improve clarity. "
    "When using bullet points, include a short explanation for each item."
)

# Cache models so you only pay load cost once per domain
_MODEL_CACHE = {}

def get_llm(domain: str):
    domain = (domain or "").lower().strip()
    cache_key = domain if domain in LORA_GGUF else "base"

    # Print which model or adapter is being used
    if cache_key == "base":
        print("No specific adapter found, using the base model.")
    else:
        print(f"Loading LoRA adapter for domain: {domain}")

    if cache_key in _MODEL_CACHE:
        print(f"Returning cached model for {cache_key}.")
        return _MODEL_CACHE[cache_key]

    # Tune these for your machine
    common_kwargs = dict(
        n_ctx=8192,
        n_threads=16,
        n_gpu_layers=80,  # start safe; increase slowly on A40
        verbose=False,
    )

    if cache_key == "base":
        print(f"Loading base model from {BASE_GGUF}")
        llm = Llama(model_path=BASE_GGUF, **common_kwargs)
    else:
        print(f"Loading LoRA adapter from {LORA_GGUF[cache_key]}")
        llm = Llama(model_path=BASE_GGUF, lora_path=LORA_GGUF[cache_key], **common_kwargs)

    _MODEL_CACHE[cache_key] = llm
    return llm


def build_prompt(user_prompt: str) -> str:
    prompt = "<|begin_of_text|>"
    prompt += "<|start_header_id|>system<|end_header_id|>\n" + SYSTEM_PROMPT + "<|eot_id|>"
    prompt += "<|start_header_id|>user<|end_header_id|>\n" + user_prompt.strip() + "<|eot_id|>"
    prompt += "<|start_header_id>assistant<|end_header_id|>\n"
    # Ensure no duplicate <|begin_of_text|> is added
    return prompt



def handler(job):
    inp = job.get("input", {})
    user_prompt = inp.get("prompt")
    domain = inp.get("domain", "")

    if not user_prompt or not isinstance(user_prompt, str):
        return {"error": "Missing input.prompt or input.prompt must be a string"}

    max_tokens = int(inp.get("max_tokens", 512))
    temperature = float(inp.get("temperature", 0.7))
    top_p = float(inp.get("top_p", 0.95))
    stop = inp.get("stop", ["<|eot_id|>"])

    # Print user input and domain
    print(f"Received prompt for domain: {domain}")

    llm = get_llm(domain)
    prompt = build_prompt(user_prompt)

    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )

    response = out["choices"][0]["text"].strip()

    # Print the first 100 characters of the response for debugging
    print(f"Generated response: {response[:100]}...")

    return {"response": response}


runpod.serverless.start({"handler": handler})
