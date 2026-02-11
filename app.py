import os
from llama_cpp import Llama
import runpod

# Paths for the base model and LoRA adapters
BASE_GGUF = "/app/model.gguf"
LORA_GGUF = {
    "cooking & food": "/app/Cooking_LoRAadapter.gguf",
    "history and world events": "/app/History_LoRAadapter.gguf",
    "geography and travel": "/app/Geography_LoRAadapter.gguf",
}

# System prompt to instruct the model
SYSTEM_PROMPT = (
    "You are AskVox, a friendly and helpful AI assistant. "
    "Explain topics in a natural, human, tutor-like way. "
    "Prefer clear paragraph-style explanations with context, reasoning, and examples. "
    "Use bullet points or numbered lists only when they genuinely improve clarity. "
    "When using bullet points, include a short explanation for each item."
)

# Cache for models to avoid reloading
_MODEL_CACHE = {}

# Function to load or return cached model
def get_llm(domain: str):
    # Normalize the domain to lowercase for consistent matching
    domain = (domain or "").lower().strip()
    print(f"Normalized Domain: {domain}")  # Debug log to check the domain

    # Select the appropriate model based on the domain
    cache_key = domain if domain in LORA_GGUF else "base"

    # Debugging: print which model or adapter is being used
    print(f"Cache Key: {cache_key}")

    # If the model is cached, return it directly
    if cache_key in _MODEL_CACHE:
        print(f"Returning cached model for {cache_key}.")
        return _MODEL_CACHE[cache_key]

    # Set model parameters for both base and LoRA models
    common_kwargs = {
        "n_ctx": 8192,          # Context length for the model
        "n_threads": 16,        # Use available CPU threads
        "n_gpu_layers": 80,     # Offload layers to GPU
        "verbose": False,       # Disable verbose logging
    }

    # Load the base model or LoRA adapter as required
    if cache_key == "base":
        print(f"Loading base model from {BASE_GGUF}")
        llm = Llama(model_path=BASE_GGUF, **common_kwargs)
    else:
        # Check if the LoRA adapter path exists
        lora_adapter_path = LORA_GGUF.get(cache_key)
        print(f"LoRA Adapter Path: {lora_adapter_path}")  # Debug log for adapter path
        if lora_adapter_path:
            print(f"Loading LoRA adapter for domain: {domain}")
            llm = Llama(model_path=BASE_GGUF, lora_path=lora_adapter_path, **common_kwargs)
        else:
            print(f"LoRA adapter not found for domain: {domain}, loading base model")
            llm = Llama(model_path=BASE_GGUF, **common_kwargs)

    # Cache the model to avoid reloading it multiple times
    _MODEL_CACHE[cache_key] = llm
    return llm

# Build the complete prompt for the model
def build_prompt(user_prompt: str) -> str:
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT.strip()}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_prompt.strip()}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

# Main handler function for processing requests
def handler(job):
    inp = job.get("input", {})
    user_prompt = inp.get("prompt")
    domain = inp.get("domain", "")

    # Validate input
    if not user_prompt or not isinstance(user_prompt, str):
        return {"error": "Missing input.prompt or input.prompt must be a string"}

    # Set default model parameters (they can be customized via job input)
    max_tokens = int(inp.get("max_tokens", 512))
    temperature = float(inp.get("temperature", 0.7))
    top_p = float(inp.get("top_p", 0.95))
    stop = inp.get("stop", ["<|eot_id|>"])

    print(f"Received prompt for domain: {domain}")

    # Get the appropriate Llama model (either base or LoRA)
    llm = get_llm(domain)

    # Build the model prompt
    prompt = build_prompt(user_prompt)

    # Generate the response from the model
    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )

    # Extract the response
    response = out["choices"][0]["text"].strip()

    # Debugging: print the first 100 characters of the response
    print(f"Generated response: {response[:100]}...")

    return {"response": response}

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})
