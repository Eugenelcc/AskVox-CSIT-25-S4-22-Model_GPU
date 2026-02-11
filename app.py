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

# System prompt to instruct the model (used ONLY when input.prompt is plain text)
SYSTEM_PROMPT = (
    "You are AskVox, a friendly and helpful AI assistant. "
    "Explain topics in a natural, human, tutor-like way. "
    "Prefer clear paragraph-style explanations with context, reasoning, and examples. "
    "Use bullet points or numbered lists only when they genuinely improve clarity. "
    "When using bullet points, include a short explanation for each item."
)

# Tunables (allow override without code edits)
N_CTX = int(os.getenv("N_CTX", "8192"))
N_THREADS = int(os.getenv("N_THREADS", str(os.cpu_count() or 16)))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "80"))

# Optional: preload models on cold start (reduces first-job latency inside handler)
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "0") == "1"

# Cache for models to avoid reloading
_MODEL_CACHE = {}


def _make_llama(**kwargs) -> Llama:
    """
    llama-cpp-python can auto-add BOS tokens depending on version/settings.
    If our prompt already includes <|begin_of_text|>, adding BOS again can trigger warnings.
    We try add_bos=False when supported; fall back gracefully if not.
    """
    try:
        return Llama(add_bos=False, **kwargs)
    except TypeError:
        # Older llama_cpp versions may not support add_bos
        return Llama(**kwargs)


# Function to load or return cached model
def get_llm(domain: str):
    domain = (domain or "").lower().strip()
    print(f"Normalized Domain: {domain}")

    cache_key = domain if domain in LORA_GGUF else "base"
    print(f"Cache Key: {cache_key}")

    if cache_key in _MODEL_CACHE:
        print(f"Returning cached model for {cache_key}.")
        return _MODEL_CACHE[cache_key]

    common_kwargs = {
        "model_path": BASE_GGUF,
        "n_ctx": N_CTX,
        "n_threads": N_THREADS,
        "n_gpu_layers": N_GPU_LAYERS,
        "verbose": False,
    }

    if cache_key == "base":
        print(f"Loading base model from {BASE_GGUF}")
        llm = _make_llama(**common_kwargs)
    else:
        lora_adapter_path = LORA_GGUF.get(cache_key)
        print(f"LoRA Adapter Path: {lora_adapter_path}")
        if lora_adapter_path:
            print(f"Loading LoRA adapter for domain: {domain}")
            llm = _make_llama(**common_kwargs, lora_path=lora_adapter_path)
        else:
            print(f"LoRA adapter not found for domain: {domain}, loading base model")
            llm = _make_llama(**common_kwargs)

    _MODEL_CACHE[cache_key] = llm
    return llm


def build_prompt(user_prompt: str) -> str:
    """
    If user_prompt already looks like a full Llama-3 chat template (starts with <|begin_of_text|>),
    do NOT wrap it again — this prevents the duplicate <|begin_of_text|> warning and preserves
    custom system prompts (e.g., your backend second-pass prompt).
    """
    raw = (user_prompt or "")
    s = raw.lstrip()
    if s.startswith("<|begin_of_text|>"):
        return s  # already templated upstream

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT.strip()}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{raw.strip()}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
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
    print(f"Generated response: {response[:100]}...")

    return {"response": response}


# Optional preload (helps reduce first-request latency inside handler)
if PRELOAD_MODELS:
    try:
        get_llm("")  # base
        for k in list(LORA_GGUF.keys()):
            get_llm(k)
        print("✅ Preloaded model(s).")
    except Exception as e:
        print(f"⚠️ Preload failed: {e}")

runpod.serverless.start({"handler": handler})