import os
import gc
import threading
import runpod
from llama_cpp import Llama


# -----------------------
# Paths
# -----------------------
BASE_GGUF = os.getenv("BASE_GGUF", "./model.gguf")

LORA_GGUF = {
    "cooking & food": os.getenv("COOKING_LORA", "./Cooking_LoRAadapter.gguf"),
    "history and world events": os.getenv("HISTORY_LORA", "./History_LoRAadapter.gguf"),
    "geography and travel": os.getenv("GEO_LORA", "./Geography_LoRAadapter.gguf"),
}

# -----------------------
# Model settings (same as your good version)
# -----------------------
N_CTX = int(os.getenv("N_CTX", "8192"))
N_THREADS = int(os.getenv("N_THREADS", "16"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "80"))

# -----------------------
# System prompt
# -----------------------
SYSTEM_PROMPT = (
    "You are AskVox, a friendly and helpful AI assistant. "
    "Explain topics in a natural, human, tutor-like way. "
    "Prefer clear paragraph-style explanations with context, reasoning, and examples. "
    "Use bullet points or numbered lists only when they genuinely improve clarity "
    "(such as comparisons, rankings, or step-by-step instructions). "
    "When using bullet points, include a short explanation for each item."
)

# -----------------------
# Global model state
# -----------------------
_LOCK = threading.RLock()
_CURRENT_KEY = None
_CURRENT_LLM = None


# -----------------------
# Domain normalization
# -----------------------
def normalize_domain(domain: str) -> str:
    d = (domain or "").strip().lower()
    d = d.replace("_", " ").replace("-", " ")
    d = " ".join(d.split())

    aliases = {
        "cooking": "cooking & food",
        "food": "cooking & food",
        "history": "history and world events",
        "world events": "history and world events",
        "geography": "geography and travel",
        "travel": "geography and travel",
    }

    return aliases.get(d, d)


# -----------------------
# Cleanup helper
# -----------------------
def safe_close(llm):
    if llm is None:
        return
    try:
        if hasattr(llm, "close"):
            llm.close()
    except Exception as e:
        print(f"[WARN] close error: {e}")
    try:
        del llm
    except:
        pass
    gc.collect()


# -----------------------
# Prompt builder (Llama-3 format)
# -----------------------
def build_prompt(user_prompt: str) -> str:
    raw = user_prompt.strip()

    if raw.startswith("<|begin_of_text|>"):
        return raw

    prompt = "<|begin_of_text|>"

    prompt += (
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
    )

    prompt += (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{raw}<|eot_id|>"
    )

    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt


# -----------------------
# Model loader
# -----------------------
def load_model(key: str) -> Llama:
    common = dict(
        model_path=BASE_GGUF,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=True,
        # IMPORTANT:
        # Do NOT set add_bos=False
        # Default behavior gives best Llama-3 quality
    )

    # Base model
    if key == "base":
        print("[LOAD] Base model")
        return Llama(**common)

    # LoRA
    lora_path = LORA_GGUF.get(key)

    if not lora_path or not os.path.exists(lora_path):
        print(f"[LOAD] LoRA for '{key}' not found, using base")
        return Llama(**common)

    print(f"[LOAD] Base + LoRA ({key}) -> {lora_path}")
    return Llama(**common, lora_path=lora_path, lora_scale=1.0)


# -----------------------
# Model switcher
# -----------------------
def get_model(domain: str) -> Llama:
    global _CURRENT_KEY, _CURRENT_LLM

    norm = normalize_domain(domain)
    key = norm if norm in LORA_GGUF else "base"

    with _LOCK:
        if _CURRENT_LLM is not None and _CURRENT_KEY == key:
            return _CURRENT_LLM

        if _CURRENT_LLM is not None:
            print(f"[SWITCH] {_CURRENT_KEY} -> {key}")
            safe_close(_CURRENT_LLM)

        _CURRENT_LLM = load_model(key)
        _CURRENT_KEY = key
        print(f"[READY] Model loaded: {key}")
        return _CURRENT_LLM


# -----------------------
# Cold start (preload base)
# -----------------------
print("Loading base model at startup...")
_CURRENT_LLM = load_model("base")
_CURRENT_KEY = "base"
print("Model ready.")


# -----------------------
# RunPod handler
# -----------------------
def handler(job):
    inp = job.get("input", {})
    user_prompt = inp.get("prompt")
    domain = inp.get("domain", "")

    if not user_prompt or not isinstance(user_prompt, str):
        return {"error": "Missing input.prompt"}

    llm = get_model(domain)
    prompt = build_prompt(user_prompt)

    output = llm(
        prompt,
        max_tokens=int(inp.get("max_tokens", 1024)),
        temperature=float(inp.get("temperature", 0.7)),
        top_p=float(inp.get("top_p", 0.95)),
        stop=inp.get("stop", ["<|eot_id|>", "<|start_header_id|>"]),
    )

    response = output["choices"][0]["text"].strip()

    return {
        "response": response,
        "domain_used": normalize_domain(domain) or "base"
    }


# -----------------------
# Start RunPod serverless
# -----------------------
runpod.serverless.start({"handler": handler})
