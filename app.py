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
# Model settings
# -----------------------
N_CTX = int(os.getenv("N_CTX", "8192"))
N_THREADS = int(os.getenv("N_THREADS", "16"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "999"))  # H200 can do full offload

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
    except Exception:
        pass
    gc.collect()

# -----------------------
# Loader with GPU->CPU fallback
# -----------------------
def load_model(key: str) -> Llama:
    common = dict(
        model_path=BASE_GGUF,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=True,
        chat_format="llama-3",
    )

    lora_path = None
    if key != "base":
        lp = LORA_GGUF.get(key)
        if lp and os.path.exists(lp):
            lora_path = lp

    try:
        if key == "base":
            print(f"[LOAD] Base (GPU try) n_ctx={N_CTX} n_gpu_layers={N_GPU_LAYERS}")
            return Llama(**common)

        if lora_path:
            print(f"[LOAD] Base+LoRA ({key}) (GPU try) -> {lora_path}")
            return Llama(**common, lora_path=lora_path, lora_scale=1.0)

        print(f"[LOAD] LoRA '{key}' missing, using base (GPU try)")
        return Llama(**common)

    except Exception as e:
        print(f"[GPU FAIL] {e}")
        print("[FALLBACK] Reload CPU (n_gpu_layers=0)")
        common["n_gpu_layers"] = 0

        if key == "base":
            return Llama(**common)
        if lora_path:
            return Llama(**common, lora_path=lora_path, lora_scale=1.0)
        return Llama(**common)

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
            _CURRENT_LLM = None

        _CURRENT_LLM = load_model(key)
        _CURRENT_KEY = key
        print(f"[READY] Model loaded: {key}")
        return _CURRENT_LLM

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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt.strip()},
    ]

    out = llm.create_chat_completion(
        messages=messages,
        max_tokens=int(inp.get("max_tokens", 1024)),
        temperature=float(inp.get("temperature", 0.7)),
        top_p=float(inp.get("top_p", 0.95)),
    )

    response = out["choices"][0]["message"]["content"].strip()

    used = normalize_domain(domain)
    if used not in LORA_GGUF:
        used = "base"

    return {"response": response, "domain_used": used, "n_ctx": N_CTX}

runpod.serverless.start({"handler": handler})
