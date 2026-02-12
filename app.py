import os
import gc
import threading
from llama_cpp import Llama
import runpod

# -----------------------
# Paths (prefer env override in RunPod)
# -----------------------
BASE_GGUF = os.getenv("BASE_GGUF", "/app/model.gguf")

LORA_GGUF = {
    "cooking & food": os.getenv("COOKING_LORA", "/app/Cooking_LoRAadapter.gguf"),
    "history and world events": os.getenv("HISTORY_LORA", "/app/History_LoRAadapter.gguf"),
    "geography and travel": os.getenv("GEO_LORA", "/app/Geography_LoRAadapter.gguf"),
}

SYSTEM_PROMPT = (
    "You are AskVox, a friendly and helpful AI assistant. "
    "Explain topics in a natural, human, tutor-like way. "
    "Prefer clear paragraph-style explanations with context, reasoning, and examples. "
    "Use bullet points or numbered lists only when they genuinely improve clarity. "
    "When using bullet points, include a short explanation for each item."
)

# Tunables
N_CTX = int(os.getenv("N_CTX", "8192"))
N_THREADS = int(os.getenv("N_THREADS", str(os.cpu_count() or 16)))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))

# Optional preload
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "0") == "1"

# -----------------------
# Global single-model state
# -----------------------
_LOCK = threading.RLock()
_CURRENT_KEY = None          # "base" or one of LORA_GGUF keys
_CURRENT_LLM = None          # the only live Llama() instance


def normalize_domain(domain: str) -> str:
    """Normalize incoming domain strings to match keys in LORA_GGUF."""
    d = (domain or "").strip().lower()

    # common variants your frontend might send
    d = d.replace("_", " ").replace("-", " ")
    d = " ".join(d.split())

    # map a few friendly aliases if needed
    aliases = {
        "cooking": "cooking & food",
        "cooking and food": "cooking & food",
        "food": "cooking & food",
        "history": "history and world events",
        "world events": "history and world events",
        "geography": "geography and travel",
        "travel": "geography and travel",
    }
    return aliases.get(d, d)


def safe_close_llm(llm: Llama):
    """
    Be defensive: llama-cpp-python sometimes throws during cleanup if init failed.
    We never want cleanup errors to kill the worker.
    """
    if llm is None:
        return
    try:
        # Some versions have .close(); some do cleanup on del
        if hasattr(llm, "close"):
            llm.close()
    except Exception as e:
        print(f"[WARN] Ignored error during llm.close(): {e}")

    try:
        del llm
    except Exception:
        pass

    gc.collect()


def build_prompt(user_prompt: str) -> str:
    raw = (user_prompt or "").strip()
    if raw.startswith("<|begin_of_text|>"):
        return raw

    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT.strip()}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{raw}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def load_llm_for_key(key: str) -> Llama:
    """
    Load base or base+one LoRA.
    IMPORTANT: we only ever have one Llama instance alive.
    """
    common_kwargs = dict(
        model_path=BASE_GGUF,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
        add_bos=False,
    )

    if key == "base":
        print(f"[LOAD] Base model: {BASE_GGUF}")
        return Llama(**common_kwargs)

    lora_path = LORA_GGUF.get(key)
    if not lora_path:
        print(f"[LOAD] Unknown key '{key}', falling back to base.")
        return Llama(**common_kwargs)

    print(f"[LOAD] Base+LoRA key='{key}' lora='{lora_path}'")
    return Llama(**common_kwargs, lora_path=lora_path)


def get_llm(domain: str) -> Llama:
    """
    Single-instance switcher:
    - If requested domain differs, unload current model and load the requested one.
    - Guarded by a lock to prevent concurrent loads/switches.
    """
    global _CURRENT_KEY, _CURRENT_LLM

    norm = normalize_domain(domain)
    key = norm if norm in LORA_GGUF else "base"

    with _LOCK:
        if _CURRENT_LLM is not None and _CURRENT_KEY == key:
            print(f"[CACHE] Using already-loaded key='{key}'")
            return _CURRENT_LLM

        # Switch model
        if _CURRENT_LLM is not None:
            print(f"[SWITCH] Unloading key='{_CURRENT_KEY}' -> loading key='{key}'")
            safe_close_llm(_CURRENT_LLM)
            _CURRENT_LLM = None
            _CURRENT_KEY = None

        _CURRENT_LLM = load_llm_for_key(key)
        _CURRENT_KEY = key
        print(f"[READY] Loaded key='{key}'")
        return _CURRENT_LLM


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

    print(f"[REQ] domain='{domain}' normalized='{normalize_domain(domain)}'")

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
    print(f"[OK] {response[:120]}...")
    return {"response": response}


# Optional preload: just load base once so first request is faster.
if PRELOAD_MODELS:
    try:
        with _LOCK:
            if _CURRENT_LLM is None:
                _CURRENT_LLM = load_llm_for_key("base")
                _CURRENT_KEY = "base"
        print("✅ Preloaded base model.")
    except Exception as e:
        print(f"⚠️ Preload failed: {e}")

runpod.serverless.start({"handler": handler})
