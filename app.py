import os
import gc
import threading
import subprocess
import runpod
from llama_cpp import Llama

# =========================
# Quick GPU diagnostics
# =========================
def log_gpu_env():
    print("[ENV] CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
    print("[ENV] NVIDIA_VISIBLE_DEVICES =", os.getenv("NVIDIA_VISIBLE_DEVICES"))
    print("[ENV] LD_LIBRARY_PATH =", os.getenv("LD_LIBRARY_PATH"))
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT).decode()
        print("[GPU] nvidia-smi -L:\n", out)
    except Exception as e:
        print("[GPU] nvidia-smi failed:", repr(e))

log_gpu_env()

# =========================
# Paths
# =========================
BASE_GGUF = os.getenv("BASE_GGUF", "./model.gguf")

LORA_GGUF = {
    "cooking & food": os.getenv("COOKING_LORA", "./Cooking_LoRAadapter.gguf"),
    "history and world events": os.getenv("HISTORY_LORA", "./History_LoRAadapter.gguf"),
    "geography and travel": os.getenv("GEO_LORA", "./Geography_LoRAadapter.gguf"),
}

# =========================
# Model settings
# =========================
# With H200 you can TRY higher N_CTX, but only if your GGUF supports it.
N_CTX = int(os.getenv("N_CTX", "8192"))
N_THREADS = int(os.getenv("N_THREADS", "16"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "80"))

# Optional: if you built a long-context GGUF and your llama.cpp supports RoPE scaling.
# Common approaches:
# - ROPE_SCALING="yarn" or "linear"
# - ROPE_SCALE (float) / ROPE_FREQ_SCALE etc. depend on build/version
ROPE_SCALING = os.getenv("ROPE_SCALING", "").strip()  # e.g. "yarn" or "linear"
ROPE_SCALE = os.getenv("ROPE_SCALE", "").strip()      # e.g. "8" (string -> float)

# =========================
# System prompt
# =========================
SYSTEM_PROMPT = (
    "You are AskVox, a friendly and helpful AI assistant. "
    "Explain topics in a natural, human, tutor-like way. "
    "Prefer clear paragraph-style explanations with context, reasoning, and examples. "
    "Use bullet points or numbered lists only when they genuinely improve clarity "
    "(such as comparisons, rankings, or step-by-step instructions). "
    "When using bullet points, include a short explanation for each item."
)

# =========================
# Global model state
# =========================
_LOCK = threading.RLock()
_CURRENT_KEY = None
_CURRENT_LLM = None

# =========================
# Domain normalization
# =========================
def normalize_domain(domain: str) -> str:
    d = (domain or "").strip().lower()
    d = d.replace("_", " ").replace("-", " ")
    d = " ".join(d.split())

    aliases = {
        "cooking": "cooking & food",
        "food": "cooking & food",
        "cooking and food": "cooking & food",
        "history": "history and world events",
        "world events": "history and world events",
        "geography": "geography and travel",
        "travel": "geography and travel",
    }
    return aliases.get(d, d)

# =========================
# Cleanup helper
# =========================
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

# =========================
# Model loader (GPU first, CPU fallback)
# =========================
def _build_common_kwargs(n_gpu_layers: int):
    kw = dict(
        model_path=BASE_GGUF,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
        chat_format="llama-3",
    )

    # Try to pass RoPE scaling only if user provided envs.
    # Not all llama_cpp builds support these keys; if unsupported it will raise -> we catch and fallback.
    if ROPE_SCALING:
        kw["rope_scaling_type"] = ROPE_SCALING  # some builds accept this
    if ROPE_SCALE:
        try:
            kw["rope_scale"] = float(ROPE_SCALE)  # some builds accept this
        except ValueError:
            print("[WARN] ROPE_SCALE is not a float; ignoring")

    return kw

def load_model(key: str) -> Llama:
    """
    Loads base or base+LoRA.
    - Tries GPU first.
    - If CUDA/driver init fails (driverInitFileInfo...), falls back to CPU.
    - Uses chat_format='llama-3' to avoid duplicate begin_of_text warnings.
    """
    # Resolve LoRA path if needed
    lora_path = None
    if key != "base":
        lp = LORA_GGUF.get(key)
        if lp and os.path.exists(lp):
            lora_path = lp

    # ---------- GPU try ----------
    try:
        common_gpu = _build_common_kwargs(N_GPU_LAYERS)

        if key == "base":
            print(f"[LOAD] Base (GPU try) n_ctx={N_CTX} n_gpu_layers={N_GPU_LAYERS}")
            return Llama(**common_gpu)

        if lora_path:
            print(f"[LOAD] Base+LoRA ({key}) (GPU try) -> {lora_path}")
            return Llama(**common_gpu, lora_path=lora_path, lora_scale=1.0)

        print(f"[LOAD] LoRA '{key}' missing, using base (GPU try)")
        return Llama(**common_gpu)

    except Exception as e:
        print(f"[GPU FAIL] {repr(e)}")
        print("[FALLBACK] Reloading on CPU (n_gpu_layers=0)")

    # ---------- CPU fallback ----------
    common_cpu = _build_common_kwargs(0)

    if key == "base":
        print(f"[LOAD] Base (CPU) n_ctx={N_CTX}")
        return Llama(**common_cpu)

    if lora_path:
        print(f"[LOAD] Base+LoRA ({key}) (CPU) -> {lora_path}")
        return Llama(**common_cpu, lora_path=lora_path, lora_scale=1.0)

    print(f"[LOAD] LoRA '{key}' missing, using base (CPU)")
    return Llama(**common_cpu)

# =========================
# Model switcher
# =========================
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

# =========================
# RunPod handler
# =========================
def handler(job):
    inp = job.get("input", {})
    user_prompt = inp.get("prompt")
    domain = inp.get("domain", "")

    if not user_prompt or not isinstance(user_prompt, str):
        return {"error": "Missing input.prompt (string)"}

    llm = get_model(domain)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt.strip()},
    ]

    try:
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=int(inp.get("max_tokens", 1024)),
            temperature=float(inp.get("temperature", 0.7)),
            top_p=float(inp.get("top_p", 0.95)),
        )

        response = output["choices"][0]["message"]["content"].strip()

        used = normalize_domain(domain)
        if used not in LORA_GGUF:
            used = "base"

        return {
            "response": response,
            "domain_used": used,
            "n_ctx": N_CTX,
        }

    except Exception as e:
        used = normalize_domain(domain)
        if used not in LORA_GGUF:
            used = "base"

        return {
            "error": str(e),
            "domain_used": used,
            "n_ctx": N_CTX,
        }

# =========================
# Start RunPod serverless
# =========================
runpod.serverless.start({"handler": handler})
