from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Literal
from vllm import LLM  # Import vLLM
import os
import time

app = FastAPI(title="AskVox LLaMA2 GPU")

# --- CONFIGURATION ---
MODEL_PATH = "./model.gguf"

N_CTX = int(os.getenv("N_CTX", "2048"))
N_THREADS = int(os.getenv("N_THREADS", str(os.cpu_count() or 4)))
N_BATCH = int(os.getenv("N_BATCH", "512"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

print("--- Initializing GPU AI ---")
print(f"Config → ctx={N_CTX}, threads={N_THREADS}, batch={N_BATCH}, max_tokens={MAX_TOKENS}")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"CRITICAL ERROR: Model not found at {MODEL_PATH}. Did the Dockerfile 'wget' command fail?")

# --- LOAD MODEL WITH GPU SUPPORT ---
try:
    print(f"Loading model from {MODEL_PATH}...")
    model = LLM(model_path=MODEL_PATH, device="cuda", max_batch_size=N_BATCH)
    print("✅ Model loaded successfully on GPU.")
except Exception as e:
    raise RuntimeError(f"Failed to load Llama model: {e}")

# ---------- CHAT SCHEMA ----------
class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[HistoryItem] = []

class ChatResponse(BaseModel):
    response: str
# -------------------------------

@app.get("/")
def root():
    return {"status": "online", "system": "Runpod GPU"}

# --- META ENDPOINT ---
@app.get("/meta")
def meta():
    return {
        "system": "Runpod GPU",
        "model_path": MODEL_PATH,
        "n_ctx": N_CTX,
        "n_threads": N_THREADS,
        "n_batch": N_BATCH,
        "max_tokens": MAX_TOKENS,
    }

# --- STANDARD CHAT ---
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    SYSTEM_PROMPT = "You are AskVox, a safe educational AI tutor."
    parts = [f"[SYSTEM] {SYSTEM_PROMPT}"]

    for h in req.history[-3:]:
        if h.role == "user":
            parts.append(f"[USER] {h.content}")
        else:
            parts.append(f"[ASSISTANT] {h.content}")

    parts.append(f"[USER] {req.message}")
    parts.append("[ASSISTANT]")

    prompt = "\n".join(parts)

    print("🧠 Runpod LLaMA: generating...")
    t0 = time.perf_counter()

    try:
        output = model.generate(prompt, max_tokens=MAX_TOKENS)
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    t1 = time.perf_counter()
    print(f"⏱️ Runpod generation time: {t1 - t0:.2f}s")

    text = output["choices"][0]["text"]
    return ChatResponse(response=text.strip())

# --- STREAMING CHAT ---
def _stream_generate(prompt: str):
    print("🧠 Runpod LLaMA: streaming...")
    try:
        for chunk in model.generate(prompt, max_tokens=MAX_TOKENS, stream=True):
            yield chunk["choices"][0]["text"]
    except Exception as e:
        yield f"\n[stream_error: {e}]"

@app.post("/chat_stream")
def chat_stream(req: ChatRequest):
    SYSTEM_PROMPT = "You are AskVox, a safe educational AI tutor."
    parts = [f"[SYSTEM] {SYSTEM_PROMPT}"]

    for h in req.history[-3:]:
        if h.role == "user":
            parts.append(f"[USER] {h.content}")
        else:
            parts.append(f"[ASSISTANT] {h.content}")

    parts.append(f"[USER] {req.message}")
    parts.append("[ASSISTANT]")

    prompt = "\n".join(parts)

    return StreamingResponse(
        _stream_generate(prompt),
        media_type="text/plain",
    )
