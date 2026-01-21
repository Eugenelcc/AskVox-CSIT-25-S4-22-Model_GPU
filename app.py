from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal
from llama_cpp import Llama
import os
import time

app = FastAPI(title="AskVox LLaMA2 CloudRun")

# --- CONFIGURATION ---
MODEL_PATH = "./model.gguf"  # Ensure the correct model path is set
N_CTX = int(os.getenv("N_CTX", "2048"))
N_THREADS = int(os.getenv("N_THREADS", "4"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
GPU_LAYER_COUNT = int(os.getenv("GPU_LAYER_COUNT", "30"))  # Set number of layers to load on GPU

# Initialize the model (this will happen during startup)
llm = None

# Define a function to load the model
def load_model():
    global llm
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"CRITICAL ERROR: Model not found at {MODEL_PATH}. Did the Dockerfile 'wget' command fail?")

    try:
        print(f"Loading model from {MODEL_PATH}...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=GPU_LAYER_COUNT,  # Ensure GPU is being utilized with specific layers
            verbose=False,
        )
        print("✅ Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load Llama model: {e}")

# Load model during FastAPI startup
@app.on_event("startup")
async def startup_event():
    load_model()

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
    # Debug print indicating server is running
    print("App.py RunPod server is online")
    return {"status": "online", "system": "RunPod GPU"}

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

    print("🧠 RunPod LLaMA: generating...")
    t0 = time.perf_counter()
    try:
        output = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    t1 = time.perf_counter()
    print(f"⏱️ RunPod generation time: {t1 - t0:.2f}s")

    try:
        text = output["choices"][0]["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bad model output: {e}")

    return ChatResponse(response=text.strip())
