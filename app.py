import runpod
from llama_cpp import Llama
from transformers import LlamaTokenizer
from peft import PeftModel
import torch

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
# Function to dynamically load LoRA adapter based on domain
# -----------------------
def load_lora_adapter(domain):
    # Mapping domains to their LoRA adapter paths
    adapter_paths = {
        "cooking": "Skybison/CookingandFoodQLoRAadapter-GGUF",
        "history": "Skybison/HistoryQLoRAadapter-GUFF",
        "geography": "Skybison/GeographyQLoRAadapter-GUFF"
    }
    adapter_path = adapter_paths.get(domain)
    if adapter_path:
        # Load the LoRA adapter if it matches the domain
        model = PeftModel.from_pretrained(llm, adapter_path)
        return model
    # If domain is not found, return the base model
    return llm

# -----------------------
# RunPod handler
# -----------------------
def handler(job):
    inp = job.get("input", {})
    user_prompt = inp.get("prompt")
    domain = inp.get("domain", "")  # Get domain for LoRA adapter

    if not user_prompt:
        return {"error": "Missing input.prompt"}

    if not isinstance(user_prompt, str):
        return {"error": "input.prompt must be a string"}

    # NEW: allow backend to tune generation per learning preference
    max_tokens = inp.get("max_tokens", 1024)
    temperature = inp.get("temperature", 0.7)
    top_p = inp.get("top_p", 0.95)

    # NEW: allow backend to override stop tokens (optional)
    stop = inp.get("stop", ["<|eot_id|>", "<|start_header_id|>"])

    # Load the appropriate LoRA adapter based on domain (this is where the function is called)
    model = load_lora_adapter(domain)

    # Build Llama-3.3 Instruct prompt
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
    output = model(
        prompt,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        stop=stop,
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }

# -----------------------
# Start RunPod serverless
# -----------------------
runpod.serverless.start({"handler": handler})
