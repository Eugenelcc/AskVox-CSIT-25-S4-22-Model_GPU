import runpod
from llama_cpp import Llama

# -----------------------
# LOAD MODEL (ONCE)
# -----------------------
llm = Llama(
    model_path="./Llama-3.2-3B.Q6_K.gguf",
    n_ctx=2048,
    n_gpu_layers=20,     
    n_threads=8,
    verbose=False,
)

# -----------------------
# PROMPT BUILDER
# -----------------------
def build_prompt(message, history):
    system = (
        "You are AskVox, a friendly and knowledgeable AI assistant. "
        "Respond naturally and clearly, with as much detail as the question requires.\n\n"
    )

    prompt = system

    # keep last 6 turns max
    for h in history[-6:]:
        role = h.get("role")
        content = (h.get("content") or "").strip()
        if not content:
            continue

        if role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"

    prompt += f"User: {message}\nAssistant:"

    return prompt


# -----------------------
# RUNPOD HANDLER
# -----------------------
def handler(job):
    inp = job.get("input", {})
    message = inp.get("message") or inp.get("prompt")
    history = inp.get("history", [])

    if not message:
        return {"error": "Missing input.message"}

    prompt = build_prompt(message, history)

    output = llm(
        prompt,
        max_tokens=1200,
        temperature=0.6,
        top_p=0.9,
        repeat_penalty=1.15,
        stop=["\nUser:"],
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }


runpod.serverless.start({"handler": handler})
