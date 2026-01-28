import runpod
from llama_cpp import Llama

# Load model once (cold start)
llm = Llama(
    model_path="./Llama-3.2-3B.Q6_K.gguf",
    n_ctx=4096,
    n_gpu_layers=20,     
    n_threads=8,
    verbose=False,
)

# Strong steering prompt for BASE models
SYSTEM_PREFIX = (
    "You are AskVox, a knowledgeable, clear, and helpful AI assistant. "
    "You answer questions accurately, step by step when needed, "
    "and keep explanations concise and friendly.\n\n"
)

def handler(job):
    inp = job.get("input", {})
    user_prompt = inp.get("prompt")

    if not user_prompt:
        return {"error": "Missing input.prompt"}

    prompt = (
        SYSTEM_PREFIX +
        f"User: {user_prompt}\n"
        "Assistant:"
    )

    output = llm(
        prompt,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["User:", "Assistant:"],
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }

runpod.serverless.start({
    "handler": handler
})
