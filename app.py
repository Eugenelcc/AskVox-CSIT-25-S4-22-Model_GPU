import runpod
from llama_cpp import Llama

llm = Llama(
    model_path="./model.gguf",  # your GGUF file
    n_ctx=4096,
    n_gpu_layers=20,
    n_threads=8,
    verbose=False,
)

SYSTEM_PROMPT = (
    "You are AskVox, a friendly and helpful AI assistant. "
    "Answer clearly and conversationally."
)

def handler(job):
    inp = job.get("input", {})
    user_prompt = inp.get("prompt")
    if not user_prompt:
        return {"error": "Missing input.prompt"}

    # Build messages in the LLaMA chat template format
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    prompt = ""
    bos_token = "<|begin_of_text|>"
    for i, msg in enumerate(messages):
        content = f"<|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content'].strip()}<|eot_id|>"
        if i == 0:
            content = bos_token + content
        prompt += content

    # Add assistant header to tell the model to respond
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

    # Run the model
    output = llm(
        prompt,
        max_tokens=1200,
        temperature=0.7,
        stop=["<|eot_id|>"],
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }

runpod.serverless.start({"handler": handler})
