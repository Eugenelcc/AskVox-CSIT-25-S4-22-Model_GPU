import runpod
from llama_cpp import Llama

llm = Llama(
    model_path="./model.gguf",
    n_ctx=1536,
    n_gpu_layers=20,   # or -1
    verbose=False,
)

def handler(job):
    if "input" not in job or "prompt" not in job["input"]:
        return {"error": "Missing input.prompt"}

    user_prompt = job["input"]["prompt"]

    prompt = (
        "Instruction: You are AskVox, a safe educational AI tutor.\n"
        "Explain the following topic clearly and in detail. "
        "Use paragraphs and examples where helpful.\n\n"
        f"Topic: {user_prompt}\n\n"
        "Explanation:"
    )

    output = llm(
        prompt,
        max_tokens=800,
        temperature=0.5,
        top_p=0.9,
        repeat_penalty=1.15,
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }

runpod.serverless.start({"handler": handler})
