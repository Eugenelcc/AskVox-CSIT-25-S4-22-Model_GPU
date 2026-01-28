import runpod
from llama_cpp import Llama

llm = Llama(
    model_path="./model.gguf",
    n_ctx=2048,
    n_gpu_layers=30,
    verbose=False,
)

def handler(job):
    if "input" not in job or "prompt" not in job["input"]:
        return {"error": "Missing input.prompt"}

    user_prompt = job["input"]["prompt"]

    full_prompt = (
        "Instruction: You are AskVox, a safe educational AI tutor.\n"
        f"Question: {user_prompt}\n"
        "Answer:"
    )

    output = llm(
        full_prompt,
        max_tokens=400,
        temperature=0.3,
        top_p=0.9,
        repeat_penalty=1.2,
        stop=["\n\n"],
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }

runpod.serverless.start({"handler": handler})
