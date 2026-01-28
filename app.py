import runpod
from llama_cpp import Llama

llm = Llama(
    model_path="./model.gguf",
    n_ctx=4096,
    n_gpu_layers=20,
    n_threads=8,
    verbose=False,
)

def handler(job):
    inp = job.get("input", {})
    user_prompt = inp.get("prompt")
    if not user_prompt:
        return {"error": "Missing input.prompt"}

    prompt = (
        "Instruction: You are AskVox, a helpful assistant.\n"
        f"Question: {user_prompt}\n"
        "Answer:"
    )

    output = llm(
        prompt,
        max_tokens=300,
        temperature=0.3,
        top_p=0.9,
        repeat_penalty=1.2,
        stop=["\n\n"],
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }

runpod.serverless.start({"handler": handler})
