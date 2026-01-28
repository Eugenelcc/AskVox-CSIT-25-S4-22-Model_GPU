import runpod
from llama_cpp import Llama

llm = Llama(
    model_path="./model.gguf",
    n_ctx=2048,
    n_gpu_layers=30,
    verbose=False,
)

def handler(job):
    inp = job.get("input", {})
    user_prompt = inp.get("prompt")
    history = inp.get("history", [])

    if not user_prompt:
        return {"error": "Missing input.prompt"}

    system_prompt = (
        "You are AskVox, a friendly and knowledgeable AI assistant. "
        "Answer clearly, helpfully, and in appropriate detail."
    )

    full_prompt = build_prompt(system_prompt, history, user_prompt)

    output = llm(
        full_prompt,
        max_tokens=800,
        temperature=0.45,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=[
            "\nUser:",
            "\nAssistant:",
        ],
    )

    return {
        "response": output["choices"][0]["text"].strip()
    }


runpod.serverless.start({"handler": handler})
