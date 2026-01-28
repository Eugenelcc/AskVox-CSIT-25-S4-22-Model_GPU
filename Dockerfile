FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download model
RUN wget -O model.gguf "https://huggingface.co/cakebut/Llama-3.3-8B-Instruct-Q5/resolve/main/Llama-3.3-8B-Instruct.Q5_K_M.gguf"

# Install deps
RUN pip3 install --upgrade pip
RUN pip3 install \
    runpod \
    llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

COPY app.py .

CMD ["python3", "app.py"]
