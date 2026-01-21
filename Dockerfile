# 1. CUDA devel image (needed for llama-cpp GPU)
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 2. Install Python + system deps
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    wget \
    git \
    build-essential \
    cmake \
 && rm -rf /var/lib/apt/lists/*

# 3. Ensure python / pip are consistent
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    python -m pip install --upgrade pip

# 4. Download model
RUN wget -O model.gguf \
    "https://huggingface.co/cakebut/askvox_api/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true"

# 5. Install web stack FIRST (guaranteed)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic

# 6. Install llama-cpp-python GPU wheel (ONLY ONCE)
RUN pip install --no-cache-dir llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# 7. Copy app
COPY app.py .

# 8. Expose + start
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
