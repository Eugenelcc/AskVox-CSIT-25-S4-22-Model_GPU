# 1. Use DEVEL image (Required for runtime libraries)
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 2. Install Python & Basic Tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Download Model
RUN wget -O model.gguf "https://huggingface.co/cakebut/askvox_api/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true"

# 4. Install General Python Deps
COPY requirements.txt .
# Upgrading pip is often required for modern wheels
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# 5. Install Llama-cpp-python via PRE-BUILT WHEEL (The Fix)
# We use the 'cu121' wheel which works on CUDA 12.x
RUN pip3 install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# 6. Copy App
COPY app.py .

# 7. Expose & Start
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]