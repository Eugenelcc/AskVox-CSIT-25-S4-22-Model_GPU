# 1. Use DEVEL image (Crucial for compiling GPU support)
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 2. Install Python & Build Tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Download Model
RUN wget -O model.gguf "https://huggingface.co/cakebut/askvox_api/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true"

# 4. Install Python Deps
# Copy requirements first to cache layers
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 5. Install Llama-cpp with CUDA (GPU) flags
# We do this separately to ensure the 'CMAKE_ARGS' are applied
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip3 install llama-cpp-python

# 6. Copy App
COPY app.py .

# 7. EXPOSE PORT (Important for LB documentation, though RunPod ignores it sometimes)
EXPOSE 8080

# 8. START COMMAND (Web Server Mode)
# We listen on 0.0.0.0 so the Load Balancer can reach us
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
