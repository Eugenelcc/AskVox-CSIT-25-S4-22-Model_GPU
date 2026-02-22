FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /app

# -----------------------
# System build dependencies
# -----------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    cmake \
    ninja-build \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# -----------------------
# Build llama-cpp-python FROM SOURCE with CUDA for H200 (SM90)
# -----------------------
ENV FORCE_CMAKE=1
ENV CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=90"

# (Optional but helps avoid surprises)
# You can pin to a version known to work well.
RUN pip3 install "runpod==1.*" "llama-cpp-python==0.3.7"

# -----------------------
# Download models (baked into image)
# -----------------------
RUN wget -O model.gguf \
"https://huggingface.co/cakebut/QLlama-3.3-70b/resolve/main/llama-3.3-70b-instruct.Q4_K_M.gguf"

RUN wget -O Cooking_LoRAadapter.gguf \
"https://huggingface.co/Skybison/CookingandFoodQLoRAadapter-GGUF/resolve/main/CookingandFoodQLoRAadapter.gguf" \
|| echo "Cooking LoRA not found, skipping"

RUN wget -O History_LoRAadapter.gguf \
"https://huggingface.co/Skybison/HistoryQLoRAadapter-GUFF/resolve/main/HistoryQLoRAadapter.gguf" \
|| echo "History LoRA not found, skipping"

RUN wget -O Geography_LoRAadapter.gguf \
"https://huggingface.co/Skybison/GeographyQLoRAadapter-GUFF/resolve/main/GeographyQLoRAadapter.gguf" \
|| echo "Geography LoRA not found, skipping"

# -----------------------
# App
# -----------------------
COPY app.py .

# Print GPU info then run
CMD ["bash", "-lc", "nvidia-smi -L || true; python3 app.py"]
