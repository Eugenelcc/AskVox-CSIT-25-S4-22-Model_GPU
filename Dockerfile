FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# Install dependencies first (better caching)
# -----------------------
RUN pip3 install --upgrade pip

# Pin llama-cpp-python and force CUDA wheel index
# (Pick a stable version; you can change the version if needed)
RUN pip3 install \
    runpod==1.* \
    "llama-cpp-python==0.3.7" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# -----------------------
# Download models (can be slow, but ok if you want baked image)
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

# (Optional) show GPU at container startup in logs
CMD ["bash", "-lc", "nvidia-smi || true && python3 app.py"]
