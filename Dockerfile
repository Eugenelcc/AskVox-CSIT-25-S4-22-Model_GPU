FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /app

# -----------------------
# System deps
# -----------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# -----------------------
# Build llama-cpp-python FROM SOURCE for H200 (SM90)
# -----------------------
ENV CMAKE_ARGS="-DLLAMA_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=90"
ENV FORCE_CMAKE=1

RUN pip3 install \
    runpod==1.* \
    llama-cpp-python

# -----------------------
# Models
# -----------------------
RUN wget -O model.gguf \
"https://huggingface.co/cakebut/QLlama-3.3-70b/resolve/main/llama-3.3-70b-instruct.Q4_K_M.gguf"

RUN wget -O Cooking_LoRAadapter.gguf \
"https://huggingface.co/Skybison/CookingandFoodQLoRAadapter-GGUF/resolve/main/CookingandFoodQLoRAadapter.gguf" \
|| echo "Cooking LoRA skipped"

RUN wget -O History_LoRAadapter.gguf \
"https://huggingface.co/Skybison/HistoryQLoRAadapter-GUFF/resolve/main/HistoryQLoRAadapter.gguf" \
|| echo "History LoRA skipped"

RUN wget -O Geography_LoRAadapter.gguf \
"https://huggingface.co/Skybison/GeographyQLoRAadapter-GUFF/resolve/main/GeographyQLoRAadapter.gguf" \
|| echo "Geography LoRA skipped"

# -----------------------
# App
# -----------------------
COPY app.py .

# Prove GPU is used at startup
CMD ["bash", "-lc", "nvidia-smi && python3 app.py"]
