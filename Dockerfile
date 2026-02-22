FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /app

# -----------------------
# System dependencies
# -----------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Make python3 default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# -----------------------
# Upgrade pip
# -----------------------
RUN pip3 install --upgrade pip

# -----------------------
# Force CUDA build of llama-cpp
# -----------------------
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80"
ENV FORCE_CMAKE=1


RUN pip3 install --no-cache-dir runpod llama-cpp-python




# -----------------------
# Download base model
# -----------------------
RUN wget -O model.gguf \
"https://huggingface.co/cakebut/QLlama-3.3-70b/resolve/main/llama-3.3-70b-instruct.Q4_K_M.gguf"

# -----------------------
# Download LoRAs from Hugging Face
# (replace URLs with your actual repos)
# -----------------------

# Cooking
RUN wget -O Cooking_LoRAadapter.gguf \
"https://huggingface.co/Skybison/CookingandFoodQLoRAadapter-GGUF/resolve/main/CookingandFoodQLoRAadapter.gguf" || echo "Cooking LoRA not found, skipping"

# History
RUN wget -O History_LoRAadapter.gguf \
"https://huggingface.co/Skybison/HistoryQLoRAadapter-GUFF/resolve/main/HistoryQLoRAadapter.gguf" || echo "History LoRA not found, skipping"

# Geography
RUN wget -O Geography_LoRAadapter.gguf \
"https://huggingface.co/Skybison/GeographyQLoRAadapter-GUFF/resolve/main/GeographyQLoRAadapter.gguf" || echo "Geography LoRA not found, skipping"

COPY app.py .

CMD ["python3", "app.py"]
