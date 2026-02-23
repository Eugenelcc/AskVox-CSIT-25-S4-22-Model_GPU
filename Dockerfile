FROM FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*


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

# -----------------------
# Install dependencies
# -----------------------
RUN pip3 install --upgrade pip

RUN pip3 install runpod

RUN pip3 install --force-reinstall --no-cache-dir \
    llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121


COPY app.py .

CMD ["python3", "app.py"]
