FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download the model (ensure the model path is correct)
RUN wget -O model.gguf "https://huggingface.co/cakebut/QLlama-3.3-70b/resolve/main/llama-3.3-70b-instruct.Q4_K_M.gguf"

# Install dependencies
RUN pip3 install --upgrade pip
RUN pip3 install \
    runpod \
    llama-cpp-python \
    peft \
    transformers \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Download LoRA adapters
RUN wget -O Cooking_LoRAadapter.gguf https://huggingface.co/Skybison/CookingandFoodQLoRAadapter-GGUF
RUN wget -O History_LoRAadapter.gguf https://huggingface.co/Skybison/HistoryQLoRAadapter-GUFF
RUN wget -O Geography_LoRAadapter.gguf https://huggingface.co/Skybison/GeographyQLoRAadapter-GUFF

# Copy application code into the container
COPY app.py .

# Command to run the application
CMD ["python3", "app.py"]
