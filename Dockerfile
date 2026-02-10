FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    curl \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Download the base model
RUN wget -O /app/model.gguf "https://huggingface.co/cakebut/QLlama-3.3-70b/resolve/main/llama-3.3-70b-instruct.Q4_K_M.gguf"

# Download LoRA adapters (Cooking, History, Geography)
RUN wget -O /app/Cooking_LoRAadapter.gguf "https://huggingface.co/Skybison/CookingandFoodQLoRAadapter-GGUF"
RUN wget -O /app/History_LoRAadapter.gguf "https://huggingface.co/Skybison/HistoryQLoRAadapter-GUFF"
RUN wget -O /app/Geography_LoRAadapter.gguf "https://huggingface.co/Skybison/GeographyQLoRAadapter-GUFF"

# Install Python dependencies
RUN pip3 install --upgrade pip \
    && pip3 install \
        runpod \
        llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Copy the application code into the container
COPY app.py .

# Set the environment variable for CUDA usage
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

 

# Command to run the application
CMD ["python3", "app.py"]
