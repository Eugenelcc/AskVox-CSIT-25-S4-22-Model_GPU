# Use a CUDA-enabled base image for GPU support
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    git \
    python3-dev \
    python3-pip \
    libomp-dev \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Download the model (assuming the model is stored on Hugging Face)
RUN wget -O model.gguf "https://huggingface.co/cakebut/askvox_api/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true"

# Copy your custom app.py logic into the container
COPY app.py .

# Set environment variable to use GPU (ensure this matches your setup on Runpod)
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
ENV PORT=8080

# Command to run the app with FastAPI and Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
