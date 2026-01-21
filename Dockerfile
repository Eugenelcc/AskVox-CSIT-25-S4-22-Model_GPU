# Use NVIDIA CUDA runtime base image with Python 3.11
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    git \
    libomp-dev \
    python3.11 \
    python3-pip \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory to /app
WORKDIR /app

# Download the model directly
RUN wget -O model.gguf "https://huggingface.co/cakebut/askvox_api/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true"

# Copy the requirements file to the container
COPY requirements.txt .

# Install Python dependencies
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_NATIVE=ON" pip install --no-cache-dir -r requirements.txt

# Copy the app.py file to the container
COPY app.py .

# Expose the port for FastAPI
ENV PORT=8080

# Set the command to run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
