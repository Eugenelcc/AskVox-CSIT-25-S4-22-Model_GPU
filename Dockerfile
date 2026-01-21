FROM python:3.11-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    git \
    libomp-dev \
    cuda-nvcc-11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download the model directly
RUN wget -O model.gguf "https://huggingface.co/cakebut/askvox_api/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true"

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_NATIVE=ON" pip install --no-cache-dir -r requirements.txt

# Copy the app file
COPY app.py .

# Expose the port for FastAPI
ENV PORT=8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
