# Use the official Python slim image as the base image
FROM python:3.11-slim

# Set DEBIAN_FRONTEND to noninteractive to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    git \
    tzdata \
 && rm -rf /var/lib/apt/lists/*

# Reset DEBIAN_FRONTEND to interactive (optional, but good practice)
ENV DEBIAN_FRONTEND=interactive

# Set the working directory
WORKDIR /app

# Download the model
RUN wget -O model.gguf "https://huggingface.co/cakebut/askvox_api/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true"

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_NATIVE=ON" pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code
COPY app.py .

# Set the environment variable for the port
ENV PORT=8080

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
