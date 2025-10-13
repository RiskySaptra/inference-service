# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.2-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements into container
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git libgl1 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Default command to run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]