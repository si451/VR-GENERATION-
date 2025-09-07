# Use Ubuntu as base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create workspace directory and verify installations
RUN mkdir -p /app/workspace \
    && python --version \
    && ffmpeg -version

# Make startup script executable
RUN chmod +x start.sh

# Set environment variables
ENV WORKSPACE_DIR=/app/workspace

# Start command
CMD bash -c "echo '🚀 Starting VR180 Backend on port $PORT' && python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
