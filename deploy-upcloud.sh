#!/bin/bash

echo "🚀 Deploying VR180 Backend to UpCloud..."

# Set environment variables
export PORT=8000
export WORKSPACE_DIR=/app/workspace

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t vr180-backend ./backend/api

# Stop any existing container
echo "🛑 Stopping existing container..."
docker stop vr180-backend-container 2>/dev/null || true
docker rm vr180-backend-container 2>/dev/null || true

# Run the new container
echo "🚀 Starting new container..."
docker run -d \
  --name vr180-backend-container \
  -p 8000:8000 \
  -e PORT=8000 \
  -e WORKSPACE_DIR=/app/workspace \
  -v /app/workspace:/app/workspace \
  vr180-backend

# Check if container is running
echo "✅ Checking container status..."
docker ps | grep vr180-backend-container

echo "🎉 Deployment complete!"
echo "🌐 Your backend is running at: http://YOUR_SERVER_IP:8000"
echo "📊 Check logs with: docker logs vr180-backend-container"
