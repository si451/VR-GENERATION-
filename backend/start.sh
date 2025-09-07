#!/bin/bash
# Railway startup script for VR180 Backend

echo "🚀 Starting VR180 Backend..."
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Available files:"
ls -la

# Install dependencies if needed
if [ ! -d ".venv" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the FastAPI server
echo "Starting FastAPI server..."
python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT

