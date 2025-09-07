#!/bin/bash

# Debug: Show environment variables
echo "🔧 Environment variables:"
echo "PORT: $PORT"
echo "PWD: $PWD"
echo "PATH: $PATH"

# Get the port from Railway environment variable, default to 8000
PORT=${PORT:-8000}

echo "🚀 Starting VR180 Backend on port $PORT"
echo "🔧 Command: python -m uvicorn main:app --host 0.0.0.0 --port $PORT"

# Start the application
python -m uvicorn main:app --host 0.0.0.0 --port $PORT
