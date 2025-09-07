#!/bin/bash

# Get the port from Railway environment variable, default to 8000
PORT=${PORT:-8000}

echo "🚀 Starting VR180 Backend on port $PORT"

# Start the application
python -m uvicorn main:app --host 0.0.0.0 --port $PORT
