@echo off
REM Railway startup script for VR180 Backend (Windows)

echo 🚀 Starting VR180 Backend...
echo Current directory: %CD%
echo Python version:
python --version
echo Available files:
dir

REM Install dependencies if needed
if not exist ".venv" (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Start the FastAPI server
echo Starting FastAPI server...
python -m uvicorn api.main:app --host 0.0.0.0 --port %PORT%

