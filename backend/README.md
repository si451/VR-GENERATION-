# VR Platform Backend

## Environment Variables

Create a `.env` file in the backend directory with the following configuration:

```bash
# Hugging Face API Configuration
HUGGINGFACE_TOKEN=your_huggingface_token_here

# API Configuration
HF_API_URL=https://api-inference.huggingface.co/models

# Model Configuration
MODEL_DEPTH=Intel/dpt-hybrid-midas
MODEL_INPAINT=stabilityai/stable-diffusion-2-inpainting
MODEL_FLOW=ddrfan/RAFT
MODEL_INTERPOLATION=hzwer/RIFE

# Processing Configuration
DOWNSCALE_WIDTH=1280
MAX_DURATION_SEC=300
```

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (for video processing)

### FFmpeg Installation
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `winget install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt install ffmpeg`

### Quick Install (Recommended)
```bash
# Run the automated installation script
python install.py

# Or on Windows:
install.bat
# Or PowerShell:
.\install.ps1
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file (copy from .env.example and update with your tokens)
cp .env.example .env
```

## Development

```bash
# Run the development server
cd api
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Production Deployment

1. Set all environment variables in your deployment platform
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `python -m uvicorn main:app --host 0.0.0.0 --port 8000`

## API Endpoints

- `POST /upload` - Upload video for processing
- `GET /status/{job_id}` - Get processing status
- `GET /download/{job_id}` - Download processed video
- `GET /jobs` - List all jobs
- `GET /test-cors` - Test CORS configuration

## CORS Configuration

The backend is configured to accept requests from:
- `http://localhost:3000` (local development)
- Your deployed frontend domain (configure in CORS middleware)
