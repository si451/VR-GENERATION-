# api/config.py
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", str(ROOT.parent / "workspace")))
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Set your HF token in environment variables
HF_API_URL = "https://api-inference.huggingface.co/models"

# Models (IDs you selected)
MODEL_DEPTH = os.getenv("MODEL_DEPTH", "Intel/dpt-hybrid-midas")
MODEL_INPAINT = os.getenv("MODEL_INPAINT", "stabilityai/stable-diffusion-2-inpainting")

# Processing defaults
DOWNSCALE_WIDTH = int(os.getenv("DOWNSCALE_WIDTH", "1280"))  # Higher resolution for better quality
KEYFRAME_STRIDE = int(os.getenv("KEYFRAME_STRIDE", "1"))
BASELINE_RATIO = float(os.getenv("BASELINE_RATIO", "0.02"))  # Reduced to minimize gaps
LDI_LAYERS = int(os.getenv("LDI_LAYERS", "3"))

# Limits
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))  # 200MB default
MAX_DURATION_SEC = int(os.getenv("MAX_DURATION_SEC", "300"))  # 5 minutes limit

# FFmpeg
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
