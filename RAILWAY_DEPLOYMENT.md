# Railway Deployment Guide

## 🚂 Deploy Backend to Railway

### Step 1: Connect to Railway

1. Go to [railway.app](https://railway.app)
2. Sign up/Login with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your repository: `si451/VR-GENERATION-`

### Step 2: Configure Service

1. **Root Directory**: `backend/api`
2. **Build Command**: Leave empty (handled by Dockerfile)
3. **Start Command**: Leave empty (handled by Dockerfile)

**Note**: The `Dockerfile` automatically installs FFmpeg and handles the build process. The requirements.txt uses `opencv-python-headless` to avoid GUI library dependencies.

### Step 2.1: Dockerfile Configuration

The `Dockerfile` in the backend/api directory configures Railway's build process:

```dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .

ENV PORT=8080
ENV WORKSPACE_DIR=/app/workspace
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

This ensures FFmpeg is installed via apt-get during the build phase.

### Step 3: Environment Variables

Add these in Railway dashboard:

```
HUGGINGFACE_TOKEN=your_huggingface_token_here
HF_API_URL=https://api-inference.huggingface.co/models
MODEL_DEPTH=Intel/dpt-hybrid-midas
MODEL_INPAINT=stabilityai/stable-diffusion-2-inpainting
MODEL_FLOW=ddrfan/RAFT
MODEL_INTERPOLATION=hzwer/RIFE
DOWNSCALE_WIDTH=1280
MAX_DURATION_SEC=300
WORKSPACE_DIR=/app/workspace
PORT=8000
```

### Step 4: Deploy

Railway will automatically deploy your service. You'll get a URL like:
`https://your-app-name-production.up.railway.app`

## 🎯 Frontend Configuration

### In Vercel:

1. Go to your Vercel project settings
2. Add environment variable:
   ```
   NEXT_PUBLIC_API_URL=https://your-app-name-production.up.railway.app
   ```
3. Redeploy your frontend

## ✅ Benefits of Railway:

- ✅ **No sleep timeout** - service stays awake 24/7
- ✅ **No keep-alive needed** - services are persistent
- ✅ **Better for video processing** - no time limits
- ✅ **Easy deployment** - connects directly to GitHub
- ✅ **Automatic HTTPS** - secure by default
- ✅ **$5 monthly credit** - usually sufficient for small projects

## 🔧 Testing:

Visit your Railway URL: `https://your-app-name-production.up.railway.app/health`

You should see:
```json
{
  "status": "healthy",
  "message": "VR180 Backend is running",
  "timestamp": "2024-01-15 14:30:00",
  "uptime": "active"
}
```

## 💡 Pro Tips:

- Railway automatically handles scaling
- No need for keep-alive mechanisms
- Better performance than Render free tier
- Perfect for long-running video processing tasks
