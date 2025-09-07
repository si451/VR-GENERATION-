# VR Platform - 2D to VR180 Video Converter

Transform your 2D movie clips into stunning VR 180 experiences with advanced AI technology. This platform uses cutting-edge depth estimation, temporal smoothing, and inpainting techniques to create high-quality VR180 videos.

## 🌟 Features

- **AI-Powered Depth Estimation**: Advanced depth mapping using Intel's DPT model with Guided Filter refinement
- **Temporal Smoothing**: Frame interpolation for smooth VR180 output using optical flow
- **High-Quality Inpainting**: Multi-scale context-aware inpainting to reduce seams and artifacts
- **Real-time Progress**: Live processing updates and status tracking
- **VR180 Metadata**: Proper VR180 format with stereo layout and spherical projection
- **Audio Preservation**: Original audio maintained in output videos
- **Batch Processing**: Optimized for faster video processing with memory management
- **Modern UI**: Beautiful glassmorphism design with responsive layout

## 🏗️ Project Structure

```
vr-platform/
├── frontend/              # Next.js React frontend
│   ├── app/              # Next.js app directory
│   ├── components/       # React components
│   │   ├── ui/          # Shadcn/ui components
│   │   ├── upload-interface.tsx
│   │   ├── processing-dashboard.tsx
│   │   ├── vr-preview.tsx
│   │   └── 3d-background.tsx
│   ├── lib/             # Utility functions
│   ├── public/          # Static assets
│   └── README.md        # Frontend documentation
├── backend/             # FastAPI Python backend
│   ├── api/            # API endpoints and core logic
│   │   ├── main.py     # FastAPI application
│   │   ├── pipeline.py # Video processing pipeline
│   │   ├── models.py   # AI model inference
│   │   ├── video_io.py # Video input/output operations
│   │   ├── inpaint_api.py # Inpainting operations
│   │   ├── flow_utils.py # Optical flow utilities
│   │   ├── status.py   # Status management
│   │   └── config.py   # Configuration
│   ├── workspace/      # Video processing workspace
│   ├── logs/          # Application logs
│   ├── install.py     # Automated installation script
│   └── README.md      # Backend documentation
└── README.md          # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (for backend)
- **Node.js 18+** (for frontend)
- **FFmpeg** (for video processing)
- **Hugging Face Token** (for AI models)

### FFmpeg Installation

- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or use `winget install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt install ffmpeg`

### 1. Clone the Repository

```bash
git clone <repository-url>
cd vr-platform
```

### 2. Backend Setup

```bash
cd backend

# Quick install (recommended)
python install.py

# Or manual install
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Hugging Face token
```

### 3. Frontend Setup

```bash
cd frontend
npm install

# Create environment file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

### 4. Run the Application

**Terminal 1 - Backend:**
```bash
cd backend/api
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### 5. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## ⚙️ Configuration

### Backend Environment Variables

Create `backend/.env`:

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

### Frontend Environment Variables

Create `frontend/.env.local`:

```bash
# For local development
NEXT_PUBLIC_API_URL=http://localhost:8000

# For production (replace with your deployed backend URL)
# NEXT_PUBLIC_API_URL=https://your-backend-domain.com
```

## 🎬 How It Works

### Video Processing Pipeline

1. **Frame Extraction**: Extract high-quality frames from input video
2. **Depth Estimation**: Generate depth maps using AI models with Guided Filter refinement
3. **Temporal Smoothing**: Apply optical flow-based temporal smoothing for consistency
4. **LDI Creation**: Build Layered Depth Images for stereo reprojection
5. **Inpainting**: Fill occluded areas using multi-scale context-aware inpainting
6. **Stereo Generation**: Create left and right eye views
7. **Video Assembly**: Combine stereo views with original audio and VR180 metadata

### AI Models Used

- **Depth Estimation**: Intel DPT (Dense Prediction Transformer)
- **Optical Flow**: RAFT (Recurrent All-Pairs Field Transforms)
- **Inpainting**: Stable Diffusion 2 Inpainting
- **Frame Interpolation**: RIFE (Real-Time Intermediate Flow Estimation)

## 📊 Performance

- **Processing Time**: 30-40 minutes for large videos (5+ minutes)
- **Output Quality**: High-quality VR180 with preserved audio
- **Memory Usage**: Optimized with batch processing and garbage collection
- **Supported Formats**: MP4, MOV, AVI (input) → MP4 VR180 (output)

## 🔧 API Endpoints

### Core Endpoints

- `POST /upload` - Upload video for processing
- `GET /status/{job_id}` - Get processing status
- `GET /download/{job_id}` - Download processed video
- `GET /jobs` - List all jobs

### Utility Endpoints

- `GET /test-cors` - Test CORS configuration
- `GET /test-video/{job_id}` - Test video streaming

## 🚀 Deployment

### Frontend Deployment (Vercel/Netlify)

1. Connect your repository to Vercel/Netlify
2. Set environment variable: `NEXT_PUBLIC_API_URL=https://your-backend-domain.com`
3. Deploy

### Backend Deployment (Railway/Heroku/DigitalOcean)

1. Set all environment variables in your deployment platform
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python -m uvicorn main:app --host 0.0.0.0 --port 8000`

### Docker Deployment (Optional)

```dockerfile
# Backend Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🛠️ Development

### Backend Development

```bash
cd backend
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend Development

```bash
cd frontend
npm run dev
```

### Testing

```bash
# Test backend dependencies
cd backend
python test_dependencies.py

# Test API endpoints
curl http://localhost:8000/test-cors
```

## 📝 Usage

1. **Upload Video**: Drag and drop or select a video file
2. **Processing**: Monitor real-time progress with detailed status updates
3. **Preview**: View your VR180 video in the built-in VR preview
4. **Download**: Download the processed VR180 video
5. **Manage**: View all your projects in the dashboard

## 🎯 Supported Video Formats

### Input Formats
- MP4, MOV, AVI, MKV
- Maximum duration: 5 minutes (configurable)
- Recommended resolution: 1080p or higher

### Output Format
- MP4 VR180 with stereo layout
- Original audio preserved
- VR180 metadata included
- Compatible with VR headsets

## 🔍 Troubleshooting

### Common Issues

1. **"Guided Filter not found"**
   - Install: `pip install opencv-contrib-python`

2. **"FFmpeg not found"**
   - Install FFmpeg and add to PATH

3. **"Module not found"**
   - Run: `cd backend && python install.py`

4. **CORS errors**
   - Check `NEXT_PUBLIC_API_URL` in frontend
   - Verify backend CORS configuration

5. **Processing fails**
   - Check Hugging Face token
   - Verify video format and size
   - Check backend logs

### Performance Optimization

- Use SSD storage for faster I/O
- Ensure sufficient RAM (8GB+ recommended)
- Close other applications during processing
- Use shorter videos for testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Intel DPT** for depth estimation
- **RAFT** for optical flow
- **Stable Diffusion** for inpainting
- **RIFE** for frame interpolation
- **FastAPI** and **Next.js** communities

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `backend/logs/`
3. Open an issue on GitHub
4. Check the API documentation at `/docs`

---

**Made with ❤️ for the VR community**
#   V R - G E N E R A T I O N -  
 