# 🎬 VR Platform - Complete Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Frontend Documentation](#frontend-documentation)
4. [Backend Documentation](#backend-documentation)
5. [2D to 3D Video Conversion Process](#2d-to-3d-video-conversion-process)
6. [Deployment](#deployment)
7. [API Endpoints](#api-endpoints)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

The VR Platform is a comprehensive web application that converts 2D videos into immersive VR180 format videos. The platform provides a user-friendly interface for uploading videos and processing them through an advanced AI-powered pipeline to create stereoscopic 3D content suitable for VR headsets.

### Key Features
- **2D to VR180 Conversion**: Transform regular videos into immersive VR180 format
- **Real-time Processing**: Live progress tracking during video conversion
- **High-Quality Output**: Maintains video quality while adding depth information
- **Web-based Interface**: Easy-to-use frontend for video upload and management
- **Scalable Backend**: Dockerized backend with 4GB RAM for efficient processing

---

## 🏗️ Architecture

### System Architecture
```
┌─────────────────┐    HTTPS     ┌─────────────────┐    HTTP     ┌─────────────────┐
│   Frontend      │ ────────────► │   Nginx Proxy   │ ──────────► │   Backend API   │
│   (Vercel)      │              │   (UpCloud)     │            │   (UpCloud)     │
└─────────────────┘              └─────────────────┘            └─────────────────┘
        │                                                               │
        │                                                               ▼
        │                                                        ┌─────────────────┐
        │                                                        │   Video         │
        │                                                        │   Processing    │
        │                                                        │   Pipeline      │
        │                                                        └─────────────────┘
        │                                                               │
        │                                                               ▼
        │                                                        ┌─────────────────┐
        │                                                        │   Output        │
        │                                                        │   VR180 Video   │
        │                                                        └─────────────────┘
```

### Technology Stack

#### Frontend
- **Framework**: Next.js 14.2.16
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Radix UI
- **Deployment**: Vercel
- **State Management**: React Hooks

#### Backend
- **Framework**: FastAPI
- **Language**: Python 3.11+
- **Image Processing**: OpenCV, PIL
- **Video Processing**: FFmpeg
- **AI/ML**: NumPy, OpenCV depth estimation
- **Deployment**: Docker on UpCloud
- **Reverse Proxy**: Nginx

---

## 🎨 Frontend Documentation

### Project Structure
```
frontend/
├── app/                    # Next.js app directory
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout component
│   ├── page.tsx           # Home page
│   ├── preview/           # Video preview page
│   ├── projects/          # Projects management page
│   └── upload/            # Video upload page
├── components/            # Reusable components
│   ├── ui/               # Base UI components (Radix UI)
│   ├── 3d-background.tsx # 3D background component
│   ├── header.tsx        # Navigation header
│   ├── sidebar.tsx       # Sidebar navigation
│   ├── upload-interface.tsx # Video upload interface
│   ├── vr-preview.tsx    # VR video preview component
│   └── processing-dashboard.tsx # Processing status dashboard
├── hooks/                # Custom React hooks
├── lib/                  # Utility functions
└── public/              # Static assets
```

### Key Components

#### 1. Upload Interface (`upload-interface.tsx`)
- **Purpose**: Handles video file uploads
- **Features**:
  - Drag & drop file upload
  - File validation (size, format)
  - Progress tracking
  - Error handling

#### 2. Processing Dashboard (`processing-dashboard.tsx`)
- **Purpose**: Real-time processing status display
- **Features**:
  - Live progress updates
  - Stage-by-stage tracking
  - Error reporting
  - Download links

#### 3. VR Preview (`vr-preview.tsx`)
- **Purpose**: Preview converted VR180 videos
- **Features**:
  - Side-by-side video display
  - VR headset compatibility
  - Playback controls

### Environment Variables
```bash
# Frontend Environment Variables
NEXT_PUBLIC_API_URL=https://213.163.196.177  # Backend API URL
NEXT_PUBLIC_APP_NAME=VR Platform
NEXT_PUBLIC_APP_VERSION=1.0.0
```

---

## ⚙️ Backend Documentation

### Project Structure
```
backend/
├── api/                   # Main API directory
│   ├── main.py           # FastAPI application
│   ├── pipeline.py       # Core video processing pipeline
│   ├── models.py         # AI/ML models and depth estimation
│   ├── video_io.py       # Video input/output operations
│   ├── flow_utils.py     # Optical flow utilities
│   ├── inpaint_api.py    # Inpainting functionality
│   ├── status.py         # Status management
│   ├── config.py         # Configuration settings
│   ├── worker.py         # Background processing worker
│   ├── requirements.txt  # Python dependencies
│   └── Dockerfile        # Docker configuration
├── tests/                # Test files
└── README.md            # Backend documentation
```

### Core Modules

#### 1. Main API (`main.py`)
- **Framework**: FastAPI
- **Features**:
  - CORS configuration
  - File upload handling
  - WebSocket support for real-time updates
  - Health check endpoints
  - Background job processing

#### 2. Video Processing Pipeline (`pipeline.py`)
- **Purpose**: Orchestrates the entire 2D to VR180 conversion process
- **Key Functions**:
  - Frame extraction
  - Depth estimation
  - Temporal smoothing
  - LDI (Layered Depth Image) generation
  - Inpainting
  - Final video encoding

#### 3. Depth Estimation (`models.py`)
- **Purpose**: Generates depth maps from 2D images
- **Algorithm**: OpenCV-based gradient magnitude + center bias
- **Features**:
  - Fast processing
  - High-quality depth maps
  - Memory efficient
  - Fallback mechanisms

---

## 🎬 2D to 3D Video Conversion Process

### Step-by-Step Conversion Pipeline

#### Stage 1: Video Analysis & Frame Extraction (0-25%)
```python
# Extract frames from input video
frames = extract_frames(video_path, target_fps=30)
# Expected: 1000-2000 frames for 1-minute video
```

**Process:**
1. **Video Probing**: Analyze video properties (fps, duration, resolution)
2. **Frame Extraction**: Extract frames at target FPS using FFmpeg
3. **Quality Settings**: High-quality extraction with minimal compression
4. **Memory Management**: Process frames in chunks to manage memory

#### Stage 2: Depth Estimation (25-50%)
```python
# Generate depth maps for each frame
for frame in frames:
    depth_map = create_local_depth_map(frame)
    # Algorithm: Gradient magnitude + center bias
```

**Algorithm Details:**
1. **Gradient Calculation**: 
   - Compute Sobel gradients (X and Y directions)
   - Calculate gradient magnitude: `√(grad_x² + grad_y²)`
2. **Center Bias**: 
   - Add realistic depth falloff from center
   - Formula: `1.0 - √((x-center_x)² + (y-center_y)²) / max_distance`
3. **Combination**: 
   - Blend gradient (70%) and center bias (30%)
   - Apply Gaussian blur for smoothness
4. **Normalization**: 
   - Normalize to 0-1 range
   - Handle edge cases and outliers

#### Stage 3: Temporal Smoothing (50-65%)
```python
# Apply temporal smoothing to depth maps
smoothed_depths = temporal_smoothing(depth_maps, window_size=5)
```

**Process:**
1. **Temporal Window**: Apply smoothing across consecutive frames
2. **Consistency Check**: Ensure depth changes are gradual
3. **Edge Preservation**: Maintain sharp edges while smoothing
4. **Memory Optimization**: Process in batches

#### Stage 4: LDI Generation & Inpainting (65-85%)
```python
# Create Layered Depth Images
ldi = build_ldi_and_reproject(frame, depth_map, baseline_ratio=0.02)
# Inpaint missing regions
inpainted = sd_inpaint(ldi, mask)
```

**LDI Process:**
1. **Layered Depth Image**: Create multiple depth layers
2. **Reprojection**: Project pixels to different viewpoints
3. **Baseline Calculation**: Use 2% of image width as stereo baseline
4. **Inpainting**: Fill missing regions using AI inpainting

#### Stage 5: Final Encoding (85-100%)
```python
# Create side-by-side VR180 video
output_video = create_side_by_side(left_view, right_view, audio_track)
```

**Encoding Process:**
1. **Side-by-Side Layout**: Arrange left and right views
2. **Audio Preservation**: Maintain original audio track
3. **Quality Settings**: High bitrate for VR quality
4. **Format**: MP4 with H.264 codec

### Technical Parameters

#### Processing Settings
```python
DOWNSCALE_WIDTH = 1280        # Target resolution width
KEYFRAME_STRIDE = 1           # Process every frame
BASELINE_RATIO = 0.02         # 2% of image width
LDI_LAYERS = 3                # Number of depth layers
```

#### Memory Management
- **Chunk Processing**: Process 10 frames at a time
- **Garbage Collection**: Aggressive cleanup between stages
- **Memory Monitoring**: Track usage to prevent OOM errors

---

## 🚀 Deployment

### Frontend Deployment (Vercel)

#### Configuration
```bash
# Vercel Environment Variables
NEXT_PUBLIC_API_URL=https://213.163.196.177
NEXT_PUBLIC_APP_NAME=VR Platform
NEXT_PUBLIC_APP_VERSION=1.0.0
```

#### Deployment Steps
1. **Connect Repository**: Link GitHub repository to Vercel
2. **Configure Environment**: Set environment variables
3. **Build Settings**: Use default Next.js build settings
4. **Deploy**: Automatic deployment on git push

### Backend Deployment (UpCloud)

#### Server Specifications
- **Provider**: UpCloud
- **Instance**: 1 vCPU, 4GB RAM, 40GB SSD
- **OS**: Ubuntu 24.04 LTS
- **Location**: Singapore (SG-SIN1)

#### Infrastructure Setup
```bash
# 1. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 2. Install Nginx
apt install nginx -y

# 3. Generate SSL Certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/ssl-cert-snakeoil.key \
    -out /etc/ssl/certs/ssl-cert-snakeoil.pem \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=213.163.196.177"
```

#### Nginx Configuration
```nginx
server {
    listen 443 ssl;
    server_name 213.163.196.177;
    
    ssl_certificate /etc/ssl/certs/ssl-cert-snakeoil.pem;
    ssl_certificate_key /etc/ssl/private/ssl-cert-snakeoil.key;
    
    location / {
        proxy_pass http://localhost:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Docker Deployment
```bash
# Build and run container
docker build -t vr180-backend ./backend/api
docker run -d \
  --name vr180-backend-container \
  -p 0.0.0.0:80:8000 \
  -e PORT=8000 \
  -e WORKSPACE_DIR=/app/workspace \
  -v /app/workspace:/app/workspace \
  vr180-backend
```

---

## 🔌 API Endpoints

### Core Endpoints

#### Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "message": "VR180 Backend is running",
  "timestamp": "2025-09-07T16:35:05",
  "uptime": "active"
}
```

#### Video Upload
```http
POST /upload
Content-Type: multipart/form-data
```
**Request:** Video file upload
**Response:**
```json
{
  "job_id": "abc123def456"
}
```

#### Status Check
```http
GET /status/{job_id}
```
**Response:**
```json
{
  "status": "running",
  "stage": "depth_estimation",
  "percent": 45,
  "message": "Estimating depth maps... 45%",
  "updated_at": "2025-09-07T16:40:00Z"
}
```

#### Job List
```http
GET /jobs
```
**Response:**
```json
{
  "jobs": [
    {
      "job_id": "abc123def456",
      "status": "done",
      "stage": "finished",
      "percent": 100,
      "output": "/app/workspace/abc123def456/abc123def456_sbs.mp4",
      "file_size": 15728640,
      "created_at": "2025-09-07T16:30:00Z"
    }
  ]
}
```

#### Video Download
```http
GET /download/{job_id}
```
**Response:** Video file stream

### WebSocket Endpoints

#### Real-time Status
```javascript
const ws = new WebSocket('wss://213.163.196.177/ws/{job_id}');
ws.onmessage = (event) => {
  const status = JSON.parse(event.data);
  console.log('Progress:', status.percent + '%');
};
```

---

## ⚙️ Configuration

### Environment Variables

#### Backend Configuration
```bash
# Processing Settings
DOWNSCALE_WIDTH=1280          # Target resolution
KEYFRAME_STRIDE=1             # Frame processing stride
BASELINE_RATIO=0.02           # Stereo baseline ratio
LDI_LAYERS=3                  # Depth layers

# Limits
MAX_UPLOAD_BYTES=209715200    # 200MB upload limit
MAX_DURATION_SEC=300          # 5 minutes duration limit

# Paths
WORKSPACE_DIR=/app/workspace  # Processing directory
PORT=8000                     # API port

# External APIs (Optional)
HUGGINGFACE_TOKEN=your_token
REPLICATE_API_TOKEN=your_token
```

#### Frontend Configuration
```bash
NEXT_PUBLIC_API_URL=https://213.163.196.177
NEXT_PUBLIC_APP_NAME=VR Platform
NEXT_PUBLIC_APP_VERSION=1.0.0
```

### Docker Configuration

#### Dockerfile
```dockerfile
FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create workspace directory
RUN mkdir -p /app/workspace

# Start command
CMD python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "Failed to fetch" Error
**Cause:** Mixed content (HTTPS frontend → HTTP backend)
**Solution:** Use HTTPS backend with nginx reverse proxy

#### 2. Memory Issues
**Cause:** Insufficient RAM for video processing
**Solution:** Use 4GB+ RAM server (UpCloud instead of Railway)

#### 3. CORS Errors
**Cause:** Duplicate CORS headers
**Solution:** Remove CORS headers from nginx, let FastAPI handle them

#### 4. SSL Certificate Warnings
**Cause:** Self-signed certificate
**Solution:** Accept browser warning or use Let's Encrypt for production

### Performance Optimization

#### Memory Management
```python
# Process frames in small chunks
chunk_size = 10
for i in range(0, len(frames), chunk_size):
    chunk = frames[i:i+chunk_size]
    process_chunk(chunk)
    del chunk
    gc.collect()
```

#### Processing Speed
- **Batch Processing**: Process multiple frames simultaneously
- **Memory Cleanup**: Aggressive garbage collection
- **Optimized Algorithms**: Use efficient OpenCV operations

### Monitoring

#### Container Health
```bash
# Check container status
docker ps

# Monitor logs
docker logs -f vr180-backend-container

# Check resource usage
docker stats vr180-backend-container
```

#### System Resources
```bash
# Memory usage
free -h

# CPU usage
top

# Disk space
df -h
```

---

## 📊 Performance Metrics

### Processing Times (4GB RAM Server)
- **1-minute video (1080p)**: ~10-15 minutes
- **2-minute video (1080p)**: ~20-30 minutes
- **5-minute video (1080p)**: ~50-75 minutes

### Memory Usage
- **Peak Memory**: 2.5-3.5GB
- **Average Memory**: 2-2.5GB
- **Minimum Required**: 2GB

### Quality Settings
- **Input Resolution**: Up to 4K
- **Output Resolution**: 1280x720 (VR180)
- **Frame Rate**: 30 FPS
- **Bitrate**: High quality (varies by content)

---

## 🎯 Future Enhancements

### Planned Features
1. **Real-time Processing**: WebRTC-based live conversion
2. **Batch Processing**: Multiple video processing
3. **Cloud Storage**: Direct cloud upload/download
4. **Advanced AI**: Better depth estimation models
5. **VR Preview**: In-browser VR preview
6. **Mobile App**: React Native mobile application

### Technical Improvements
1. **GPU Acceleration**: CUDA support for faster processing
2. **Distributed Processing**: Multi-server processing
3. **Caching**: Redis-based result caching
4. **Monitoring**: Advanced logging and metrics
5. **Auto-scaling**: Dynamic resource allocation

---

## 📝 License

This project is proprietary software. All rights reserved.

---

## 👥 Contributing

For development and contribution guidelines, please contact the development team.

---

## 📞 Support

For technical support or questions:
- **Documentation**: This file
- **Issues**: GitHub Issues
- **Contact**: Development Team

---

*Last Updated: September 7, 2025*
*Version: 1.0.0*
