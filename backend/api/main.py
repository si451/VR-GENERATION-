# api/main.py
import uvicorn
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uuid
import asyncio
import subprocess
import sys
import threading
import time
import requests
# from .config import WORKSPACE_DIR, MAX_UPLOAD_BYTES
from status import StatusManager
from video_io import probe_video
import os
ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024))) 
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", str(ROOT.parent / "workspace")))
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="VR180 Backend")

# Check FFmpeg availability on startup
@app.on_event("startup")
async def startup_event():
    """Check system dependencies on startup"""
    import shutil
    import glob
    
    print("🔧 DEBUG: Startup event triggered")
    print(f"🔧 DEBUG: PORT environment variable: {os.getenv('PORT', 'NOT SET')}")
    print(f"🔧 DEBUG: Current working directory: {os.getcwd()}")
    print("🔧 Checking system dependencies...")
    
    # Check FFmpeg with comprehensive search
    ffmpeg_path = None
    
    # Try which command first
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"✅ FFmpeg found via which: {ffmpeg_path}")
    else:
        print("❌ FFmpeg not found in PATH")
        
        # Search in common locations
        search_paths = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg"
        ]
        
        # Search in Nix store
        nix_paths = glob.glob("/nix/store/*/bin/ffmpeg")
        search_paths.extend(nix_paths)
        
        for path in search_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                ffmpeg_path = path
                print(f"✅ FFmpeg found at: {path}")
                break
    
    if ffmpeg_path:
        try:
            result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"✅ FFmpeg version: {version_line}")
            else:
                print(f"⚠️ FFmpeg found but not working: {result.stderr}")
        except Exception as e:
            print(f"⚠️ FFmpeg check failed: {e}")
    else:
        print("❌ FFmpeg not found in any location")
        print("Available binaries:")
        for path in ["/usr/bin", "/usr/local/bin"]:
            if os.path.exists(path):
                print(f"  {path}: {os.listdir(path)[:5]}...")
        
        # Check Nix store
        nix_bin_dirs = glob.glob("/nix/store/*/bin")
        if nix_bin_dirs:
            print(f"  Nix store bin directories found: {len(nix_bin_dirs)}")
            for bin_dir in nix_bin_dirs[:3]:  # Show first 3
                try:
                    files = os.listdir(bin_dir)
                    ffmpeg_files = [f for f in files if 'ffmpeg' in f.lower()]
                    if ffmpeg_files:
                        print(f"    {bin_dir}: {ffmpeg_files}")
                except:
                    pass
    
    print("🚀 VR180 Backend startup completed")

# Add custom CORS middleware
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    
    # Add CORS headers to all responses
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, HEAD"
    response.headers["Access-Control-Allow-Headers"] = "Range, Content-Range, Content-Length, Content-Type, Authorization"
    response.headers["Access-Control-Expose-Headers"] = "Content-Range, Accept-Ranges, Content-Length"
    
    return response

# Also add the standard CORS middleware as backup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

status_mgr = StatusManager(WORKSPACE_DIR)
connections = {}  # job_id -> set of websockets

# # Keep-alive mechanism for free tier
# def keep_alive_ping():
#     """Ping the server every 10 minutes to prevent it from sleeping"""
#     while True:
#         try:
#             # Get the server URL from environment or use localhost
#             port = os.getenv("PORT", "8000")
#             server_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{port}")
            
#             # Ping the health endpoint
#             response = requests.get(f"{server_url}/health", timeout=10)
#             if response.status_code == 200:
#                 print(f"✅ Keep-alive ping successful: {time.strftime('%Y-%m-%d %H:%M:%S')}")
#             else:
#                 print(f"⚠️ Keep-alive ping failed with status: {response.status_code}")
                
#         except Exception as e:
#             print(f"❌ Keep-alive ping error: {e}")
        
#         # Wait 10 minutes (600 seconds) before next ping
#         time.sleep(600)

# def internal_keep_alive():
#     """Internal keep-alive that runs every 5 minutes to keep the process active"""
#     while True:
#         try:
#             # Just log that we're alive
#             print(f"💓 Internal keep-alive: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
#             # Do some lightweight work to keep the process active
#             _ = len(str(time.time()))
            
#         except Exception as e:
#             print(f"❌ Internal keep-alive error: {e}")
        
#         # Wait 5 minutes (300 seconds) before next internal ping
#         time.sleep(300)

# # Start keep-alive threads
# keep_alive_thread = threading.Thread(target=keep_alive_ping, daemon=True)
# keep_alive_thread.start()

# internal_keep_alive_thread = threading.Thread(target=internal_keep_alive, daemon=True)
# internal_keep_alive_thread.start()

print("🔄 Keep-alive mechanisms started:")
print("   - External ping every 10 minutes")
print("   - Internal keep-alive every 5 minutes")

@app.on_event("startup")
async def startup_event():
    """Startup event to ensure keep-alive is running"""
    print("🚀 VR180 Backend started successfully")
    print("💡 Ready to process VR180 videos")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "VR180 Backend is running",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "uptime": "active"
    }

@app.get("/debug/files")
async def debug_files():
    """Debug endpoint to check file structure"""
    import os
    from pathlib import Path
    
    current_dir = os.getcwd()
    root_dir = ROOT
    
    # Check for worker.py in various locations
    worker_locations = {
        "current_dir": os.path.join(current_dir, "worker.py"),
        "current_dir_explicit": os.path.join(current_dir, "./worker.py"),
        "parent_dir": os.path.join(current_dir, "../worker.py"),
        "root_parent": str(ROOT.parent / "worker.py"),
        "api_dir": str(ROOT / "worker.py"),
        "railway_app": "/app/worker.py",
        "railway_backend": "/app/backend/worker.py"
    }
    
    results = {}
    for name, path in worker_locations.items():
        results[name] = {
            "path": path,
            "exists": os.path.exists(path),
            "is_file": os.path.isfile(path) if os.path.exists(path) else False
        }
    
    return {
        "current_working_directory": current_dir,
        "root_directory": str(root_dir),
        "directory_contents": os.listdir(current_dir),
        "worker_locations": results
    }

@app.get("/test-video/{job_id}")
async def test_video(job_id: str):
    """Test endpoint to check if video file exists and is accessible"""
    from pathlib import Path
    
    # First check if status has output path
    s = status_mgr.read(job_id)
    out = s.get("output")
    
    # If no output in status, look for the standard output file pattern
    if not out:
        job_dir = WORKSPACE_DIR / job_id
        if not job_dir.exists():
            return {"exists": False, "error": "Job directory not found"}
        
        # Look for files matching the pattern: {job_id}_sbs.mp4
        output_pattern = f"{job_id}_sbs.mp4"
        output_path = job_dir / output_pattern
        if not output_path.exists():
            return {"exists": False, "error": "Output file not found"}
    else:
        output_path = Path(out)
        if not output_path.exists():
            return {"exists": False, "error": "Output file not found"}
    
    file_size = output_path.stat().st_size
    return {
        "exists": True, 
        "path": str(output_path),
        "size": file_size,
        "download_url": f"http://localhost:{os.getenv('PORT', '8000')}/download/{job_id}"
    }

@app.get("/test-cors")
async def test_cors():
    """Test endpoint to verify CORS is working"""
    return {
        "message": "CORS test successful",
        "timestamp": "2024-01-01T00:00:00Z",
        "cors_headers": "Should be present in response"
    }

def save_upload_stream(upload: UploadFile, target: Path, size_limit: int = MAX_UPLOAD_BYTES):
    # synchronous write (called using run_in_executor)
    with open(target, "wb") as f:
        total = 0
        for chunk in iter(lambda: upload.file.read(1024*1024), b""):
            if not chunk:
                break
            total += len(chunk)
            if total > size_limit:
                raise HTTPException(status_code=413, detail="File too large")
            f.write(chunk)
    return total

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex
    job_dir = WORKSPACE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / "input.mp4"
    loop = asyncio.get_running_loop()
    
    # stream save
    await loop.run_in_executor(None, save_upload_stream, file, input_path)
    
    # quick probe and limit
    try:
        meta = probe_video(input_path)
        duration = float(meta["format"].get("duration", 0.0))
        if duration > float(os.getenv("MAX_DURATION_SEC", "300")):  # 5 minutes limit
            status_mgr.update(job_id, {"status":"failed", "message":"video too long"})
            return {"job_id": job_id, "status": "rejected", "message": "duration exceeds limit"}
    except Exception:
        pass
    
    status_mgr.update(job_id, {"status":"queued", "stage":"queued"})
    
    print(f"Starting worker for job {job_id} with input: {input_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"ROOT directory: {ROOT}")
    
    # Start worker process instead of processing in server
    def start_worker():
        try:
            # Start worker process and capture output
            # Try multiple possible paths for worker.py
            worker_paths = [
                "worker.py",     # Same directory (Railway runs from backend/)
                
                
            ]
            
            print(f"Searching for worker.py in the following paths:")
            for path in worker_paths:
                exists = os.path.exists(path)
                print(f"  {path}: {'EXISTS' if exists else 'NOT FOUND'}")
            
            worker_cmd = None
            for worker_path in worker_paths:
                if os.path.exists(worker_path):
                    worker_cmd = [sys.executable, worker_path, job_id, str(input_path)]
                    print(f"✅ Found worker.py at: {worker_path}")
                    break
            
            if not worker_cmd:
                print("❌ ERROR: worker.py not found in any expected location")
                print(f"Current working directory: {os.getcwd()}")
                print(f"Directory contents: {os.listdir('.')}")
                status_mgr.update(job_id, {"status":"failed", "message":"Worker script not found"})
                return
            
            process = subprocess.Popen(worker_cmd, 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True,
                                    cwd=os.getcwd())
            
            # Start a thread to read worker output and update status
            import threading
            def read_worker_output():
                try:
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            line = line.strip()
                            print(f"Worker output: {line}")
                            
                            # Parse worker output and update status
                            if "Starting worker with job ID:" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"starting", "percent":1, "message":"Starting video processing..."})
                            elif "Processing:" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"starting", "percent":2, "message":"Processing video file..."})
                            elif "Copied input to:" in line or "Input already in job directory:" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"starting", "percent":3, "message":"Preparing video for processing..."})
                            elif "FFprobe detected:" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"probe_video", "percent":5, "message":"Analyzing video properties..."})
                            elif "Using FPS:" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"probe_video", "percent":7, "message":"Detecting frame rate..."})
                            elif "Video duration:" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"probe_video", "percent":8, "message":"Calculating video duration..."})
                            elif "Expected total frames:" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"probe_video", "percent":10, "message":"Calculating total frames..."})
                            elif "Extracting frames with high quality settings..." in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"extract_frames", "percent":15, "message":"Extracting video frames..."})
                            elif "Extracting frames:" in line and "%" in line:
                                # Extract progress from tqdm output like "Extracting frames: 45%|████▌     | 647/1437 [00:08<00:10, 73.28frame/s]"
                                import re
                                progress_match = re.search(r'(\d+)%', line)
                                if progress_match:
                                    progress = int(progress_match.group(1))
                                    # Only update every 10% to reduce file locking issues
                                    if progress % 10 == 0 or progress >= 95:
                                        # Map 0-100% to 15-25% of total progress
                                        total_progress = 15 + int(progress * 0.1)
                                        status_mgr.update(job_id, {"status":"running", "stage":"extract_frames", "percent":total_progress, "message":f"Extracting frames... {progress}%"})
                            elif "Extracted" in line and "frames successfully" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"extract_frames", "percent":25, "message":"Frame extraction completed"})
                            elif "Total frames extracted:" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"extract_frames", "percent":27, "message":"Frame extraction verified"})
                            elif "Starting depth estimation" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"depth_estimation", "percent":30, "message":"Starting depth estimation..."})
                            elif "Depth estimation:" in line and "%" in line:
                                # Extract progress from tqdm output
                                import re
                                progress_match = re.search(r'(\d+)%', line)
                                if progress_match:
                                    progress = int(progress_match.group(1))
                                    # Only update every 20% to reduce file locking issues
                                    if progress % 20 == 0 or progress >= 95:
                                        # Map 0-100% to 30-50% of total progress
                                        total_progress = 30 + int(progress * 0.2)
                                        status_mgr.update(job_id, {"status":"running", "stage":"depth_estimation", "percent":total_progress, "message":f"Estimating depth maps... {progress}%"})
                            elif "Depth estimation completed" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"depth_estimation", "percent":50, "message":"Depth estimation completed"})
                            elif "Starting temporal smoothing" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"temporal_smoothing", "percent":55, "message":"Starting temporal smoothing..."})
                            elif "Temporal smoothing:" in line and "%" in line:
                                # Extract progress from tqdm output
                                import re
                                progress_match = re.search(r'(\d+)%', line)
                                if progress_match:
                                    progress = int(progress_match.group(1))
                                    # Only update every 20% to reduce file locking issues
                                    if progress % 20 == 0 or progress >= 95:
                                        # Map 0-100% to 55-65% of total progress
                                        total_progress = 55 + int(progress * 0.1)
                                        status_mgr.update(job_id, {"status":"running", "stage":"temporal_smoothing", "percent":total_progress, "message":f"Applying temporal smoothing... {progress}%"})
                            elif "Temporal smoothing completed" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"temporal_smoothing", "percent":65, "message":"Temporal smoothing completed"})
                            elif "Starting LDI reprojection and inpainting" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"ldi_reprojection", "percent":70, "message":"Starting VR180 view creation..."})
                            elif "LDI & Inpainting:" in line and "%" in line:
                                # Extract progress from tqdm output
                                import re
                                progress_match = re.search(r'(\d+)%', line)
                                if progress_match:
                                    progress = int(progress_match.group(1))
                                    # Only update every 20% to reduce file locking issues
                                    if progress % 20 == 0 or progress >= 95:
                                        # Map 0-100% to 70-85% of total progress
                                        total_progress = 70 + int(progress * 0.15)
                                        status_mgr.update(job_id, {"status":"running", "stage":"ldi_reprojection", "percent":total_progress, "message":f"Creating VR180 views... {progress}%"})
                            elif "LDI reprojection and inpainting completed" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"ldi_reprojection", "percent":85, "message":"VR180 reprojection completed"})
                            elif "Creating final VR180 video" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"encode", "percent":90, "message":"Creating final VR180 video..."})
                            elif "Creating side-by-side video" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"encode", "percent":92, "message":"Creating side-by-side video..."})
                            elif "Encoding video with high quality settings..." in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"encode", "percent":95, "message":"Encoding final video..."})
                            elif "Side-by-side video created:" in line:
                                status_mgr.update(job_id, {"status":"running", "stage":"encode", "percent":98, "message":"Video encoding completed"})
                            elif "Output video created:" in line:
                                # Extract output path from the line
                                import re
                                path_match = re.search(r'Output video created: ([^(]+)', line)
                                if path_match:
                                    output_path = path_match.group(1).strip()
                                    status_mgr.update(job_id, {"status":"running", "stage":"encode", "percent":99, "message":"Final video created", "output": output_path})
                                else:
                                    status_mgr.update(job_id, {"status":"running", "stage":"encode", "percent":99, "message":"Final video created"})
                            elif "Job" in line and "completed successfully" in line:
                                status_mgr.update(job_id, {"status":"done", "stage":"finished", "percent":100, "message":"Video processing completed successfully!"})
                            elif "Results saved in:" in line:
                                status_mgr.update(job_id, {"status":"done", "stage":"finished", "percent":100, "message":"Processing completed! Results saved."})
                            elif "failed" in line.lower() or "error" in line.lower():
                                status_mgr.update(job_id, {"status":"failed", "message":line})
        
                except Exception as e:
                                            print(f"Error reading worker output: {e}")
                                            status_mgr.update(job_id, {"status":"failed", "message":f"Worker output error: {e}"})
            
            # Start the output reader thread
            output_thread = threading.Thread(target=read_worker_output)
            output_thread.daemon = True
            output_thread.start()
            
            print(f"Started worker process for job {job_id}")
        except Exception as e:
            print(f"Failed to start worker: {e}")
            status_mgr.update(job_id, {"status":"failed", "message":f"Worker start failed: {e}"})
    
    # Start worker in background
    await loop.run_in_executor(None, start_worker)
    
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    return status_mgr.read(job_id)

@app.get("/jobs")
async def list_jobs():
    """List all jobs in the workspace"""
    jobs = []
    for job_dir in WORKSPACE_DIR.iterdir():
        if job_dir.is_dir():
            status = status_mgr.read(job_dir.name)
            
            # Get input file info
            input_file = job_dir / "input.mp4"
            file_size = 0
            if input_file.exists():
                file_size = input_file.stat().st_size
            
            # Get output file info - look for the actual output video file
            output_file = None
            output_filename = None
            
            # First check if status has output path
            if status.get("output"):
                output_path = Path(status["output"])
                if output_path.exists():
                    output_file = str(output_path)
                    output_filename = output_path.name
            
            # If no output in status, look for the standard output file pattern
            if not output_file:
                # Look for files matching the pattern: {job_id}_sbs.mp4
                output_pattern = f"{job_dir.name}_sbs.mp4"
                output_path = job_dir / output_pattern
                if output_path.exists():
                    output_file = str(output_path)
                    output_filename = output_path.name
            
            jobs.append({
                "job_id": job_dir.name,
                "status": status.get("status", "unknown"),
                "stage": status.get("stage", "unknown"),
                "percent": status.get("percent", 0),
                "message": status.get("message", ""),
                "output": output_file,
                "output_filename": output_filename,
                "file_size": file_size,
                "created_at": status.get("updated_at", ""),
                "input_file": "input.mp4"
            })
    return {"jobs": jobs}

@app.get("/download/{job_id}")
async def download(job_id: str):
    from fastapi.responses import FileResponse
    from fastapi import Response
    
    # First check if status has output path
    s = status_mgr.read(job_id)
    out = s.get("output")
    
    # If no output in status, look for the standard output file pattern
    if not out:
        job_dir = WORKSPACE_DIR / job_id
        if not job_dir.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Look for files matching the pattern: {job_id}_sbs.mp4
        output_pattern = f"{job_id}_sbs.mp4"
        output_path = job_dir / output_pattern
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Output file not found")
    else:
        output_path = Path(out)
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Output file not found")
    
    # Create response with explicit CORS headers
    response = FileResponse(
        path=str(output_path),
        filename=output_path.name,
        media_type="video/mp4"
    )
    
    # Add explicit CORS headers
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Range, Content-Range, Content-Length, Content-Type"
    response.headers["Access-Control-Expose-Headers"] = "Content-Range, Accept-Ranges, Content-Length"
    response.headers["Accept-Ranges"] = "bytes"
    response.headers["Cache-Control"] = "public, max-age=3600"
    
    return response

@app.options("/download/{job_id}")
async def download_options(job_id: str):
    from fastapi.responses import Response
    
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Range, Content-Range, Content-Length, Content-Type"
    response.headers["Access-Control-Expose-Headers"] = "Content-Range, Accept-Ranges, Content-Length"
    response.headers["Accept-Ranges"] = "bytes"
    return response

@app.websocket("/ws/{job_id}")
async def ws_job(websocket: WebSocket, job_id: str):
    await websocket.accept()
    conns = connections.setdefault(job_id, set())
    conns.add(websocket)
    try:
        while True:
            await asyncio.sleep(1.0)
            # push status file periodically
            status = status_mgr.read(job_id)
            await websocket.send_json(status)
    except WebSocketDisconnect:
        conns.discard(websocket)

# Server startup is handled by start.sh script
