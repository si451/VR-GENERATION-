# api/pipeline.py
import asyncio
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import os
import subprocess
import time
from typing import List, Dict, Any
from tqdm import tqdm
import gc  # For garbage collection
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# from config import WORKSPACE_DIR, DOWNSCALE_WIDTH, KEYFRAME_STRIDE, BASELINE_RATIO, LDI_LAYERS
from status import StatusManager
from models import depth_from_hf, create_local_depth_map
from video_io import probe_video
def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video information using FFprobe"""
    try:
        meta = probe_video(video_path)
        duration = float(meta["format"].get("duration", 0.0))
        fps = 30.0  # Default FPS
        
        # Try to get FPS from stream
        if "streams" in meta and len(meta["streams"]) > 0:
            stream = meta["streams"][0]
            if "r_frame_rate" in stream:
                fps_str = stream["r_frame_rate"]
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den) if float(den) != 0 else 30.0
                else:
                    fps = float(fps_str)
        
        frame_count = int(duration * fps)
        
        return {
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "width": int(stream.get("width", 1920)),
            "height": int(stream.get("height", 1080))
        }
    except Exception as e:
        print(f"Error getting video info: {e}")
        # Return default values
        return {
            "duration": 60.0,
            "fps": 30.0,
            "frame_count": 1800,
            "width": 1920,
            "height": 1080
        }
from inpaint_api import sd_inpaint
from video_io import extract_frames, create_side_by_side
from flow_utils import compute_flow_cv2, warp_image_with_flow, interpolate_occlusion_aware, forward_backward_consistency_mask
import math
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", str(ROOT.parent / "workspace")))
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
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
status_mgr = StatusManager(WORKSPACE_DIR)

def normalize_depth(d: np.ndarray) -> np.ndarray:
    d = d.astype(np.float32)
    mn = np.nanpercentile(d, 2)
    mx = np.nanpercentile(d, 98)
    d = np.clip(d, mn, mx)
    if mx - mn < 1e-6:
        return np.zeros_like(d)
    return (d - mn) / (mx - mn)

def build_ldi_and_reproject(frame_rgb: np.ndarray, depth_norm: np.ndarray, baseline_ratio: float, n_layers: int = 7):
    h, w = depth_norm.shape
    
    # Optimized depth layers for faster processing
    edges = np.linspace(0.0, 1.0, n_layers+1)
    x_coords = np.arange(w)
    y_coords = np.arange(h)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    baseline_px = baseline_ratio * w
    
    # Initialize accumulators with original image as base
    left_accum = frame_rgb.astype(np.float32).copy()
    right_accum = frame_rgb.astype(np.float32).copy()
    left_weight = np.ones((h, w), dtype=np.float32)
    right_weight = np.ones((h, w), dtype=np.float32)
    
    # Process each depth layer with improved depth-aware weighting
    for i in range(n_layers):
        lo, hi = edges[i], edges[i+1]
        mask = ((depth_norm >= lo) & (depth_norm <= hi)).astype(np.float32)
        
        if mask.sum() == 0:
            continue
            
        # Create masked layer with smooth blending
        layer_color = frame_rgb.astype(np.float32) * mask[..., None]
        layer_depth = depth_norm * mask
        
        # Optimized stereo shift calculation for faster processing
        depth_factor = 1.0 - layer_depth  # Linear depth mapping
        shift_x = baseline_px * depth_factor * 0.2  # Conservative shift for speed
        
        # Calculate left and right positions
        left_x = grid_x + shift_x
        right_x = grid_x - shift_x
        
        # Create hole masks (only for extreme cases)
        hole_left = (left_x < -w*0.1) | (left_x >= w*1.1)
        hole_right = (right_x < -w*0.1) | (right_x >= w*1.1)
        
        # Clip coordinates with padding to avoid edge artifacts
        left_xc = np.clip(left_x, -w*0.1, w*1.1).astype(np.float32)
        right_xc = np.clip(right_x, -w*0.1, w*1.1).astype(np.float32)
        
        # Create coordinate maps for remapping
        map_y = np.repeat(np.arange(h)[:, None], w, axis=1).astype(np.float32)
        
        # Optimized remapping with linear interpolation for speed
        remapped_left = np.zeros_like(layer_color)
        remapped_right = np.zeros_like(layer_color)
        
        for c in range(3):
            remapped_left[..., c] = cv2.remap(
                layer_color[..., c], left_xc, map_y, 
                interpolation=cv2.INTER_LINEAR,  # Linear interpolation for speed
                borderMode=cv2.BORDER_REFLECT_101
            )
            remapped_right[..., c] = cv2.remap(
                layer_color[..., c], right_xc, map_y, 
                interpolation=cv2.INTER_LINEAR,  # Linear interpolation for speed
                borderMode=cv2.BORDER_REFLECT_101
            )
        
        # Improved depth-aware weighting
        # Give more weight to layers with more significant depth changes
        depth_variance = np.var(layer_depth[mask > 0]) if mask.sum() > 0 else 0
        layer_weight = mask * (1.0 / n_layers) * (1.0 + depth_variance * 0.5)
        
        # Adaptive blending based on depth complexity
        blend_factor = 0.08 + depth_variance * 0.02  # Adaptive blending
        left_accum = (1 - blend_factor) * left_accum + blend_factor * remapped_left
        right_accum = (1 - blend_factor) * right_accum + blend_factor * remapped_right
        
        left_weight += layer_weight
        right_weight += layer_weight
    
    # Normalize by weights to avoid division by zero
    left_weight = np.maximum(left_weight, 1e-6)
    right_weight = np.maximum(right_weight, 1e-6)
    
    # Convert to uint8
    left_accum = np.clip(left_accum, 0, 255).astype(np.uint8)
    right_accum = np.clip(right_accum, 0, 255).astype(np.uint8)
    
    # Create hole masks for inpainting (only for very small areas)
    left_mask_acc = (left_weight < 0.05).astype(np.uint8) * 255
    right_mask_acc = (right_weight < 0.05).astype(np.uint8) * 255
    
    # Apply morphological operations to reduce small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    left_mask_acc = cv2.morphologyEx(left_mask_acc, cv2.MORPH_CLOSE, kernel)
    right_mask_acc = cv2.morphologyEx(right_mask_acc, cv2.MORPH_CLOSE, kernel)
    
    return left_accum, right_accum, left_mask_acc, right_mask_acc

async def process_job(job_id: str, input_path: Path, use_inpaint_sd: bool = True):
    job_dir = WORKSPACE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Create stage-specific directories for disk storage
    frames_dir = job_dir / "01_frames"
    depths_dir = job_dir / "02_depths"
    smoothed_dir = job_dir / "03_smoothed"
    left_dir = job_dir / "04_left"
    right_dir = job_dir / "05_right"
    
    for d in [frames_dir, depths_dir, smoothed_dir, left_dir, right_dir]:
        d.mkdir(parents=True, exist_ok=True)

    status_mgr.update(job_id, {"status":"running", "stage":"probe_video", "percent":1})
    # Probe and short-circuit if too long
    meta = probe_video(input_path)
    # find duration
    duration = float(meta["format"].get("duration", 0.0))
    if duration > float(os.getenv("MAX_DURATION_SEC", "300")):  # Increased to 5 minutes
        status_mgr.update(job_id, {"status":"failed", "message":"Video too long"})
        return
    # fps - get actual FPS from video with improved accuracy
    stream = next((s for s in meta["streams"] if s["codec_type"] == "video"), None)
    r_frame_rate = stream.get("r_frame_rate", "30/1")
    
    # Parse frame rate properly with better precision
    if '/' in r_frame_rate:
        numerator, denominator = r_frame_rate.split('/')
        fps = float(numerator) / float(denominator)
    else:
        fps = float(r_frame_rate)
    
    # Round to reasonable precision to avoid floating point issues
    fps = round(fps, 3)
    
    print(f"Using FPS: {fps:.3f} from {r_frame_rate}")
    print(f"Video duration: {duration:.2f}s")
    print(f"Expected total frames: {int(duration * fps)}")
    # Stage 0: Extract frames - START
    status_mgr.update(job_id, {
        "status":"running", 
        "stage":"extract_frames", 
        "percent":5,
        "message":"Starting frame extraction... (Estimated time: 2-3 minutes)",
        "stage_started": True
    })
    
    # Extract frames with tqdm progress
    print(f"Extracting frames from video...")
    with tqdm(desc="Extracting Frames", unit="frame") as pbar:
        await extract_frames(input_path, frames_dir, downscale_width=DOWNSCALE_WIDTH, progress_callback=lambda c, t: pbar.update(1) if c > 0 else None)
    
    # Get list of extracted frames
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    n_frames = len(frame_files)
    print(f"Total frames extracted: {n_frames}")
    
    # Stage 0: Extract frames - END
    status_mgr.update(job_id, {
        "status":"running", 
        "stage":"extract_frames", 
        "percent":10,
        "message":"Frame extraction completed. Starting depth estimation...",
        "stage_completed": True
    })
    # Stage 1: Depth Estimation - START
    status_mgr.update(job_id, {
        "status":"running", 
        "stage":"depth_estimation", 
        "percent":10, 
        "frames": n_frames,
        "message":"Starting depth estimation... (Estimated time: 15-20 minutes)",
        "stage_started": True
    })
    print(f"Starting depth estimation for {n_frames} frames...")
    
    # Process frames one by one and save to disk
    with tqdm(total=n_frames, desc="Depth Estimation", unit="frame", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        for i, frame_path in enumerate(frame_files):
            try:
                # Load frame
                img = Image.open(frame_path).convert("RGB")
                if img.size[0] > DOWNSCALE_WIDTH:
                    img = img.resize((DOWNSCALE_WIDTH, int(img.size[1] * DOWNSCALE_WIDTH / img.size[0])), Image.Resampling.LANCZOS)
                
                # Estimate depth
                depth_arr = create_local_depth_map(img)
                depth_arr = normalize_depth(depth_arr)
                
                # Save depth map to disk
                depth_path = depths_dir / f"depth_{i:06d}.npy"
                np.save(depth_path, depth_arr)
                
                # Update progress
                pbar.update(1)
                
                # Clean up memory
                del img, depth_arr
                gc.collect()
                        
            except Exception as e:
                print(f"❌ Depth estimation failed for frame {i}: {e}")
                # Create dummy depth map
                dummy_depth = np.ones((DOWNSCALE_WIDTH//2, DOWNSCALE_WIDTH//2), dtype=np.float32) * 0.5
                depth_path = depths_dir / f"depth_{i:06d}.npy"
                np.save(depth_path, dummy_depth)
                pbar.update(1)
    
    print(f"Depth estimation completed for {n_frames} frames")
    # Free memory after depth estimation
    gc.collect()
    
    # Stage 1: Depth Estimation - END
    status_mgr.update(job_id, {
        "status":"running", 
        "stage":"depth_estimation", 
        "percent":50, 
        "message":"Depth estimation completed. Starting temporal smoothing...",
        "stage_completed": True
    })
    
    # Stage 2: Temporal Smoothing - START
    status_mgr.update(job_id, {
        "status":"running", 
        "stage":"temporal_smoothing", 
        "percent":52,
        "message":"Starting temporal smoothing... (Estimated time: 5-8 minutes)",
        "stage_started": True
    })
    print(f"Starting temporal smoothing...")
    
    with tqdm(total=n_frames, desc="Temporal Smoothing", unit="frame",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for i in range(n_frames):
            # Load current depth from disk
            depth_path = depths_dir / f"depth_{i:06d}.npy"
            depth_cur = np.load(depth_path)
            
            prev_idx = max(i-1, 0)
            next_idx = min(i+1, n_frames-1)
            
            # Load neighbor depths from disk
            depth_prev_path = depths_dir / f"depth_{prev_idx:06d}.npy"
            depth_next_path = depths_dir / f"depth_{next_idx:06d}.npy"
            depth_prev = np.load(depth_prev_path)
            depth_next = np.load(depth_next_path)
            
            # Simple temporal smoothing (average with neighbors)
            try:
                # Ensure all depths have same shape
                if depth_prev.shape != depth_cur.shape:
                    depth_prev = cv2.resize(depth_prev, (depth_cur.shape[1], depth_cur.shape[0]))
                if depth_next.shape != depth_cur.shape:
                    depth_next = cv2.resize(depth_next, (depth_cur.shape[1], depth_cur.shape[0]))
                
                # Simple temporal fusion
                fused = 0.6 * depth_cur + 0.2 * depth_prev + 0.2 * depth_next
                smoothed_depth = normalize_depth(fused)
                
            except Exception as e:
                print(f"Temporal smoothing failed for frame {i}: {e}, using original depth")
                smoothed_depth = depth_cur
            
            # Save smoothed depth to disk
            smoothed_path = smoothed_dir / f"smoothed_{i:06d}.npy"
            np.save(smoothed_path, smoothed_depth)
            
            pbar.update(1)
            
            # Clean up memory
            del depth_cur, depth_prev, depth_next, smoothed_depth
            gc.collect()
    
    print(f"Temporal smoothing completed")
    # Free memory after temporal smoothing
    gc.collect()
    
    # Stage 2: Temporal Smoothing - END
    status_mgr.update(job_id, {
        "status":"running", 
        "stage":"temporal_smoothing", 
        "percent":65,
        "message":"Temporal smoothing completed. Starting VR180 view creation...",
        "stage_completed": True
    })
    
    # Stage 3: LDI Reprojection - START
    status_mgr.update(job_id, {
        "status":"running", 
        "stage":"ldi_reprojection", 
        "percent":65,
        "message":"Starting VR180 view creation... (Estimated time: 8-12 minutes)",
        "stage_started": True
    })
    print(f"Starting VR180 view creation...")
    
    with tqdm(total=n_frames, desc="VR180 View Creation", unit="frame",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for i in range(n_frames):
            # Load frame from disk
            img = Image.open(frame_files[i]).convert("RGB")
            if img.size[0] > DOWNSCALE_WIDTH:
                img = img.resize((DOWNSCALE_WIDTH, int(img.size[1] * DOWNSCALE_WIDTH / img.size[0])), Image.Resampling.LANCZOS)
            frame_rgb = np.array(img).astype(np.uint8)
            
            # Load smoothed depth from disk
            smoothed_path = smoothed_dir / f"smoothed_{i:06d}.npy"
            depth_norm = np.load(smoothed_path)
            
            # Build LDI and reproject
            left_img, right_img, left_hmask, right_hmask = build_ldi_and_reproject(frame_rgb, depth_norm, BASELINE_RATIO, LDI_LAYERS)
            
            # Convert to PIL for inpainting
            left_mask_pil = Image.fromarray(left_hmask).convert("L")
            right_mask_pil = Image.fromarray(right_hmask).convert("L")
            left_pil = Image.fromarray(left_img)
            right_pil = Image.fromarray(right_img)
            
            # Inpainting for left view
            if left_hmask.sum() > 0:
                if use_inpaint_sd:
                    try:
                        left_inp = sd_inpaint(left_pil, left_mask_pil)
                        left_inp.save(left_dir / f"frame_{i:06d}.png")
                    except Exception as e:
                        print(f"Inpainting failed for left frame {i}: {e}")
                        left_pil.save(left_dir / f"frame_{i:06d}.png")
                else:
                    left_pil.save(left_dir / f"frame_{i:06d}.png")
            else:
                left_pil.save(left_dir / f"frame_{i:06d}.png")

            # Inpainting for right view
            if right_hmask.sum() > 0:
                if use_inpaint_sd:
                    try:
                        right_inp = sd_inpaint(right_pil, right_mask_pil)
                        right_inp.save(right_dir / f"frame_{i:06d}.png")
                    except Exception as e:
                        print(f"Inpainting failed for right frame {i}: {e}")
                        right_pil.save(right_dir / f"frame_{i:06d}.png")
                else:
                    right_pil.save(right_dir / f"frame_{i:06d}.png")
            else:
                right_pil.save(right_dir / f"frame_{i:06d}.png")

            pbar.update(1)
            
            # Clean up memory
            del img, frame_rgb, depth_norm, left_img, right_img, left_hmask, right_hmask
            del left_pil, right_pil, left_mask_pil, right_mask_pil
            gc.collect()
    
    print(f"LDI reprojection and inpainting completed")
    
    # Stage 3: LDI Reprojection - END
    status_mgr.update(job_id, {
        "status":"running", 
        "stage":"ldi_reprojection", 
        "percent":85,
        "message":"VR180 view creation completed. Starting final video encoding...",
        "stage_completed": True
    })
    
    # Stage 4: Final Encoding - START
    status_mgr.update(job_id, {
        "status":"running", 
        "stage":"encode", 
        "percent":88,
        "message":"Starting final video encoding... (Estimated time: 2-3 minutes)",
        "stage_started": True
    })
    out_path = job_dir / f"{job_id}_sbs.mp4"
    
    # Count frames for verification
    left_frames = sorted(left_dir.glob("frame_*.png"))
    right_frames = sorted(right_dir.glob("frame_*.png"))
    
    print(f"Creating final VR180 video with audio from original...")
    print(f"Using {len(left_frames)} frames at {fps:.3f} fps")
    print(f"Target duration: {len(left_frames) / fps:.3f}s")
    
    try:
        await create_side_by_side(left_dir, right_dir, out_path, fps, input_path)
        
        # Verify the output file was created successfully
        if out_path.exists():
            file_size = out_path.stat().st_size
            print(f"Output video created: {out_path} ({file_size / (1024*1024):.1f} MB)")
            
            # Get bitrate information for status
            try:
                output_meta = probe_video(out_path)
                bitrate = output_meta["format"].get("bit_rate", "N/A")
                if bitrate != "N/A":
                    bitrate_mbps = int(bitrate) / 1000000
                    bitrate_info = f"{bitrate_mbps:.2f} Mbps"
                else:
                    bitrate_info = "N/A"
            except:
                bitrate_info = "N/A"
            
            # Stage 4: Final Encoding - SUCCESS
            status_mgr.update(job_id, {
                "status":"done", 
                "stage":"finished", 
                "percent":100, 
                "output": str(out_path), 
                "message": "Video processing completed successfully!",
                "bitrate": bitrate_info,
                "file_size_mb": round(file_size / (1024*1024), 1),
                "stage_completed": True
            })
        else:
            print(f"Output video was not created!")
            status_mgr.update(job_id, {
                "status":"failed", 
                "stage":"encode", 
                "percent":95, 
                "message": "Failed to create output video file",
                "stage_completed": True
            })
            
    except Exception as e:
        print(f"❌ Final encoding failed: {e}")
        import traceback
        traceback.print_exc()
        status_mgr.update(job_id, {
            "status":"failed", 
            "stage":"encode", 
            "percent":95, 
            "message": f"Final encoding failed: {str(e)}",
            "stage_completed": True
        })

# Parallel processing functions removed - using sequential disk-based processing

# All parallel processing functions removed - using sequential disk-based processing
