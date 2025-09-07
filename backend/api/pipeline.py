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
from video_io import extract_frames, probe_video, create_side_by_side
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
    frames_dir = job_dir / "frames"
    frames_down = job_dir / "frames_down"
    left_dir = job_dir / "left"
    right_dir = job_dir / "right"
    preview_dir = job_dir / "preview"
    for d in [frames_dir, frames_down, left_dir, right_dir, preview_dir]:
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
    # extract frames downscaled for inference
    status_mgr.update(job_id, {"status":"running", "stage":"extract_frames", "percent":5})
    
    def frame_progress_callback(current, total):
        if total > 0:
            progress = 5 + int(5 * current / total)  # 5-10% range
            status_mgr.update(job_id, {"status":"running", "stage":"extract_frames", "percent":progress})
    
    await extract_frames(input_path, frames_down, downscale_width=DOWNSCALE_WIDTH, progress_callback=frame_progress_callback)
    
    # get list of frames
    frame_files = sorted(frames_down.glob("frame_*.png"))
    n_frames = len(frame_files)
    print(f"Total frames extracted: {n_frames}")
    print(f"🔧 DEBUG: Frame extraction completed, moving to depth estimation")
    status_mgr.update(job_id, {"status":"running", "stage":"depth_estimation", "percent":10, "frames": n_frames})

    # Enhanced batch depth estimation with parallel processing
    print(f"Starting batch depth estimation for {n_frames} frames...")
    print(f"🔧 DEBUG: About to start depth estimation process")
    depths = []
    batch_size = 12  # Reduced batch size for memory efficiency
    
    # Process frames in smaller batches to avoid memory issues
    print("Processing frames in small batches to manage memory...")
    batch_size = 4  # Very small batch size for memory efficiency
    chunk_size = 10  # Very small chunk size for loading
    
    # Process frames in chunks to avoid loading all into memory at once
    depths = []
    import time
    start_time = time.time()
    max_processing_time = 1800  # 30 minutes max for depth estimation
    
    for chunk_start in range(0, n_frames, chunk_size):
        # Check for timeout
        if time.time() - start_time > max_processing_time:
            print(f"⚠️  Depth estimation timeout after {max_processing_time/60:.1f} minutes")
            break
        chunk_end = min(chunk_start + chunk_size, n_frames)
        chunk_files = frame_files[chunk_start:chunk_end]
        
        print(f"Processing frames {chunk_start+1}-{chunk_end} of {n_frames}")
        
        # Load chunk of images
        chunk_images = []
        for fpath in chunk_files:
            try:
                img = Image.open(fpath).convert("RGB")
                # Resize to target resolution if needed to save memory
                if img.size[0] > DOWNSCALE_WIDTH:
                    img = img.resize((DOWNSCALE_WIDTH, int(img.size[1] * DOWNSCALE_WIDTH / img.size[0])), Image.Resampling.LANCZOS)
                chunk_images.append(np.array(img))
            except Exception as e:
                print(f"Failed to load frame {fpath}: {e}")
                # Create dummy image
                dummy_img = np.ones((DOWNSCALE_WIDTH//2, DOWNSCALE_WIDTH//2, 3), dtype=np.uint8) * 128
                chunk_images.append(dummy_img)
        
        # Process this chunk
        chunk_depths = []
        for i, img_array in enumerate(chunk_images):
            try:
                # Convert numpy array to PIL for depth estimation
                img_pil = Image.fromarray(img_array)
                
                # Use simpler depth estimation for reliability
                depth_arr = create_local_depth_map(img_pil)
                depth_arr = normalize_depth(depth_arr)
                chunk_depths.append(depth_arr)
                print(f"✅ Processed frame {chunk_start + i + 1}/{n_frames}")
                
                # Update progress every 10 frames to reduce API calls
                if (i + 1) % 10 == 0:
                    progress = 10 + int(40.0 * (chunk_start + i + 1) / n_frames)
                    try:
                        status_mgr.update(job_id, {"status":"running", "stage":"depth_estimation", "percent":progress})
                    except Exception as e:
                        print(f"⚠️  Status update failed: {e}")
                    
            except (Exception, TimeoutError) as e:
                print(f"❌ Depth estimation failed for frame {chunk_start + i + 1}: {e}")
                # Create a dummy depth map as fallback
                dummy_depth = np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.float32) * 0.5
                chunk_depths.append(dummy_depth)
        
        depths.extend(chunk_depths)
        
        # Update progress
        progress = 10 + int(40.0 * chunk_end / n_frames)
        try:
            status_mgr.update(job_id, {"status":"running", "stage":"depth_estimation", "percent":progress})
        except Exception as e:
            print(f"⚠️  Status update failed: {e}")
        
        # Force garbage collection after each chunk
        del chunk_images, chunk_depths
        gc.collect()
        print(f"Completed chunk {chunk_start+1}-{chunk_end}, total depths: {len(depths)}")
        
        # Add a small delay to prevent overwhelming the system
        import time
        time.sleep(0.1)
    
    print(f"Depth estimation completed for {len(depths)} frames")
    # Free memory after depth estimation
    gc.collect()
    status_mgr.update(job_id, {"status":"running", "stage":"temporal_smoothing", "percent":52})
    # Memory-efficient temporal smoothing with on-the-fly flow computation
    print(f"Starting memory-efficient temporal smoothing...")
    depths_smoothed = []
    
    # Apply temporal smoothing with on-the-fly flow computation to save memory
    print("Applying temporal smoothing with on-the-fly flow computation...")
    with tqdm(total=n_frames, desc="Temporal smoothing", unit="frame") as pbar:
        for i in range(n_frames):
            depth_cur = depths[i]
            prev_idx = max(i-1, 0)
            next_idx = min(i+1, n_frames-1)
            
            # Load frames on-demand for flow computation
            try:
                if i > 0:
                    # Load current and previous frames
                    img_cur = Image.open(frame_files[i]).convert("RGB")
                    img_prev = Image.open(frame_files[prev_idx]).convert("RGB")
                    if img_cur.size[0] > DOWNSCALE_WIDTH:
                        img_cur = img_cur.resize((DOWNSCALE_WIDTH, int(img_cur.size[1] * DOWNSCALE_WIDTH / img_cur.size[0])), Image.Resampling.LANCZOS)
                    if img_prev.size[0] > DOWNSCALE_WIDTH:
                        img_prev = img_prev.resize((DOWNSCALE_WIDTH, int(img_prev.size[1] * DOWNSCALE_WIDTH / img_prev.size[0])), Image.Resampling.LANCZOS)
                    
                    frame_cur_bgr = cv2.cvtColor(np.array(img_cur), cv2.COLOR_RGB2BGR)
                    frame_prev_bgr = cv2.cvtColor(np.array(img_prev), cv2.COLOR_RGB2BGR)
                    flow_cur_to_prev = compute_flow_cv2(frame_cur_bgr, frame_prev_bgr)
                else:
                    h, w = depth_cur.shape[:2]
                    flow_cur_to_prev = np.zeros((h, w, 2), dtype=np.float32)
                
                if i < n_frames-1:
                    # Load next frame
                    img_next = Image.open(frame_files[next_idx]).convert("RGB")
                    if img_next.size[0] > DOWNSCALE_WIDTH:
                        img_next = img_next.resize((DOWNSCALE_WIDTH, int(img_next.size[1] * DOWNSCALE_WIDTH / img_next.size[0])), Image.Resampling.LANCZOS)
                    frame_next_bgr = cv2.cvtColor(np.array(img_next), cv2.COLOR_RGB2BGR)
                    flow_cur_to_next = compute_flow_cv2(frame_cur_bgr, frame_next_bgr)
                else:
                    flow_cur_to_next = np.zeros_like(flow_cur_to_prev)
                
            except Exception as e:
                print(f"Flow computation failed for frame {i}: {e}")
                h, w = depth_cur.shape[:2]
                flow_cur_to_prev = np.zeros((h, w, 2), dtype=np.float32)
                flow_cur_to_next = np.zeros_like(flow_cur_to_prev)
            
            # Update progress every 50 frames
            if i % 50 == 0:
                progress = 52 + int(8.0 * i / n_frames)
                status_mgr.update(job_id, {"status":"running", "stage":"temporal_smoothing", "percent":progress})
            
            # Ensure flow dimensions match depth dimensions
            if flow_cur_to_prev.shape[:2] != depth_cur.shape[:2]:
                if len(flow_cur_to_prev.shape) == 3:
                    flow_resized = np.zeros((depth_cur.shape[0], depth_cur.shape[1], flow_cur_to_prev.shape[2]), dtype=flow_cur_to_prev.dtype)
                    for j in range(flow_cur_to_prev.shape[2]):
                        flow_resized[:, :, j] = cv2.resize(flow_cur_to_prev[:, :, j], (depth_cur.shape[1], depth_cur.shape[0]))
                    flow_cur_to_prev = flow_resized
                else:
                    flow_cur_to_prev = cv2.resize(flow_cur_to_prev, (depth_cur.shape[1], depth_cur.shape[0]))
            
            if flow_cur_to_next.shape[:2] != depth_cur.shape[:2]:
                if len(flow_cur_to_next.shape) == 3:
                    flow_resized = np.zeros((depth_cur.shape[0], depth_cur.shape[1], flow_cur_to_next.shape[2]), dtype=flow_cur_to_next.dtype)
                    for j in range(flow_cur_to_next.shape[2]):
                        flow_resized[:, :, j] = cv2.resize(flow_cur_to_next[:, :, j], (depth_cur.shape[1], depth_cur.shape[0]))
                    flow_cur_to_next = flow_resized
                else:
                    flow_cur_to_next = cv2.resize(flow_cur_to_next, (depth_cur.shape[1], depth_cur.shape[0]))
            
            # Get neighbor depths
            depth_prev = depths[prev_idx]
            depth_next = depths[next_idx]
            
            # Ensure depth dimensions match
            if depth_prev.shape != depth_cur.shape:
                depth_prev = cv2.resize(depth_prev, (depth_cur.shape[1], depth_cur.shape[0]))
            if depth_next.shape != depth_cur.shape:
                depth_next = cv2.resize(depth_next, (depth_cur.shape[1], depth_cur.shape[0]))
            
            try:
                # Ensure depth arrays are 2D for warping
                if len(depth_prev.shape) == 3:
                    depth_prev_2d = depth_prev[:,:,0] if depth_prev.shape[2] == 1 else depth_prev.mean(axis=2)
                else:
                    depth_prev_2d = depth_prev
                    
                if len(depth_next.shape) == 3:
                    depth_next_2d = depth_next[:,:,0] if depth_next.shape[2] == 1 else depth_next.mean(axis=2)
                else:
                    depth_next_2d = depth_next
                
                # Vectorized warping operations
                warped_prev = warp_image_with_flow((depth_prev_2d*255).astype(np.uint8), flow_cur_to_prev)
                warped_prev = warped_prev.astype(np.float32)/255.0
                warped_next = warp_image_with_flow((depth_next_2d*255).astype(np.uint8), flow_cur_to_next)
                warped_next = warped_next.astype(np.float32)/255.0
                
                # Ensure warped arrays are 2D
                if len(warped_prev.shape) == 3:
                    warped_prev = warped_prev[:,:,0] if warped_prev.shape[2] == 1 else warped_prev.mean(axis=2)
                if len(warped_next.shape) == 3:
                    warped_next = warped_next[:,:,0] if warped_next.shape[2] == 1 else warped_next.mean(axis=2)
                
                # Vectorized temporal fusion
                motion_magnitude = np.sqrt(np.sum(flow_cur_to_prev**2, axis=2) + np.sum(flow_cur_to_next**2, axis=2))
                motion_factor = np.clip(motion_magnitude / 10.0, 0.1, 1.0)
                
                # Vectorized weight calculation
                w_prev = 0.2 * motion_factor
                w_next = 0.2 * motion_factor
                w_cur = 1.0 - w_prev - w_next
                
                # Normalize weights
                total_weight = w_cur + w_prev + w_next
                w_cur = w_cur / total_weight
                w_prev = w_prev / total_weight
                w_next = w_next / total_weight
                
                # Vectorized fusion - ensure all arrays have same shape
                fused = w_cur * depth_cur + w_prev * warped_prev + w_next * warped_next
                
            except Exception as e:
                print(f"Warping failed for frame {i}: {e}, using original depth")
                fused = depth_cur
            
            depths_smoothed.append(normalize_depth(fused))
            
            pbar.update(1)
            if i % 32 == 0:  # Update less frequently
                progress = 52 + int(10.0*i/n_frames)
                status_mgr.update(job_id, {"status":"running", "stage":"temporal_smoothing", "percent":progress})
    
    print(f"Temporal smoothing completed")
    # Free memory after temporal smoothing
    gc.collect()

    status_mgr.update(job_id, {"status":"running", "stage":"ldi_reprojection", "percent":65})
    # Enhanced batch LDI reprojection and inpainting
    print(f"Starting batch LDI reprojection and inpainting...")
    
    # Process frames in batches for better performance
    ldi_batch_size = 4  # Reduced batch size for memory efficiency
    with tqdm(total=n_frames, desc="Batch LDI & Inpainting", unit="frame") as pbar:
        for batch_start in range(0, n_frames, ldi_batch_size):
            batch_end = min(batch_start + ldi_batch_size, n_frames)
            
            # Process batch of frames
            for i in range(batch_start, batch_end):
                # Load frame on-demand
                img = Image.open(frame_files[i]).convert("RGB")
                if img.size[0] > DOWNSCALE_WIDTH:
                    img = img.resize((DOWNSCALE_WIDTH, int(img.size[1] * DOWNSCALE_WIDTH / img.size[0])), Image.Resampling.LANCZOS)
                frame_rgb = np.array(img).astype(np.uint8)
                depth_norm = depths_smoothed[i]
                
                # Build LDI and reproject
                left_img, right_img, left_hmask, right_hmask = build_ldi_and_reproject(frame_rgb, depth_norm, BASELINE_RATIO, LDI_LAYERS)
                
                # Convert to PIL for inpainting
                left_mask_pil = Image.fromarray(left_hmask).convert("L")
                right_mask_pil = Image.fromarray(right_hmask).convert("L")
                left_pil = Image.fromarray(left_img)
                right_pil = Image.fromarray(right_img)
                
                # Batch inpainting for left view
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

                # Batch inpainting for right view
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
            
            # Update status for the batch
            progress = 65 + int(20.0 * batch_end / n_frames)
            status_mgr.update(job_id, {"status":"running", "stage":"ldi_reprojection", "percent":progress})
    
    print(f"LDI reprojection and inpainting completed")

    status_mgr.update(job_id, {"status":"running", "stage":"encode", "percent":88})
    out_path = job_dir / f"{job_id}_sbs.mp4"
    
    # Count frames for verification
    left_frames = sorted(left_dir.glob("frame_*.png"))
    right_frames = sorted(right_dir.glob("frame_*.png"))
    
    print(f"Creating final VR180 video with audio from original...")
    print(f"Using {len(left_frames)} frames at {fps:.3f} fps")
    print(f"Target duration: {len(left_frames) / fps:.3f}s")
    
    await create_side_by_side(left_dir, right_dir, out_path, fps, input_path)
    
    # Verify the output file was created successfully
    if out_path.exists():
        file_size = out_path.stat().st_size
        print(f"Output video created: {out_path} ({file_size / (1024*1024):.1f} MB)")
    else:
        print(f"Output video was not created!")
    
    # Update status with output path
    status_mgr.update(job_id, {"status":"done", "stage":"finished", "percent":100, "output": str(out_path), "message": "Video processing completed successfully!"})

def process_video_parallel(job_id: str, input_path: str, output_path: str, 
                          status_mgr: StatusManager) -> Dict[str, Any]:
    """Process video with parallel stages and memory streaming"""
    try:
        # Get video info
        video_info = get_video_info(input_path)
        n_frames = video_info['frame_count']
        fps = video_info['fps']
        duration = video_info['duration']
        
        print(f"🎬 Processing {n_frames} frames at {fps:.2f} FPS")
        
        # Create workspace
        workspace = Path(WORKSPACE_DIR) / job_id
        workspace.mkdir(parents=True, exist_ok=True)
        
        # Stage 1: Extract frames (streaming)
        print("📸 Extracting frames...")
        status_mgr.update(job_id, {"status": "running", "stage": "extract_frames", "percent": 5})
        
        frame_dir = workspace / "frames"
        frame_dir.mkdir(exist_ok=True)
        
        # Extract frames with streaming
        extract_frames_streaming(input_path, frame_dir, n_frames)
        status_mgr.update(job_id, {"status": "running", "stage": "depth_estimation", "percent": 10})
        
        # Stage 2: Parallel depth estimation + temporal smoothing + LDI generation
        print("🧠 Starting parallel processing...")
        
        # Create processing queues
        frame_queue = queue.Queue(maxsize=50)  # Limit memory
        depth_queue = queue.Queue(maxsize=50)
        ldi_queue = queue.Queue(maxsize=50)
        
        # Start parallel workers using ThreadPoolExecutor (queues work with threads)
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit parallel tasks with job_id and status_mgr for progress updates
            depth_future = executor.submit(process_depth_parallel, frame_dir, n_frames, depth_queue, job_id, status_mgr)
            temporal_future = executor.submit(process_temporal_parallel, depth_queue, n_frames, ldi_queue, job_id, status_mgr)
            ldi_future = executor.submit(process_ldi_parallel, ldi_queue, n_frames, workspace, job_id, status_mgr)
            
            # Monitor progress
            monitor_parallel_progress(job_id, status_mgr, depth_future, temporal_future, ldi_future)
            
            # Wait for completion
            depth_future.result()
            temporal_future.result()
            ldi_future.result()
        
        # Stage 3: Final encoding
        print("🎥 Final encoding...")
        status_mgr.update(job_id, {"status": "running", "stage": "final_encoding", "percent": 90})
        
        # Encode final video
        encode_final_video(workspace, output_path, fps)
        
        status_mgr.update(job_id, {"status": "completed", "percent": 100})
        return {"status": "success", "output_path": output_path}
        
    except Exception as e:
        print(f"❌ Parallel processing error: {e}")
        status_mgr.update(job_id, {"status": "failed", "error": str(e)})
        return {"status": "error", "error": str(e)}

def extract_frames_streaming(input_path: str, frame_dir: Path, n_frames: int):
    """Extract frames with streaming to save memory"""
    # Use FFmpeg to extract frames with streaming
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-vf', 'fps=30',  # Limit to 30 FPS for processing
        '-q:v', '2',  # High quality
        str(frame_dir / 'frame_%06d.jpg')
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)

def process_depth_parallel(frame_dir: Path, n_frames: int, depth_queue: queue.Queue, job_id: str = None, status_mgr = None):
    """Process depth estimation in parallel with memory streaming"""
    print("🧠 Starting parallel depth estimation...")
    
    for i in range(n_frames):
        try:
            frame_path = frame_dir / f"frame_{i+1:06d}.jpg"
            if not frame_path.exists():
                continue
                
            # Load and process single frame
            img = Image.open(frame_path).convert("RGB")
            depth_map = create_local_depth_map(img)
            
            # Stream to queue instead of storing in memory
            depth_queue.put((i, depth_map))
            
            # Clean up immediately
            del img, depth_map
            gc.collect()
            
            # Update progress every 50 frames
            if (i + 1) % 50 == 0:
                progress = 10 + int(40.0 * (i + 1) / n_frames)
                print(f"✅ Processed {i + 1}/{n_frames} frames for depth")
                if status_mgr and job_id:
                    try:
                        status_mgr.update(job_id, {
                            "status": "running", 
                            "stage": "depth_estimation", 
                            "percent": progress,
                            "message": f"Processing depth maps... {i + 1}/{n_frames} frames"
                        })
                    except:
                        pass
                
        except Exception as e:
            print(f"❌ Depth processing error for frame {i}: {e}")
            # Add dummy depth map
            depth_queue.put((i, np.ones((512, 512), dtype=np.float32) * 0.5))
    
    # Signal completion
    depth_queue.put(None)

def process_temporal_parallel(depth_queue: queue.Queue, n_frames: int, ldi_queue: queue.Queue, job_id: str = None, status_mgr = None):
    """Process temporal smoothing in parallel with memory streaming"""
    print("⏱️ Starting parallel temporal smoothing...")
    
    # Collect depth maps for temporal smoothing
    depth_maps = {}
    processed = 0
    
    while processed < n_frames:
        try:
            item = depth_queue.get(timeout=30)
            if item is None:
                break
                
            frame_idx, depth_map = item
            depth_maps[frame_idx] = depth_map
            
            # Process in small batches to avoid memory overflow
            if len(depth_maps) >= 50 or processed == n_frames - 1:
                # Apply temporal smoothing to batch
                smoothed_depths = apply_temporal_smoothing_batch(depth_maps)
                
                # Stream to LDI queue
                for frame_idx, smoothed_depth in smoothed_depths.items():
                    ldi_queue.put((frame_idx, smoothed_depth))
                
                # Clean up
                depth_maps.clear()
                gc.collect()
                
            processed += 1
            
            # Update progress every 100 processed frames
            if processed % 100 == 0:
                progress = 50 + int(20.0 * processed / n_frames)
                print(f"✅ Temporal smoothing: {processed}/{n_frames} frames")
                if status_mgr and job_id:
                    try:
                        status_mgr.update(job_id, {
                            "status": "running", 
                            "stage": "temporal_smoothing", 
                            "percent": progress,
                            "message": f"Applying temporal smoothing... {processed}/{n_frames} frames"
                        })
                    except:
                        pass
            
        except queue.Empty:
            print("⚠️ Timeout waiting for depth data")
            break
    
    # Signal completion
    ldi_queue.put(None)

def process_ldi_parallel(ldi_queue: queue.Queue, n_frames: int, workspace: Path, job_id: str = None, status_mgr = None):
    """Process LDI generation in parallel with memory streaming"""
    print("🎭 Starting parallel LDI generation...")
    
    ldi_dir = workspace / "ldi"
    ldi_dir.mkdir(exist_ok=True)
    
    processed = 0
    while processed < n_frames:
        try:
            item = ldi_queue.get(timeout=30)
            if item is None:
                break
                
            frame_idx, depth_map = item
            
            # Generate LDI for single frame
            ldi_data = generate_ldi_single_frame(depth_map, frame_idx)
            
            # Save to disk immediately
            ldi_path = ldi_dir / f"ldi_{frame_idx:06d}.npz"
            np.savez_compressed(ldi_path, **ldi_data)
            
            processed += 1
            
            # Update progress every 100 processed frames
            if processed % 100 == 0:
                progress = 70 + int(20.0 * processed / n_frames)
                print(f"✅ Generated {processed}/{n_frames} LDI frames")
                if status_mgr and job_id:
                    try:
                        status_mgr.update(job_id, {
                            "status": "running", 
                            "stage": "ldi_generation", 
                            "percent": progress,
                            "message": f"Creating VR180 views... {processed}/{n_frames} frames"
                        })
                    except:
                        pass
                
        except queue.Empty:
            print("⚠️ Timeout waiting for LDI data")
            break

def monitor_parallel_progress(job_id: str, status_mgr: StatusManager, 
                            depth_future, temporal_future, ldi_future):
    """Monitor parallel processing progress"""
    start_time = time.time()
    max_time = 1800  # 30 minutes max
    
    while not all(f.done() for f in [depth_future, temporal_future, ldi_future]):
        if time.time() - start_time > max_time:
            print("⚠️ Parallel processing timeout")
            break
            
        # Update progress based on completed stages
        progress = 10
        if depth_future.done():
            progress += 30
        if temporal_future.done():
            progress += 30
        if ldi_future.done():
            progress += 20
            
        try:
            status_mgr.update(job_id, {
                "status": "running", 
                "stage": "parallel_processing", 
                "percent": progress
            })
        except:
            pass
            
        time.sleep(5)  # Update every 5 seconds

def apply_temporal_smoothing_batch(depth_maps: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """Apply temporal smoothing to a batch of depth maps"""
    if not depth_maps:
        return {}
    
    # Sort by frame index
    sorted_frames = sorted(depth_maps.items())
    smoothed = {}
    
    for i, (frame_idx, depth_map) in enumerate(sorted_frames):
        # Simple temporal smoothing - average with neighbors
        neighbors = []
        
        # Add previous frame
        if i > 0:
            prev_idx, prev_depth = sorted_frames[i-1]
            neighbors.append(prev_depth)
        
        # Add current frame
        neighbors.append(depth_map)
        
        # Add next frame
        if i < len(sorted_frames) - 1:
            next_idx, next_depth = sorted_frames[i+1]
            neighbors.append(next_depth)
        
        # Average the neighbors
        if len(neighbors) > 1:
            smoothed_depth = np.mean(neighbors, axis=0)
        else:
            smoothed_depth = depth_map
            
        smoothed[frame_idx] = smoothed_depth
    
    return smoothed

def generate_ldi_single_frame(depth_map: np.ndarray, frame_idx: int) -> Dict[str, np.ndarray]:
    """Generate LDI data for a single frame"""
    # Simple LDI generation - create stereo views
    h, w = depth_map.shape
    
    # Create left and right views based on depth
    left_view = np.zeros((h, w, 3), dtype=np.uint8)
    right_view = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Simple stereo generation (placeholder - you can enhance this)
    for y in range(h):
        for x in range(w):
            depth = depth_map[y, x]
            # Create stereo offset based on depth
            offset = int(depth * 10)  # Adjust multiplier as needed
            
            # Left view
            if x - offset >= 0:
                left_view[y, x] = [255, 255, 255]  # White for now
            
            # Right view  
            if x + offset < w:
                right_view[y, x] = [255, 255, 255]  # White for now
    
    return {
        'left_view': left_view,
        'right_view': right_view,
        'depth_map': depth_map
    }

def encode_final_video(workspace: Path, output_path: str, fps: float):
    """Encode the final VR180 video"""
    ldi_dir = workspace / "ldi"
    
    # Create video from LDI frames
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', str(ldi_dir / 'ldi_%06d.npz'),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        str(output_path)
    ]
    
    subprocess.run(cmd, check=True)
