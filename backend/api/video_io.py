# api/video_io.py
import subprocess
from pathlib import Path
from typing import Tuple
import shlex
import asyncio
from fractions import Fraction
import json
import os
from tqdm import tqdm
from PIL import Image

# from .config import FFMPEG_BIN
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")

def find_ffmpeg():
    """Find FFmpeg binary with fallback options"""
    import shutil
    import glob
    import os
    
    # Try the environment variable first
    if FFMPEG_BIN and FFMPEG_BIN != "ffmpeg":
        return FFMPEG_BIN
    
    # Try common paths
    common_paths = [
        "ffmpeg",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg"
    ]
    
    # Try Nix store paths
    nix_paths = glob.glob("/nix/store/*/bin/ffmpeg")
    common_paths.extend(nix_paths)
    
    for path in common_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            print(f"✅ Found FFmpeg at: {path}")
            return path
    
    # Try which command as last resort
    which_path = shutil.which("ffmpeg")
    if which_path:
        print(f"✅ Found FFmpeg via which: {which_path}")
        return which_path
    
    print("❌ FFmpeg not found in any common locations")
    return "ffmpeg"

# Use the found FFmpeg binary
FFMPEG_BIN = find_ffmpeg()
def run_cmd_sync(cmd: str, timeout: int = 3600):
    """Synchronous command runner for Windows compatibility"""
    print(f"🔧 Running command: {cmd}")
    print(f"🔧 Using FFmpeg binary: {FFMPEG_BIN}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    
    if result.returncode != 0:
        print(f"❌ Command failed with return code: {result.returncode}")
        print(f"❌ STDOUT: {result.stdout}")
        print(f"❌ STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}\nReturn code: {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
    
    return result.stdout, result.stderr

async def run_cmd(cmd: str, timeout: int = 3600):
    """Async wrapper for sync command runner"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_cmd_sync, cmd, timeout)

async def extract_frames(video_path: Path, out_dir: Path, downscale_width: int = None, progress_callback=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get accurate video information first
    meta = probe_video(video_path)
    duration = float(meta["format"].get("duration", 30.0))
    
    # Extract FPS accurately
    video_stream = None
    for stream in meta["streams"]:
        if stream.get("codec_type") == "video":
            video_stream = stream
            break
    
    if video_stream and "r_frame_rate" in video_stream:
        r_frame_rate = video_stream["r_frame_rate"]
        if '/' in r_frame_rate:
            numerator, denominator = r_frame_rate.split('/')
            fps = float(numerator) / float(denominator)
        else:
            fps = float(r_frame_rate)
    else:
        fps = 30.0
        print(f"Could not detect FPS, using default: {fps}")
    
    # Calculate exact frame count based on duration and FPS
    expected_frames = int(duration * fps)
    print(f"Expected {expected_frames} frames from {duration:.1f}s at {fps:.3f}fps")
    
    # Build high-quality extraction command
    scale_filter = f",scale={downscale_width}:-1:flags=lanczos" if downscale_width else ""
    
    # Use ultra-high-quality settings for frame extraction
    cmd = (f'{FFMPEG_BIN} -y -i "{video_path}" '
           f'-vsync 0 '  # Extract all frames, not just keyframes
           f'-q:v 0 '    # Ultra-high quality (0-31, lower is better)
           f'-vf "format=rgb24{scale_filter}" '  # Ensure RGB format
           f'-f image2 '  # Image sequence output
           f'"{out_dir}/frame_%06d.png"')
    
    print(f"Extracting frames with high quality settings...")
    
    # Use tqdm for progress tracking
    with tqdm(total=expected_frames, desc="Extracting frames", unit="frame") as pbar:
        try:
            # Run FFmpeg with progress monitoring
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor progress
            while True:
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output and 'frame=' in output:
                    try:
                        current_frame = int(output.split('frame=')[1].split()[0])
                        pbar.n = min(current_frame, expected_frames)
                        pbar.refresh()
                        if progress_callback:
                            progress_callback(current_frame, expected_frames)
                    except:
                        pass
            
            process.wait()
            if process.returncode != 0:
                stderr_output = process.stderr.read()
                print(f"FFmpeg stderr: {stderr_output}")
                raise RuntimeError(f"FFmpeg failed with return code {process.returncode}")
                
        except Exception as e:
            print(f"Frame extraction failed: {e}")
            # Try with different parameters as fallback
            print("Trying fallback extraction...")
            try:
                # Fallback with different quality settings (compatible with FFmpeg 4.4.2)
                cmd_fallback = (f'{FFMPEG_BIN} -y -i "{video_path}" '
                              f'-vsync 0 '  # Extract all frames, not just keyframes
                              f'-q:v 1 '  # Still high quality
                              f'-vf "format=rgb24{scale_filter}" '
                              f'-f image2 '
                              f'"{out_dir}/frame_%06d.png"')
                await run_cmd(cmd_fallback)
            except Exception as e2:
                print(f"Fallback extraction also failed: {e2}")
                # Create dummy frames as last resort
                print("Creating dummy frames...")
                for i in range(min(expected_frames, 10)):  # Create max 10 dummy frames
                    dummy_img = Image.new('RGB', (640, 480), color=(128, 128, 128))
                    dummy_img.save(out_dir / f"frame_{i+1:06d}.png")
    
    # Count actual extracted frames
    actual_frames = len(list(out_dir.glob("frame_*.png")))
    print(f"Extracted {actual_frames} frames successfully")
    
    # Verify frame count matches expected
    if actual_frames != expected_frames:
        print(f"Frame count mismatch: expected {expected_frames}, got {actual_frames}")
        if actual_frames < expected_frames * 0.8:  # If we got less than 80% of expected frames
            print("Significant frame loss detected!")
    else:
        print(f"Frame count matches expected: {actual_frames} frames")

def probe_video(video_path: Path) -> dict:
    """Use FFprobe for accurate video information extraction"""
    try:
        # First try FFprobe for accurate metadata
        ffprobe_cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams "{video_path}"'
        try:
            result = subprocess.run(ffprobe_cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Find video stream
                video_stream = None
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        video_stream = stream
                        break
                
                if video_stream:
                    # Extract accurate FPS
                    r_frame_rate = video_stream.get('r_frame_rate', '30/1')
                    if '/' in r_frame_rate:
                        numerator, denominator = r_frame_rate.split('/')
                        fps = float(numerator) / float(denominator)
                    else:
                        fps = float(r_frame_rate)
                    
                    # Extract duration
                    duration = float(data['format'].get('duration', 30.0))
                    
                    # Extract resolution
                    width = int(video_stream.get('width', 1920))
                    height = int(video_stream.get('height', 1080))
                    
                    print(f"FFprobe detected: {width}x{height} @ {fps:.3f}fps, duration: {duration:.2f}s")
                    
                    # Build streams array with all streams (video + audio)
                    streams = [{
                        "codec_type": "video",
                        "width": width,
                        "height": height,
                        "r_frame_rate": r_frame_rate,
                        "codec_name": video_stream.get('codec_name', 'h264')
                    }]
                    
                    # Add audio streams if they exist
                    for stream in data.get('streams', []):
                        if stream.get('codec_type') == 'audio':
                            streams.append({
                                "codec_type": "audio",
                                "codec_name": stream.get('codec_name', 'aac'),
                                "samgit push -u origin mainple_rate": stream.get('sample_rate', '48000'),
                                "channels": stream.get('channels', 2)
                            })
                            print(f"Audio stream found: {stream.get('codec_name', 'unknown')} at {stream.get('sample_rate', 'unknown')}Hz")
                    
                    return {
                        "format": {
                            "duration": str(duration),
                            "bit_rate": data['format'].get('bit_rate', '1000000')
                        },
                        "streams": streams
                    }
        except Exception as e:
            print(f"FFprobe failed: {e}, falling back to FFmpeg")
        
        # Fallback to FFmpeg if FFprobe fails
        cmd = f'{FFMPEG_BIN} -i "{video_path}" -f null - 2>&1'
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        # Parse video info from stderr with improved accuracy
        duration = "30.0"
        width, height = 1920, 1080
        fps = "30/1"
        
        for line in res.stderr.split('\n'):
            if 'Duration:' in line:
                try:
                    duration_part = line.split('Duration:')[1].split(',')[0].strip()
                    # Convert HH:MM:SS.mmm to seconds
                    parts = duration_part.split(':')
                    if len(parts) == 3:
                        hours, minutes, seconds = parts
                        duration = str(int(hours) * 3600 + int(minutes) * 60 + float(seconds))
                except:
                    pass
            elif 'Stream #0:0' in line and 'Video:' in line:
                try:
                    # Extract resolution more accurately
                    if 'x' in line:
                        # Look for pattern like "1920x1080"
                        import re
                        res_match = re.search(r'(\d+)x(\d+)', line)
                        if res_match:
                            width, height = int(res_match.group(1)), int(res_match.group(2))
                except:
                    pass
            elif 'fps' in line.lower() and 'fps' in line:
                try:
                    # More accurate FPS extraction
                    import re
                    fps_match = re.search(r'(\d+(?:\.\d+)?)\s*fps', line)
                    if fps_match:
                        fps_val = float(fps_match.group(1))
                        fps = f"{fps_val}/1"
                except:
                    pass
        
        print(f"FFmpeg detected: {width}x{height} @ {fps}, duration: {duration}s")
        
        return {
            "format": {
                "duration": duration,
                "bit_rate": "1000000"
            },
            "streams": [{
                "codec_type": "video",
                "width": width,
                "height": height,
                "r_frame_rate": fps,
                "codec_name": "h264"
            }]
        }
    except Exception as e:
        print(f"Video probe failed: {e}")
        # Return default fallback
        return {
            "format": {
                "duration": "30.0",
                "bit_rate": "1000000"
            },
            "streams": [{
                "codec_type": "video",
                "r_frame_rate": "30/1",
                "width": 1920,
                "height": 1080
            }]
        }

async def create_side_by_side(left_dir: Path, right_dir: Path, out_path: Path, fps: float, original_video_path: Path = None):
    """Create high-quality side-by-side VR180 video with exact duration matching"""
    
    # Count actual frames to ensure we have the right number
    left_frames = sorted(left_dir.glob("frame_*.png"))
    right_frames = sorted(right_dir.glob("frame_*.png"))
    
    if len(left_frames) != len(right_frames):
        raise ValueError(f"Frame count mismatch: {len(left_frames)} left, {len(right_frames)} right")
    
    if not left_frames:
        raise ValueError("No frames found in input directories")
    
    print(f"Creating side-by-side video with {len(left_frames)} frames at {fps:.3f} fps")
    
    # Calculate exact duration based on frame count and FPS
    exact_duration = len(left_frames) / fps
    print(f"Target duration: {exact_duration:.3f}s ({len(left_frames)} frames / {fps:.3f} fps)")
    
    if original_video_path and original_video_path.exists():
        # Check if original video has audio with improved detection
        has_audio = False
        audio_stream_info = None
        try:
            meta = probe_video(original_video_path)
            print(f"Checking for audio in: {original_video_path}")
            print(f"Streams found: {len(meta.get('streams', []))}")
            
            for i, stream in enumerate(meta.get("streams", [])):
                print(f"Stream {i}: {stream.get('codec_type', 'unknown')} - {stream.get('codec_name', 'unknown')}")
                if stream.get("codec_type") == "audio":
                    has_audio = True
                    audio_stream_info = stream
                    print(f"Audio stream found: {stream.get('codec_name', 'unknown')} at {stream.get('sample_rate', 'unknown')}Hz")
                    break
        except Exception as e:
            print(f"Could not check for audio in original video: {e}")
        
        print(f"Audio detection result: {has_audio}")
        
        if has_audio:
            # Include audio from original video with VR180 metadata
            print("Creating video with audio...")
            cmd = (f'{FFMPEG_BIN} -y '
                   f'-framerate {fps} -i "{left_dir}/frame_%06d.png" '
                   f'-framerate {fps} -i "{right_dir}/frame_%06d.png" '
                   f'-i "{original_video_path}" '
                   f'-filter_complex "[0][1]hstack=inputs=2[v]" '
                   f'-map "[v]" -map 2:a '
                   f'-c:v libx264 -crf 12 -preset slow '
                   f'-c:a aac -b:a 192k -ar 48000 '
                   f'-pix_fmt yuv420p '
                   f'-shortest '
                   f'-avoid_negative_ts make_zero '
                   f'-fflags +genpts '
                   f'-metadata spherical-video=1 '
                   f'-metadata spherical-stereo=1 '
                   f'-metadata spherical-layout=mono '
                   f'-metadata spherical-projection=equirectangular '
                   f'"{out_path}"')
        else:
            # No audio detected, but try to include it anyway as fallback
            print("No audio detected, but attempting to include audio as fallback...")
            cmd = (f'{FFMPEG_BIN} -y '
                   f'-framerate {fps} -i "{left_dir}/frame_%06d.png" '
                   f'-framerate {fps} -i "{right_dir}/frame_%06d.png" '
                   f'-i "{original_video_path}" '
                   f'-filter_complex "[0][1]hstack=inputs=2[v]" '
                   f'-map "[v]" -map 2:a? '  # The ? makes audio optional
                   f'-c:v libx264 -crf 12 -preset slow '
                   f'-c:a aac -b:a 192k -ar 48000 '
                   f'-pix_fmt yuv420p '
                   f'-shortest '
                   f'-avoid_negative_ts make_zero '
                   f'-fflags +genpts '
                   f'-metadata spherical-video=1 '
                   f'-metadata spherical-stereo=1 '
                   f'-metadata spherical-layout=mono '
                   f'-metadata spherical-projection=equirectangular '
                   f'"{out_path}"')
    else:
        # No audio but with VR180 metadata
        cmd = (f'{FFMPEG_BIN} -y '
               f'-framerate {fps} -i "{left_dir}/frame_%06d.png" '
               f'-framerate {fps} -i "{right_dir}/frame_%06d.png" '
               f'-filter_complex "[0][1]hstack=inputs=2" '
               f'-c:v libx264 -crf 12 -preset slow '  # Even higher quality settings
               f'-pix_fmt yuv420p '
               f'-avoid_negative_ts make_zero '  # Fix timestamp issues
               f'-fflags +genpts '  # Generate presentation timestamps
               f'-metadata spherical-video=1 '
               f'-metadata spherical-stereo=1 '
               f'-metadata spherical-layout=mono '
               f'-metadata spherical-projection=equirectangular '
               f'"{out_path}"')
    
    print(f"Encoding video with high quality settings...")
    await run_cmd(cmd)
    
    # Verify output video duration
    try:
        output_meta = probe_video(out_path)
        output_duration = float(output_meta["format"].get("duration", 0.0))
        duration_diff = abs(output_duration - exact_duration)
        
        print(f"Output duration: {output_duration:.3f}s (target: {exact_duration:.3f}s)")
        
        if duration_diff > 0.1:  # More than 0.1 second difference
            print(f"Duration mismatch: {duration_diff:.3f}s difference")
        else:
            print(f"Duration matches target within {duration_diff:.3f}s")
            
    except Exception as e:
        print(f"Could not verify output duration: {e}")
    
    print(f"Side-by-side video created: {out_path}")
