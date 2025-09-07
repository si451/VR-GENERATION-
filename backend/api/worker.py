#!/usr/bin/env python3
"""
VR180 Processing Worker
Handles heavy video processing in a separate process
"""
import sys
import os
import asyncio
import uuid
from pathlib import Path

# Add the current directory to Python path so we can import from api
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from pipeline import process_job, process_video_parallel
from config import WORKSPACE_DIR

def main():
    if len(sys.argv) != 3:
        print("Usage: python worker.py <job_id> <input_video_path>")
        sys.exit(1)
    
    job_id = sys.argv[1]
    input_video_path = Path(sys.argv[2])
    
    if not input_video_path.exists():
        print(f"❌ Input video not found: {input_video_path}")
        sys.exit(1)
    
    print(f"Starting worker with job ID: {job_id}")
    print(f"Processing: {input_video_path}")
    
    # Create job directory
    job_dir = WORKSPACE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy input video to job directory (if not already there)
    input_path = job_dir / "input.mp4"
    import shutil
    
    # Check if the file is already in the correct location
    if input_video_path.resolve() != input_path.resolve():
        shutil.copy2(input_video_path, input_path)
        print(f"Copied input to: {input_path}")
    else:
        print(f"Input already in job directory: {input_path}")
    
    try:
        # Import status manager
        from api.status import StatusManager
        status_mgr = StatusManager(WORKSPACE_DIR)
        
        # Create output path
        output_path = job_dir / f"{job_id}_sbs.mp4"
        
        # Run the parallel processing job
        result = process_video_parallel(job_id, str(input_path), str(output_path), status_mgr)
        
        if result["status"] == "success":
            print(f"Job {job_id} completed successfully")
            print(f"Results saved in: {job_dir}")
            
            # List output files
            output_files = list(job_dir.glob("*"))
            print(f"Generated files:")
            for file in output_files:
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"  - {file.name} ({size_mb:.1f} MB)")
        else:
            print(f"Job {job_id} failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
                
    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
