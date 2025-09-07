# api/status.py
import json
from pathlib import Path
from datetime import datetime
import os
from threading import Lock
from typing import Dict, Any

class StatusManager:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self._lock = Lock()

    def job_dir(self, job_id: str) -> Path:
        p = self.workspace / job_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def status_file(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "status.json"

    def update(self, job_id: str, data: Dict[str, Any]):
        with self._lock:
            sf = self.status_file(job_id)
            data.setdefault("updated_at", datetime.utcnow().isoformat() + "Z")
            
            # Try multiple approaches to handle Windows file locking
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Method 1: Try direct write with retry
                    with open(sf, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    break  # Success, exit retry loop
                    
                except (PermissionError, OSError) as e:
                    if attempt < max_retries - 1:
                        # Wait a bit before retrying
                        import time
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    else:
                        # Last attempt failed, try alternative approach
                        try:
                            # Method 2: Try with a unique temp file
                            import uuid
                            tmp = sf.with_suffix(f".tmp.{uuid.uuid4().hex}")
                            with open(tmp, "w", encoding="utf-8") as f:
                                json.dump(data, f, indent=2)
                                f.flush()
                                os.fsync(f.fileno())
                            
                            # Try to replace the original file
                            try:
                                os.replace(tmp, sf)
                            except PermissionError:
                                # If replace fails, just leave the temp file
                                # The read method will handle this
                                pass
                            break
                            
                        except Exception as e2:
                            print(f"Status update failed after {max_retries} attempts: {e2}")
                            # Don't fail the entire process for status update issues
                            break

    def read(self, job_id: str) -> Dict[str, Any]:
        sf = self.status_file(job_id)
        
        # Try to read the main status file first
        if sf.exists():
            try:
                with open(sf, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (PermissionError, OSError, json.JSONDecodeError):
                # If main file fails, try to find a temp file
                pass
        
        # Look for temp files as fallback
        job_dir = self.job_dir(job_id)
        temp_files = list(job_dir.glob("status.tmp.*"))
        if temp_files:
            # Use the most recent temp file
            latest_temp = max(temp_files, key=lambda f: f.stat().st_mtime)
            try:
                with open(latest_temp, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (PermissionError, OSError, json.JSONDecodeError):
                pass
        
        return {}
