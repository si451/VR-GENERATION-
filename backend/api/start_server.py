#!/usr/bin/env python3
"""
Railway startup script for VR180 Backend
Handles port binding properly for Railway deployment
"""
import os
import uvicorn

if __name__ == "__main__":
    # Get port from Railway environment variable
    port = int(os.getenv("PORT", "8000"))
    
    print(f"🚀 Starting VR180 Backend on port {port}")
    print(f"🔧 Host: 0.0.0.0")
    print(f"🔧 Port: {port}")
    
    # Start the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info"
    )
