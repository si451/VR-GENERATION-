#!/usr/bin/env python3
"""
VR Platform Backend Installation Script
This script installs all necessary dependencies and sets up the backend environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    if not run_command(f"{sys.executable} -m pip install -r {requirements_file}", "Installing requirements"):
        return False
    
    return True

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    print("\n🎬 Checking FFmpeg installation...")
    try:
        result = subprocess.run("ffmpeg -version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is installed")
            return True
        else:
            print("❌ FFmpeg not found")
            return False
    except:
        print("❌ FFmpeg not found")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    print("\n⚙️ Setting up environment file...")
    env_file = Path(__file__).parent / ".env"
    env_example = Path(__file__).parent / ".env.example"
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        # Copy example to .env
        with open(env_example, 'r') as f:
            content = f.read()
        with open(env_file, 'w') as f:
            f.write(content)
        print("✅ Created .env file from .env.example")
        print("⚠️  Please update the .env file with your actual configuration")
    else:
        # Create basic .env file
        env_content = """# Hugging Face API Configuration
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
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created basic .env file")
        print("⚠️  Please update the .env file with your actual configuration")
    
    return True

def main():
    """Main installation function"""
    print("🚀 VR Platform Backend Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed during dependency installation")
        sys.exit(1)
    
    # Check FFmpeg
    ffmpeg_ok = check_ffmpeg()
    if not ffmpeg_ok:
        print("\n⚠️  FFmpeg is not installed. Please install FFmpeg:")
        print("   Windows: Download from https://ffmpeg.org/download.html")
        print("   macOS: brew install ffmpeg")
        print("   Ubuntu: sudo apt install ffmpeg")
    
    # Create .env file
    if not create_env_file():
        print("\n❌ Failed to create .env file")
        sys.exit(1)
    
    print("\n🎉 Backend installation completed!")
    print("\n📋 Next steps:")
    print("1. Update the .env file with your Hugging Face token")
    print("2. Install FFmpeg if not already installed")
    print("3. Run the backend: cd api && python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload")
    
    if not ffmpeg_ok:
        print("\n⚠️  Remember to install FFmpeg before running the backend!")

if __name__ == "__main__":
    main()
