# api/models.py
import os
import base64
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any
from config import HUGGINGFACE_TOKEN, HF_API_URL, MODEL_DEPTH

HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"} if HUGGINGFACE_TOKEN else {}

def pil_to_bytes(p: Image.Image, fmt="PNG") -> bytes:
    buf = BytesIO()
    p.save(buf, fmt)
    return buf.getvalue()

def depth_from_hf(image_pil: Image.Image, model_id: str = MODEL_DEPTH, return_exr: bool=False) -> np.ndarray:
    """
    Use local CV2 depth estimation instead of Hugging Face API.
    Returns normalized depth map.
    """
    return create_local_depth_map(image_pil)

def create_local_depth_map(image_pil: Image.Image) -> np.ndarray:
    """Create a fast and reliable depth map using simple OpenCV methods"""
    import cv2
    import numpy as np
    
    try:
        # Convert to numpy array
        img_array = np.array(image_pil.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Simple and fast depth estimation using gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient magnitude
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
        # Invert gradient (edges are closer, smooth areas are farther)
        depth_map = 1.0 - gradient_magnitude
        
        # Add center bias for more realistic depth
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        center_bias = 1.0 - np.sqrt((x - center_x)**2 + (y - center_y)**2) / np.sqrt(center_x**2 + center_y**2)
        
        # Combine gradient and center bias
        depth_map = 0.7 * depth_map + 0.3 * center_bias
        
        # Apply Gaussian blur for smoothness
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 1.0)
        
        # Clean up memory
        del grad_x, grad_y, gradient_magnitude, center_bias
        
        return np.clip(depth_map, 0, 1).astype(np.float32)
        
    except Exception as e:
        print(f"❌ Depth estimation error: {e}")
        # Return a simple center bias depth map as fallback
        h, w = image_pil.size[1], image_pil.size[0]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        depth_map = 1.0 - np.sqrt((x - center_x)**2 + (y - center_y)**2) / np.sqrt(center_x**2 + center_y**2)
        return np.clip(depth_map, 0, 1).astype(np.float32)
