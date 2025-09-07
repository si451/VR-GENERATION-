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
    """Create a high-quality depth map using improved OpenCV methods with Guided Filter refinement"""
    import cv2
    import numpy as np
    
    # Import the guided filter from the contrib module
    try:
        from cv2.ximgproc import guidedFilter
    except ImportError:
        print("Guided Filter not found. Please install opencv-contrib-python.")
        # Fallback to basic depth estimation
        img_array = np.array(image_pil.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        # Simple center bias fallback
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        depth_map = 1.0 - np.sqrt((x - center_x)**2 + (y - center_y)**2) / np.sqrt(center_x**2 + center_y**2)
        return np.clip(depth_map, 0, 1).astype(np.float32)

    img_array = np.array(image_pil.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Method 1: Enhanced gradient-based depth estimation
    # Use smaller Sobel kernels for sharper edges
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Apply lighter Gaussian blur to preserve more detail
    gradient_magnitude = cv2.GaussianBlur(gradient_magnitude, (5, 5), 0)
    
    # Normalize gradient magnitude
    if gradient_magnitude.max() > 0:
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
    
    # Method 2: Center bias (objects in center are closer)
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    center_bias = 1.0 - np.sqrt((x - center_x)**2 + (y - center_y)**2) / np.sqrt(center_x**2 + center_y**2)
    center_bias = np.clip(center_bias, 0.1, 1.0)
    
    # Method 3: Laplacian for edge-based depth
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.abs(laplacian)
    if laplacian.max() > 0:
        laplacian = laplacian / laplacian.max()
    
    # Combine all methods with weights
    initial_depth = (
        0.4 * gradient_magnitude +  # Edge-based depth
        0.4 * center_bias +         # Center bias
        0.2 * laplacian             # Laplacian edges
    )
    
    # Apply lighter bilateral filter to preserve more detail
    initial_depth = cv2.bilateralFilter(initial_depth.astype(np.float32), 5, 50, 50)
    
    # Apply lighter morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    initial_depth = cv2.morphologyEx(initial_depth, cv2.MORPH_CLOSE, kernel)
    
    # Apply lighter Gaussian blur for smoother but sharper depth transitions
    initial_depth = cv2.GaussianBlur(initial_depth, (3, 3), 0)
    
    # Ensure depth map is in 0-1 range
    initial_depth = np.clip(initial_depth, 0, 1)
    
    # Invert depth (closer objects = higher values)
    initial_depth = 1.0 - initial_depth
    
    # Convert image and depth map to 32-bit floating point for the filter
    img_cv_float = np.float32(img_array)
    initial_depth_float = np.float32(initial_depth)

    # Apply the Guided Filter to refine the depth map
    # The first parameter is the guide image (your original image)
    # The second is the input image (your initial depth map)
    # The third is the radius, and the fourth is the regularization parameter (epsilon)
    try:
        refined_depth = guidedFilter(guide=img_cv_float, src=initial_depth_float, radius=5, eps=1e-3)
        
        # Normalize the final depth map to the 0-1 range
        refined_depth = (refined_depth - refined_depth.min()) / (refined_depth.max() - refined_depth.min())
        
        return refined_depth.astype(np.float32)
        
    except Exception as e:
        print(f"Guided filter failed, using original depth map: {e}")
        # Fallback to original depth map if guided filter fails
        return initial_depth.astype(np.float32)
