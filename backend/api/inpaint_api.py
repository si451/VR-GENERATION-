# api/inpaint_api.py
import requests
from PIL import Image
from io import BytesIO
from typing import Tuple
import base64
from .config import HF_API_URL, HUGGINGFACE_TOKEN, MODEL_INPAINT
from .models import pil_to_bytes

HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"} if HUGGINGFACE_TOKEN else {}

def sd_inpaint(image: Image.Image, mask: Image.Image, prompt: str = "Photorealistic inpainting"):
    """
    Use local CV2 inpainting instead of Hugging Face API.
    Returns PIL Image.
    Expects mask as white areas=holes to fill (255).
    """
    return local_inpaint(image, mask)

def local_inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Multi-scale context-aware inpainting using OpenCV with improved blending"""
    import cv2
    import numpy as np
    
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask_cv = np.array(mask)
    
    # Apply Gaussian blur to mask for smoother inpainting
    mask_blurred = cv2.GaussianBlur(mask_cv, (5, 5), 0)
    
    # Multi-scale inpainting approach
    # First pass: Inpaint using both TELEA and NS methods
    inpainted_telea = cv2.inpaint(img_cv, mask_blurred, 5, cv2.INPAINT_TELEA)
    inpainted_ns = cv2.inpaint(img_cv, mask_blurred, 5, cv2.INPAINT_NS)
    
    # Second pass: Refine the results with different parameters
    # Use smaller radius for more detailed inpainting
    inpainted_telea_refined = cv2.inpaint(inpainted_telea, mask_blurred, 3, cv2.INPAINT_TELEA)
    inpainted_ns_refined = cv2.inpaint(inpainted_ns, mask_blurred, 3, cv2.INPAINT_NS)
    
    # Create a gradient mask for context-aware blending
    # This creates a smooth transition from the edge of the hole to the center
    dist_transform = cv2.distanceTransform(255 - mask_blurred, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Create a more sophisticated blending mask
    # This gives more weight to the NS method near the edges (where it's better)
    # and more weight to the TELEA method in the interior
    alpha = np.clip(dist_norm * 2.0, 0, 1.0)
    alpha = cv2.merge([alpha, alpha, alpha])
    
    # Blend the refined results based on distance from edge
    blended_result = (inpainted_ns_refined * (1 - alpha)) + (inpainted_telea_refined * alpha)
    blended_result = np.clip(blended_result, 0, 255).astype(np.uint8)
    
    # Apply additional smoothing to reduce artifacts
    # Use a more conservative bilateral filter
    blended_result = cv2.bilateralFilter(blended_result, 7, 50, 50)
    
    # Final refinement: Apply a light Gaussian blur to smooth any remaining artifacts
    blended_result = cv2.GaussianBlur(blended_result, (3, 3), 0)
    
    # Convert back to PIL
    inpainted_rgb = cv2.cvtColor(blended_result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(inpainted_rgb)
