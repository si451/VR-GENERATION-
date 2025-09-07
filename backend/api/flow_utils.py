# api/flow_utils.py
import cv2
import numpy as np
from typing import Tuple

cv2.setNumThreads(0)

def compute_flow_cv2(prev_bgr: np.ndarray, next_bgr: np.ndarray, method: str = "DIS") -> np.ndarray:
    prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    next_ = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
    if method == "DIS":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc(prev, next_, None)
    elif method == "TVL1":
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = tvl1.calc(prev, next_, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(prev, next_, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    return flow.astype(np.float32)

def warp_image_with_flow(src_img: np.ndarray, flow_target_to_src: np.ndarray) -> np.ndarray:
    h, w = src_img.shape[:2]
    
    # Ensure flow dimensions match image dimensions
    if flow_target_to_src.shape[:2] != (h, w):
        # For 3D flow arrays, we need to resize each channel separately
        if len(flow_target_to_src.shape) == 3:
            flow_resized = np.zeros((h, w, flow_target_to_src.shape[2]), dtype=flow_target_to_src.dtype)
            for i in range(flow_target_to_src.shape[2]):
                flow_resized[:, :, i] = cv2.resize(flow_target_to_src[:, :, i], (w, h))
            flow_target_to_src = flow_resized
        else:
            flow_target_to_src = cv2.resize(flow_target_to_src, (w, h))
    
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow_target_to_src[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_target_to_src[..., 1]).astype(np.float32)
    warped = cv2.remap(src_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped

def forward_backward_consistency_mask(flow01: np.ndarray, flow10: np.ndarray, thr: float = 1.0) -> np.ndarray:
    h, w = flow01.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    fwd_x = grid_x + flow01[..., 0]
    fwd_y = grid_y + flow01[..., 1]
    # sample flow10 at forward positions
    flow10_x = cv2.remap(flow10[...,0].astype(np.float32), fwd_x.astype(np.float32), fwd_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    flow10_y = cv2.remap(flow10[...,1].astype(np.float32), fwd_x.astype(np.float32), fwd_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    round_trip_x = flow01[...,0] + flow10_x
    round_trip_y = flow01[...,1] + flow10_y
    err = np.sqrt(round_trip_x**2 + round_trip_y**2)
    return (err < thr)

def interpolate_occlusion_aware(frame0: np.ndarray, frame1: np.ndarray, flow01: np.ndarray, flow10: np.ndarray, t: float=0.5) -> np.ndarray:
    h, w = frame0.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    f0t_x = flow01[...,0] * t
    f0t_y = flow01[...,1] * t
    f1t_x = flow10[...,0] * (1 - t)
    f1t_y = flow10[...,1] * (1 - t)
    map0_x = (grid_x + f0t_x).astype(np.float32)
    map0_y = (grid_y + f0t_y).astype(np.float32)
    map1_x = (grid_x + f1t_x).astype(np.float32)
    map1_y = (grid_y + f1t_y).astype(np.float32)
    warp0 = cv2.remap(frame0, map0_x, map0_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    warp1 = cv2.remap(frame1, map1_x, map1_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask_consistent = forward_backward_consistency_mask(flow01, flow10, thr=1.0).astype(np.float32)
    conf0 = mask_consistent[..., None]
    conf1 = 1.0 - conf0
    out = conf0 * warp0.astype(np.float32) + conf1 * warp1.astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out
