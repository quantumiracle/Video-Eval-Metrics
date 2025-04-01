import torch
import torchvision.transforms as T
from torchvision.io import read_video
import imageio
import os
import multiprocessing as mp
from tqdm import tqdm
import queue
import cv2
import numpy as np

def get_canny_edges(frames, low_threshold=None, high_threshold=None):
    """
    Get Canny edge detection for a list of frames
    frames: list of frames (T, C, H, W)
    returns: list of edge images (T, C, H, W)
    """
    edge_frames = []

    for frame in frames:
        # Convert to numpy and correct format for OpenCV
        frame_np = frame.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        if low_threshold is None or high_threshold is None:
            # Otsu’s method or percentile-based selection for auto thresholding
            sigma = 0.33
            median = np.median(frame_np)
            low_threshold = int(max(0, (1.0 - sigma) * median))
            high_threshold = int(min(255, (1.0 + sigma) * median))

        # Convert to uint8 (required by Canny)
        if frame_np.dtype != np.uint8:
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
        
        # Convert to grayscale if it's RGB
        if frame_np.shape[2] == 3:
            gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame_np.squeeze()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # Convert back to RGB for visualization
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Convert back to torch tensor
        edge_tensor = torch.from_numpy(edges_rgb).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        edge_frames.append(edge_tensor)
    
    return torch.stack(edge_frames)