import os
import torch
import numpy as np
from torchvision.io import read_video
from tqdm import tqdm
import math
# Import align_video_shapes function
from eval_edge import align_video_shapes


def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images.
    
    Args:
        img1: First image tensor (C, H, W)
        img2: Second image tensor (C, H, W)
        
    Returns:
        PSNR value in dB
    """
    # Convert to float and range [0, 1]
    img1 = img1.float() / 255.0
    img2 = img2.float() / 255.0
    
    # Calculate MSE (Mean Squared Error)
    mse = torch.mean((img1 - img2) ** 2)
    
    # Handle division by zero
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse.item()))
    
    return psnr


def calculate_video_psnr(folder1, folder2):
    """
    Calculate the PSNR score between videos in two folders.
    
    Args:
        folder1: path to the first folder containing videos
        folder2: path to the second folder containing videos
        
    Returns: 
        average PSNR score
    """
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith('.mp4')])
    files2 = sorted([f for f in os.listdir(folder2) if f.endswith('.mp4')])
    
    assert len(files1) == len(files2), "The number of videos in both folders must be the same and aligned."
    print(f"Calculating PSNR score for {len(files1)} videos")
    
    psnr_scores = []
    # progress bar
    for file1, file2 in tqdm(zip(files1, files2), total=len(files1), desc="Calculating PSNR score"):
        video_path1 = os.path.join(folder1, file1)
        video_path2 = os.path.join(folder2, file2)
        
        frames1, _, _ = read_video(video_path1)
        frames2, _, _ = read_video(video_path2)

        # Align video shapes
        frames1, frames2 = align_video_shapes(frames1, frames2) # (T, C, H, W)
        
        # Calculate PSNR for each frame pair
        frame_psnr_scores = []
        for f1, f2 in zip(frames1, frames2):
            psnr = calculate_psnr(f1, f2)
            frame_psnr_scores.append(psnr)
        
        # Average PSNR across frames
        video_psnr = np.mean(frame_psnr_scores)
        psnr_scores.append(video_psnr)
    
    return sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0.0


if __name__ == "__main__":
    folder1 = "/path/to/generated_videos"  # Replace with actual path
    folder2 = "/path/to/reference_videos"  # Replace with actual path
    psnr_score = calculate_video_psnr(folder1, folder2)
    print(f"PSNR score: {psnr_score}") 