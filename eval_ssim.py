import os
import torch
import sys
from torchvision.io import read_video
from tqdm import tqdm
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import numpy as np


def align_video_shapes(frames1, frames2, verbose=False):
    """
    Align two video tensors to have the same spatial and temporal dimensions.
    
    Args:
        frames1: First video tensor of shape (T, H, W, C)
        frames2: Second video tensor of shape (T, H, W, C)
        verbose: Whether to print shape information
        
    Returns:
        Tuple of aligned video tensors (aligned_frames1, aligned_frames2), (T, C, H, W)
    """
    if verbose:
        print('original shapes, frames1:', frames1.shape, 'frames2:', frames2.shape)
    
    # Match spatial dimensions (height and width)
    # make t,h,w,c to t,c,h,w
    frames1 = frames1.permute(0, 3, 1, 2)
    frames2 = frames2.permute(0, 3, 1, 2)
    # If frames2 (target) is larger, resize frames2 to match frames1
    if frames2.shape[2] > frames1.shape[2] and frames2.shape[3] > frames1.shape[3]:
        frames2 = transforms.Resize(frames1.shape[2:4])(frames2)
    elif frames2.shape[2] < frames1.shape[2] and frames2.shape[3] < frames1.shape[3]:
        frames2 = transforms.Resize(frames1.shape[2:4])(frames2)
    elif frames2.shape[2] < frames1.shape[2]:
        frames2 = transforms.Resize(frames1.shape[2:4])(frames2)
    
    # Match temporal dimension (number of frames)
    # If frames1 is longer, take the first frames2.shape[0] frames
    if frames1.shape[0] > frames2.shape[0]:
        frames1 = frames1[:frames2.shape[0], :, :, :]
    # If frames2 is longer, take the first frames1.shape[0] frames
    elif frames2.shape[0] > frames1.shape[0]:
        frames2 = frames2[:frames1.shape[0], :, :, :]
    
    if verbose:
        print('aligned shapes, frames1:', frames1.shape, 'frames2:', frames2.shape)
    
    return frames1, frames2



def calculate_blur_ssim(folder1, folder2, blur=False, blur_kernel_size=11):
    """
    Calculate the Blur SSIM score between videos in two folders.
    
    Args:
        folder1: path to the first folder containing videos
        folder2: path to the second folder containing videos
        blur_kernel_size: size of the Gaussian blur kernel
        
    Returns: 
        average Blur SSIM score
    """
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith('.mp4')])
    files2 = sorted([f for f in os.listdir(folder2) if f.endswith('.mp4')])
    
    assert len(files1) == len(files2), "The number of videos in both folders must be the same and aligned."
    print(f"Calculating Blur SSIM score for {len(files1)} videos")
    
    # Create blur transform
    blur = transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=3.0) if blur else None
    
    ssim_scores = []
    # progress bar
    for file1, file2 in tqdm(zip(files1, files2), total=len(files1), desc="Calculating Blur SSIM score"):
        video_path1 = os.path.join(folder1, file1)
        video_path2 = os.path.join(folder2, file2)
        
        frames1, _, _ = read_video(video_path1)
        frames2, _, _ = read_video(video_path2)

        # Align video shapes
        frames1, frames2 = align_video_shapes(frames1, frames2) # (T, C, H, W)
            
        # Apply blur to both videos
        blurred_frames1 = torch.stack([
            blur(frame).permute(1, 2, 0) if blur else frame.permute(1, 2, 0)
            for frame in frames1
        ])
        
        blurred_frames2 = torch.stack([
            blur(frame).permute(1, 2, 0) if blur else frame.permute(1, 2, 0)
            for frame in frames2
        ])
        
        # Calculate SSIM for each frame pair
        frame_ssim_scores = []
        for bf1, bf2 in zip(blurred_frames1, blurred_frames2):
            # Convert to numpy and to grayscale for SSIM calculation
            bf1_np = bf1.cpu().numpy().mean(axis=2).astype(np.float32)
            bf2_np = bf2.cpu().numpy().mean(axis=2).astype(np.float32)
            
            # Calculate SSIM
            score = ssim(bf1_np, bf2_np, data_range=255.0)
            frame_ssim_scores.append(score)
        
        # Average SSIM across frames
        video_ssim = np.mean(frame_ssim_scores)
        ssim_scores.append(video_ssim)
    
    return sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0.0


if __name__ == "__main__":
    folder1 = "/path/to/generated_videos"  # Replace with actual path
    folder2 = "/path/to/reference_videos"  # Replace with actual path
    ssim_score = calculate_blur_ssim(folder1, folder2, blur=False)
    print(f"(Blur) SSIM score: {ssim_score}")