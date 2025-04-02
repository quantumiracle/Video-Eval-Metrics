import os
import torch
import sys
import numpy as np
from torchvision.io import read_video
from tqdm import tqdm
import lpips
from PIL import Image
import torchvision.transforms as transforms

# Get current directory and import parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))


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


def calculate_lpips(folder1, folder2, net_type='alex'):
    """
    Calculate the LPIPS score between videos in two folders.
    
    Args:
        folder1: path to the first folder containing videos
        folder2: path to the second folder containing videos
        net_type: Network backbone to use for LPIPS ('alex', 'vgg', or 'squeeze')
        
    Returns: 
        average LPIPS score
    """
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith('.mp4')])
    files2 = sorted([f for f in os.listdir(folder2) if f.endswith('.mp4')])
    
    assert len(files1) == len(files2), "The number of videos in both folders must be the same and aligned."
    print(f"Calculating LPIPS score for {len(files1)} videos using {net_type} backbone")
    
    # Load LPIPS model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_model = lpips.LPIPS(net=net_type).to(device)
    
    # Preprocessing transform to normalize images to [-1, 1] range for LPIPS
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    lpips_scores = []
    # progress bar
    for file1, file2 in tqdm(zip(files1, files2), total=len(files1), desc="Calculating LPIPS score"):
        video_path1 = os.path.join(folder1, file1)
        video_path2 = os.path.join(folder2, file2)
        
        frames1, _, _ = read_video(video_path1)
        frames2, _, _ = read_video(video_path2)

        # Align video shapes
        frames1, frames2 = align_video_shapes(frames1, frames2)  # (T, C, H, W)
            
        # Calculate LPIPS for each frame pair
        frame_lpips_scores = []
        for f1, f2 in zip(frames1, frames2):
            # Preprocess images to the required format for LPIPS
            # LPIPS expects inputs in range [-1, 1]
            f1_norm = preprocess(f1 / 255.0).unsqueeze(0).to(device)
            f2_norm = preprocess(f2 / 255.0).unsqueeze(0).to(device)
            
            # Calculate LPIPS distance
            with torch.no_grad():
                distance = lpips_model(f1_norm, f2_norm)
            
            frame_lpips_scores.append(distance.item())
        
        # Average LPIPS across frames
        video_lpips = np.mean(frame_lpips_scores)
        lpips_scores.append(video_lpips)
    
    return sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0.0


if __name__ == "__main__":
    # Example usage
    folder1 = "/path/to/generated_videos"
    folder2 = "/path/to/reference_videos"
    
    lpips_score = calculate_lpips(folder1, folder2)
    print(f"LPIPS score: {lpips_score}")
    
    # Optionally run with different backbones
    lpips_vgg = calculate_lpips(folder1, folder2, net_type='vgg')
    print(f"LPIPS score (VGG backbone): {lpips_vgg}") 
