import os
import torch
import numpy as np
from torchvision.io import read_video
from tqdm import tqdm
import torch.nn.functional as F
from scipy import linalg
import sys
from PIL import Image
import warnings

# Import align_video_shapes function
from eval_edge import align_video_shapes

try:
    # Try to import pytorch_i3d if available
    sys.path.append('./pytorch-i3d')  # Adjust if your I3D implementation is elsewhere
    from pytorch_i3d import InceptionI3d
except ImportError:
    warnings.warn("Could not import InceptionI3d. Please install the pytorch_i3d package or provide your own I3D implementation.")
    print("Try 'git clone https://github.com/piergiaj/pytorch-i3d.git' first")


def load_i3d_model(device='cuda'):
    """
    Load a pre-trained I3D model for feature extraction.
    
    Args:
        device: Device to load the model on
    
    Returns:
        Pre-trained I3D model
    """
    try:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('./pytorch-i3d/models/rgb_imagenet.pt', map_location=device))
        i3d = i3d.to(device)
        i3d.eval()
        return i3d
    except:
        print("Could not load I3D model. Please ensure you have downloaded the pre-trained weights.")
        print("You can download them from: https://github.com/piergiaj/pytorch-i3d/tree/master/models")
        raise


def preprocess_video(frames, target_size=(224, 224), clip_len=32, stride=32):
    """
    Preprocess video frames for I3D model and split into clips.
    
    Args:
        frames: Video frames tensor (T, C, H, W)
        target_size: Target size for resizing
        clip_len: Length of each clip in frames
        stride: Stride between consecutive clips
        
    Returns:
        Preprocessed frames (B, C, T, H, W) where B is number of clips
    """
    # Ensure frames have the correct shape (T, C, H, W)
    if frames.shape[1] != 3:
        frames = frames.permute(0, 3, 1, 2)
    
    # Resize to target size
    frames = F.interpolate(frames, size=target_size, mode='bilinear', align_corners=False)
    
    # Normalize to [-1, 1]
    frames = frames / 127.5 - 1
    
    # Split into clips
    T = frames.shape[0]
    clips = []
    for start in range(0, T - clip_len + 1, stride):
        clip = frames[start:start + clip_len]  # (clip_len, C, H, W)
        # Reshape to (1, C, T, H, W) as expected by I3D
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0)
        clips.append(clip)
    
    # Concatenate all clips along batch dimension
    return torch.cat(clips, dim=0)  # (B, C, T, H, W)


def extract_features(model, video_frames, batch_size=16, device='cuda'):
    """
    Extract features from video frames using the I3D model.
    
    Args:
        model: Pre-trained I3D model
        video_frames: Video frames tensor (T, C, H, W)
        batch_size: Batch size for processing
        device: Device to perform extraction on
        
    Returns:
        Features tensor
    """
    # Preprocess frames
    frames = preprocess_video(video_frames, target_size=(224, 224)).to(device)
    
    # Extract features
    with torch.no_grad():
        features = model.extract_features(frames) # (B, 1024, T, 1, 1)
        features = torch.mean(features, dim=[2, 3, 4])  # (B, 1024)
    
    return features.cpu().numpy()


def calculate_fvd(features1, features2):
    """
    Calculate Fréchet Video Distance between two sets of features.
    
    Args:
        features1: Features from first set of videos (N, D)
        features2: Features from second set of videos (N, D)
        
    Returns:
        FVD score (lower is better)
    """
    # Calculate mean and covariance for each feature set
    mu1 = np.mean(features1, axis=0)
    sigma1 = np.cov(features1, rowvar=False)
    
    mu2 = np.mean(features2, axis=0)
    sigma2 = np.cov(features2, rowvar=False)
    
    # Calculate FVD (similar to FID)
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    fvd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fvd


def calculate_video_fvd(folder1, folder2):
    """
    Calculate the FVD score between videos in two folders.
    
    Args:
        folder1: path to the first folder containing videos
        folder2: path to the second folder containing videos
        
    Returns: 
        FVD score (lower is better)
    """
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith('.mp4')])
    files2 = sorted([f for f in os.listdir(folder2) if f.endswith('.mp4')])
    
    assert len(files1) == len(files2), "The number of videos in both folders must be the same and aligned."
    print(f"Calculating FVD score for {len(files1)} videos")
    
    # Load I3D model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        i3d_model = load_i3d_model(device)
    except Exception as e:
        print(f"Error loading I3D model: {e}")
        print("Using a placeholder. FVD results will not be accurate.")
        return float('inf')
    
    all_features1 = []
    all_features2 = []
    
    # Extract features from each video
    for file1, file2 in tqdm(zip(files1, files2), total=len(files1), desc="Extracting features"):
        video_path1 = os.path.join(folder1, file1)
        video_path2 = os.path.join(folder2, file2)
        
        frames1, _, _ = read_video(video_path1)
        frames2, _, _ = read_video(video_path2)

        # Align video shapes
        frames1, frames2 = align_video_shapes(frames1, frames2)
        
        # Extract features
        features1 = extract_features(i3d_model, frames1, device=device)
        features2 = extract_features(i3d_model, frames2, device=device)
        
        all_features1.append(features1)
        all_features2.append(features2)
    
    # Concatenate features
    all_features1 = np.concatenate(all_features1, axis=0)
    all_features2 = np.concatenate(all_features2, axis=0)
    
    # Calculate FVD
    fvd_score = calculate_fvd(all_features1, all_features2)
    
    return fvd_score


if __name__ == "__main__":
    folder1 = "/path/to/generated_videos"  # Replace with actual path
    folder2 = "/path/to/reference_videos"  # Replace with actual path
    fvd_score = calculate_video_fvd(folder1, folder2)
    print(f"FVD score: {fvd_score}") 