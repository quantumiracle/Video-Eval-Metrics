import os
import torch
import sys
import numpy as np
from torchvision.io import read_video
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch.nn.functional as F
# Import align_video_shapes function
from eval_edge import align_video_shapes


def scale_invariant_rmse(pred_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute scale-invariant RMSE between predicted and ground-truth depth maps.

    Args:
        pred_depth (torch.Tensor): Predicted depth map, shape (..., H, W)
        gt_depth (torch.Tensor): Ground-truth depth map, shape (..., H, W)
        mask (torch.Tensor, optional): Optional binary mask (same shape) to exclude invalid regions.

    Returns:
        torch.Tensor: The scalar si-RMSE value.

    Formula (Eigen et al., 2014):
        Let D_p be the predicted depth, D_g the ground-truth depth.
        Define:
            d_i = log(D_p_i) - log(D_g_i)
            n = number of valid pixels

        Then:
            si-RMSE = sqrt( (1/n) * sum_i d_i^2 - (1/n^2) * (sum_i d_i)^2 )

        This penalizes relative errors and is invariant to global scale.
    """
    # Flatten for simplicity
    pred = pred_depth.flatten()
    gt = gt_depth.flatten()

    if mask is not None:
        mask = mask.flatten()
        pred = pred[mask]
        gt = gt[mask]

    # Avoid log(0) or negative depths
    eps = 1e-8
    pred = torch.clamp(pred, min=eps)
    gt = torch.clamp(gt, min=eps)

    log_diff = torch.log(pred) - torch.log(gt)
    n = log_diff.numel()

    mse = torch.mean(log_diff ** 2)             # (1/n) * sum_i d_i^2
    square_mean = torch.mean(log_diff) ** 2     # ( (1/n) * sum_i d_i )^2

    si_rmse = torch.sqrt(mse - square_mean)
    
    return si_rmse


def calculate_depth_si_rmse(folder1, folder2):
    """
    Calculate the Depth si-RMSE score between videos in two folders.
    
    Args:
        folder1: path to the first folder containing videos
        folder2: path to the second folder containing videos
        
    Returns: 
        average Depth si-RMSE score
    """
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith('.mp4')])
    files2 = sorted([f for f in os.listdir(folder2) if f.endswith('.mp4')])
    
    assert len(files1) == len(files2), "The number of videos in both folders must be the same and aligned."
    print(f"Calculating Depth si-RMSE score for {len(files1)} videos")
    
    # Load DepthAnything model - using the correct model ID
    try:
        # Use the correct model ID based on the example
        processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        print("Successfully loaded Depth Anything V2 Small model")
    except Exception as e:
        print(f"Could not load Depth Anything V2 Small model: {e}. Trying alternative model.")
        try:
            # Try fallback to Base model
            processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-Base-hf")
            model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-Base-hf")
            print("Successfully loaded Depth Anything Base model")
        except Exception as e2:
            print(f"Could not load alternative model: {e2}. Using the smallest model as final fallback.")
            # Final fallback to the smallest model
            processor = AutoImageProcessor.from_pretrained("LiheYoung/depth_anything_small")
            model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth_anything_small")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    si_rmse_scores = []
    # progress bar
    for file1, file2 in tqdm(zip(files1, files2), total=len(files1), desc="Calculating Depth si-RMSE score"):
        video_path1 = os.path.join(folder1, file1)
        video_path2 = os.path.join(folder2, file2)
        
        frames1, _, _ = read_video(video_path1)
        frames2, _, _ = read_video(video_path2)

        # Align video shapes
        frames1, frames2 = align_video_shapes(frames1, frames2)
        
        # Calculate depth for each frame pair
        frame_si_rmse_scores = []
        for f1, f2 in zip(frames1, frames2):
            # Process each frame for depth estimation
            with torch.no_grad():
                try:
                    # CHW to HWC
                    f1_np = f1.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    f2_np = f2.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    
                    # Convert to PIL images
                    f1_pil = Image.fromarray(f1_np)
                    f2_pil = Image.fromarray(f2_np)
                    
                    # Process with depth model
                    inputs1 = processor(images=f1_pil, return_tensors="pt").to(device)
                    inputs2 = processor(images=f2_pil, return_tensors="pt").to(device)
                    
                    outputs1 = model(**inputs1)
                    outputs2 = model(**inputs2)
                    
                    # Post-process depths to original size
                    original_size = (f1.shape[0], f1.shape[1])  # (H, W)
                    processed1 = processor.post_process_depth_estimation(
                        outputs1, target_sizes=[original_size]
                    )
                    processed2 = processor.post_process_depth_estimation(
                        outputs2, target_sizes=[original_size]
                    )
                    
                    # Get the predicted depths
                    depth1 = processed1[0]["predicted_depth"]
                    depth2 = processed2[0]["predicted_depth"]
                    
                    # Calculate si-RMSE
                    score = scale_invariant_rmse(depth1, depth2)
                    # Only add valid scores (not NaN)
                    if torch.isnan(score).item():
                        print(f"Warning: NaN score detected for a frame")
                    else:
                        frame_si_rmse_scores.append(score.item())
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
        # Average si-RMSE across frames (if we have any valid scores)
        if frame_si_rmse_scores:
            video_si_rmse = np.mean(frame_si_rmse_scores)
            si_rmse_scores.append(video_si_rmse)
        else:
            print(f"Warning: No valid frames to calculate si-RMSE for {file1} and {file2}")
    
    return sum(si_rmse_scores) / len(si_rmse_scores) if si_rmse_scores else 0.0


if __name__ == "__main__":
    # Calculate depth si-RMSE for two video folders, paired by name
    folder1 = **
    folder2 = **
    si_rmse_score = calculate_depth_si_rmse(folder1, folder2)
    print(f"Depth si-RMSE score: {si_rmse_score}")
