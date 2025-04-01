import os
import torch
from sklearn.metrics import f1_score
# current dir
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current dir: {current_dir}")
# import parent dir
import sys
sys.path.append(os.path.dirname(current_dir))
from utils.canny_edge import get_canny_edges
from torchvision.io import read_video
from tqdm import tqdm
from torchvision import transforms


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


def calculate_edge_f1(folder1, folder2, low_threshold=50, high_threshold=150):
    """
    Calculate the Edge F1 score between videos in two folders.
    folder1: path to the first folder containing videos
    folder2: path to the second folder containing videos
    returns: average Edge F1 score
    """
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith('.mp4')])
    files2 = sorted([f for f in os.listdir(folder2) if f.endswith('.mp4')])
    
    assert len(files1) == len(files2), "The number of videos in both folders must be the same and aligned."
    print(f"Calculating Edge F1 score for {len(files1)} videos")
    f1_scores = []
    # progress bar
    for file1, file2 in tqdm(zip(files1, files2), total=len(files1), desc="Calculating Edge F1 score"):
        video_path1 = os.path.join(folder1, file1)
        video_path2 = os.path.join(folder2, file2)
        
        frames1, _, _ = read_video(video_path1)
        frames2, _, _ = read_video(video_path2)

        # Align video shapes
        frames1, frames2 = align_video_shapes(frames1, frames2, verbose=True)
        
        # same threshold for canny edge detection
        edges1 = get_canny_edges(frames1, low_threshold=low_threshold, high_threshold=high_threshold) # (T, C, H, W)
        edges2 = get_canny_edges(frames2, low_threshold=low_threshold, high_threshold=high_threshold)
        
        # Flatten the edge tensors and calculate F1 score
        edges1_flat = edges1.flatten().numpy()
        edges2_flat = edges2.flatten().numpy()
        
        f1 = f1_score(edges1_flat, edges2_flat, average='binary', pos_label=255) # 255 is the value of the edge
        f1_scores.append(f1)
    
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


if __name__ == "__main__":
    folder1 = **
    folder2 = **
    # the same threshold for canny edge detection
    # threshold values should be set based on the dataset
    f1_result = calculate_edge_f1(folder1, folder2, low_threshold=50, high_threshold=150)
    print(f"Edge F1 score: {f1_result}")

    