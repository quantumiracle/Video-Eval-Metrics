# Video Evaluation Metrics

This repository contains a collection of metrics designed to evaluate video generation alignment between input sample videos and generated videos. The metrics focus on different aspects of visual alignment to provide a comprehensive evaluation framework.

## Installation

```bash
pip install -r requirements.txt

# For FVD evaluation, you'll need the pytorch-i3d package:
git clone https://github.com/piergiaj/pytorch-i3d.git
```

## Metrics Overview

### Blur SSIM (Structural Similarity Index Measure)

**File**: `eval_ssim.py`

**Description**: Applies the same blurring operation to both the input sample videos and the generated videos, then computes their Structural Similarity Index Measure (SSIM). The scores are averaged over all samples within the dataset.

**Interpretation**: Higher values indicate better visual alignment.

**Reference**: Wang et al., 2004

### Edge F1 (Edge Alignment)

**File**: `eval_edge.py`

**Description**: Applies Canny edge extraction to both the input sample videos and the generated videos, then computes a classification F1 score on the black-and-white pixel classification. The F1 scores are averaged over the dataset.

**Interpretation**: Higher values indicate better edge alignment.

**Reference**: van Rijsbergen, 1979

### Depth si-RMSE (Depth Alignment)

**File**: `eval_depth.py`

**Description**: Computes the scale-invariant Root Mean Squared Error (si-RMSE) between the depth maps extracted from both the input sample videos and the generated videos using DepthAnythingV2. This metric evaluates how well the generated videos preserve the spatial structure of the input videos.

**Interpretation**: Lower values indicate better depth alignment.

**Reference**: Eigen et al., 2014; Yang et al., 2024 (DepthAnythingV2)

### PSNR (Peak Signal-to-Noise Ratio)

**File**: `eval_psnr.py`

**Description**: Calculates the Peak Signal-to-Noise Ratio between the input sample videos and the generated videos. PSNR is a pixel-level fidelity metric based on mean squared error that measures the reconstruction quality of the generated videos.

**Interpretation**: Higher values indicate better pixel-level fidelity.

**Reference**: Huynh-Thu & Ghanbari, 2008

### LPIPS (Learned Perceptual Image Patch Similarity)

**File**: `eval_lpips.py`

**Description**: Measures the perceptual similarity between input sample videos and generated videos using a deep neural network trained to predict human perceptual judgments. LPIPS aims to capture perceptual differences that traditional metrics like PSNR and SSIM might miss.

**Interpretation**: Lower values indicate better perceptual similarity.

**Reference**: Zhang et al., 2018

### FVD (Fréchet Video Distance)

**File**: `eval_fvd.py`

**Description**: Computes the Fréchet Video Distance between two sets of videos by comparing the distributions of features extracted from a pre-trained I3D model. Unlike other metrics that compare individual video pairs, FVD evaluates the distribution-level similarity between two sets of videos, measuring both visual quality and temporal consistency of the generated videos.

**Better have 50 to 100 videos for each folder for FVD evaluation**

**Interpretation**: Lower values indicate better video quality and temporal consistency.

**Reference**: Unterthiner et al., 2019

## Usage

Each metric can be calculated by running the respective Python script:

```bash
python eval_ssim.py  # Calculate Blur SSIM
python eval_edge.py  # Calculate Edge F1
python eval_depth.py  # Calculate Depth si-RMSE
python eval_psnr.py  # Calculate PSNR
python eval_lpips.py  # Calculate LPIPS
python eval_fvd.py   # Calculate FVD; git clone https://github.com/piergiaj/pytorch-i3d.git first
```

Before running, you need to specify the input directories in the main section of each script:

```python
if __name__ == "__main__":
    folder1 = "path/to/input/videos"  # Original sample videos
    folder2 = "path/to/generated/videos"  # Generated videos
    # Calculate metrics
    score = calculate_metric(folder1, folder2)
    print(f"Score: {score}")
```

### Expected Folder Structure

The evaluation scripts expect a specific folder structure where the input sample videos and the generated videos have corresponding filenames:

```
folder1/                           folder2/
├── video1.mp4                     ├── video1.mp4
├── video2.mp4                     ├── video2.mp4
├── video3.mp4                     ├── video3.mp4
└── ...                            └── ...
```

The scripts will automatically match and compare videos with the same filename from both folders. The number of videos in both folders must be the same, and they must be aligned by name.

## Requirements

- PyTorch
- torchvision
- scikit-image (for SSIM)
- scikit-learn (for F1 score)
- OpenCV (for edge detection)
- transformers (for depth estimation using DepthAnythingV2)
- lpips (for LPIPS calculation)
- pytorch-i3d (for FVD calculation, install via `git clone https://github.com/piergiaj/pytorch-i3d.git`)
- scipy (for FVD calculation)
- numpy
- tqdm

## Implementation Details

- All metrics handle videos of different shapes by aligning them before computation
- For Edge F1, Canny edge detection is used with configurable thresholds
- For Depth si-RMSE, DepthAnythingV2 is used to extract depth maps
- For LPIPS, a pre-trained AlexNet, VGG, or SqueezeNet model can be used as the backbone
- For FVD, a pre-trained I3D model is used to extract features from video frames
- For PSNR, pixel-wise mean squared error is calculated in image space

## References

- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing.
- van Rijsbergen, C. J. (1979). Information Retrieval (2nd ed.). Butterworth-Heinemann.
- Eigen, D., Puhrsch, C., & Fergus, R. (2014). Depth Map Prediction from a Single Image using a Multi-Scale Deep Network. Advances in Neural Information Processing Systems.
- Yang, L., Wang, Z., Long, C., He, H., Cheng, D., Xiao, A., Zhao, Y., & Dong, J. (2024). DepthAnythingV2: Zero-shot pixel-perfect depth estimation using an aligned visual model.
- Huynh-Thu, Q., & Ghanbari, M. (2008). Scope of validity of PSNR in image/video quality assessment. Electronics letters, 44(13), 800-801.
- Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
- Unterthiner, T., van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., & Gelly, S. (2019). FVD: A new metric for video generation. In International Conference on Learning Representations. 