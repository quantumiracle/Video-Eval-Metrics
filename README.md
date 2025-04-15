# Video Evaluation Metrics

This repository contains a collection of metrics designed to evaluate video generation alignment between input sample videos and generated videos. The metrics focus on different aspects of visual alignment to provide a comprehensive evaluation framework.

## Installation

```bash
pip install -r requirements.txt
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

## Usage

Each metric can be calculated by running the respective Python script:

```bash
python eval_ssim.py  # Calculate Blur SSIM
python eval_edge.py  # Calculate Edge F1
python eval_depth.py  # Calculate Depth si-RMSE
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
- numpy
- tqdm

## Implementation Details

- All metrics handle videos of different shapes by aligning them before computation
- For Edge F1, Canny edge detection is used with configurable thresholds
- For Depth si-RMSE, DepthAnythingV2 is used to extract depth maps

## References

- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing.
- van Rijsbergen, C. J. (1979). Information Retrieval (2nd ed.). Butterworth-Heinemann.
- Eigen, D., Puhrsch, C., & Fergus, R. (2014). Depth Map Prediction from a Single Image using a Multi-Scale Deep Network. Advances in Neural Information Processing Systems.
- Yang, L., Wang, Z., Long, C., He, H., Cheng, D., Xiao, A., Zhao, Y., & Dong, J. (2024). DepthAnythingV2: Zero-shot pixel-perfect depth estimation using an aligned visual model. 