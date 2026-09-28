# Explainable AI: Grad-CAM Visualization from Scratch

A CNN classifier trained on CIFAR-10, paired with a from-scratch implementation
of Grad-CAM to visualize which image regions drive each prediction.

## Pipeline
1. Train a simple CNN classifier (4 conv blocks + global average pooling)
2. Implement Grad-CAM via forward/backward hooks on the last conv layer
3. Compute channel-wise importance weights from gradients (GAP over spatial dims)
4. Generate class activation heatmaps and overlay them on input images

## How Grad-CAM works
- Forward hook captures activations of the target conv layer
- Backward hook captures gradients of the predicted class w.r.t. those activations
- Weights = global-average-pooled gradients per channel
- Heatmap = ReLU(weighted sum of activation channels)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/train.py
python src/evaluate.py
```

## Results
- `results/training_curves.png` — training/test loss & accuracy
- `results/gradcam_visualizations.png` — original images + Grad-CAM overlays

## Author
Hessam Kaveh — Research Fellow, Italian Institute of Technology
2026

