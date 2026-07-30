import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import get_dataloaders, CLASS_NAMES, MEAN, STD
from model import SimpleCNN
from gradcam import GradCAM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_cnn.pt")


def denormalize(img_tensor):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = (img_tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def overlay_heatmap(img_np, cam):
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.5 * img_np + 0.5 * heatmap
    return np.clip(overlay, 0, 1)


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_dataloaders(batch_size=8)

    model = SimpleCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # آخرین لایه کانولوشنی به عنوان target layer برای Grad-CAM
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    images, labels = next(iter(test_loader))
    n_samples = min(6, images.size(0))

    fig, axes = plt.subplots(2, n_samples, figsize=(2.2 * n_samples, 5))

    for i in range(n_samples):
        img_tensor = images[i:i+1].to(device)
        cam, pred_class = gradcam.generate(img_tensor)

        img_np = denormalize(images[i])
        overlay = overlay_heatmap(img_np, cam)

        true_label = CLASS_NAMES[labels[i].item()]
        pred_label = CLASS_NAMES[pred_class]
        correct = "✓" if pred_class == labels[i].item() else "✗"

        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"True: {true_label}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"Pred: {pred_label} {correct}", fontsize=9)
        axes[1, i].axis("off")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "gradcam_visualizations.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved Grad-CAM visualizations to {out_path}")


if __name__ == "__main__":
    evaluate()
