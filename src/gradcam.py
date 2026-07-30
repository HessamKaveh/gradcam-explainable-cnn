import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    """
    پیاده‌سازی Grad-CAM از صفر:
    1. Forward hook برای گرفتن activation های لایه هدف
    2. Backward hook برای گرفتن gradient های همون لایه
    3. وزن‌دهی activation ها بر اساس میانگین gradient (global average pooling)
    4. ترکیب وزنی + ReLU برای ساخت heatmap
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # وزن هر کانال = میانگین gradient روی آن کانال (Global Average Pooling)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, class_idx
