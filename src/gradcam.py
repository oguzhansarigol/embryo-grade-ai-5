"""Grad-CAM heatmaps for ConvNeXt — required by doc §4.3 and §5.1."""
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from . import config as cfg
from .model import find_gradcam_target_layer


def _convnext_reshape(tensor: torch.Tensor) -> torch.Tensor:
    """ConvNeXt's stage outputs are (B, C, H, W) so no reshape is needed."""
    return tensor


def make_cam(model: nn.Module) -> GradCAM:
    """Hook Grad-CAM onto the deepest ConvNeXt stage."""
    target = find_gradcam_target_layer(model)
    return GradCAM(model=model, target_layers=[target], reshape_transform=_convnext_reshape)


def _preprocess_for_cam(image_bgr: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
    """Return a normalised tensor and the raw [0,1] RGB image for overlay."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (cfg.IMG_SIZE, cfg.IMG_SIZE))
    image_float = image_resized.astype(np.float32) / 255.0

    mean = np.array(cfg.IMAGENET_MEAN, dtype=np.float32)
    std = np.array(cfg.IMAGENET_STD, dtype=np.float32)
    normed = (image_float - mean) / std
    tensor = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, image_float


def generate_gradcam(model: nn.Module, image_path: Path, device: torch.device,
                     target_class: Optional[int] = None,
                     save_path: Optional[Path] = None) -> Tuple[np.ndarray, int, float]:
    """Run a single image through Grad-CAM; optionally save the overlay.

    Returns: (overlay_rgb_uint8, predicted_class_idx, predicted_confidence).
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to load {image_path}")
    tensor, raw_float = _preprocess_for_cam(image_bgr)
    tensor = tensor.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_conf = float(probs[pred_idx])
    cam_target = ClassifierOutputTarget(target_class if target_class is not None else pred_idx)

    cam = make_cam(model)
    grayscale_cam = cam(input_tensor=tensor, targets=[cam_target])[0]
    overlay = show_cam_on_image(raw_float, grayscale_cam, use_rgb=True)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return overlay, pred_idx, pred_conf


def gradcam_attention_centrality(model: nn.Module, image_path: Path,
                                 device: torch.device) -> float:
    """Fraction of Grad-CAM activation falling inside the central 50% of the image.

    A high value means the model is attending to the embryo body (centre of the
    petri dish) rather than peripheral artefacts. Used in the morphology report.
    """
    image_bgr = cv2.imread(str(image_path))
    tensor, _ = _preprocess_for_cam(image_bgr)
    tensor = tensor.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        pred = int(logits.argmax(dim=1).item())

    cam = make_cam(model)
    grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred)])[0]
    h, w = grayscale_cam.shape
    cy0, cy1 = h // 4, 3 * h // 4
    cx0, cx1 = w // 4, 3 * w // 4
    central = grayscale_cam[cy0:cy1, cx0:cx1].sum()
    total = grayscale_cam.sum() + 1e-8
    return float(central / total)
