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


def _imread_any(path: Path) -> np.ndarray | None:
    """Robust imread for Windows unicode paths.

    OpenCV can intermittently fail on Windows when the path contains non-ASCII
    characters. `np.fromfile` + `cv2.imdecode` is more reliable.
    """
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(str(path))


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
    image_bgr = _imread_any(image_path)
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
        bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("Failed to encode Grad-CAM PNG")
        # More reliable than cv2.imwrite on Windows unicode paths
        buf.tofile(str(save_path))

    return overlay, pred_idx, pred_conf


def gradcam_focus_hint(model: nn.Module, image_path: Path, device: torch.device,
                       target_class: Optional[int] = None) -> str:
    """Best-effort focus region hint: ICM/TE/background.

    Fast heuristic built on top of Grad-CAM without any extra training:
    - Build a coarse embryo mask (largest contour) at 224x224.
    - Split into INNER (eroded mask) vs OUTER RING (mask - inner) vs OUTSIDE.
    - Compute Grad-CAM energy share in each region and pick the dominant one.

    Note: This is *not* medical-grade segmentation; it's only an explainability hint.
    """
    image_bgr = _imread_any(image_path)
    if image_bgr is None:
        return "Model odağı belirlenemedi."

    tensor, raw_float = _preprocess_for_cam(image_bgr)
    tensor = tensor.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    cam_target = ClassifierOutputTarget(target_class if target_class is not None else pred_idx)

    cam = make_cam(model)
    cam_map = cam(input_tensor=tensor, targets=[cam_target])[0]  # (H,W), float in [0,1]

    # Coarse embryo mask from grayscale image (same resized scale as cam_map)
    gray = cv2.cvtColor(cv2.resize(image_bgr, (cfg.IMG_SIZE, cfg.IMG_SIZE)), cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # Otsu threshold; invert if needed
    _, th = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure embryo region is white: pick the larger of th vs inverted
    if th.mean() > 127:
        th = cv2.bitwise_not(th)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Model odağı belirlenemedi."
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(th)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)

    # Region split: inner vs outer ring vs outside
    bin_mask = (mask > 0).astype(np.uint8)
    area = int(bin_mask.sum())
    if area < 100:
        return "Model odağı belirlenemedi."

    # Erode proportional to embryo size: keep an outer ring thick enough to be measurable
    radius = (area / np.pi) ** 0.5
    erode_px = int(max(3, round(radius * 0.06)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    inner = cv2.erode(bin_mask, kernel, iterations=1)
    outer = (bin_mask - inner).clip(0, 1)
    outside = (1 - bin_mask).clip(0, 1)

    cam_pos = np.clip(cam_map, 0.0, None).astype(np.float32)
    total = float(cam_pos.sum())
    if total <= 1e-8:
        return "Model odağı belirgin değil (yaygın aktivasyon)."

    s_inner = float((cam_pos * inner).sum())
    s_outer = float((cam_pos * outer).sum())
    s_out = float((cam_pos * outside).sum())

    p_inner = s_inner / total
    p_outer = s_outer / total
    p_out = s_out / total

    # Decision with mild thresholds to avoid always picking INNER
    if p_out >= 0.45:
        return f"Model ağırlıklı olarak boşluk/arka plan bölgesine odaklanmıştır. (pay={p_out:.0%})"
    # If inner/outer are close, report a mixed focus (more honest than forcing TE/ICM).
    if p_inner >= 0.25 and p_outer >= 0.25 and abs(p_inner - p_outer) <= 0.10:
        dominant = "ICM" if p_inner >= p_outer else "TE"
        return (f"Model odağı ICM ve TE bölgeleri arasında dağılmış görünüyor "
                f"(ICM={p_inner:.0%}, TE={p_outer:.0%}); daha çok {dominant} tarafına yakındır.")

    # TE is an outer-ring structure; treat as "near TE" when outer ring is substantial.
    if p_outer >= 0.30 and p_outer > p_inner:
        return f"Model ağırlıklı olarak TE (trofektoderm) bölgesine yakın odaklanmıştır. (pay={p_outer:.0%})"

    return f"Model ağırlıklı olarak ICM (iç hücre kütlesi) bölgesine yakın odaklanmıştır. (pay={p_inner:.0%})"


def gradcam_attention_centrality(model: nn.Module, image_path: Path,
                                 device: torch.device) -> float:
    """Fraction of Grad-CAM activation falling inside the central 50% of the image.

    A high value means the model is attending to the embryo body (centre of the
    petri dish) rather than peripheral artefacts. Used in the morphology report.
    """
    image_bgr = _imread_any(image_path)
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
