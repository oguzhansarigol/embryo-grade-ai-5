"""Morphological feature extraction for the XAI report (doc §5.3).

We compute simple, interpretable image-level descriptors and correlate them
with prediction correctness + Grad-CAM attention. The aim is to answer:
"is the model basing decisions on classical embryology cues (symmetry,
texture, edges) or on something else?"
"""
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as cfg
from .gradcam import gradcam_attention_centrality


def _load_gray(image_path: Path, size: int = cfg.IMG_SIZE) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to load {image_path}")
    return cv2.resize(img, (size, size))


def brightness(gray: np.ndarray) -> float:
    return float(gray.mean() / 255.0)


def contrast(gray: np.ndarray) -> float:
    return float(gray.std() / 255.0)


def horizontal_symmetry(gray: np.ndarray) -> float:
    """1.0 = perfect L/R symmetry; lower = more asymmetric."""
    flipped = cv2.flip(gray, 1)
    diff = np.abs(gray.astype(np.int32) - flipped.astype(np.int32))
    return float(1.0 - diff.mean() / 255.0)


def edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels classified as edges by Canny — proxy for cell boundary clarity."""
    edges = cv2.Canny(gray, 60, 160)
    return float((edges > 0).mean())


def fragmentation_proxy(gray: np.ndarray) -> float:
    """Variance of local Laplacian — high in fragmented (noisy) cytoplasm."""
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return float(lap.var() / 1000.0)  # scaled to keep magnitudes comparable


def vacuole_proxy(gray: np.ndarray) -> float:
    """Count of bright circular blobs — crude proxy for vacuoles."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blob_count = sum(1 for c in contours if 10 < cv2.contourArea(c) < 400)
    return float(blob_count)


FEATURES = {
    "brightness": brightness,
    "contrast": contrast,
    "symmetry": horizontal_symmetry,
    "edge_density": edge_density,
    "fragmentation": fragmentation_proxy,
    "vacuole_count": vacuole_proxy,
}


def extract_features(image_path: Path) -> Dict[str, float]:
    gray = _load_gray(image_path)
    return {name: fn(gray) for name, fn in FEATURES.items()}


def build_morphology_report(model: nn.Module, predictions_csv: Path,
                            device: torch.device,
                            save_dir: Optional[Path] = None) -> pd.DataFrame:
    """Combine morphology features + Grad-CAM centrality with prediction correctness.

    Outputs:
      - morphology_features.csv (one row per image)
      - morphology_correlation.png (correctness vs each feature, boxplots)
      - gradcam_centrality_dist.png (correct vs incorrect)
    """
    save_dir = save_dir or cfg.REPORT_DIR
    df_pred = pd.read_csv(predictions_csv)

    rows = []
    for _, row in df_pred.iterrows():
        path = Path(row["image_path"])
        feats = extract_features(path)
        try:
            feats["gradcam_centrality"] = gradcam_attention_centrality(model, path, device)
        except Exception as e:  # Grad-CAM is best-effort; don't fail the whole report
            feats["gradcam_centrality"] = float("nan")
            print(f"[morphology] Grad-CAM failed for {path.name}: {e}")
        feats["correct"] = int(row["y_true"] == row["y_pred"])
        feats["y_true"] = row["y_true"]
        feats["y_pred"] = row["y_pred"]
        feats["confidence"] = row["confidence"]
        feats["image_path"] = str(path)
        rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "morphology_features.csv", index=False)

    feature_cols = list(FEATURES.keys()) + ["gradcam_centrality"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, col in zip(axes.flat, feature_cols):
        sns.boxplot(data=df, x="correct", y=col, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("Correct prediction")
    for ax in axes.flat[len(feature_cols):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(cfg.FIGURE_DIR / "morphology_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="gradcam_centrality", hue="correct",
                 element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title("Grad-CAM centrality distribution by correctness")
    plt.tight_layout()
    plt.savefig(cfg.FIGURE_DIR / "gradcam_centrality_dist.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = df.groupby("correct")[feature_cols].mean().T
    summary.columns = ["mean_when_wrong", "mean_when_correct"]
    summary["delta"] = summary["mean_when_correct"] - summary["mean_when_wrong"]
    summary.to_csv(save_dir / "morphology_summary.csv")

    return df
