"""Single and batch inference with the confidence-threshold warning (doc §5.2)."""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch

from . import config as cfg
from .evaluate import load_checkpoint
from .gradcam import generate_gradcam


@dataclass
class Prediction:
    image_path: str
    predicted_class: str
    confidence: float
    warning: Optional[str]
    probabilities: dict
    gradcam_path: Optional[str] = None


def _warning_for(confidence: float) -> Optional[str]:
    if confidence < cfg.CONFIDENCE_THRESHOLD:
        return ("Bu tahmin düşük güvenilirliktedir, lütfen manuel kontrol yapınız. "
                f"(güven={confidence:.2f} < {cfg.CONFIDENCE_THRESHOLD})")
    return None


class EmbryoPredictor:
    """Wraps a trained model with the public predict_single / predict_batch API."""

    def __init__(self, checkpoint_path: Path = None, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = Path(checkpoint_path or (cfg.CHECKPOINT_DIR / "final_model.pth"))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        self.model = load_checkpoint(ckpt_path, self.device)

    def predict_single(self, image_path: Path,
                       gradcam_save_path: Optional[Path] = None) -> Prediction:
        image_path = Path(image_path)
        overlay, pred_idx, pred_conf = generate_gradcam(
            self.model, image_path, self.device, save_path=gradcam_save_path,
        )

        # Re-run a clean forward to grab the full probability vector
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (cfg.IMG_SIZE, cfg.IMG_SIZE)).astype(np.float32) / 255.0
        image = (image - cfg.IMAGENET_MEAN) / cfg.IMAGENET_STD
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1).cpu().numpy()[0]

        prob_dict = {cfg.IDX_TO_CLASS[i]: float(probs[i]) for i in range(cfg.NUM_CLASSES)}
        return Prediction(
            image_path=str(image_path),
            predicted_class=cfg.IDX_TO_CLASS[pred_idx],
            confidence=float(pred_conf),
            warning=_warning_for(pred_conf),
            probabilities=prob_dict,
            gradcam_path=str(gradcam_save_path) if gradcam_save_path else None,
        )

    def predict_batch(self, folder: Path,
                      gradcam_dir: Optional[Path] = None,
                      csv_out: Optional[Path] = None) -> List[Prediction]:
        folder = Path(folder)
        gradcam_dir = Path(gradcam_dir) if gradcam_dir else None
        if gradcam_dir:
            gradcam_dir.mkdir(parents=True, exist_ok=True)

        results = []
        # Accept .bmp + common image extensions for flexibility
        exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts)
        for img_path in images:
            cam_path = gradcam_dir / f"{img_path.stem}_gradcam.png" if gradcam_dir else None
            results.append(self.predict_single(img_path, gradcam_save_path=cam_path))

        if csv_out:
            rows = []
            for r in results:
                row = {k: v for k, v in asdict(r).items() if k != "probabilities"}
                row.update({f"prob_{c}": r.probabilities[c] for c in cfg.CLASSES})
                rows.append(row)
            pd.DataFrame(rows).to_csv(csv_out, index=False)

        return results


def export_gradcam_samples(predictor: EmbryoPredictor, n_per_class: int = 3,
                           out_dir: Path = cfg.GRADCAM_DIR) -> None:
    """Generate a small gallery of Grad-CAM samples for the analysis report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls in cfg.CLASSES:
        cls_dir = cfg.DATA_DIR / cls
        images = sorted(cls_dir.glob("*.bmp"))[:n_per_class]
        for img in images:
            save = out_dir / f"{cls}_{img.stem}_gradcam.png"
            predictor.predict_single(img, gradcam_save_path=save)
