"""Evaluation: confusion matrix, classification report, training curves, learning curve."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize

from . import config as cfg
from .data import EmbryoDataset, build_transforms, build_test_loader
from .model import build_model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_loader(model: nn.Module, loader, device: torch.device
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Run model on loader; return (y_true, y_pred, probs, image_paths)."""
    y_true, y_pred, probs, paths = [], [], [], []
    model.eval()
    for images, labels, img_paths in loader:
        images = images.to(device)
        logits = model(images)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(p.argmax(axis=1).tolist())
        probs.append(p)
        paths.extend(img_paths)
    return np.array(y_true), np.array(y_pred), np.concatenate(probs), paths


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_history(csv_path: Path, save_path: Path) -> None:
    """Plot train/val accuracy and loss vs epoch (doc §4.1)."""
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train", marker="o", markersize=3)
    axes[0].plot(df["epoch"], df["val_loss"], label="val", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss — {csv_path.stem}")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["train_acc"], label="train", marker="o", markersize=3)
    axes[1].plot(df["epoch"], df["val_acc"], label="val", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Accuracy — {csv_path.stem}")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          save_path: Path, title: str = "Confusion Matrix",
                          normalize: bool = False) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg.NUM_CLASSES)))
    cm_show = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_show, annot=True, fmt=".2f" if normalize else "d",
                cmap="Blues", xticklabels=cfg.CLASSES, yticklabels=cfg.CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


def plot_roc_curves(y_true: np.ndarray, probs: np.ndarray, save_path: Path) -> Dict[str, float]:
    """One-vs-rest ROC for each class."""
    y_bin = label_binarize(y_true, classes=list(range(cfg.NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(6, 5))
    aucs: Dict[str, float] = {}
    for i, cls in enumerate(cfg.CLASSES):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        auc = roc_auc_score(y_bin[:, i], probs[:, i])
        aucs[cls] = auc
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return aucs


# ---------------------------------------------------------------------------
# Single-run test evaluation
# ---------------------------------------------------------------------------

def evaluate_test(train_result: Dict, device: torch.device) -> Dict:
    """Evaluate the single training run's checkpoint on the held-out test set.

    Produces:
      - figures/confusion_matrix_test.png       (raw counts)
      - figures/confusion_matrix_test_norm.png  (row-normalised)
      - figures/roc_curves.png                  (one-vs-rest, per class AUC)
      - figures/history.png                     (train/val acc-loss vs epoch)
      - reports/metrics_summary.json
      - outputs/predictions_test.csv            (consumed by morphology.py)
    """
    ds_eval = EmbryoDataset(transform=build_transforms(train=False))
    test_idx = np.array(train_result["test_idx"])
    test_loader = build_test_loader(test_idx, ds_eval)

    model = load_checkpoint(Path(train_result["checkpoint_path"]), device)
    y_true, y_pred, probs, paths = predict_loader(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    cm = plot_confusion_matrix(
        y_true, y_pred, cfg.FIGURE_DIR / "confusion_matrix_test.png",
        title="Confusion Matrix — Test (15%)",
    )
    plot_confusion_matrix(
        y_true, y_pred, cfg.FIGURE_DIR / "confusion_matrix_test_norm.png",
        title="Confusion Matrix — Test (normalised)", normalize=True,
    )
    aucs = plot_roc_curves(y_true, probs, cfg.FIGURE_DIR / "roc_curves.png")
    plot_history(Path(train_result["history_path"]),
                 cfg.FIGURE_DIR / "history.png")

    report = classification_report(
        y_true, y_pred, target_names=cfg.CLASSES,
        output_dict=True, zero_division=0,
    )

    summary = {
        "split": {
            "train_frac": cfg.TRAIN_FRAC,
            "val_frac": cfg.VAL_FRAC,
            "test_frac": cfg.TEST_FRAC,
            "seed": cfg.SEED,
            "n_train": len(train_result["train_idx"]),
            "n_val": len(train_result["val_idx"]),
            "n_test": len(test_idx),
        },
        "test_metrics": {
            "accuracy": acc,
            "precision_w": prec,
            "recall_w": rec,
            "f1_w": f1,
            "best_val_loss": train_result["best_val_loss"],
        },
        "test_classification_report": report,
        "test_confusion_matrix": cm.tolist(),
        "roc_auc_per_class": aucs,
    }

    with open(cfg.REPORT_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame({
        "image_path": paths,
        "y_true": [cfg.IDX_TO_CLASS[i] for i in y_true],
        "y_pred": [cfg.IDX_TO_CLASS[i] for i in y_pred],
        "confidence": probs.max(axis=1),
        **{f"prob_{cls}": probs[:, i] for i, cls in enumerate(cfg.CLASSES)},
    }).to_csv(cfg.OUTPUT_DIR / "predictions_test.csv", index=False)

    return summary


# ---------------------------------------------------------------------------
# Learning curve (doc §4.1)
# ---------------------------------------------------------------------------

def _stratified_subsample(indices: np.ndarray, labels_at_indices: np.ndarray,
                          n_target: int, seed: int) -> np.ndarray:
    """Take ~n_target indices, sampled per-class proportionally (≥1 per class).

    Avoids `StratifiedShuffleSplit`'s constraint that the leftover set must hold
    at least one sample per class — that breaks at fractions close to 1.0 on tiny
    datasets.
    """
    if n_target >= len(indices):
        return indices.copy()
    rng = np.random.RandomState(seed)
    classes = np.unique(labels_at_indices)
    selected: List[int] = []
    for c in classes:
        cls_indices = indices[labels_at_indices == c]
        share = max(1, int(round(n_target * len(cls_indices) / len(indices))))
        share = min(share, len(cls_indices))
        chosen = rng.choice(cls_indices, size=share, replace=False)
        selected.extend(chosen.tolist())
    return np.array(selected)


def learning_curve(device: torch.device, fractions=(0.25, 0.5, 0.75, 1.0),
                   epochs: int = 15) -> pd.DataFrame:
    """Train on increasing fractions of training data; plot val acc."""
    from sklearn.model_selection import StratifiedShuffleSplit
    from .train import train_single_run

    ds_train = EmbryoDataset(transform=build_transforms(train=True))
    ds_eval = EmbryoDataset(transform=build_transforms(train=False))
    labels = ds_train.get_labels()

    # Hold out 20% as the evaluation set; subsample the remaining 80%.
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.SEED)
    train_full_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

    rows = []
    saved_warmup = cfg.WARMUP_EPOCHS
    saved_ft = cfg.FINETUNE_EPOCHS
    cfg.WARMUP_EPOCHS = max(2, epochs // 5)
    cfg.FINETUNE_EPOCHS = epochs - cfg.WARMUP_EPOCHS

    try:
        for frac in fractions:
            n = max(cfg.NUM_CLASSES, int(len(train_full_idx) * frac))
            sub_train_idx = _stratified_subsample(
                train_full_idx, labels[train_full_idx], n, seed=cfg.SEED,
            )

            print(f"\n[learning curve] fraction={frac:.2f} n_train={len(sub_train_idx)}")
            res = train_single_run(
                train_idx=sub_train_idx, val_idx=val_idx,
                ds_train=ds_train, ds_eval=ds_eval, device=device,
                log_dir=cfg.LOG_DIR, ckpt_dir=cfg.CHECKPOINT_DIR / "lc_tmp",
                run_name=f"lc_{int(frac * 100)}",
            )
            df = pd.read_csv(res["history_path"])
            rows.append({"fraction": frac, "n_train": len(sub_train_idx),
                         "best_val_acc": df["val_acc"].max(),
                         "final_val_acc": df["val_acc"].iloc[-1]})
    finally:
        cfg.WARMUP_EPOCHS = saved_warmup
        cfg.FINETUNE_EPOCHS = saved_ft

    df = pd.DataFrame(rows)
    df.to_csv(cfg.REPORT_DIR / "learning_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["n_train"], df["best_val_acc"], marker="o", label="best val acc")
    ax.plot(df["n_train"], df["final_val_acc"], marker="s", label="final val acc")
    ax.set_xlabel("# training samples"); ax.set_ylabel("Validation Accuracy")
    ax.set_title("Learning Curve")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.FIGURE_DIR / "learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df


def export_final_model() -> Path:
    """Copy `best_model.pth` -> `final_model.pth` (canonical name).

    Keeps `infer.py` and `app/app.py` unchanged — they consume `final_model.pth`.
    """
    import shutil
    src = cfg.CHECKPOINT_DIR / "best_model.pth"
    dst = cfg.CHECKPOINT_DIR / "final_model.pth"
    shutil.copy(src, dst)
    return dst
