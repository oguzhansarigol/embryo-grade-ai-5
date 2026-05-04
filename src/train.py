"""Two-stage k-fold training loop with MixUp, class weights, AMP, and early stopping."""
import csv
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm

from . import config as cfg
from .data import (
    EmbryoDataset, build_transforms, get_kfold_splits,
    get_class_weights, build_dataloaders,
)
from .model import build_model, freeze_backbone, unfreeze_all, get_param_groups


def set_seed(seed: int = cfg.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@dataclass
class EpochMetrics:
    epoch: int
    phase: str
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr_head: float
    lr_backbone: float


class EarlyStopping:
    """Stop when validation loss stops improving for `patience` epochs."""

    def __init__(self, patience: int = cfg.EARLY_STOP_PATIENCE):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        improved = val_loss < self.best - 1e-4
        if improved:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def _run_epoch(model, loader, optimizer, criterion_train, criterion_eval,
               scaler, mixup_fn, device, train: bool) -> Dict[str, float]:
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in tqdm(loader, leave=False, desc="train" if train else "val "):
            images, labels, _ = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train and mixup_fn is not None:
                mixed_images, mixed_targets = mixup_fn(images, labels)
            else:
                mixed_images, mixed_targets = images, labels

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(mixed_images)
                if train and mixup_fn is not None:
                    loss = criterion_train(logits, mixed_targets)
                else:
                    loss = criterion_eval(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_n += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return {"loss": total_loss / total_n, "acc": total_correct / total_n}


def _make_optimizer(model, lr_head: float, lr_backbone: Optional[float]) -> AdamW:
    groups = get_param_groups(model, lr_head=lr_head, lr_backbone=lr_backbone or lr_head)
    return AdamW(groups, weight_decay=cfg.WEIGHT_DECAY)


def _lr_of(optimizer, name: str) -> float:
    for g in optimizer.param_groups:
        if g.get("name") == name:
            return g["lr"]
    return 0.0


def train_one_fold(fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray,
                   ds_train: EmbryoDataset, ds_eval: EmbryoDataset,
                   device: torch.device, log_dir: Path = cfg.LOG_DIR,
                   ckpt_dir: Path = cfg.CHECKPOINT_DIR) -> Dict:
    """Run phase-1 (head warmup) + phase-2 (full fine-tune) for a single fold."""
    print(f"\n=== Fold {fold_idx + 1}/{cfg.K_FOLDS} ===")
    train_loader, val_loader = build_dataloaders(train_idx, val_idx, ds_train, ds_eval)

    train_labels = ds_train.get_labels()[train_idx]
    class_weights = get_class_weights(train_labels).to(device)

    model = build_model().to(device)
    criterion_eval = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.LABEL_SMOOTHING)
    # timm's Mixup asserts that at least one alpha > 0, so build it conditionally.
    if cfg.MIXUP_ALPHA > 0 or cfg.CUTMIX_ALPHA > 0:
        mixup_fn = Mixup(
            mixup_alpha=cfg.MIXUP_ALPHA, cutmix_alpha=cfg.CUTMIX_ALPHA,
            label_smoothing=cfg.LABEL_SMOOTHING, num_classes=cfg.NUM_CLASSES,
        )
        criterion_train = SoftTargetCrossEntropy()  # mixup uses soft targets
    else:
        mixup_fn = None
        criterion_train = criterion_eval  # plain CE with class weights
    scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))

    history: List[EpochMetrics] = []
    best_val_loss = float("inf")
    best_state = None
    early_stop = EarlyStopping()

    # ---------- Phase 1: head warmup ----------
    freeze_backbone(model)
    optimizer = _make_optimizer(model, lr_head=cfg.LR_HEAD_WARMUP, lr_backbone=None)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.WARMUP_EPOCHS)

    for epoch in range(cfg.WARMUP_EPOCHS):
        t0 = time.time()
        train_m = _run_epoch(model, train_loader, optimizer, criterion_train,
                             criterion_eval, scaler, mixup_fn, device, train=True)
        val_m = _run_epoch(model, val_loader, optimizer, criterion_train,
                           criterion_eval, scaler, None, device, train=False)
        scheduler.step()

        em = EpochMetrics(epoch=epoch, phase="warmup",
                          train_loss=train_m["loss"], train_acc=train_m["acc"],
                          val_loss=val_m["loss"], val_acc=val_m["acc"],
                          lr_head=_lr_of(optimizer, "head"),
                          lr_backbone=_lr_of(optimizer, "backbone"))
        history.append(em)
        improved = early_stop.step(val_m["loss"])
        if improved:
            best_val_loss = val_m["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[warmup {epoch+1}/{cfg.WARMUP_EPOCHS}] "
              f"train_loss={train_m['loss']:.4f} train_acc={train_m['acc']:.4f} | "
              f"val_loss={val_m['loss']:.4f} val_acc={val_m['acc']:.4f} | "
              f"{time.time()-t0:.1f}s")

    # ---------- Phase 2: full fine-tune ----------
    unfreeze_all(model)
    optimizer = _make_optimizer(model, lr_head=cfg.LR_HEAD_FINETUNE,
                                lr_backbone=cfg.LR_BACKBONE_FINETUNE)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.FINETUNE_EPOCHS)

    for epoch in range(cfg.FINETUNE_EPOCHS):
        if early_stop.should_stop:
            print(f"Early stopping at fine-tune epoch {epoch}")
            break
        t0 = time.time()
        train_m = _run_epoch(model, train_loader, optimizer, criterion_train,
                             criterion_eval, scaler, mixup_fn, device, train=True)
        val_m = _run_epoch(model, val_loader, optimizer, criterion_train,
                           criterion_eval, scaler, None, device, train=False)
        scheduler.step()

        em = EpochMetrics(epoch=cfg.WARMUP_EPOCHS + epoch, phase="finetune",
                          train_loss=train_m["loss"], train_acc=train_m["acc"],
                          val_loss=val_m["loss"], val_acc=val_m["acc"],
                          lr_head=_lr_of(optimizer, "head"),
                          lr_backbone=_lr_of(optimizer, "backbone"))
        history.append(em)
        improved = early_stop.step(val_m["loss"])
        if improved:
            best_val_loss = val_m["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[finetune {epoch+1}/{cfg.FINETUNE_EPOCHS}] "
              f"train_loss={train_m['loss']:.4f} train_acc={train_m['acc']:.4f} | "
              f"val_loss={val_m['loss']:.4f} val_acc={val_m['acc']:.4f} | "
              f"{time.time()-t0:.1f}s")

    # ---------- Save artefacts ----------
    log_path = log_dir / f"fold_{fold_idx}_history.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(history[0]).keys()))
        writer.writeheader()
        for em in history:
            writer.writerow(asdict(em))

    ckpt_path = ckpt_dir / f"fold_{fold_idx}_best.pth"
    torch.save({
        "fold": fold_idx,
        "state_dict": best_state,
        "best_val_loss": best_val_loss,
        "classes": cfg.CLASSES,
        "model_name": cfg.MODEL_NAME,
        "img_size": cfg.IMG_SIZE,
    }, ckpt_path)

    return {"fold": fold_idx, "best_val_loss": best_val_loss,
            "history_path": str(log_path), "checkpoint_path": str(ckpt_path),
            "val_idx": val_idx.tolist()}


def run_kfold(device: Optional[torch.device] = None) -> List[Dict]:
    """Train all K folds end-to-end, returning a summary dict per fold."""
    set_seed()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds_train = EmbryoDataset(transform=build_transforms(train=True))
    ds_eval = EmbryoDataset(transform=build_transforms(train=False))
    labels = ds_train.get_labels()
    print(f"Total samples: {len(ds_train)} | classes: {cfg.CLASSES}")

    splits = get_kfold_splits(labels)
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        result = train_one_fold(fold_idx, train_idx, val_idx, ds_train, ds_eval, device)
        fold_results.append(result)

    return fold_results