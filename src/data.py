"""Dataset, augmentation transforms, stratified train/val/test split, class weighting."""
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2

from . import config as cfg


class EmbryoDataset(Dataset):
    """Loads .bmp blastocyst images organised in class-named subfolders."""

    def __init__(self, data_dir: Path = cfg.DATA_DIR, transform: Optional[A.Compose] = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.classes = cfg.CLASSES

        for cls_name in self.classes:
            cls_dir = self.data_dir / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {cls_dir}")
            for img_path in sorted(cls_dir.glob("*.bmp")):
                self.samples.append((img_path, cfg.CLASS_TO_IDX[cls_name]))

        if not self.samples:
            raise RuntimeError(f"No .bmp images found under {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def get_labels(self) -> np.ndarray:
        return np.array([label for _, label in self.samples])

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        # cv2 returns BGR; convert to RGB so it matches ImageNet normalisation
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label, str(img_path)


def build_transforms(train: bool) -> A.Compose:
    """Train uses moderate augmentation; eval just resizes + normalises.

    Notes for fine-grained Gardner grading on a tiny dataset:
      - No `RandomResizedCrop`: the leading digit (3 vs 4) encodes blastocyst
        expansion stage, i.e. apparent size. Random rescaling destroys that cue.
        We do `Resize -> RandomCrop` instead — small spatial jitter, fixed scale.
      - No `VerticalFlip`: ICM/TE position is morphologically meaningful.
      - Rotation kept small (±10°) for the same reason.
      - `CoarseDropout` removed: it can hide ICM packing detail that A vs C
        depends on.
    """
    if train:
        return A.Compose([
            A.Resize(cfg.IMG_SIZE + 24, cfg.IMG_SIZE + 24),
            A.RandomCrop(cfg.IMG_SIZE, cfg.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
        ToTensorV2(),
    ])


def get_train_val_test_split(
    labels: np.ndarray,
    val_frac: float = cfg.VAL_FRAC,
    test_frac: float = cfg.TEST_FRAC,
    seed: int = cfg.SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-stage stratified split → (train_idx, val_idx, test_idx).

    1) Hold out `test_frac` of N as test (stratified). Test is touched ONLY at
       final reporting time — never during early stopping or hyperparameter
       selection.
    2) Within the remaining (1 - test_frac), hold out `val_frac / (1 - test_frac)`
       as val (stratified). The rest is train.

    Deterministic with a fixed `seed`.
    """
    idx_all = np.arange(len(labels))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(idx_all, labels))

    val_rel = val_frac / (1.0 - test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed)
    sub_tr, sub_vl = next(sss2.split(trainval_idx, labels[trainval_idx]))
    train_idx = trainval_idx[sub_tr]
    val_idx = trainval_idx[sub_vl]
    return train_idx, val_idx, test_idx


def print_split_distribution(labels: np.ndarray, train_idx: np.ndarray,
                             val_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Print per-class counts and sanity-check that no class collapses to <2 in test."""
    print(f"{'Class':<10} {'train':>6} {'val':>5} {'test':>5} {'total':>6}")
    for i, cls in enumerate(cfg.CLASSES):
        n_train = int((labels[train_idx] == i).sum())
        n_val = int((labels[val_idx] == i).sum())
        n_test = int((labels[test_idx] == i).sum())
        print(f"{cls:<10} {n_train:>6} {n_val:>5} {n_test:>5} {n_train + n_val + n_test:>6}")
        if n_test < 2:
            raise RuntimeError(
                f"Stratification produced <2 test samples for class '{cls}'. "
                f"Check seed or class balance."
            )


def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for the cross-entropy loss."""
    classes = np.arange(cfg.NUM_CLASSES)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32)


def build_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Oversample rare classes so each batch sees them roughly equally."""
    class_counts = np.bincount(labels, minlength=cfg.NUM_CLASSES)
    sample_weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )


def build_dataloaders(train_idx, val_idx, full_dataset_train: EmbryoDataset,
                      full_dataset_eval: EmbryoDataset, use_sampler: bool = True):
    """Wrap split indices into (train, val) DataLoaders.

    Two dataset instances are passed so train/val can use different transforms
    even when sharing the same underlying file list.
    """
    train_subset = Subset(full_dataset_train, train_idx.tolist())
    val_subset = Subset(full_dataset_eval, val_idx.tolist())

    train_labels = full_dataset_train.get_labels()[train_idx]
    sampler = build_sampler(train_labels) if use_sampler else None

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_test_loader(test_idx: np.ndarray,
                      full_dataset_eval: EmbryoDataset) -> DataLoader:
    """Held-out test loader: deterministic order, no augmentation, no sampler."""
    test_subset = Subset(full_dataset_eval, test_idx.tolist())
    return DataLoader(
        test_subset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
