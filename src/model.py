"""ConvNeXt-Base builder + freeze/unfreeze utilities for staged transfer learning."""
from typing import List, Dict
import timm
import torch
import torch.nn as nn

from . import config as cfg


def build_model(num_classes: int = cfg.NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ConvNeXt-Base with ImageNet-22k pretraining and a fresh 4-class head."""
    model = timm.create_model(
        cfg.MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=cfg.DROPOUT,
        drop_path_rate=cfg.DROP_PATH,
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Phase-1 warmup: freeze stem + first three stages, train only stage 3 + head."""
    for p in model.parameters():
        p.requires_grad = False
    # Always train the head
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    # Also unfreeze the deepest stage so it can adapt while head warms up
    if hasattr(model, "stages"):
        for p in model.stages[-1].parameters():
            p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Phase-2: unfreeze every parameter for full fine-tuning."""
    for p in model.parameters():
        p.requires_grad = True


def get_param_groups(model: nn.Module, lr_head: float, lr_backbone: float) -> List[Dict]:
    """Discriminative learning rates: small LR for pretrained backbone, larger for head."""
    head_params, backbone_params = [], []
    head_module = model.get_classifier()
    head_param_ids = {id(p) for p in head_module.parameters()}

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_param_ids:
            head_params.append(p)
        else:
            backbone_params.append(p)

    groups = [{"params": head_params, "lr": lr_head, "name": "head"}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "name": "backbone"})
    return groups


def find_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the last ConvNeXt stage block — the canonical Grad-CAM hook point."""
    if hasattr(model, "stages"):
        return model.stages[-1]
    raise AttributeError("Model has no `stages` attribute; cannot locate Grad-CAM target.")


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
