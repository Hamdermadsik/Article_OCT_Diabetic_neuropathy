"""Shared model loading, paths and preprocessing constants.

Single source of truth for things that were previously copy-pasted across
scripts: the checkpoint loader, the project root (which used to be hardcoded to
one machine's absolute path in several files), and the model input size.
"""

from __future__ import annotations

import os
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

# Model input size as (height, width). ``configs/config.yaml`` is authoritative
# for Hydra-driven training; this default serves the standalone inference
# scripts, which have no config plumbing.
DEFAULT_TARGET_SIZE = (512, 512)

ENCODER_NAME = "resnet18"

# Training normalizes to [-1, 1]; inference must apply the identical mapping.
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5


def project_root() -> Path:
    """Repo root, overridable via ``OCT_PROJECT_ROOT`` for cluster runs."""
    override = os.environ.get("OCT_PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


def default_model_path() -> Path:
    return project_root() / "models" / "best_unet_model.pth"


def build_unet(encoder_weights: str | None = None) -> smp.Unet:
    """Construct the segmentation model with the project's fixed architecture."""
    return smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=encoder_weights,
        in_channels=1,
        classes=1,
        activation=None,
    )


def load_pytorch_model(model_path: str | Path, device: torch.device | str) -> smp.Unet:
    """Load a trained checkpoint, accepting bare or wrapped state dicts."""
    model = build_unet(encoder_weights=None)
    checkpoint = torch.load(str(model_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model
