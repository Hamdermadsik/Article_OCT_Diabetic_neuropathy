"""Shared fixtures. Tests use synthetic images so they need no real data."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture
def frame_factory(tmp_path):
    """Write a synthetic image/mask pair and return its manifest row.

    Frames are deliberately non-square (like real B-scans) so that
    aspect-ratio bugs show up rather than being hidden by a square input.
    """

    def _make(
        frame_id: str,
        patient_id: str = "P-001",
        height: int = 40,
        width: int = 100,
        marker=None,
        with_mask: bool = True,
    ) -> dict:
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)

        image = np.zeros((height, width), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        # A horizontal band, the way the real papillary-dermis masks look.
        mask[height // 3 : 2 * height // 3, :] = 255
        image[height // 3 : 2 * height // 3, :] = 200
        if marker is not None:
            image[marker] = 255
            mask[marker] = 255

        image_path = images_dir / f"{frame_id}.png"
        mask_path = masks_dir / f"mask_{frame_id}.png"
        cv2.imwrite(str(image_path), image)
        if with_mask:
            cv2.imwrite(str(mask_path), mask)

        return {
            "frame_id": frame_id,
            "png_path": str(image_path),
            "mask_path": str(mask_path) if with_mask else None,
            "patient_id": patient_id,
        }

    return _make


@pytest.fixture
def frames_table(frame_factory):
    """Three patients contributing three frames each."""

    def _make(n_patients: int = 3, frames_per_patient: int = 3) -> pd.DataFrame:
        rows = []
        for patient in range(n_patients):
            for frame in range(frames_per_patient):
                rows.append(
                    frame_factory(
                        frame_id=f"p{patient}_f{frame}",
                        patient_id=f"P-{patient:03d}",
                    )
                )
        return pd.DataFrame(rows)

    return _make
