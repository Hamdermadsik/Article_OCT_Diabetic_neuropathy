"""Dataset contract: shapes, value ranges, and image/mask alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from dataloaders import OCT_SKIN, frames_from_directories


def test_returns_expected_shapes_and_ranges(frames_table):
    frames = frames_table(n_patients=2, frames_per_patient=2)

    dataset = OCT_SKIN(frames, target_size=(64, 64), augment=False)
    image, mask = dataset[0]

    assert image.shape == (1, 64, 64)
    assert mask.shape == (1, 64, 64)
    assert image.dtype == torch.float32 and mask.dtype == torch.float32
    assert image.min() >= -1.0 and image.max() <= 1.0
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_mask_is_binary_after_letterbox(frames_table):
    frames = frames_table(n_patients=1, frames_per_patient=1)

    dataset = OCT_SKIN(frames, target_size=(128, 128), augment=False)
    _, mask = dataset[0]

    # Interpolating a mask with anything but nearest-neighbour would introduce
    # fractional values here.
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_image_and_mask_stay_aligned_under_augmentation(frame_factory):
    # The marker sits in both image and mask; any transform applied to one but
    # not the other (or applied differently) separates them.
    marker = (slice(5, 9), slice(20, 24))
    frames = pd.DataFrame([frame_factory("aligned", marker=marker)])

    dataset = OCT_SKIN(frames, target_size=(64, 64), augment=True)
    for _ in range(12):
        image, mask = dataset[0]
        bright = (image > image.max() * 0.7).squeeze(0)
        positive = mask.squeeze(0) > 0
        assert (bright & positive).sum() > 0, "marker lost alignment with its mask"


def test_frames_missing_a_mask_are_excluded_not_crashed(frame_factory):
    frames = pd.DataFrame(
        [
            frame_factory("has_mask", patient_id="P-1"),
            frame_factory("no_mask", patient_id="P-2", with_mask=False),
        ]
    )

    dataset = OCT_SKIN(frames, target_size=(32, 32), augment=False)

    assert len(dataset) == 1
    assert dataset.frames.iloc[0]["frame_id"] == "has_mask"


def test_missing_required_columns_is_an_explicit_error():
    with pytest.raises(ValueError, match="missing column"):
        OCT_SKIN(pd.DataFrame({"frame_id": ["a"]}))


def test_directory_pairing_reports_unmatched_images(frame_factory, caplog):
    row = frame_factory("1", with_mask=True)
    frame_factory("2", with_mask=False)
    images_dir = type(row["png_path"])  # noqa: F841 - keep linters quiet
    from pathlib import Path

    images_dir = Path(row["png_path"]).parent
    masks_dir = Path(row["mask_path"]).parent

    with caplog.at_level("WARNING"):
        frames = frames_from_directories(images_dir, masks_dir)

    assert len(frames) == 1
    assert "no matching mask" in caplog.text


def test_geometry_mismatch_between_image_and_mask_is_detected(frame_factory, tmp_path):
    import cv2

    row = frame_factory("mismatch", height=40, width=100)
    # Rewrite the mask at a different size than its image.
    cv2.imwrite(row["mask_path"], np.zeros((20, 50), dtype=np.uint8))

    dataset = OCT_SKIN(pd.DataFrame([row]), target_size=(32, 32), augment=False)
    with pytest.raises(RuntimeError, match="geometry mismatch"):
        _ = dataset[0]


def test_patient_ids_expose_the_grouping_key(frames_table):
    frames = frames_table(n_patients=3, frames_per_patient=2)

    dataset = OCT_SKIN(frames, target_size=(32, 32), augment=False)

    assert len(set(dataset.patient_ids)) == 3
