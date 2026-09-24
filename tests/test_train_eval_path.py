"""Regression tests for the silently-dead test-evaluation path.

Two bugs made held-out evaluation impossible, and neither surfaced as an error:
the test set was discovered with a ``*.jpeg`` glob while the files on disk are
``.png`` (so the branch never ran), and the dataset was constructed with
``n_samples``/``noise_sigma`` keyword arguments it does not accept (so the
branch would have raised had it ever run). Every recorded training history
therefore has ``test_dice = None``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from dataloaders import OCT_SKIN, frames_from_directories


def test_test_set_discovery_finds_png_files(frame_factory):
    row = frame_factory("125")
    images_dir = Path(row["png_path"]).parent
    masks_dir = Path(row["mask_path"]).parent

    frames = frames_from_directories(images_dir, masks_dir)

    assert len(frames) == 1, "test-set discovery must match the .png files on disk"


def test_dataset_accepts_exactly_the_arguments_training_passes():
    # The old call site passed n_samples/noise_sigma, which raised TypeError.
    parameters = inspect.signature(OCT_SKIN.__init__).parameters

    assert {"frames", "target_size", "augment"} <= set(parameters)
    assert "n_samples" not in parameters
    assert "noise_sigma" not in parameters


def test_constructing_the_test_dataset_does_not_raise(frame_factory):
    frames = pd.DataFrame([frame_factory("125"), frame_factory("126")])

    dataset = OCT_SKIN(frames, target_size=(32, 32), augment=False)

    assert len(dataset) == 2
    image, mask = dataset[0]
    assert image.shape == (1, 32, 32) and mask.shape == (1, 32, 32)


def test_training_module_no_longer_globs_for_jpeg():
    source = (Path(__file__).resolve().parents[1] / "src" / "train.py").read_text()

    assert ".jpeg" not in source, "test-set discovery must not look for .jpeg files"


def test_empty_test_directory_is_handled_without_crashing(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    frames = frames_from_directories(images_dir, masks_dir)

    assert len(frames) == 0
