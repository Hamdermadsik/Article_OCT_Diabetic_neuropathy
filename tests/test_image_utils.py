"""Letterbox resize must preserve geometry and invert cleanly.

The bug being guarded against: resizing a 460x1174 B-scan straight to 512x512
scales depth and lateral axes by different factors, so tissue is distorted
before the model sees it and predictions are distorted on the way back out.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from image_utils import letterbox_resize, letterbox_unresize


@pytest.mark.parametrize(
    "shape", [(460, 1174), (460, 1129), (100, 100), (1174, 460), (37, 211)]
)
def test_output_is_exactly_target_size(shape):
    image = np.zeros(shape, dtype=np.uint8)

    output, _ = letterbox_resize(image, (512, 512))

    assert output.shape == (512, 512)


def test_aspect_ratio_is_preserved():
    image = np.zeros((460, 1174), dtype=np.uint8)

    _, params = letterbox_resize(image, (512, 512))

    original_ratio = 1174 / 460
    resized_ratio = params.resized_width / params.resized_height
    assert resized_ratio == pytest.approx(original_ratio, rel=0.01)


def test_a_square_stays_square_through_the_resize():
    # A plain resize to a square would stretch this 20x20 block into a
    # rectangle; letterboxing must not.
    image = np.zeros((200, 600), dtype=np.uint8)
    image[90:110, 290:310] = 255

    output, _ = letterbox_resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)

    rows = np.where(output.any(axis=1))[0]
    cols = np.where(output.any(axis=0))[0]
    height = rows[-1] - rows[0] + 1
    width = cols[-1] - cols[0] + 1
    assert height == pytest.approx(width, abs=2)


def test_round_trip_returns_original_geometry():
    mask = np.zeros((460, 1174), dtype=np.float32)
    mask[150:300, :] = 1.0

    resized, params = letterbox_resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    restored = letterbox_unresize(resized, params, interpolation=cv2.INTER_NEAREST)

    assert restored.shape == mask.shape
    # The band's vertical extent must survive the round trip.
    original_rows = np.where(mask.any(axis=1))[0]
    restored_rows = np.where(restored.any(axis=1))[0]
    assert restored_rows[0] == pytest.approx(original_rows[0], abs=3)
    assert restored_rows[-1] == pytest.approx(original_rows[-1], abs=3)


def test_marker_position_survives_the_round_trip():
    image = np.zeros((300, 900), dtype=np.float32)
    image[100, 450] = 1.0

    resized, params = letterbox_resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
    restored = letterbox_unresize(resized, params, interpolation=cv2.INTER_NEAREST)

    rows, cols = np.where(restored > 0)
    assert len(rows) > 0
    assert rows.mean() == pytest.approx(100, abs=4)
    assert cols.mean() == pytest.approx(450, abs=4)


def test_padding_does_not_bleed_into_content():
    image = np.full((200, 600), 255, dtype=np.uint8)

    output, params = letterbox_resize(image, (512, 512), pad_value=0)

    content = output[
        params.pad_top : params.pad_top + params.resized_height,
        params.pad_left : params.pad_left + params.resized_width,
    ]
    assert (content == 255).all()
    assert output.sum() == content.sum()
