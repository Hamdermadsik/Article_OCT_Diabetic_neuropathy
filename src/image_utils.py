"""Aspect-preserving resize helpers.

B-scans are strongly non-square (e.g. 460x1174) and the two axes are not
interchangeable: one is depth, the other lateral position. Resizing straight to
a square stretches the axes by different factors, distorting tissue geometry
before the network ever sees it -- and again, differently, at inference time.

Letterboxing scales both axes by the same factor and pads the remainder, so a
structure's shape is preserved regardless of the source geometry (which does
vary between acquisition cohorts).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxParams:
    """Everything needed to map coordinates back to the original image."""

    scale: float
    pad_top: int
    pad_left: int
    resized_height: int
    resized_width: int
    original_height: int
    original_width: int


def letterbox_resize(
    image: np.ndarray,
    target_size: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
    pad_value: float | int = 0,
) -> tuple[np.ndarray, LetterboxParams]:
    """Resize to ``target_size`` (height, width) preserving aspect ratio.

    Returns the padded image and the parameters needed to invert the operation.
    """
    target_height, target_width = target_size
    original_height, original_width = image.shape[:2]

    scale = min(target_height / original_height, target_width / original_width)
    resized_height = max(1, min(target_height, round(original_height * scale)))
    resized_width = max(1, min(target_width, round(original_width * scale)))

    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

    pad_top = (target_height - resized_height) // 2
    pad_left = (target_width - resized_width) // 2
    pad_bottom = target_height - resized_height - pad_top
    pad_right = target_width - resized_width - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value,
    )
    params = LetterboxParams(
        scale=scale,
        pad_top=pad_top,
        pad_left=pad_left,
        resized_height=resized_height,
        resized_width=resized_width,
        original_height=original_height,
        original_width=original_width,
    )
    return padded, params


def letterbox_unresize(
    image: np.ndarray,
    params: LetterboxParams,
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """Invert :func:`letterbox_resize`, returning an image at original size.

    The padding must be cropped before rescaling; resizing the padded image
    directly would squeeze the padding into the content and shift every
    predicted boundary.
    """
    cropped = image[
        params.pad_top : params.pad_top + params.resized_height,
        params.pad_left : params.pad_left + params.resized_width,
    ]
    return cv2.resize(
        cropped,
        (params.original_width, params.original_height),
        interpolation=interpolation,
    )
