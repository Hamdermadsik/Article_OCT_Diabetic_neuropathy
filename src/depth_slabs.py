"""Depth boundaries for slabs relative to an epidermis segmentation."""

import numpy as np


def extract_dej(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dej_px, surface_px, valid_columns) from a 2D binary mask.

    Rows increase with depth. Each valid column must contain exactly one
    nonempty, contiguous foreground run. The epidermis occupies the half-open
    interval [surface_px, dej_px): DEJ is the boundary immediately AFTER the
    last foreground row, not that row's index. Boundaries use pixel-edge
    coordinates and may equal the image height.

    Empty or disconnected columns receive NaN boundaries and False validity.
    Validity checks only mask continuity, not anatomical accuracy or whether
    sufficient image depth remains for a requested slab. Input must contain
    only 0 and 1 (or booleans); probabilities must be thresholded beforehand.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or 0 in mask.shape:
        raise ValueError("mask must be a nonempty 2D array")
    if mask.dtype.kind not in "buif" or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must contain only binary values (0 or 1)")

    foreground = mask.astype(bool)
    height, _ = foreground.shape
    surface = np.argmax(foreground, axis=0)
    dej = height - np.argmax(foreground[::-1], axis=0)
    foreground_count = np.count_nonzero(foreground, axis=0)
    # A contiguous run fills every row between its first and last pixel.
    valid = (foreground_count > 0) & (foreground_count == dej - surface)

    return np.where(valid, dej, np.nan), np.where(valid, surface, np.nan), valid


def project_slab(
    flow: np.ndarray,
    dej_px: np.ndarray,
    valid_columns: np.ndarray,
    start_px: int,
    stop_px: int,
    *,
    surface_px: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one B-scan's [DEJ + start_px, DEJ + stop_px) slab.

    Return a float MIP row and a boolean validity row, each of length width.
    Offsets are integer pixels, positive into the dermis. No interpolation,
    clipping, resizing or intensity normalization is performed. Stack rows
    from successive B-scans to form an en-face MIP.

    A column is invalid (NaN output) if its full interval is unavailable or
    contains nonfinite signal. Zero signal is valid. Negative start offsets
    require surface_px to exclude air above the segmented skin surface.
    Boundaries must be integer pixel edges, as returned by extract_dej.
    """
    flow = np.asarray(flow)
    if flow.ndim != 2 or 0 in flow.shape or flow.dtype.kind not in "buif":
        raise ValueError("flow must be a nonempty real numeric 2D array")
    for offset in (start_px, stop_px):
        if isinstance(offset, (bool, np.bool_)) or not isinstance(offset, (int, np.integer)):
            raise ValueError("slab offsets must be integer pixels")
    if start_px >= stop_px:
        raise ValueError("start_px must be less than stop_px")
    if start_px < 0 and surface_px is None:
        raise ValueError("surface_px is required for negative start offsets")

    height, width = flow.shape
    dej_px = np.asarray(dej_px)
    valid_columns = np.asarray(valid_columns)
    if dej_px.shape != (width,) or dej_px.dtype.kind not in "uif":
        raise ValueError("dej_px must be a numeric vector matching flow width")
    if valid_columns.shape != (width,) or valid_columns.dtype.kind != "b":
        raise ValueError("valid_columns must be a boolean vector matching flow width")
    if surface_px is not None:
        surface_px = np.asarray(surface_px)
        if surface_px.shape != (width,) or surface_px.dtype.kind not in "uif":
            raise ValueError("surface_px must be a numeric vector matching flow width")

    mip_row = np.full(width, np.nan)
    valid_row = np.zeros(width, dtype=bool)
    for col in np.flatnonzero(valid_columns):
        dej = dej_px[col]
        if not np.isfinite(dej) or not 0 < dej <= height or dej != np.floor(dej):
            continue
        start, stop = int(dej) + int(start_px), int(dej) + int(stop_px)
        if start < 0 or stop > height:
            continue
        if surface_px is not None:
            surface = surface_px[col]
            if (
                not np.isfinite(surface)
                or not 0 <= surface < dej
                or surface != np.floor(surface)
                or start < surface
            ):
                continue
        signal = flow[start:stop, col]
        if not np.isfinite(signal).all():
            continue
        mip_row[col] = np.max(signal)
        valid_row[col] = True

    return mip_row, valid_row
