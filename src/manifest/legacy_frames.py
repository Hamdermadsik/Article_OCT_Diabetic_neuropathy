"""Bring the original hand-picked, hand-annotated frames into the manifest.

These 63 frames predate any provenance tracking: they were pulled manually from
various acquisitions and numbered sequentially, with no record of which
acquisition each came from. Their patient identity is therefore *inferred*, not
known -- frames were pulled roughly three per participant, so consecutive runs
of up to three frame numbers are treated as one participant.

This is explicitly a heuristic. Every row is stamped
``patient_id_confidence="heuristic_consecutive_run"`` so it can never be
mistaken for resolved identity, and so a patient-grouped split treats these
frames conservatively rather than assuming they are independent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cohort_specs import LEGACY_SOURCE

LEGACY_COHORT = "legacy"
FRAMES_PER_PARTICIPANT = 3


def _frame_numbers(images_dir: Path) -> list[int]:
    numbers = []
    for path in images_dir.glob("*.png"):
        try:
            numbers.append(int(path.stem))
        except ValueError:
            continue
    return sorted(numbers)


def group_consecutive_runs(numbers: list[int], max_run: int = FRAMES_PER_PARTICIPANT) -> list[list[int]]:
    """Split a sorted number list into runs of at most ``max_run`` consecutives."""
    groups: list[list[int]] = []
    index = 0
    while index < len(numbers):
        run = [numbers[index]]
        while (
            index + 1 < len(numbers)
            and numbers[index + 1] == run[-1] + 1
            and len(run) < max_run
        ):
            run.append(numbers[index + 1])
            index += 1
        groups.append(run)
        index += 1
    return groups


def build_legacy_frames(
    train_images: Path,
    train_masks: Path,
    test_images: Path,
    test_masks: Path,
) -> pd.DataFrame:
    """One manifest row per legacy annotated frame."""
    sources = [(train_images, train_masks, "train"), (test_images, test_masks, "test")]

    available: dict[int, dict] = {}
    for images_dir, masks_dir, split_hint in sources:
        if not images_dir.exists():
            continue
        for number in _frame_numbers(images_dir):
            image_path = images_dir / f"{number}.png"
            mask_path = masks_dir / f"mask_{number}.png"
            available[number] = {
                "png_path": str(image_path),
                "mask_path": str(mask_path) if mask_path.exists() else None,
                "legacy_split": split_hint,
            }

    if not available:
        return pd.DataFrame()

    # Group across the combined train+test sequence: both splits came from one
    # numbering run, so a participant's frames can straddle the boundary.
    groups = group_consecutive_runs(sorted(available))

    rows = []
    for group_index, numbers in enumerate(groups, start=1):
        patient_id = f"LEGACY-{group_index:03d}"
        for number in numbers:
            info = available[number]
            rows.append(
                {
                    "frame_id": f"legacy_{number:04d}",
                    "scan_id": None,
                    "frame_index": None,
                    "patient_id": patient_id,
                    "cohort": LEGACY_COHORT,
                    "source": LEGACY_SOURCE,
                    "patient_id_confidence": "heuristic_consecutive_run",
                    "extraction_status": "extracted",
                    "png_path": info["png_path"],
                    "annotation_status": "verified" if info["mask_path"] else "none",
                    "upper_line_path": None,
                    "lower_line_path": None,
                    "mask_path": info["mask_path"],
                    "legacy_split": info["legacy_split"],
                    "legacy_frame_number": number,
                }
            )
    return pd.DataFrame(rows)
