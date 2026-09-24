"""Patient-grouped dataset splitting.

Frames from one patient are near-duplicates of each other: adjacent B-scans
through the same tissue. Splitting them at random puts some of a patient's
frames in train and others in validation, so the model is scored partly on
tissue it has already seen. The resulting score measures memorisation of a
patient rather than generalisation to a new one.

Splitting by patient instead of by frame removes that. The cost is a noisier
score (fewer independent units); the benefit is that the score means what it
claims to.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

log = logging.getLogger(__name__)

PATIENT_COLUMN = "patient_id"


def grouped_split_indices(
    groups, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split positions into (train, val) so no group spans both sides."""
    groups = np.asarray(groups)
    if len(groups) == 0:
        return [], []

    unique_groups = pd.unique(groups)
    if len(unique_groups) < 2:
        log.warning(
            "Only %d distinct patient group(s); cannot hold out a patient-disjoint "
            "validation set. All samples assigned to train.",
            len(unique_groups),
        )
        return list(range(len(groups))), []

    # GroupShuffleSplit allocates whole groups, so the realised fraction of
    # frames is only approximately val_fraction.
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
    return train_idx.tolist(), val_idx.tolist()


def split_frames(
    frames: pd.DataFrame,
    val_fraction: float,
    seed: int,
    patient_column: str = PATIENT_COLUMN,
) -> tuple[list[int], list[int]]:
    """Patient-grouped train/val split over a frames table.

    Falls back to a per-frame split only if the table carries no patient
    column, and says so loudly -- that situation reintroduces leakage.
    """
    if patient_column not in frames.columns:
        log.warning(
            "No '%s' column: falling back to an ungrouped split. Frames from the same "
            "patient may appear in both train and validation.",
            patient_column,
        )
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(len(frames))
        n_val = int(val_fraction * len(frames))
        return shuffled[n_val:].tolist(), shuffled[:n_val].tolist()

    return grouped_split_indices(frames[patient_column].to_numpy(), val_fraction, seed)


def assert_no_patient_overlap(
    frames: pd.DataFrame,
    train_idx,
    val_idx,
    patient_column: str = PATIENT_COLUMN,
) -> None:
    """Raise if any patient appears on both sides of a split."""
    if patient_column not in frames.columns:
        return
    train_patients = set(frames.iloc[list(train_idx)][patient_column])
    val_patients = set(frames.iloc[list(val_idx)][patient_column])
    overlap = train_patients & val_patients
    if overlap:
        raise AssertionError(
            f"{len(overlap)} patient(s) appear in both train and validation: "
            f"{sorted(map(str, overlap))[:10]}"
        )


def describe_split(
    frames: pd.DataFrame, train_idx, val_idx, patient_column: str = PATIENT_COLUMN
) -> str:
    """One-line human-readable summary of a split."""
    if patient_column in frames.columns:
        n_train_patients = frames.iloc[list(train_idx)][patient_column].nunique()
        n_val_patients = frames.iloc[list(val_idx)][patient_column].nunique()
        return (
            f"train={len(train_idx)} frames / {n_train_patients} patients | "
            f"val={len(val_idx)} frames / {n_val_patients} patients"
        )
    return f"train={len(train_idx)} frames | val={len(val_idx)} frames (ungrouped)"
