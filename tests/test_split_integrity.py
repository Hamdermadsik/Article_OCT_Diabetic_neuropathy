"""The regression test for the leakage bug this work exists to fix.

Historically the split was a flat permutation over frames with no notion of
patient, so a patient's near-identical adjacent B-scans landed on both sides
and inflated the validation score. If this test ever fails, that is back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from splits import assert_no_patient_overlap, grouped_split_indices, split_frames


def test_no_patient_appears_in_both_splits(frames_table):
    frames = frames_table(n_patients=10, frames_per_patient=3)

    train_idx, val_idx = split_frames(frames, val_fraction=0.3, seed=42)

    train_patients = set(frames.iloc[train_idx]["patient_id"])
    val_patients = set(frames.iloc[val_idx]["patient_id"])
    assert train_patients and val_patients
    assert not (train_patients & val_patients)


def test_every_frame_is_assigned_exactly_once(frames_table):
    frames = frames_table(n_patients=8, frames_per_patient=3)

    train_idx, val_idx = split_frames(frames, val_fraction=0.25, seed=0)

    assert sorted(train_idx + val_idx) == list(range(len(frames)))


def test_a_patients_frames_are_never_separated():
    # Mirrors the real failure: consecutive frames belonging to one patient.
    groups = np.repeat([f"P-{i:03d}" for i in range(12)], 3)

    train_idx, val_idx = grouped_split_indices(groups, val_fraction=0.25, seed=7)

    for patient in set(groups):
        positions = {i for i, g in enumerate(groups) if g == patient}
        assert positions <= set(train_idx) or positions <= set(val_idx)


def test_split_is_deterministic_for_a_given_seed(frames_table):
    frames = frames_table(n_patients=6, frames_per_patient=3)

    first = split_frames(frames, val_fraction=0.3, seed=123)
    second = split_frames(frames, val_fraction=0.3, seed=123)

    assert first == second


def test_assert_no_patient_overlap_raises_on_a_leaking_split(frames_table):
    frames = frames_table(n_patients=4, frames_per_patient=3)
    # Deliberately leak: frames 0 and 1 are the same patient.
    leaking_train = [0] + list(range(2, len(frames)))
    leaking_val = [1]

    with pytest.raises(AssertionError, match="both train and validation"):
        assert_no_patient_overlap(frames, leaking_train, leaking_val)


def test_single_patient_cannot_produce_a_disjoint_validation_set(frames_table):
    frames = frames_table(n_patients=1, frames_per_patient=5)

    train_idx, val_idx = split_frames(frames, val_fraction=0.4, seed=1)

    # Better to hold out nothing than to report a leaked validation score.
    assert val_idx == []
    assert len(train_idx) == len(frames)


def test_ungrouped_fallback_is_used_only_without_patient_ids(frames_table):
    frames = frames_table(n_patients=5, frames_per_patient=2).drop(columns=["patient_id"])

    train_idx, val_idx = split_frames(frames, val_fraction=0.2, seed=3)

    assert sorted(train_idx + val_idx) == list(range(len(frames)))
