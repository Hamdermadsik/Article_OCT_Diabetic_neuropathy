"""Image selection, blinding and detector parity for the Fiji loop export."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.transform import resize

from export_loop_annotation import blind, detector_loops, detector_view, select_images
from papi import EN_FACE_SIZE, identify_and_plot_dots


def _scans(patients):
    rows = []
    for patient, group in patients:
        for site in ("Hand", "Foot"):
            phases = ["Baseline"] * 5 + ["Isc"] + ["PORH"] * 3
            for i, phase in enumerate(phases):
                rows.append({"visit_id": f"{patient} | {site} | 2021-05-12", "study_group": group, "phase": phase,
                             "scan_number": i, "mean_slab_flow": 10 + (5 if phase == "PORH" and i == 7 else 0),
                             "filename": f"{patient}_{site}_{i}.dcm"})
    return pd.DataFrame(rows)


PATIENTS = [(f"C{i}", "Control") for i in range(4)] + [(f"T{i}", "T2DM") for i in range(4)]


def test_one_baseline_and_the_porh_peak_per_chosen_participant():
    selected = select_images(_scans(PATIENTS), per_group=2, seed=0)

    assert len(selected) == 8
    assert (selected["site"] == "Hand").all()
    assert selected.groupby("study_group")["patient_id"].nunique().to_dict() == {"Control": 2, "T2DM": 2}
    baseline = selected[selected["phase_pick"] == "Baseline"]
    assert set(baseline["filename"].str.extract(r"_(\d)\.dcm")[0]) == {"2"}  # middle of 5 repeats
    assert set(selected.loc[selected["phase_pick"] == "PORH peak", "scan_number"]) == {7}


def test_names_reveal_nothing_and_repeats_point_to_their_original():
    selected = select_images(_scans(PATIENTS), per_group=2, seed=0)

    images = blind(selected, repeats=2, seed=0)

    assert len(images) == 10
    assert images["image"].is_unique
    assert images["image"].str.fullmatch(r"loop_\d{2}").all()
    repeats = images[images["is_repeat"]]
    originals = images.set_index("image")
    for _, row in repeats.iterrows():
        original = originals.loc[row["repeat_of"]]
        assert original["filename"] == row["filename"]
        assert row["repeat_of"] != row["image"]
        assert (original["tile_x0"], original["tile_y0"]) == (row["tile_x0"], row["tile_y0"])  # same tile


def test_tiles_fit_inside_the_en_face_image():
    from export_loop_annotation import choose_tiles

    corners = choose_tiles(200, 400, seed=3)

    assert (corners >= 0).all()
    assert (corners[:, 0] + 400 <= EN_FACE_SIZE[1]).all() and (corners[:, 1] + 400 <= EN_FACE_SIZE[0]).all()


def test_exported_image_gives_the_detector_the_same_count_as_the_runs():
    rng = np.random.default_rng(1)
    mip = rng.gamma(2, 8, size=(120, 300)).astype(np.float32)
    for r, c in rng.integers((5, 10), (115, 290), size=(25, 2)):
        mip[r, c - 3:c + 3] = 255  # loop-like bright dashes

    in_runs, _ = identify_and_plot_dots(resize(mip, EN_FACE_SIZE, anti_aliasing=True, order=1), min_size=30,
                                        intensity_threshold=75, circularity_threshold=0.45, plot=False)

    view = detector_view(mip)
    assert view.dtype == np.uint8 and view.shape == EN_FACE_SIZE
    assert len(detector_loops(view)) == len(in_runs)
