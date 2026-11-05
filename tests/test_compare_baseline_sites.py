"""Scan selection for the hand-vs-foot comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from compare_baseline_sites import select_scans


def _scan(patient, site, scan_number, phase="Baseline", condition="Regular", date="2021-05-10",
          excluded=False, bodypart=None):
    return {
        "patient_id": patient,
        "protocol_area": site,
        "bodypart": bodypart or ("Hand" if site == "Hand" else "Hallux"),
        "scan_number": scan_number,
        "phase": phase,
        "condition": condition,
        "date": date,
        "excluded": excluded,
        "filename": f"{patient}_{scan_number}.dcm",
        "filepath": f"/mnt/d/{patient}_{scan_number}.dcm",
    }


def _patients(*rows):
    return pd.DataFrame([{"patient_id": p, "group": g} for p, g in rows])


def _visit(patient, start, date="2021-05-10"):
    """Five repeated hand acquisitions, then five foot ones, as the protocol records."""
    return [_scan(patient, "Hand", start + i, date=date) for i in range(5)] + [
        _scan(patient, "Foot", start + 5 + i, date=date) for i in range(5)
    ]


def test_one_scan_per_patient_and_site():
    scans = pd.DataFrame(_visit("A", 100) + _visit("B", 200))

    selected = select_scans(scans, _patients(("A", "HP"), ("B", "DP")), n_patients=2)

    assert len(selected) == 4
    assert selected.groupby("patient_id")["protocol_area"].apply(set).to_dict() == {
        "A": {"Hand", "Foot"},
        "B": {"Hand", "Foot"},
    }


def test_the_middle_repeat_is_chosen():
    selected = select_scans(pd.DataFrame(_visit("A", 100)), _patients(("A", "HP")), n_patients=1)

    assert sorted(selected["scan_number"]) == [102, 107]


def test_patients_without_both_sites_are_skipped():
    scans = pd.DataFrame(_visit("A", 100) + [_scan("B", "Hand", 300)])

    selected = select_scans(scans, _patients(("A", "HP"), ("B", "HP")), n_patients=5)

    assert set(selected["patient_id"]) == {"A"}


def test_other_phases_and_excluded_scans_are_ignored():
    scans = pd.DataFrame(
        _visit("A", 100)
        + [_scan("B", "Hand", 300, phase="PORH"), _scan("B", "Foot", 301, phase="PORH")]
        + [_scan("C", "Hand", 400, excluded=True), _scan("C", "Foot", 401)]
    )

    selected = select_scans(scans, _patients(("A", "HP"), ("B", "HP"), ("C", "HP")), n_patients=5)

    assert set(selected["patient_id"]) == {"A"}


def test_both_sites_come_from_the_same_visit_when_possible():
    scans = pd.DataFrame(
        [_scan("A", "Hand", 100, date="2021-05-10")]
        + _visit("A", 200, date="2021-05-20")
    )

    selected = select_scans(scans, _patients(("A", "HP")), n_patients=1)

    assert set(selected["date"]) == {"2021-05-20"}


def test_groups_are_drawn_evenly():
    scans = pd.DataFrame(
        [row for i, p in enumerate(["D1", "D2", "D3", "H1", "H2", "H3"]) for row in _visit(p, 100 * (i + 1))]
    )
    patients = _patients(("D1", "DP"), ("D2", "DP"), ("D3", "DP"), ("H1", "HP"), ("H2", "HP"), ("H3", "HP"))

    selected = select_scans(scans, patients, n_patients=4)

    counts = selected.drop_duplicates("patient_id")["group"].value_counts().to_dict()
    assert counts == {"DP": 2, "HP": 2}


def test_no_eligible_patients_yields_nothing():
    scans = pd.DataFrame([_scan("A", "Hand", 100)])

    assert select_scans(scans, _patients(("A", "HP"))).empty


class TestResume:
    def _saved(self, tmp_path, **arrays):
        path = tmp_path / "scan.npz"
        np.savez(path, mip=np.zeros((2, 2)), **arrays)
        return path

    def test_missing_results_are_processed(self, tmp_path):
        from compare_baseline_sites import needs_processing

        assert needs_processing(tmp_path / "absent.npz", (0, 15), recompute=False)

    def test_results_from_the_same_slab_are_reused(self, tmp_path):
        from compare_baseline_sites import needs_processing

        path = self._saved(tmp_path, slab=np.array([-5, 20]))

        assert not needs_processing(path, (-5, 20), recompute=False)

    def test_results_from_a_different_slab_are_never_reused(self, tmp_path):
        from compare_baseline_sites import needs_processing

        path = self._saved(tmp_path, slab=np.array([0, 15]))

        assert needs_processing(path, (-5, 20), recompute=False)

    def test_results_saved_before_slabs_were_recorded_count_as_0_to_15(self, tmp_path):
        from compare_baseline_sites import needs_processing

        path = self._saved(tmp_path)

        assert not needs_processing(path, (0, 15), recompute=False)
        assert needs_processing(path, (-5, 20), recompute=False)

    def test_recompute_forces_processing(self, tmp_path):
        from compare_baseline_sites import needs_processing

        path = self._saved(tmp_path, slab=np.array([0, 15]))

        assert needs_processing(path, (0, 15), recompute=True)
