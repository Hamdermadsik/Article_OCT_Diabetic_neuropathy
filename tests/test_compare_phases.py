"""Visit selection, timing and summary for the PORH time-course analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compare_phases import seconds_from_occlusion, select_visits, summarise_visit


def _visit_scans(patient, site, start, date="2021-05-12", condition="Regular", t0="10:00:00"):
    """5 Baseline, 1 Isc, 6 PORH scans, 47 s apart, as the protocol records them."""
    phases = ["Baseline"] * 5 + ["Isc"] + ["PORH"] * 6
    clock = pd.Timestamp(f"{date} {t0}")
    rows = []
    for i, phase in enumerate(phases):
        rows.append(
            {
                "patient_id": patient,
                "protocol_area": site,
                "bodypart": "Hand" if site == "Hand" else "Hallux",
                "date": date,
                "phase": phase,
                "condition": condition,
                "excluded": False,
                "scan_number": start + i,
                "filename": f"{patient}_{start + i}.dcm",
                "filepath": f"/mnt/d/{patient}_{start + i}.dcm",
                "acquisition_datetime": (clock + pd.Timedelta(seconds=47 * i)).isoformat(),
            }
        )
    return rows


def _patients(*rows):
    return pd.DataFrame([{"patient_id": p, "group": g} for p, g in rows])


class TestSelectVisits:
    def test_takes_every_protocol_scan_of_one_visit_per_site(self):
        scans = pd.DataFrame(_visit_scans("A", "Hand", 100) + _visit_scans("A", "Foot", 200))

        selected = select_visits(scans, _patients(("A", "HP")), n_patients=1)

        assert len(selected) == 24
        assert selected["visit_id"].nunique() == 2
        assert selected.groupby("visit_id")["phase"].apply(set).map(len).eq(3).all()

    def test_patients_without_both_sites_are_skipped(self):
        scans = pd.DataFrame(
            _visit_scans("A", "Hand", 100) + _visit_scans("A", "Foot", 200) + _visit_scans("B", "Hand", 300)
        )

        selected = select_visits(scans, _patients(("A", "HP"), ("B", "HP")), n_patients=5)

        assert set(selected["patient_id"]) == {"A"}

    def test_flavanol_runs_are_left_out(self):
        # Real case: VF's foot went through the protocol again after flavanol.
        scans = pd.DataFrame(
            _visit_scans("A", "Hand", 100)
            + _visit_scans("A", "Foot", 200)
            + _visit_scans("A", "Foot", 300, condition="Flavanol", t0="12:00:00")
        )

        selected = select_visits(scans, _patients(("A", "HP")), n_patients=1)

        assert set(selected["condition"]) == {"Regular"}
        assert len(selected) == 24

    def test_earliest_qualifying_visit_is_used(self):
        scans = pd.DataFrame(
            _visit_scans("A", "Hand", 100, date="2021-05-12")
            + _visit_scans("A", "Foot", 200, date="2021-05-12")
            + _visit_scans("A", "Hand", 500, date="2021-06-01")
        )

        selected = select_visits(scans, _patients(("A", "HP")), n_patients=1)

        assert set(selected["date"]) == {"2021-05-12"}

    def test_no_eligible_patients_yields_nothing(self):
        scans = pd.DataFrame(_visit_scans("A", "Hand", 100))

        assert select_visits(scans, _patients(("A", "HP"))).empty


class TestTiming:
    def test_time_is_measured_from_the_occlusion_scan(self):
        visit = pd.DataFrame(_visit_scans("A", "Hand", 100))

        seconds = seconds_from_occlusion(visit)

        assert seconds[visit["phase"] == "Isc"].iloc[0] == 0
        assert seconds.iloc[0] == -5 * 47
        assert seconds.iloc[-1] == 6 * 47

    def test_without_an_occlusion_scan_the_first_porh_scan_is_zero(self):
        visit = pd.DataFrame([r for r in _visit_scans("A", "Hand", 100) if r["phase"] != "Isc"])

        seconds = seconds_from_occlusion(visit)

        assert seconds[visit["phase"] == "PORH"].min() == 0


class TestSummariseVisit:
    def _visit(self, baseline, isc, porh):
        visit = pd.DataFrame(_visit_scans("A", "Hand", 100)).assign(group="HP", visit_id="A | Hand")
        values = list(baseline) + [isc] + list(porh)
        visit["blobs"] = values
        visit["mean_slab_flow"] = values
        visit["seconds"] = seconds_from_occlusion(visit)
        return visit

    def test_baseline_level_spread_and_porh_peak(self):
        visit = self._visit(baseline=[90, 95, 100, 105, 110], isc=40, porh=[300, 250, 200, 150, 120, 100])

        summary = summarise_visit(visit)

        assert summary["blobs_baseline"] == 100
        assert summary["blobs_baseline_sd"] == pytest.approx(np.std([90, 95, 100, 105, 110], ddof=1), abs=0.01)
        assert summary["blobs_isc"] == 40
        assert summary["blobs_porh_peak"] == 300
        assert summary["blobs_peak_ratio"] == 3.0
        assert summary["blobs_peak_time_s"] == 47

    def test_peak_is_expressed_in_baseline_standard_deviations(self):
        visit = self._visit(baseline=[90, 95, 100, 105, 110], isc=40, porh=[110, 105, 100, 100, 100, 100])

        summary = summarise_visit(visit)

        sd = np.std([90, 95, 100, 105, 110], ddof=1)
        assert summary["blobs_peak_vs_baseline_sd"] == pytest.approx(10 / sd, abs=0.1)
