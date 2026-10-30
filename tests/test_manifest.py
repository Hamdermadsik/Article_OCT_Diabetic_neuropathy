"""Manifest parsing and matching behaviour.

These cover the failure modes found in the previous pipeline: participant IDs
spelled differently in different sheets, initials shared by two people, and
malformed scan-range cells expanding into hundreds of phantom scan numbers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from manifest.demographics import (
    build_identity_review,
    find_ambiguous_initials,
    resolve_demographics,
)
from manifest.excel_utils import (
    expand_scan_field,
    expand_scan_field_with_issues,
    extract_initials,
    normalize_participant_id,
    parse_dicom_filename,
)
from manifest.legacy_frames import group_consecutive_runs
from manifest.matching import find_phase_overlaps, mark_duplicates, match_scans


class TestParticipantIdNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("DP-06-JF", "DP-06-JF"),
            ("DP-02DG", "DP-02-DG"),
            ("DP15NL", "DP-15-NL"),
            ("hp-53RM", "HP-53-RM"),
            ("HP-22-DMS", "HP-22-DMS"),
            ("AR", "AR"),
            ("  hp-03mb  ", "HP-03-MB"),
        ],
    )
    def test_spellings_collapse_to_one_key(self, raw, expected):
        assert normalize_participant_id(raw) == expected

    def test_the_two_sheet_spellings_of_one_person_agree(self):
        # The scan log writes DP-02-DG; the demographics sheet writes DP-02DG.
        # If these disagree, demographics never join and one person becomes two.
        assert normalize_participant_id("DP-02-DG") == normalize_participant_id("DP-02DG")

    def test_blank_values_yield_none(self):
        assert normalize_participant_id(None) is None
        assert normalize_participant_id("") is None

    def test_initials_come_from_the_trailing_letters(self):
        assert extract_initials("HP-52CSM") == "CSM"
        assert extract_initials("AR") == "AR"


class TestScanFieldExpansion:
    def test_expands_ranges_and_lists(self):
        assert expand_scan_field("193-196,198") == [193, 194, 195, 196, 198]

    def test_tolerates_decimals_and_stray_characters(self):
        assert expand_scan_field("2178.0") == [2178]
        assert expand_scan_field("249, 251-254") == [249, 251, 252, 253, 254]

    def test_blank_cells_expand_to_nothing(self):
        assert expand_scan_field(None) == []
        assert expand_scan_field("") == []
        assert expand_scan_field(float("nan")) == []

    def test_implausibly_wide_ranges_are_rejected_and_reported(self):
        # Real cell from the May log; almost certainly a typo for 2179-2184.
        scans, issues = expand_scan_field_with_issues("2179-2484")

        assert scans == []
        assert any("spans 306 scans" in issue for issue in issues)

    def test_a_missing_comma_does_not_invent_thousands_of_scans(self):
        # Real cell: "2955,29562959-2961" - the comma after 2956 was dropped.
        scans, issues = expand_scan_field_with_issues("2955,29562959-2961")

        assert 2955 in scans
        assert len(scans) < 10
        assert issues

    def test_descending_ranges_are_flagged(self):
        _, issues = expand_scan_field_with_issues("2316-2312")
        assert any("descending" in issue for issue in issues)


class TestFilenameParsing:
    def test_parses_a_standard_acquisition_name(self):
        parsed = parse_dicom_filename("AC_Right Hallux_L1051_S2397__20_05_2021.dcm")

        assert parsed["initials"] == "AC"
        assert parsed["bodypart"] == "Hallux"
        assert parsed["protocol_area"] == "Foot"
        assert parsed["scan_number"] == 2397
        assert parsed["copy_index"] == 0

    def test_windows_copy_suffix_is_captured_not_rejected(self):
        parsed = parse_dicom_filename("RS_Right Hand_L1074_S3851__02_06_2021 (1).dcm")

        assert parsed is not None, "copy suffix must not push the file into 'unparsed'"
        assert parsed["copy_index"] == 1
        assert parsed["scan_number"] == 3851

    def test_hand_maps_to_hand_and_foot_sites_map_to_foot(self):
        assert parse_dicom_filename("X_Right Hand_L1_S1__01_01_2021.dcm")["protocol_area"] == "Hand"
        assert parse_dicom_filename("X_Left Instep_L1_S1__01_01_2021.dcm")["protocol_area"] == "Foot"

    def test_finger_sites_parse_and_count_as_hand(self):
        # Real files: "BB_Right Middle Finger_L1047_S2245__19_05_2021.dcm".
        parsed = parse_dicom_filename("BB_Right Middle Finger_L1047_S2245__19_05_2021.dcm")

        assert parsed is not None
        assert parsed["bodypart"] == "Middle Finger"
        assert parsed["protocol_area"] == "Hand"

    def test_unrecognized_names_return_none(self):
        assert parse_dicom_filename("notes.txt") is None
        assert parse_dicom_filename("ProcessMichelson_Main.m") is None


class TestMatching:
    def _long_row(self, **overrides):
        row = {
            "participant_id": "HP-01-AB",
            "initials": "AB",
            "group": "HP",
            "date": pd.Timestamp("2021-05-10"),
            "scan_number": 100,
            "phase": "Baseline",
            "protocol_area": "Hand",
            "condition": "Regular",
            "excel_column": "Hand Baseline",
            "cohort": "may_2021",
        }
        row.update(overrides)
        return row

    def _file_row(self, **overrides):
        row = {
            "filename": "AB_Right Hand_L1_S100__10_05_2021.dcm",
            "scan_id": "AB_scan",
            "initials": "AB",
            "date": pd.Timestamp("2021-05-10"),
            "scan_number": 100,
            "protocol_area": "Hand",
            "cohort": "may_2021",
            "copy_index": 0,
            "file_size": 100,
        }
        row.update(overrides)
        return row

    def test_exact_match_is_preferred(self):
        result = match_scans(
            pd.DataFrame([self._file_row()]), pd.DataFrame([self._long_row()])
        )
        assert result.iloc[0]["match_method"] == "exact"
        assert result.iloc[0]["participant_id"] == "HP-01-AB"

    def test_falls_back_to_a_looser_key_when_the_date_disagrees(self):
        files = pd.DataFrame([self._file_row(date=pd.Timestamp("2021-05-11"))])
        result = match_scans(files, pd.DataFrame([self._long_row()]))

        assert result.iloc[0]["match_method"] == "initials_scan_area"
        assert result.iloc[0]["participant_id"] == "HP-01-AB"

    def test_a_scan_claimed_by_two_participants_is_left_unresolved(self):
        # Never guess which of two people a scan belongs to.
        long_df = pd.DataFrame(
            [
                self._long_row(participant_id="HP-01-AB", initials="ZZ"),
                self._long_row(participant_id="HP-02-CD", initials="ZZ"),
            ]
        )
        files = pd.DataFrame([self._file_row(initials="ZZ")])

        result = match_scans(files, long_df)

        assert result.iloc[0]["match_method"] == "ambiguous"
        assert result.iloc[0]["participant_id"] is None

    def test_identity_inferred_from_scan_number_alone_is_flagged(self):
        files = pd.DataFrame([self._file_row(initials="QQ")])
        result = match_scans(files, pd.DataFrame([self._long_row()]))

        assert result.iloc[0]["match_method"] == "scan_area_only"
        assert "scan number alone" in result.iloc[0]["match_detail"]

    def test_unmatched_files_are_reported_not_dropped(self):
        files = pd.DataFrame([self._file_row(scan_number=999999, initials="QQ")])
        result = match_scans(files, pd.DataFrame([self._long_row()]))

        assert len(result) == 1
        assert result.iloc[0]["match_method"] == "unmatched"

    def test_a_scan_in_two_phases_keeps_its_patient_but_not_a_guessed_phase(self):
        # Real case: NB's Foot PORH cell "2200-2211" overlaps Foot Baseline
        # 2200-2204. Taking the first listing would mislabel a PORH scan as
        # Baseline and wrongly make it a segmentation candidate.
        long_df = pd.DataFrame(
            [
                self._long_row(phase="Baseline", excel_column="Hand Baseline"),
                self._long_row(phase="PORH", excel_column="Hand PORH"),
            ]
        )

        result = match_scans(pd.DataFrame([self._file_row()]), long_df)

        row = result.iloc[0]
        assert row["participant_id"] == "HP-01-AB"
        assert row["phase"] is None
        assert "phase ambiguous" in row["match_detail"]

    def test_windows_copies_are_marked_as_duplicates(self):
        scans = pd.DataFrame(
            [
                self._file_row(scan_id="original", copy_index=0),
                self._file_row(scan_id="copy", copy_index=1),
            ]
        )
        marked = mark_duplicates(scans)

        by_id = marked.set_index("scan_id")["is_duplicate_of"]
        assert by_id["original"] is None
        assert by_id["copy"] == "original"


def _scan_log(*participants):
    """Minimal long-format scan log: (cohort, participant_id) pairs."""
    rows = []
    for cohort, participant_id in participants:
        rows.append(
            {
                "cohort": cohort,
                "participant_id": participant_id,
                "initials": participant_id.split("-")[-1],
                "group": participant_id.split("-")[0] if "-" in participant_id else None,
            }
        )
    return pd.DataFrame(rows)


def _sheet(*rows):
    """Demographics-sheet rows: (participant_id, sex, age)."""
    records = []
    for i, (participant_id, sex, age) in enumerate(rows):
        initials = participant_id.split("-")[-1]
        records.append(
            {
                "participant_id": participant_id,
                "initials": initials,
                "group": participant_id.split("-")[0] if "-" in participant_id else None,
                "bare_initials": participant_id == initials,
                "sex": sex,
                "age": age,
                "sheet_row": i + 1,
            }
        )
    return pd.DataFrame(records)


def _october(*rows):
    return pd.DataFrame(
        [{"participant_id": p, "sex": s, "age": a} for p, s, a in rows],
        columns=["participant_id", "sex", "age"],
    )


class TestDemographics:
    def test_session_comes_from_the_scan_log_not_the_demographics_sheet(self):
        # Regression: the demographics sheet lists both sessions. Someone scanned
        # only in October must not acquire a phantom May record just because
        # their row sits in that sheet.
        log = _scan_log(("may_2021", "HP-01-AB"), ("october_2020", "AF"))
        sheet = _sheet(("HP-01-AB", "M", 40), ("HP-08-AF", "M", 33))

        participants, _ = resolve_demographics(log, sheet, _october(("AF", "M", 33)))

        may = participants[participants["cohort"] == "may_2021"]
        assert set(may["participant_id"]) == {"HP-01-AB"}
        assert build_identity_review(participants).empty

    def test_may_participants_match_the_sheet_on_full_id(self):
        log = _scan_log(("may_2021", "HP-01-AB"))
        sheet = _sheet(("HP-01-AB", "F", 50))

        participants, issues = resolve_demographics(log, sheet, _october())

        row = participants.iloc[0]
        assert (row["sex"], row["age"]) == ("F", 50)
        assert row["demographics_source"] == "demographics_sheet"
        assert issues.empty

    def test_conflicting_rows_are_left_blank_not_guessed(self):
        # Real case: HP-35-LM listed at both 54 and 45.
        log = _scan_log(("may_2021", "HP-35-LM"))
        sheet = _sheet(("HP-35-LM", "F", 54), ("HP-35-LM", "F", 45))

        participants, issues = resolve_demographics(log, sheet, _october())

        assert pd.isna(participants.iloc[0]["age"])
        assert participants.iloc[0]["demographics_source"] == "conflicting_rows"
        assert list(issues["issue"]) == ["conflicting_demographics"]

    def test_a_conflict_matching_the_october_person_says_so(self):
        # Real case: HP-53-RM listed at 26 and 65; October's RM is 26.
        log = _scan_log(("may_2021", "HP-53-RM"), ("october_2020", "RM"))
        sheet = _sheet(("HP-53-RM", "M", 26), ("HP-53-RM", "M", 65))

        _, issues = resolve_demographics(log, sheet, _october(("RM", "M", 26)))

        detail = issues[issues["issue"] == "conflicting_demographics"].iloc[0]["detail"]
        assert "October participant RM" in detail

    def test_bare_initials_row_is_used_only_as_a_flagged_fallback(self):
        log = _scan_log(("may_2021", "HP-48-MP"))
        sheet = _sheet(("MP", "M", 64))

        participants, issues = resolve_demographics(log, sheet, _october())

        assert participants.iloc[0]["age"] == 64
        assert participants.iloc[0]["demographics_source"] == "demographics_sheet_by_initials"
        assert list(issues["issue"]) == ["matched_by_initials_only"]

    def test_missing_demographics_suggest_likely_typos(self):
        # Real case: the log has DP-12-RS, the sheet has DP-12-SR.
        log = _scan_log(("may_2021", "DP-12-RS"))
        sheet = _sheet(("DP-12-SR", "F", 71))

        participants, issues = resolve_demographics(log, sheet, _october())

        assert pd.isna(participants.iloc[0]["age"])
        assert "DP-12-SR" in issues.iloc[0]["detail"]

    def test_october_disagreement_with_the_sheet_is_reported(self):
        # Real case: October's PM is M/27 in the scan log, F/27 in the sheet.
        log = _scan_log(("october_2020", "PM"))
        sheet = _sheet(("HP-09-PM", "F", 27))

        participants, issues = resolve_demographics(log, sheet, _october(("PM", "M", 27)))

        assert participants.iloc[0]["sex"] == "M"
        assert list(issues["issue"]) == ["sources_disagree"]

    def test_missing_october_sex_is_filled_not_reported_as_a_disagreement(self):
        # Real case: October's log has GW at age 22 with no sex; the sheet has
        # HP-18-GW as F/22. That is a gap, not a contradiction.
        log = _scan_log(("october_2020", "GW"))
        sheet = _sheet(("HP-18-GW", "F", 22))

        participants, issues = resolve_demographics(log, sheet, _october(("GW", None, 22)))

        assert participants.iloc[0]["sex"] == "F"
        assert list(issues["issue"]) == ["sex_filled_from_sheet"]

    def test_missing_october_sex_is_not_filled_when_ages_disagree(self):
        log = _scan_log(("october_2020", "GW"))
        sheet = _sheet(("HP-18-GW", "F", 60))

        participants, issues = resolve_demographics(log, sheet, _october(("GW", None, 22)))

        assert pd.isna(participants.iloc[0]["sex"])
        assert list(issues["issue"]) == ["sources_disagree"]

    def test_identity_review_pairs_only_initials_scanned_in_both_sessions(self):
        log = _scan_log(
            ("may_2021", "HP-20-CH"),
            ("october_2020", "CH"),
            ("may_2021", "HP-49-MB"),
            ("october_2020", "MB"),
            ("may_2021", "HP-01-AB"),
        )
        sheet = _sheet(("HP-20-CH", "M", 46), ("HP-49-MB", "M", 44), ("HP-01-AB", "F", 30))
        october = _october(("CH", "M", 46), ("MB", "F", 28))

        participants, _ = resolve_demographics(log, sheet, october)
        review = build_identity_review(participants).set_index("initials")

        assert set(review.index) == {"CH", "MB"}
        assert "same person" in review.loc["CH", "assessment"]
        assert "different people" in review.loc["MB", "assessment"]

    def test_initials_shared_within_one_session_are_detected(self):
        participants = pd.DataFrame(
            [
                {"cohort": "may_2021", "participant_id": "HP-09-PM", "initials": "PM"},
                {"cohort": "may_2021", "participant_id": "HP-48-PM", "initials": "PM"},
                {"cohort": "may_2021", "participant_id": "HP-01-AB", "initials": "AB"},
            ]
        )

        assert set(find_ambiguous_initials(participants)["initials"]) == {"PM"}


class TestLegacyGrouping:
    def test_consecutive_frames_group_into_threes(self):
        assert group_consecutive_runs([1, 2, 3, 4, 5, 6]) == [[1, 2, 3], [4, 5, 6]]

    def test_a_gap_starts_a_new_group(self):
        assert group_consecutive_runs([1, 2, 10, 11]) == [[1, 2], [10, 11]]

    def test_runs_never_exceed_the_maximum(self):
        groups = group_consecutive_runs(list(range(1, 11)))
        assert all(len(group) <= 3 for group in groups)


class TestSessionAssignment:
    def test_session_follows_the_acquisition_date_not_the_folder(self):
        # Real case: TL scans dated 13 October 2020 filed in the May 2021 folder.
        from manifest.build_manifest import session_for_date

        assert session_for_date(pd.Timestamp("2020-10-13"), "may_2021") == "october_2020"
        assert session_for_date(pd.Timestamp("2021-05-20"), "may_2021") == "may_2021"

    def test_undated_files_fall_back_to_their_folder(self):
        from manifest.build_manifest import session_for_date

        assert session_for_date(pd.NaT, "october_2020") == "october_2020"


class TestPhaseOverlaps:
    def _row(self, scan, column, area, phase="Baseline", date="2021-05-19"):
        return {
            "cohort": "may_2021",
            "participant_id": "HP-26-NB",
            "initials": "NB",
            "group": "HP",
            "date": pd.Timestamp(date),
            "scan_number": scan,
            "phase": phase,
            "protocol_area": area,
            "condition": "Regular",
            "excel_column": column,
        }

    def test_same_site_overlap_needs_fixing(self):
        long_df = pd.DataFrame(
            [self._row(n, "Foot Baseline", "Foot") for n in range(2200, 2205)]
            + [self._row(n, "Foot PORH", "Foot", "PORH") for n in range(2200, 2212)]
        )

        overlaps = find_phase_overlaps(long_df)

        assert len(overlaps) == 1
        assert bool(overlaps.iloc[0]["needs_fix"])
        assert overlaps.iloc[0]["raw_value"] == "2200-2204"

    def test_cross_site_overlap_is_informational(self):
        # A file's body site tells Hand and Foot listings apart.
        long_df = pd.DataFrame(
            [self._row(1545, "Foot Baseline", "Foot"), self._row(1545, "Hand PORH.1", "Hand", "PORH")]
        )

        overlaps = find_phase_overlaps(long_df)

        assert len(overlaps) == 1
        assert not bool(overlaps.iloc[0]["needs_fix"])

    def test_separate_visits_do_not_count_as_overlapping(self):
        long_df = pd.DataFrame(
            [
                self._row(2300, "Foot Baseline", "Foot", date="2021-05-20"),
                self._row(2300, "Foot PORH", "Foot", "PORH", date="2021-05-27"),
            ]
        )

        assert find_phase_overlaps(long_df).empty

    def test_no_overlaps_yields_an_empty_report(self):
        long_df = pd.DataFrame([self._row(n, "Foot Baseline", "Foot") for n in range(5)])

        assert find_phase_overlaps(long_df).empty
