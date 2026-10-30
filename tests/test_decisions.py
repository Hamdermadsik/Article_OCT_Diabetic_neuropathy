"""Recorded identity decisions and the saved drive index."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from manifest.decisions import (
    apply_aliases,
    apply_file_initials,
    apply_identity_decisions,
    decision_label,
    drop_superseded_issues,
    load_decisions,
)


def _participants(*rows):
    """(cohort, participant_id, sex, age, demographics_source)."""
    return pd.DataFrame(
        [
            {
                "cohort": cohort,
                "participant_id": pid,
                "initials": pid.split("-")[-1],
                "group": pid.split("-")[0] if "-" in pid else None,
                "sex": sex,
                "age": age,
                "demographics_source": source,
            }
            for cohort, pid, sex, age, source in rows
        ]
    )


def _sheet(*rows):
    return pd.DataFrame(
        [
            {"participant_id": pid, "initials": pid.split("-")[-1], "sex": sex, "age": age}
            for pid, sex, age in rows
        ]
    )


def _same(may, october, same=True):
    return {"may": may, "october": october, "same_person": same}


class TestIdentityDecisions:
    def test_same_person_shares_one_patient_id(self):
        participants = _participants(
            ("may_2021", "HP-20-CH", "M", 46, "demographics_sheet"),
            ("october_2020", "CH", "M", 46, "october_scan_log"),
        )

        result, _ = apply_identity_decisions(participants, [_same("HP-20-CH", "CH")])

        assert result["patient_id"].nunique() == 1
        assert set(result["identity_confidence"]) == {"confirmed_same_person"}

    def test_different_people_stay_separate_but_are_marked_decided(self):
        participants = _participants(
            ("may_2021", "HP-49-MB", "M", 44, "demographics_sheet"),
            ("october_2020", "MB", "F", 28, "october_scan_log"),
        )

        result, _ = apply_identity_decisions(participants, [_same("HP-49-MB", "MB", False)])

        assert result["patient_id"].nunique() == 2
        assert set(result["identity_confidence"]) == {"confirmed_different_people"}

    def test_undecided_pairs_stay_separate_and_unresolved(self):
        participants = _participants(
            ("may_2021", "HP-53-RM", None, None, "conflicting_rows"),
            ("october_2020", "RM", "M", 26, "october_scan_log"),
        )

        result, _ = apply_identity_decisions(participants, [])

        assert result["patient_id"].nunique() == 2
        assert set(result["identity_confidence"]) == {"unresolved_kept_separate"}

    def test_a_conflict_is_settled_by_the_confirmed_october_record(self):
        # Real case: HP-35-LM listed as F/54 and F/45; October LM is F/54.
        participants = _participants(
            ("may_2021", "HP-35-LM", None, None, "conflicting_rows"),
            ("october_2020", "LM", "F", 54, "october_scan_log"),
        )
        sheet = _sheet(("HP-35-LM", "F", 54), ("HP-35-LM", "F", 45))

        result, issues = apply_identity_decisions(
            participants, [_same("HP-35-LM", "LM")], sheet
        )

        may = result[result["cohort"] == "may_2021"].iloc[0]
        assert (may["sex"], may["age"]) == ("F", 54)
        assert any(i["issue"] == "resolved_conflict" for i in issues)

    def test_missing_demographics_are_taken_from_the_same_person(self):
        # Real case: DH has no age in the October log; May records M/66.
        participants = _participants(
            ("may_2021", "DP-13-DH", "M", 66, "demographics_sheet"),
            ("october_2020", "DH", None, None, "none"),
        )

        result, issues = apply_identity_decisions(participants, [_same("DP-13-DH", "DH")])

        october = result[result["cohort"] == "october_2020"].iloc[0]
        assert (october["sex"], october["age"]) == ("M", 66)
        assert any(i["issue"] == "resolved_missing" for i in issues)

    def test_same_person_with_contradictory_records_is_reported(self):
        participants = _participants(
            ("may_2021", "HP-01-AB", "M", 40, "demographics_sheet"),
            ("october_2020", "AB", "F", 40, "october_scan_log"),
        )

        _, issues = apply_identity_decisions(participants, [_same("HP-01-AB", "AB")])

        assert any(i["issue"] == "same_person_disagrees" for i in issues)

    def test_a_decision_naming_an_unknown_participant_is_reported(self):
        participants = _participants(("may_2021", "HP-20-CH", "M", 46, "demographics_sheet"))

        _, issues = apply_identity_decisions(participants, [_same("HP-20-CH", "XX")])

        assert [i["issue"] for i in issues] == ["decision_unmatched"]

    def test_resolved_issues_replace_the_original_complaint(self):
        original = pd.DataFrame(
            [{"cohort": "may_2021", "participant_id": "HP-35-LM", "initials": "LM",
              "issue": "conflicting_demographics", "detail": "54 vs 45"}]
        )
        resolution = [{"cohort": "may_2021", "participant_id": "HP-35-LM", "initials": "LM",
                       "issue": "resolved_conflict", "detail": "kept 54"}]

        merged = drop_superseded_issues(original, resolution)

        assert list(merged["issue"]) == ["resolved_conflict"]

    def test_review_rows_show_the_recorded_decision(self):
        review = pd.DataFrame(
            [{"may_participant_id": "HP-20-CH", "october_participant_id": "CH"},
             {"may_participant_id": "HP-53-RM", "october_participant_id": "RM"}]
        )

        labels = decision_label(review, [_same("HP-20-CH", "CH")])

        assert list(labels) == ["same person", "undecided"]


class TestAliases:
    def _log(self, *rows):
        return pd.DataFrame(
            [{"cohort": c, "participant_id": p, "initials": p.split("-")[-1],
              "group": p.split("-")[0]} for c, p in rows]
        )

    def test_alias_renames_the_participant_and_their_files(self):
        # Real case: the log says DP-12-RS; agreed ID is DP-12-SR. The files on
        # the drive are labelled RS and must follow the rename.
        log = self._log(("may_2021", "DP-12-RS"), ("may_2021", "HP-01-AB"))

        renamed, file_initials, issues = apply_aliases(log, {"DP-12-RS": "DP-12-SR"})
        files = apply_file_initials(
            pd.DataFrame([{"cohort": "may_2021", "initials": "RS"}]), file_initials
        )

        assert "DP-12-SR" in set(renamed["participant_id"])
        assert files.iloc[0]["initials"] == "SR"
        assert files.iloc[0]["initials_raw"] == "RS"
        assert issues == []

    def test_files_are_not_renamed_when_the_initials_belong_to_someone_else_too(self):
        log = self._log(("may_2021", "DP-12-RS"), ("may_2021", "HP-30-RS"))

        _, file_initials, issues = apply_aliases(log, {"DP-12-RS": "DP-12-SR"})

        assert file_initials == {}
        assert [i["issue"] for i in issues] == ["alias_initials_shared"]

    def test_an_alias_for_an_unknown_id_is_reported(self):
        _, _, issues = apply_aliases(self._log(("may_2021", "HP-01-AB")), {"XX-99-ZZ": "X"})

        assert [i["issue"] for i in issues] == ["decision_unmatched"]


class TestDecisionsFile:
    def test_missing_file_means_no_decisions(self, tmp_path):
        decisions = load_decisions(tmp_path / "absent.yaml")

        assert decisions.aliases == {} and decisions.identity == []

    def test_ids_are_normalized_like_the_spreadsheets(self, tmp_path):
        path = tmp_path / "decisions.yaml"
        path.write_text(
            "participant_id_aliases:\n  dp-12rs: DP-12-SR\n"
            "identity:\n  - {may: hp-20ch, october: ch, same_person: true}\n"
        )

        decisions = load_decisions(path)

        assert decisions.aliases == {"DP-12-RS": "DP-12-SR"}
        assert decisions.identity == [_same("HP-20-CH", "CH")]

    def test_the_projects_decisions_file_parses(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "manifest_decisions.yaml"

        decisions = load_decisions(path)

        assert decisions.identity, "expected recorded identity decisions"


class TestDriveCache:
    def test_saved_headers_are_used_when_the_drive_is_not_attached(self, tmp_path):
        from manifest.build_manifest import index_drive

        cache = tmp_path / "drive_index.csv"
        pd.DataFrame(
            [{"filepath": "/mnt/d/x/AB_Right Hand_L1_S100__10_05_2021.dcm",
              "filename": "AB_Right Hand_L1_S100__10_05_2021.dcm",
              "header_status": "ok", "folder_cohort": "may_2021"}]
        ).to_csv(cache, index=False)

        files, unparsed = index_drive({"may_2021": tmp_path / "unplugged"}, cache)

        assert len(files) == 1 and unparsed.empty
        assert files.iloc[0]["scan_number"] == 100
        assert files.iloc[0]["cohort"] == "may_2021"

    def test_filenames_are_reparsed_from_the_cache_not_trusted(self, tmp_path):
        # Parsing rules can improve after the drive was read; a cached file
        # must benefit from that without re-reading the drive.
        from manifest.build_manifest import index_drive

        cache = tmp_path / "drive_index.csv"
        pd.DataFrame(
            [{"filepath": "p", "filename": "BB_Right Middle Finger_L1047_S2245__19_05_2021.dcm",
              "header_status": "ok", "folder_cohort": "may_2021"}]
        ).to_csv(cache, index=False)

        files, _ = index_drive({"may_2021": tmp_path / "unplugged"}, cache)

        assert files.iloc[0]["protocol_area"] == "Hand"
