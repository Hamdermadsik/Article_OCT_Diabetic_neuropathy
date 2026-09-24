"""Attach age/sex to the participants who were actually scanned.

Two sources exist:

* the "Demographic" sheet in ``2021-05-OCTAAgeing-hand.xlsx``. Despite the
  filename it lists participants from *both* sessions, so a row in it says
  nothing about which session the person belongs to;
* October's scan log (Sheet1), which records sex and age inline.

Session membership therefore comes from the scan logs, never from the
demographics sheet. May participants take their demographics from the sheet
(matched on full participant ID); October participants take theirs from
Sheet1, and the sheet is only used to cross-check them.

Where the sources are ambiguous or disagree, nothing is picked silently: the
value is left blank and the problem is written to the issues report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cohort_specs import (
    MAY_COHORT,
    OCTOBER_AGE_COLUMN,
    OCTOBER_COHORT,
    OCTOBER_PARTICIPANT_COLUMN,
    OCTOBER_SEX_COLUMN,
)
from .excel_utils import extract_group, extract_initials, normalize_participant_id

_VALID_SEX = {"M", "F"}

# Ages recorded seven months apart can legitimately differ by a year.
AGE_TOLERANCE_YEARS = 1

RESOLVED_COLUMNS = [
    "cohort",
    "participant_id",
    "initials",
    "group",
    "sex",
    "age",
    "demographics_source",
]
ISSUE_COLUMNS = ["cohort", "participant_id", "initials", "issue", "detail"]


def _coerce_age(value) -> float | None:
    try:
        age = float(value)
    except (TypeError, ValueError):
        return None
    if not 0 < age < 120:
        return None
    return age


def _coerce_sex(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    sex = str(value).strip().upper()
    return sex if sex in _VALID_SEX else None


def _present(value) -> bool:
    return value is not None and not pd.isna(value)


def _describe(sex, age) -> str:
    sex_text = sex if _present(sex) else "?"
    age_text = f"{age:g}" if _present(age) else "?"
    return f"{sex_text}/{age_text}"


def load_demographics_sheet(path: Path, sheet_name: str = "Demographic") -> pd.DataFrame:
    """Read the combined demographics sheet, without assigning a session.

    The sheet's header row is shifted one column left of its data: the column
    labelled "Initials" holds sex and the one labelled "Sex" holds age, so
    columns are read by position.
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    records = []
    for sheet_row, row in raw.iterrows():
        raw_id = row.get(0)
        participant_id = normalize_participant_id(raw_id)
        if not participant_id or participant_id in {"PARTICIPANT ID", "INITIALS"}:
            continue

        sex = _coerce_sex(row.get(1))
        age = _coerce_age(row.get(2))
        if sex is None and age is None:
            continue

        records.append(
            {
                "participant_id": participant_id,
                "initials": extract_initials(participant_id),
                "group": extract_group(participant_id),
                # Rows like "AR" carry no group or number to match on.
                "bare_initials": participant_id == extract_initials(participant_id),
                "sex": sex,
                "age": age,
                "sheet_row": int(sheet_row) + 1,
            }
        )

    return pd.DataFrame(
        records,
        columns=["participant_id", "initials", "group", "bare_initials", "sex", "age", "sheet_row"],
    )


def load_october_demographics(sheet1: pd.DataFrame) -> pd.DataFrame:
    """Read the sex/age October's scan log records at each participant block."""
    records = []
    for _, row in sheet1.iterrows():
        participant_id = normalize_participant_id(row.get(OCTOBER_PARTICIPANT_COLUMN))
        if not participant_id:
            continue
        records.append(
            {
                "participant_id": participant_id,
                "sex": _coerce_sex(row.get(OCTOBER_SEX_COLUMN)),
                "age": _coerce_age(row.get(OCTOBER_AGE_COLUMN)),
            }
        )
    return pd.DataFrame(records, columns=["participant_id", "sex", "age"]).drop_duplicates(
        subset=["participant_id"]
    )


def scanned_participants(long_df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Participants that appear in a session's scan log."""
    subset = long_df[long_df["cohort"] == cohort]
    return (
        subset[["participant_id", "initials", "group"]]
        .drop_duplicates(subset=["participant_id"])
        .sort_values("participant_id")
        .reset_index(drop=True)
    )


def _values_disagree(rows: pd.DataFrame) -> bool:
    ages = rows["age"].dropna()
    return rows["sex"].dropna().nunique() > 1 or (
        not ages.empty and ages.max() - ages.min() > AGE_TOLERANCE_YEARS
    )


def resolve_may_demographics(
    may_participants: pd.DataFrame,
    sheet: pd.DataFrame,
    october_demographics: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict], set[int]]:
    """Demographics for May participants, the issues found, and sheet rows used."""
    resolved, issues, claimed = [], [], set()
    initials_in_may = may_participants["initials"].value_counts()
    october_by_initials = {
        extract_initials(row.participant_id): row for row in october_demographics.itertuples()
    }

    for participant in may_participants.itertuples():
        record = {
            "cohort": MAY_COHORT,
            "participant_id": participant.participant_id,
            "initials": participant.initials,
            "group": participant.group,
            "sex": None,
            "age": None,
            "demographics_source": "none",
        }

        rows = sheet[sheet["participant_id"] == participant.participant_id]
        if not rows.empty:
            claimed.update(rows["sheet_row"])
            if _values_disagree(rows):
                values = ", ".join(_describe(r.sex, r.age) for r in rows.itertuples())
                detail = f"sheet lists this ID {len(rows)} times with different values: {values}"
                october = october_by_initials.get(participant.initials)
                if october is not None:
                    matching = [
                        r for r in rows.itertuples()
                        if r.sex == october.sex and r.age == october.age
                    ]
                    if matching:
                        detail += (
                            f". The {_describe(october.sex, october.age)} row matches October "
                            f"participant {october.participant_id}, so it may be that person's "
                            "row filed under this ID"
                        )
                record["demographics_source"] = "conflicting_rows"
                issues.append(
                    {
                        "cohort": MAY_COHORT,
                        "participant_id": participant.participant_id,
                        "initials": participant.initials,
                        "issue": "conflicting_demographics",
                        "detail": detail,
                    }
                )
            else:
                first = rows.iloc[0]
                record.update(
                    sex=first["sex"], age=first["age"], demographics_source="demographics_sheet"
                )
            resolved.append(record)
            continue

        # Fall back to a bare-initials row, but only when nothing else could
        # claim it: exactly one such row, and no other May participant shares
        # the initials.
        bare = sheet[sheet["bare_initials"] & (sheet["initials"] == participant.initials)]
        if len(bare) == 1 and initials_in_may.get(participant.initials, 0) == 1:
            only = bare.iloc[0]
            claimed.add(int(only["sheet_row"]))
            record.update(
                sex=only["sex"],
                age=only["age"],
                demographics_source="demographics_sheet_by_initials",
            )
            issues.append(
                {
                    "cohort": MAY_COHORT,
                    "participant_id": participant.participant_id,
                    "initials": participant.initials,
                    "issue": "matched_by_initials_only",
                    "detail": (
                        f"no row for {participant.participant_id}; used the row listed only as "
                        f"'{only['participant_id']}' ({_describe(only['sex'], only['age'])})"
                    ),
                }
            )
            resolved.append(record)
            continue

        hints = []
        number = participant.participant_id.split("-")[1] if participant.participant_id.count("-") == 2 else None
        if number:
            same_number = sheet[
                sheet["participant_id"].str.startswith(f"{participant.group}-{number}-", na=False)
            ]
            hints += [
                f"{r.participant_id} ({_describe(r.sex, r.age)}) has the same number"
                for r in same_number.itertuples()
            ]
        same_initials = sheet[sheet["initials"] == participant.initials]
        hints += [
            f"{r.participant_id} ({_describe(r.sex, r.age)}) has the same initials"
            for r in same_initials.itertuples()
        ]
        issues.append(
            {
                "cohort": MAY_COHORT,
                "participant_id": participant.participant_id,
                "initials": participant.initials,
                "issue": "no_demographics",
                "detail": "no matching row in the demographics sheet"
                + (f"; possible typo: {'; '.join(hints)}" if hints else ""),
            }
        )
        resolved.append(record)

    return pd.DataFrame(resolved, columns=RESOLVED_COLUMNS), issues, claimed


def resolve_october_demographics(
    october_participants: pd.DataFrame,
    october_demographics: pd.DataFrame,
    sheet: pd.DataFrame,
    claimed_sheet_rows: set[int],
) -> tuple[pd.DataFrame, list[dict]]:
    """Demographics for October participants, cross-checked against the sheet."""
    resolved, issues = [], []
    inline = october_demographics.set_index("participant_id")
    unclaimed = sheet[~sheet["sheet_row"].isin(claimed_sheet_rows)]

    for participant in october_participants.itertuples():
        record = {
            "cohort": OCTOBER_COHORT,
            "participant_id": participant.participant_id,
            "initials": participant.initials,
            "group": participant.group,
            "sex": None,
            "age": None,
            "demographics_source": "none",
        }

        if participant.participant_id in inline.index:
            own = inline.loc[participant.participant_id]
            record.update(sex=own["sex"], age=own["age"], demographics_source="october_scan_log")
        else:
            issues.append(
                {
                    "cohort": OCTOBER_COHORT,
                    "participant_id": participant.participant_id,
                    "initials": participant.initials,
                    "issue": "no_demographics",
                    "detail": "not listed with sex/age in October's scan log (Sheet1)",
                }
            )

        candidates = unclaimed[unclaimed["initials"] == participant.initials]
        if len(candidates) == 1 and record["demographics_source"] == "october_scan_log":
            other = candidates.iloc[0]
            ages_known = _present(other["age"]) and _present(record["age"])
            ages_agree = ages_known and abs(other["age"] - record["age"]) <= AGE_TOLERANCE_YEARS

            if not _present(record["sex"]) and _present(other["sex"]) and ages_agree:
                # The October log left sex blank. The only unclaimed sheet row
                # with these initials has the same age, so take sex from it --
                # visibly, since this is an initials-based link.
                record["sex"] = other["sex"]
                record["demographics_source"] = "october_scan_log+sheet_sex"
                issues.append(
                    {
                        "cohort": OCTOBER_COHORT,
                        "participant_id": participant.participant_id,
                        "initials": participant.initials,
                        "issue": "sex_filled_from_sheet",
                        "detail": (
                            f"October log has no sex; took {other['sex']} from demographics "
                            f"sheet row {other['participant_id']} (same initials and age)"
                        ),
                    }
                )
            else:
                sex_differs = (
                    _present(other["sex"])
                    and _present(record["sex"])
                    and other["sex"] != record["sex"]
                )
                if sex_differs or (ages_known and not ages_agree):
                    issues.append(
                        {
                            "cohort": OCTOBER_COHORT,
                            "participant_id": participant.participant_id,
                            "initials": participant.initials,
                            "issue": "sources_disagree",
                            "detail": (
                                f"October scan log says {_describe(record['sex'], record['age'])}; "
                                f"demographics sheet row {other['participant_id']} says "
                                f"{_describe(other['sex'], other['age'])}"
                            ),
                        }
                    )
        resolved.append(record)

    return pd.DataFrame(resolved, columns=RESOLVED_COLUMNS), issues


def resolve_demographics(
    long_df: pd.DataFrame,
    sheet: pd.DataFrame,
    october_demographics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Demographics for every scanned participant, plus an issues table."""
    may_resolved, may_issues, claimed = resolve_may_demographics(
        scanned_participants(long_df, MAY_COHORT), sheet, october_demographics
    )
    october_resolved, october_issues = resolve_october_demographics(
        scanned_participants(long_df, OCTOBER_COHORT), october_demographics, sheet, claimed
    )
    resolved = pd.concat([may_resolved, october_resolved], ignore_index=True)
    issues = pd.DataFrame(may_issues + october_issues, columns=ISSUE_COLUMNS)
    return resolved, issues


def find_ambiguous_initials(participants: pd.DataFrame) -> pd.DataFrame:
    """Initials shared by more than one scanned participant within a session."""
    if participants.empty:
        return participants.iloc[0:0].copy()
    counts = participants.groupby(["cohort", "initials"])["participant_id"].nunique()
    ambiguous = counts[counts > 1].reset_index()[["cohort", "initials"]]
    if ambiguous.empty:
        return participants.iloc[0:0].copy()
    return participants.merge(ambiguous, on=["cohort", "initials"]).sort_values(
        ["cohort", "initials", "participant_id"]
    )


def _assess_pair(may, october) -> str:
    if may["demographics_source"] == "conflicting_rows":
        return "cannot compare: May demographics conflict (see demographics_issues.csv)"
    if not all(_present(v) for v in (may["age"], october["age"], may["sex"], october["sex"])):
        return "cannot compare: demographics missing"
    if may["sex"] != october["sex"]:
        return "different sex: different people"
    if abs(may["age"] - october["age"]) > AGE_TOLERANCE_YEARS:
        return f"age differs ({may['age']:g} vs {october['age']:g}): likely different people"
    return "same sex and age: likely the same person"


def build_identity_review(participants: pd.DataFrame) -> pd.DataFrame:
    """One row per May/October pair sharing initials, with an assessment."""
    columns = [
        "initials",
        "may_participant_id",
        "may_sex",
        "may_age",
        "october_participant_id",
        "october_sex",
        "october_age",
        "assessment",
    ]
    may = participants[participants["cohort"] == MAY_COHORT]
    october = participants[participants["cohort"] == OCTOBER_COHORT]

    rows = []
    for initials in sorted(set(may["initials"]) & set(october["initials"])):
        for _, m in may[may["initials"] == initials].iterrows():
            for _, o in october[october["initials"] == initials].iterrows():
                rows.append(
                    {
                        "initials": initials,
                        "may_participant_id": m["participant_id"],
                        "may_sex": m["sex"],
                        "may_age": m["age"],
                        "october_participant_id": o["participant_id"],
                        "october_sex": o["sex"],
                        "october_age": o["age"],
                        "assessment": _assess_pair(m, o),
                    }
                )
    return pd.DataFrame(rows, columns=columns)
