"""Apply recorded human decisions about participant identity.

Some identity questions cannot be answered from the spreadsheets: whether the
"CH" scanned in October is the "HP-20-CH" scanned in May, or which of two IDs a
person should go by. Those answers live in ``configs/manifest_decisions.yaml``
and are applied here on every build.

Every decision must refer to a participant that actually exists; a decision
that matches nothing is reported rather than silently ignored, since it almost
always means a typo in the decisions file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from .cohort_specs import MAY_COHORT, OCTOBER_COHORT
from .demographics import AGE_TOLERANCE_YEARS, ISSUE_COLUMNS
from .excel_utils import extract_group, extract_initials, normalize_participant_id

COHORT_PREFIX = {MAY_COHORT: "MAY", OCTOBER_COHORT: "OCT"}


@dataclass
class Decisions:
    aliases: dict[str, str] = field(default_factory=dict)
    identity: list[dict] = field(default_factory=list)


def load_decisions(path: Path) -> Decisions:
    """Read the decisions file; a missing file means no decisions yet."""
    if not path.exists():
        return Decisions()

    raw = yaml.safe_load(path.read_text()) or {}
    aliases = {
        normalize_participant_id(old): normalize_participant_id(new)
        for old, new in (raw.get("participant_id_aliases") or {}).items()
    }
    identity = [
        {
            "may": normalize_participant_id(entry["may"]),
            "october": normalize_participant_id(entry["october"]),
            "same_person": bool(entry["same_person"]),
        }
        for entry in raw.get("identity") or []
    ]
    return Decisions(aliases=aliases, identity=identity)


def canonical_patient_id(cohort: str, participant_id: str) -> str:
    """Session-scoped patient key, used until a person is confirmed in both."""
    return f"{COHORT_PREFIX.get(cohort, cohort.upper())}-{participant_id}"


def _issue(cohort, participant_id, issue, detail) -> dict:
    return {
        "cohort": cohort,
        "participant_id": participant_id,
        "initials": extract_initials(participant_id),
        "issue": issue,
        "detail": detail,
    }


def apply_aliases(
    long_df: pd.DataFrame, aliases: dict[str, str]
) -> tuple[pd.DataFrame, dict[tuple[str, str], str], list[dict]]:
    """Rename participants in the scan logs.

    Returns the updated log, a map of (cohort, old_initials) -> new_initials for
    renaming the matching files, and issues for aliases that matched nothing.
    The drive's filenames carry only initials, so a rename that changes the
    initials is applied to files too -- but only where those initials belong to
    that one participant, so no one else's files are swept up.
    """
    long_df = long_df.copy()
    file_initials: dict[tuple[str, str], str] = {}
    issues = []

    for old_id, new_id in aliases.items():
        rows = long_df["participant_id"] == old_id
        if not rows.any():
            issues.append(
                _issue(None, old_id, "decision_unmatched",
                       f"alias {old_id} -> {new_id} matches no participant in the scan logs")
            )
            continue

        for cohort in long_df.loc[rows, "cohort"].unique():
            old_initials = extract_initials(old_id)
            new_initials = extract_initials(new_id)
            if old_initials == new_initials:
                continue
            in_cohort = long_df["cohort"] == cohort
            owners = long_df.loc[in_cohort & (long_df["initials"] == old_initials), "participant_id"]
            if set(owners) == {old_id}:
                file_initials[(cohort, old_initials)] = new_initials
            else:
                issues.append(
                    _issue(cohort, old_id, "alias_initials_shared",
                           f"files labelled {old_initials} also belong to "
                           f"{sorted(set(owners) - {old_id})}; files were not renamed")
                )

        long_df.loc[rows, "participant_id"] = new_id
        long_df.loc[rows, "initials"] = extract_initials(new_id)
        long_df.loc[rows, "group"] = extract_group(new_id)

    return long_df, file_initials, issues


def apply_file_initials(files_df: pd.DataFrame, file_initials: dict) -> pd.DataFrame:
    """Rename initials on indexed files, keeping the original in ``initials_raw``."""
    if files_df.empty:
        return files_df
    files_df = files_df.copy()
    files_df["initials_raw"] = files_df["initials"]
    for (cohort, old_initials), new_initials in file_initials.items():
        rows = (files_df["cohort"] == cohort) & (files_df["initials"] == old_initials)
        files_df.loc[rows, "initials"] = new_initials
    return files_df


def _present(value) -> bool:
    return value is not None and not pd.isna(value)


def _merge_demographics(participants, may_idx, october_idx, sheet, issues) -> None:
    """Reconcile demographics for two records confirmed to be one person."""
    may = participants.loc[may_idx]
    october = participants.loc[october_idx]

    # A May record with conflicting sheet rows can be settled by the October
    # record of the same person: keep the sheet row that agrees with it.
    if may["demographics_source"] == "conflicting_rows" and sheet is not None:
        candidates = sheet[sheet["participant_id"] == may["participant_id"]]
        agreeing = candidates[
            candidates["age"].sub(october["age"]).abs().le(AGE_TOLERANCE_YEARS)
            & (candidates["sex"] == october["sex"])
        ] if _present(october["age"]) and _present(october["sex"]) else candidates.iloc[0:0]
        if len(agreeing) == 1:
            chosen = agreeing.iloc[0]
            participants.loc[may_idx, ["sex", "age"]] = [chosen["sex"], chosen["age"]]
            participants.loc[may_idx, "demographics_source"] = "sheet_row_confirmed_by_october"
            rejected = ", ".join(
                f"{r.sex}/{r.age:g}" for r in candidates.itertuples() if r.Index != chosen.name
            )
            issues.append(
                _issue(MAY_COHORT, may["participant_id"], "resolved_conflict",
                       f"same person as October {october['participant_id']}; kept "
                       f"{chosen['sex']}/{chosen['age']:g}, discarded {rejected}")
            )
            may = participants.loc[may_idx]

    for source_idx, target_idx in ((may_idx, october_idx), (october_idx, may_idx)):
        source, target = participants.loc[source_idx], participants.loc[target_idx]
        filled = []
        for column in ("sex", "age"):
            if not _present(target[column]) and _present(source[column]):
                participants.loc[target_idx, column] = source[column]
                filled.append(column)
        if filled:
            participants.loc[target_idx, "demographics_source"] = (
                f"from_same_person_{source['cohort']}"
            )
            issues.append(
                _issue(target["cohort"], target["participant_id"], "resolved_missing",
                       f"same person as {source['participant_id']}; took "
                       f"{'/'.join(filled)} from that record")
            )

    may, october = participants.loc[may_idx], participants.loc[october_idx]
    if all(_present(v) for v in (may["sex"], october["sex"])) and may["sex"] != october["sex"]:
        issues.append(
            _issue(MAY_COHORT, may["participant_id"], "same_person_disagrees",
                   f"recorded as {may['sex']} in May but {october['sex']} in October")
        )
    if all(_present(v) for v in (may["age"], october["age"])) and (
        abs(may["age"] - october["age"]) > AGE_TOLERANCE_YEARS
    ):
        issues.append(
            _issue(MAY_COHORT, may["participant_id"], "same_person_disagrees",
                   f"age {may['age']:g} in May but {october['age']:g} in October")
        )


def apply_identity_decisions(
    participants: pd.DataFrame,
    identity: list[dict],
    sheet: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Assign ``patient_id`` and ``identity_confidence`` to every participant.

    Confirmed same-person pairs share one patient ID (the May participant ID),
    which is what keeps their scans on one side of a train/validation split.
    """
    participants = participants.copy().reset_index(drop=True)
    issues: list[dict] = []

    participants["patient_id"] = [
        canonical_patient_id(r.cohort, r.participant_id) for r in participants.itertuples()
    ]
    by_cohort = participants.groupby("cohort")["initials"].apply(set).to_dict()
    in_both = by_cohort.get(MAY_COHORT, set()) & by_cohort.get(OCTOBER_COHORT, set())
    participants["identity_confidence"] = [
        "unresolved_kept_separate" if initials in in_both else "single_cohort"
        for initials in participants["initials"]
    ]

    def locate(cohort, participant_id):
        hits = participants.index[
            (participants["cohort"] == cohort) & (participants["participant_id"] == participant_id)
        ]
        return hits[0] if len(hits) else None

    for decision in identity:
        may_idx = locate(MAY_COHORT, decision["may"])
        october_idx = locate(OCTOBER_COHORT, decision["october"])
        missing = [
            f"{label} participant {pid}"
            for label, pid, idx in (
                ("May", decision["may"], may_idx),
                ("October", decision["october"], october_idx),
            )
            if idx is None
        ]
        if missing:
            issues.append(
                _issue(None, decision["may"], "decision_unmatched",
                       f"decision {decision['may']} / {decision['october']} skipped: "
                       f"no {' or '.join(missing)} in the scan logs")
            )
            continue

        if decision["same_person"]:
            shared_id = decision["may"]
            participants.loc[[may_idx, october_idx], "patient_id"] = shared_id
            participants.loc[[may_idx, october_idx], "identity_confidence"] = "confirmed_same_person"
            _merge_demographics(participants, may_idx, october_idx, sheet, issues)
        else:
            participants.loc[[may_idx, october_idx], "identity_confidence"] = (
                "confirmed_different_people"
            )

    return participants, issues


def decision_label(review: pd.DataFrame, identity: list[dict]) -> pd.Series:
    """The recorded decision for each identity-review row."""
    decided = {(d["may"], d["october"]): d["same_person"] for d in identity}
    labels = []
    for row in review.itertuples():
        verdict = decided.get((row.may_participant_id, row.october_participant_id))
        labels.append(
            "undecided" if verdict is None else ("same person" if verdict else "different people")
        )
    return pd.Series(labels, index=review.index, dtype=object)


def drop_superseded_issues(issues: pd.DataFrame, resolutions: list[dict]) -> pd.DataFrame:
    """Remove issues that a decision has since resolved, keeping the resolution."""
    resolved = {
        (r["cohort"], r["participant_id"])
        for r in resolutions
        if r["issue"] in {"resolved_conflict", "resolved_missing"}
    }
    superseded = issues.apply(
        lambda r: (r["cohort"], r["participant_id"]) in resolved
        and r["issue"] in {"conflicting_demographics", "no_demographics"},
        axis=1,
    ) if not issues.empty else pd.Series(dtype=bool)
    kept = issues[~superseded] if not issues.empty else issues
    return pd.concat([kept, pd.DataFrame(resolutions, columns=ISSUE_COLUMNS)], ignore_index=True)
