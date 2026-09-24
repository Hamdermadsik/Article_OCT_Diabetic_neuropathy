"""Resolve DICOM files against the Excel scan logs.

Matching is tried from most to least specific. Every scan records *how* it was
matched, so a row resolved by the loosest fallback is never mistaken for a
confident one -- the previous pipeline collapsed both into the same output with
no way to tell them apart.

Ambiguity is reported, not guessed: if a key matches several participants, the
scan is left unresolved rather than silently assigned to whichever row sorted
first.
"""

from __future__ import annotations

import pandas as pd

# Ordered most specific -> least specific.
MATCH_STRATEGIES = [
    ("exact", ("initials", "date", "scan_number", "protocol_area")),
    ("initials_scan_area", ("initials", "scan_number", "protocol_area")),
    ("initials_scan", ("initials", "scan_number")),
    ("scan_area_only", ("scan_number", "protocol_area")),
]

# Strategies that do not constrain on participant identity. A hit here is a
# guess about *who* the scan belongs to, not just which phase it was.
IDENTITY_UNCONSTRAINED = {"scan_area_only"}

RESOLVED_FIELDS = ["participant_id", "group", "phase", "condition", "excel_column"]


def _key_of(row, fields) -> tuple:
    return tuple(row.get(field) for field in fields)


def _build_indexes(long_df: pd.DataFrame) -> dict[str, dict[tuple, list[dict]]]:
    indexes: dict[str, dict[tuple, list[dict]]] = {name: {} for name, _ in MATCH_STRATEGIES}
    records = long_df.to_dict("records")
    for record in records:
        for name, fields in MATCH_STRATEGIES:
            key = _key_of(record, fields)
            if any(pd.isna(part) for part in key):
                continue
            indexes[name].setdefault(key, []).append(record)
    return indexes


def match_scans(
    files_df: pd.DataFrame,
    long_df: pd.DataFrame,
    ambiguous_initials: set[str] | None = None,
) -> pd.DataFrame:
    """Attach Excel-log identity to each parsed DICOM file.

    Adds ``participant_id``, ``phase``, ``condition``, ``match_method`` and
    ``match_detail``. ``match_method`` is one of the strategy names, or
    ``unmatched`` / ``ambiguous``.
    """
    ambiguous_initials = ambiguous_initials or set()
    if files_df.empty:
        return files_df.assign(match_method="unmatched", match_detail="no files")

    indexes = _build_indexes(long_df) if not long_df.empty else {n: {} for n, _ in MATCH_STRATEGIES}

    resolved_rows = []
    for file_row in files_df.to_dict("records"):
        resolution = {field: None for field in RESOLVED_FIELDS}
        resolution["match_method"] = "unmatched"
        resolution["match_detail"] = ""

        for name, fields in MATCH_STRATEGIES:
            key = _key_of(file_row, fields)
            if any(pd.isna(part) or part is None for part in key):
                continue

            candidates = indexes[name].get(key, [])
            if not candidates:
                continue

            participants = {candidate["participant_id"] for candidate in candidates}
            if len(participants) > 1:
                resolution["match_method"] = "ambiguous"
                resolution["match_detail"] = (
                    f"{name} matched {len(participants)} participants: "
                    + ",".join(sorted(str(p) for p in participants))
                )
                break

            chosen = candidates[0]
            for field in RESOLVED_FIELDS:
                resolution[field] = chosen.get(field)
            resolution["match_method"] = name

            details = []
            phases = {candidate["phase"] for candidate in candidates}
            if len(phases) > 1:
                details.append("scan listed under multiple phases: " + ",".join(sorted(map(str, phases))))
            if name in IDENTITY_UNCONSTRAINED:
                details.append("identity inferred from scan number alone")
            if file_row.get("initials") in ambiguous_initials:
                details.append(f"initials {file_row.get('initials')} are shared by multiple participants")
            resolution["match_detail"] = "; ".join(details)
            break

        resolved_rows.append({**file_row, **resolution})

    return pd.DataFrame(resolved_rows)


def mark_duplicates(scans: pd.DataFrame) -> pd.DataFrame:
    """Flag repeat acquisitions of the same scan via ``is_duplicate_of``.

    Windows "name (1).dcm" copies are the common case. The canonical row is the
    one without a copy suffix, tie-broken by larger file size so a truncated
    partial copy never wins.
    """
    scans = scans.copy()
    scans["is_duplicate_of"] = None
    if scans.empty:
        return scans

    key_fields = ["cohort", "initials", "date", "scan_number", "protocol_area"]
    for _, group in scans.groupby(key_fields, dropna=False):
        if len(group) < 2:
            continue
        ordered = group.sort_values(
            ["copy_index", "file_size"], ascending=[True, False]
        )
        canonical_id = ordered.iloc[0]["scan_id"]
        for scan_id in ordered.iloc[1:]["scan_id"]:
            scans.loc[scans["scan_id"] == scan_id, "is_duplicate_of"] = canonical_id
    return scans


def find_scan_number_collisions(long_df: pd.DataFrame) -> pd.DataFrame:
    """Scan numbers claimed by more than one participant within a cohort.

    The loosest matching strategy assumes scan numbers identify a participant
    on their own. This reports where that assumption does not hold.
    """
    if long_df.empty:
        return long_df.iloc[0:0].copy()

    unique = long_df.drop_duplicates(subset=["cohort", "scan_number", "participant_id"])
    counts = unique.groupby(["cohort", "scan_number"])["participant_id"].nunique()
    colliding = counts[counts > 1].reset_index()[["cohort", "scan_number"]]
    if colliding.empty:
        return unique.iloc[0:0].copy()
    return unique.merge(colliding, on=["cohort", "scan_number"], how="inner").sort_values(
        ["cohort", "scan_number", "participant_id"]
    )


def find_missing_expected_scans(long_df: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """Scans the Excel log expects but for which no file was found."""
    if long_df.empty:
        return long_df.iloc[0:0].copy()

    found = matched[matched["participant_id"].notna()][
        ["cohort", "participant_id", "scan_number"]
    ].drop_duplicates()
    merged = long_df.merge(
        found, on=["cohort", "participant_id", "scan_number"], how="left", indicator=True
    )
    return (
        merged[merged["_merge"] == "left_only"]
        .drop(columns=["_merge"])
        .sort_values(["cohort", "participant_id", "scan_number"])
        .reset_index(drop=True)
    )
