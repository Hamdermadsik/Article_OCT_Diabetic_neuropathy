"""Shared parsing helpers for Excel scan logs and DICOM filenames.

Canonical home for logic previously duplicated (and drifted) between
``src/exl.py`` and ``notebooks/Excel.ipynb``. Where the two disagreed, the
version verified against the real spreadsheets wins.
"""

from __future__ import annotations

import re

import pandas as pd

# Matches e.g. "AC_Right Hallux_L1051_S2397__20_05_2021.dcm" and the Windows
# copy variant "RS_Right Hand_L1074_S3851__02_06_2021 (1).dcm". The copy suffix
# is captured rather than rejected so duplicate acquisitions stay visible to
# duplicate detection instead of silently landing in the unparsed bucket.
FILENAME_PATTERN = re.compile(
    r"^(?P<initials>[A-Za-z]+)_"
    r"(?P<side>Right|Left)\s+"
    r"(?P<bodypart>Hand|Hallux|Instep|Foot|(?:[A-Za-z]+\s+)?Finger)_"
    r"(?P<lnum>L\d+)_"
    r"[Ss](?P<scan_number>\d+)__"
    r"(?P<date>\d{2}_\d{2}_\d{4})"
    r"(?:\s*\((?P<copy>\d+)\))?"
    r"\.dcm$",
    re.IGNORECASE,
)

_TRAILING_LETTERS = re.compile(r"([A-Za-z]+)\s*$")
_LEADING_GROUP = re.compile(r"^\s*(DP|HP)", re.IGNORECASE)

# Participant IDs are spelled inconsistently across sheets: the scan log writes
# "DP-06-JF" while the demographics sheet writes "DP-02DG", "DP15NL" or even
# "hp-53RM". All are (group, number, initials) and must collapse to one key, or
# the same person is counted as two patients and demographics never join.
_PARTICIPANT_ID = re.compile(
    r"^\s*(?:(?P<group>DP|HP)\s*-?\s*)?(?:(?P<number>\d+)\s*-?\s*)?(?P<initials>[A-Za-z]+)\s*$",
    re.IGNORECASE,
)

FOOT_BODYPARTS = {"HALLUX", "INSTEP", "FOOT"}


def make_unique_columns(columns) -> list[str]:
    """Strip whitespace from column names and disambiguate repeats as ``name.1``.

    The May sheet genuinely relies on this: it has two columns named "Isc" and
    two named "Foot PORH", where the second of each pair is the Flavanol arm.
    """
    counts: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        col = str(col).strip()
        if col not in counts:
            counts[col] = 0
            out.append(col)
        else:
            counts[col] += 1
            out.append(f"{col}.{counts[col]}")
    return out


def extract_initials(participant_id) -> str | None:
    """Trailing letter-run of a participant ID.

    Handles every format present across the sheets: ``DP-06-JF``, ``DP-02DG``,
    ``DP15NL``, ``hp-53RM`` and bare ``AR`` all yield their initials.
    """
    if participant_id is None or pd.isna(participant_id):
        return None
    match = _TRAILING_LETTERS.search(str(participant_id).strip())
    return match.group(1).upper() if match else None


def extract_group(participant_id) -> str | None:
    """Cohort group prefix (``DP`` diabetic / ``HP`` healthy), if present."""
    if participant_id is None or pd.isna(participant_id):
        return None
    match = _LEADING_GROUP.match(str(participant_id))
    return match.group(1).upper() if match else None


def normalize_participant_id(participant_id) -> str | None:
    """Canonical participant key, identical across every sheet's spelling.

    ``DP-06-JF``, ``DP-02DG``, ``DP15NL`` and ``hp-53RM`` all normalize to
    ``GROUP-NN-INITIALS``. IDs with no group/number (October's bare initials,
    and a few May rows) normalize to just their initials. Anything that does not
    fit the pattern is returned uppercased and whitespace-collapsed rather than
    dropped, so unexpected formats stay visible instead of vanishing.
    """
    if participant_id is None or pd.isna(participant_id):
        return None

    collapsed = " ".join(str(participant_id).split()).upper()
    if not collapsed:
        return None

    match = _PARTICIPANT_ID.match(collapsed)
    if not match:
        return collapsed

    initials = match.group("initials").upper()
    group = match.group("group")
    number = match.group("number")
    if group and number is not None:
        return f"{group.upper()}-{int(number):02d}-{initials}"
    if group:
        return f"{group.upper()}-{initials}"
    return initials


def protocol_area_for_bodypart(bodypart) -> str | None:
    """Map an anatomical site to the Excel logs' Hand/Foot protocol grouping."""
    if bodypart is None or pd.isna(bodypart):
        return None
    part = str(bodypart).strip().upper()
    if part == "HAND" or part.endswith("FINGER"):
        return "Hand"
    if part in FOOT_BODYPARTS:
        return "Foot"
    return None


# A single (participant, phase) cell records one acquisition block. Across the
# May log the median cell expands to 5 scans and only six exceed 20 -- each of
# those six is a visibly malformed cell (e.g. "2179-2484", almost certainly
# "2179-2184", or the descending "3099-3014"). Expanding them anyway invents
# hundreds of scan numbers that then collide with other participants' real
# scans, which is what made scan numbers look non-unique across the cohort.
MAX_SCAN_RANGE_SPAN = 24


def expand_scan_field_with_issues(
    scan_field, max_span: int = MAX_SCAN_RANGE_SPAN
) -> tuple[list[int], list[str]]:
    """Expand a scan-number cell, reporting fragments that were rejected.

    ``"193-196,198"`` -> ``[193, 194, 195, 196, 198]``; ``"2178.0"`` -> ``[2178]``;
    blanks and unparseable fragments -> ``[]``. Tolerates the stray ``?``,
    newlines and semicolons that appear in the hand-maintained sheets.

    Ranges wider than ``max_span`` are rejected rather than expanded, and
    reported so the underlying typo can be fixed at source.
    """
    if scan_field is None:
        return [], []

    if isinstance(scan_field, pd.Series):
        values = [v for v in scan_field.tolist() if pd.notna(v) and str(v).strip()]
        if not values:
            return [], []
        scan_field = values[0]

    if pd.isna(scan_field):
        return [], []

    text = str(scan_field).replace("?", "")
    for sep in ("\n", "\r", ";"):
        text = text.replace(sep, ",")
    text = re.sub(r"[^0-9,\-.]", "", text.replace(" ", ""))
    text = text.replace("--", "-").strip(",-")
    if not text:
        return [], []

    scans: list[int] = []
    issues: list[str] = []
    for part in text.split(","):
        if not part:
            continue
        try:
            if "-" in part:
                bounds = part.split("-")
                if len(bounds) != 2 or not all(bounds):
                    issues.append(f"unparseable range {part!r}")
                    continue
                start, end = int(float(bounds[0])), int(float(bounds[1]))
                span = abs(start - end) + 1
                if span > max_span:
                    issues.append(f"range {part!r} spans {span} scans (max {max_span})")
                    continue
                if start > end:
                    issues.append(f"range {part!r} is descending")
                step = 1 if start <= end else -1
                scans.extend(range(start, end + step, step))
            else:
                scans.append(int(float(part)))
        except (ValueError, TypeError):
            issues.append(f"unparseable value {part!r}")
            continue
    return scans, issues


def expand_scan_field(scan_field, max_span: int = MAX_SCAN_RANGE_SPAN) -> list[int]:
    """Expand a scan-number cell into explicit scan numbers."""
    return expand_scan_field_with_issues(scan_field, max_span)[0]


def parse_dicom_filename(filename: str) -> dict | None:
    """Parse an acquisition filename into its component fields, or ``None``."""
    match = FILENAME_PATTERN.match(filename.strip())
    if not match:
        return None

    date = pd.to_datetime(match.group("date"), format="%d_%m_%Y", errors="coerce")
    bodypart = match.group("bodypart").title()
    return {
        "filename": filename,
        "initials": match.group("initials").upper(),
        "side": match.group("side").title(),
        "bodypart": bodypart,
        "protocol_area": protocol_area_for_bodypart(bodypart),
        "lnum": match.group("lnum").upper(),
        "scan_number": int(match.group("scan_number")),
        "date": date.normalize() if pd.notna(date) else pd.NaT,
        "copy_index": int(match.group("copy")) if match.group("copy") else 0,
    }
