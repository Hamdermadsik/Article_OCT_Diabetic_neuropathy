"""Parse the October 2020 scan log into the normalized long format.

Shape differs from May in three ways: participant identity is a forward-filled
block header rather than a per-row value, (area, phase) is encoded as a
two-letter code rather than one column per combination, and a second sheet
holds freeform "additional" acquisitions.

The log records no per-scan acquisition dates. Rows therefore carry ``NaT``
rather than the fabricated placeholder date the previous pipeline stamped on
them, so downstream code can distinguish "unknown" from a real value.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cohort_specs import (
    LONG_SCHEMA,
    OCTOBER_COHORT,
    OCTOBER_PARTICIPANT_COLUMN,
    OCTOBER_SETTING_CODES,
    OCTOBER_SHEET1_BLOCKS,
    OCTOBER_SHEET2_COLUMNS,
    OCTOBER_SHEET2_SKIP_LABELS,
)
from .excel_utils import (
    expand_scan_field,
    extract_group,
    extract_initials,
    make_unique_columns,
    normalize_participant_id,
)


def load_october_sheets(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both October sheets with column names stripped/disambiguated."""
    sheet1 = pd.read_excel(path, sheet_name="Sheet1")
    sheet2 = pd.read_excel(path, sheet_name="Sheet2")
    sheet1.columns = make_unique_columns(sheet1.columns)
    sheet2.columns = make_unique_columns(sheet2.columns)
    return sheet1, sheet2


def _decode_setting(setting) -> tuple[str | None, str | None]:
    if setting is None or pd.isna(setting):
        return None, None
    code = str(setting).strip().upper()
    return OCTOBER_SETTING_CODES.get(code, (None, None))


def _parse_sheet1(sheet1: pd.DataFrame) -> list[dict]:
    df = sheet1.copy()
    if OCTOBER_PARTICIPANT_COLUMN not in df.columns:
        return []
    df[OCTOBER_PARTICIPANT_COLUMN] = df[OCTOBER_PARTICIPANT_COLUMN].ffill()

    rows = []
    for _, row in df.iterrows():
        participant_id = normalize_participant_id(row.get(OCTOBER_PARTICIPANT_COLUMN))
        if not participant_id:
            continue

        for setting_col, scan_col in OCTOBER_SHEET1_BLOCKS:
            if setting_col not in df.columns or scan_col not in df.columns:
                continue
            area, phase = _decode_setting(row.get(setting_col))
            if area is None:
                continue
            for scan_number in expand_scan_field(row.get(scan_col)):
                rows.append(
                    {
                        "participant_id": participant_id,
                        "initials": extract_initials(participant_id),
                        "group": extract_group(participant_id),
                        "date": pd.NaT,
                        "scan_number": int(scan_number),
                        "phase": phase,
                        "protocol_area": area,
                        "condition": "Regular",
                        "excel_column": setting_col,
                        "cohort": OCTOBER_COHORT,
                    }
                )
    return rows


def _parse_sheet2(sheet2: pd.DataFrame) -> list[dict]:
    participant_col, label_col, scan_col = OCTOBER_SHEET2_COLUMNS
    if not {participant_col, label_col, scan_col}.issubset(sheet2.columns):
        return []

    df = sheet2.copy()
    df[participant_col] = df[participant_col].ffill()
    # Forward-fill the context label *within* each participant block. Filling it
    # globally (as the previous pipeline did) leaks one participant's label
    # across the boundary into the next participant's rows.
    df[label_col] = df.groupby(participant_col, dropna=False)[label_col].ffill()

    rows = []
    for _, row in df.iterrows():
        raw_participant = str(row.get(participant_col, "")).strip()
        if raw_participant.lower() in OCTOBER_SHEET2_SKIP_LABELS:
            continue
        participant_id = normalize_participant_id(raw_participant)
        if not participant_id:
            continue

        scan_value = row.get(scan_col)
        if pd.isna(scan_value) or str(scan_value).strip().lower() in OCTOBER_SHEET2_SKIP_LABELS:
            continue

        label = str(row.get(label_col, "")).strip()
        label_upper = label.upper()
        if "HAND" in label_upper:
            area = "Hand"
        elif "FOOT" in label_upper:
            area = "Foot"
        else:
            area = None

        for scan_number in expand_scan_field(scan_value):
            rows.append(
                {
                    "participant_id": participant_id,
                    "initials": extract_initials(participant_id),
                    "group": extract_group(participant_id),
                    "date": pd.NaT,
                    "scan_number": int(scan_number),
                    "phase": label or "Unknown",
                    "protocol_area": area,
                    "condition": "Additional",
                    "excel_column": "Sheet2",
                    "cohort": OCTOBER_COHORT,
                }
            )
    return rows


def parse_october(sheet1: pd.DataFrame, sheet2: pd.DataFrame) -> pd.DataFrame:
    """Expand both October sheets to one row per scan number."""
    rows = _parse_sheet1(sheet1) + _parse_sheet2(sheet2)
    return pd.DataFrame(rows, columns=LONG_SCHEMA)
