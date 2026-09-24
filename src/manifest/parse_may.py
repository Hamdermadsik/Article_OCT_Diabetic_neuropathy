"""Parse the May 2021 scan log into the normalized long format.

Shape: one row per participant visit, with one column per
(protocol_area, condition, phase) holding a scan-number range string.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cohort_specs import (
    LONG_SCHEMA,
    MAY_COHORT,
    MAY_DATE_COLUMN,
    MAY_PARTICIPANT_COLUMN,
    MAY_PHASE_SPECS,
)
from .excel_utils import (
    expand_scan_field_with_issues,
    extract_group,
    extract_initials,
    make_unique_columns,
    normalize_participant_id,
)


def load_may_sheet(path: Path, sheet_name: int | str = 0) -> pd.DataFrame:
    """Read the May log with column names stripped and repeats disambiguated."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = make_unique_columns(df.columns)
    return df


def parse_may(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand the May log to one row per scan number.

    Returns the long-format table plus a table of source-data problems:
    configured phase columns absent from the sheet (which would mean scans are
    being silently skipped), and cells whose contents could not be fully
    expanded.
    """
    df = df.copy()
    if MAY_DATE_COLUMN in df.columns:
        df[MAY_DATE_COLUMN] = pd.to_datetime(
            df[MAY_DATE_COLUMN], errors="coerce", dayfirst=True
        )

    present = [spec for spec in MAY_PHASE_SPECS if spec["column"] in df.columns]
    issues = [
        {
            "cohort": MAY_COHORT,
            "participant_id": None,
            "excel_column": spec["column"],
            "raw_value": None,
            "issue": "configured phase column missing from sheet",
        }
        for spec in MAY_PHASE_SPECS
        if spec["column"] not in df.columns
    ]

    rows = []
    for _, row in df.iterrows():
        participant_id = normalize_participant_id(row.get(MAY_PARTICIPANT_COLUMN))
        if not participant_id:
            continue

        date_value = row.get(MAY_DATE_COLUMN)
        date_value = (
            pd.to_datetime(date_value).normalize() if pd.notna(date_value) else pd.NaT
        )
        initials = extract_initials(participant_id)
        group = extract_group(participant_id)

        for spec in present:
            raw_value = row[spec["column"]]
            scan_numbers, cell_issues = expand_scan_field_with_issues(raw_value)
            issues.extend(
                {
                    "cohort": MAY_COHORT,
                    "participant_id": participant_id,
                    "excel_column": spec["column"],
                    "raw_value": str(raw_value),
                    "issue": issue,
                }
                for issue in cell_issues
            )
            for scan_number in scan_numbers:
                rows.append(
                    {
                        "participant_id": participant_id,
                        "initials": initials,
                        "group": group,
                        "date": date_value,
                        "scan_number": int(scan_number),
                        "phase": spec["phase"],
                        "protocol_area": spec["protocol_area"],
                        "condition": spec["condition"],
                        "excel_column": spec["column"],
                        "cohort": MAY_COHORT,
                    }
                )

    issues_df = pd.DataFrame(
        issues, columns=["cohort", "participant_id", "excel_column", "raw_value", "issue"]
    )
    return pd.DataFrame(rows, columns=LONG_SCHEMA), issues_df
