"""Per-cohort configuration data (no parsing logic lives here).

October is not a config variant of May -- its sheet has a different *shape* and
needs its own parser (see ``parse_october.py``). Only genuinely data-shaped,
shared constants belong in this module.
"""

from __future__ import annotations

MAY_COHORT = "may_2021"
OCTOBER_COHORT = "october_2020"
LEGACY_SOURCE = "legacy_handpicked"
DRIVE_SOURCE = "drive_v2"

# Normalized long-format schema both cohort parsers must produce. Everything
# downstream (filename matching, demographics join, manifest writing) is
# cohort-agnostic once both parsers agree on these columns.
LONG_SCHEMA = [
    "participant_id",
    "initials",
    "group",
    "date",
    "scan_number",
    "phase",
    "protocol_area",
    "condition",
    "excel_column",
    "cohort",
]

# May: one row per participant visit, one column per (area, condition, phase),
# each cell holding a scan-number range string.
#
# Column names are post-strip/post-uniquify (see ``make_unique_columns``). The
# sheet has two columns literally named "Isc" and two named "Foot PORH"; the
# second of each pair is the Flavanol arm, hence the ".1" suffixes. All 12 were
# verified present in MAY.xlsx -- the notebook's variant looked up five of them
# under names ("Hand Isc Flavanol", "Foot PORH Flavanol", ...) that do not exist
# in the sheet and silently dropped those scans.
MAY_PHASE_SPECS = [
    {"column": "Hand Baseline", "phase": "Baseline", "protocol_area": "Hand", "condition": "Regular"},
    {"column": "Hand Isc", "phase": "Isc", "protocol_area": "Hand", "condition": "Regular"},
    {"column": "Hand PORH", "phase": "PORH", "protocol_area": "Hand", "condition": "Regular"},
    {"column": "Foot Baseline", "phase": "Baseline", "protocol_area": "Foot", "condition": "Regular"},
    {"column": "Foot Isc", "phase": "Isc", "protocol_area": "Foot", "condition": "Regular"},
    {"column": "Foot PORH", "phase": "PORH", "protocol_area": "Foot", "condition": "Regular"},
    {"column": "Hand Baseline Flavanol", "phase": "Baseline", "protocol_area": "Hand", "condition": "Flavanol"},
    {"column": "Isc", "phase": "Isc", "protocol_area": "Hand", "condition": "Flavanol"},
    {"column": "Hand PORH.1", "phase": "PORH", "protocol_area": "Hand", "condition": "Flavanol"},
    {"column": "Foot baseline flavanol", "phase": "Baseline", "protocol_area": "Foot", "condition": "Flavanol"},
    {"column": "Isc.1", "phase": "Isc", "protocol_area": "Foot", "condition": "Flavanol"},
    {"column": "Foot PORH.1", "phase": "PORH", "protocol_area": "Foot", "condition": "Flavanol"},
]

MAY_PARTICIPANT_COLUMN = "Participant ID"
MAY_DATE_COLUMN = "Date"

# October Sheet1 encodes (area, phase) as a two-letter code instead of one
# column per combination.
OCTOBER_SETTING_CODES = {
    "HB": ("Hand", "Baseline"),
    "HI": ("Hand", "Isc"),
    "HP": ("Hand", "PORH"),
    "FB": ("Foot", "Baseline"),
    "FI": ("Foot", "Isc"),
    "FP": ("Foot", "PORH"),
}

# Sheet1 lays two acquisition blocks side by side per row.
OCTOBER_SHEET1_BLOCKS = [
    ("Imaging settings", "Scan #"),
    ("Imaging settings.1", "Scan #.1"),
]
OCTOBER_PARTICIPANT_COLUMN = "Perticipant ID"  # typo is in the source sheet
OCTOBER_SEX_COLUMN = "M/F"
OCTOBER_AGE_COLUMN = "Age"

# Sheet2 is freeform "Additional Data": participant in col 0, context label in
# col 1 (e.g. Hand / Foot / Laying Down), scan number in col 2.
OCTOBER_SHEET2_COLUMNS = ("Additional Data", "Unnamed: 1", "Unnamed: 2")
OCTOBER_SHEET2_SKIP_LABELS = {"seating positions", "scan #", "nan", ""}

# The October log records no per-scan acquisition dates. Rows are stamped with
# this sentinel so downstream code can tell "unknown" from a real date instead
# of silently trusting a fabricated one.
OCTOBER_DATE_IS_PLACEHOLDER = True

# Expected acquisition geometry, verified across sample volumes from both
# cohorts. Deviations are flagged rather than dropped.
EXPECTED_ROWS = 460
EXPECTED_COLUMNS = 1174
EXPECTED_FRAMES = 120
EXPECTED_SAMPLES_PER_PIXEL = 3
EXPECTED_BITS_ALLOCATED = 8
UNCOMPRESSED_TRANSFER_SYNTAXES = {
    "1.2.840.10008.1.2",  # Implicit VR Little Endian
    "1.2.840.10008.1.2.1",  # Explicit VR Little Endian
}
