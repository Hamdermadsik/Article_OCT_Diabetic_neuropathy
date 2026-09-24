import re
import os
from pathlib import Path
import pandas as pd

# =========================================================
# HELPERS
# =========================================================
def make_unique_columns(columns):
    counts = {}
    new_cols = []

    for col in columns:
        col = str(col).strip()
        if col not in counts:
            counts[col] = 0
            new_cols.append(col)
        else:
            counts[col] += 1
            new_cols.append(f"{col}.{counts[col]}")
    return new_cols


def get_initials(participant_id: str):
    if pd.isna(participant_id):
        return None
    return str(participant_id).strip().split("-")[-1]


def get_group(participant_id: str):
    if pd.isna(participant_id):
        return None
    return str(participant_id).strip().split("-")[0]


def expand_scan_field(scan_field):
    """
    Examples:
        '193-196,198' -> [193,194,195,196,198]
        '249, 251-254' -> [249,251,252,253,254]
        '173' -> [173]
        '' / NaN -> []
    """
    if scan_field is None:
        return []

    if isinstance(scan_field, pd.Series):
        vals = [v for v in scan_field.tolist() if pd.notna(v) and str(v).strip() != ""]
        if not vals:
            return []
        if len(vals) > 1:
            raise ValueError(f"Expected one cell, got multiple values: {vals}")
        scan_field = vals[0]

    if pd.isna(scan_field):
        return []

    s = str(scan_field).replace(" ", "")
    if not s:
        return []

    scans = []
    for part in s.split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            a, b = int(a), int(b)
            if a <= b:
                scans.extend(range(a, b + 1))
            else:
                scans.extend(range(a, b - 1, -1))
        else:
            scans.append(int(part))

    return scans


# =========================================================
# PATHS
# =========================================================
DATA_PATH = Path("/mnt/d/OCTA DICOM files may 2021")
EXCEL_FILE = Path.cwd().parent / "excel_filer" / "MAY.xlsx"
OUTPUT_FILE = Path.cwd().parent / "excel_filer" / "matched_filenames_full.xlsx"
UNMATCHED_FILE = Path.cwd().parent / "excel_filer" / "unmatched_filenames_full.xlsx"
MISSING_EXPECTED_FILE = Path.cwd().parent / "excel_filer" / "missing_expected_scans.xlsx"

# =========================================================
# READ EXCEL
# =========================================================
df = pd.read_excel(EXCEL_FILE)
df.columns = make_unique_columns(df.columns)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

print("Excel columns:")
for c in df.columns:
    print(c)
print(f"\nExcel rows: {len(df)}")

# =========================================================
# DEFINE COLUMN MAPPING
# IMPORTANT:
# We intentionally use BOTH Isc columns:
#   Isc   = Hand Isc Flavanol
#   Isc.1 = Foot Isc Flavanol
# And both PORH columns:
#   Hand PORH   = Hand regular
#   Hand PORH.1 = Hand flavanol
#   Foot PORH   = Foot regular
#   Foot PORH.1 = Foot flavanol
# =========================================================
phase_specs = [
    # -------------------------
    # Regular Hand
    # -------------------------
    {
        "column": "Hand Baseline",
        "phase": "Baseline",
        "protocol_area": "Hand",
        "condition": "Regular",
    },
    {
        "column": "Hand Isc",
        "phase": "Isc",
        "protocol_area": "Hand",
        "condition": "Regular",
    },
    {
        "column": "Hand PORH",
        "phase": "PORH",
        "protocol_area": "Hand",
        "condition": "Regular",
    },

    # -------------------------
    # Regular Foot
    # -------------------------
    {
        "column": "Foot Baseline",
        "phase": "Baseline",
        "protocol_area": "Foot",
        "condition": "Regular",
    },
    {
        "column": "Foot Isc",
        "phase": "Isc",
        "protocol_area": "Foot",
        "condition": "Regular",
    },
    {
        "column": "Foot PORH",
        "phase": "PORH",
        "protocol_area": "Foot",
        "condition": "Regular",
    },

    # -------------------------
    # Flavanol Hand
    # -------------------------
    {
        "column": "Hand Baseline Flavanol",
        "phase": "Baseline",
        "protocol_area": "Hand",
        "condition": "Flavanol",
    },
    {
        "column": "Isc",
        "phase": "Isc",
        "protocol_area": "Hand",
        "condition": "Flavanol",
    },
    {
        "column": "Hand PORH.1",
        "phase": "PORH",
        "protocol_area": "Hand",
        "condition": "Flavanol",
    },

    # -------------------------
    # Flavanol Foot
    # -------------------------
    {
        "column": "Foot baseline flavanol",
        "phase": "Baseline",
        "protocol_area": "Foot",
        "condition": "Flavanol",
    },
    {
        "column": "Isc.1",
        "phase": "Isc",
        "protocol_area": "Foot",
        "condition": "Flavanol",
    },
    {
        "column": "Foot PORH.1",
        "phase": "PORH",
        "protocol_area": "Foot",
        "condition": "Flavanol",
    },
]

# keep only specs that actually exist
existing_phase_specs = [spec for spec in phase_specs if spec["column"] in df.columns]

print("\nColumns used:")
for spec in existing_phase_specs:
    print(
        f"{spec['column']:<24} -> "
        f"{spec['protocol_area']:<4} | {spec['condition']:<8} | {spec['phase']}"
    )

# =========================================================
# BUILD LONG EXCEL TABLE
# One row per scan number
# =========================================================
excel_long_rows = []

for _, row in df.iterrows():
    participant_id = row["Participant ID"]
    date_value = row["Date"]
    initials = get_initials(participant_id)
    group = get_group(participant_id)

    if pd.isna(date_value) or initials is None:
        continue

    for spec in existing_phase_specs:
        scan_numbers = expand_scan_field(row[spec["column"]])

        for scan_number in scan_numbers:
            excel_long_rows.append({
                "participant_id": participant_id,
                "DP_or_HP": group,
                "initials": initials,
                "date": pd.to_datetime(date_value).normalize(),
                "scan_number": int(scan_number),
                "phase": spec["phase"],
                "protocol_area": spec["protocol_area"],
                "condition": spec["condition"],
                "excel_column": spec["column"],
            })

excel_long_df = pd.DataFrame(excel_long_rows)

print(f"\nExpanded Excel rows: {len(excel_long_df)}")

# =========================================================
# PARSE FILENAMES
# Handles:
#   AC_Right Hallux_L1051_S2397__20_05_2021.dcm
#   BB_Right Instep_L1046_S2232__19_05_2021.dcm
#   JF_Right Hand_L1011_S193__07_05_2021.dcm
# =========================================================
filename_pattern = re.compile(
    r'^(?P<initials>[A-Z]+)_'
    r'(?P<side>Right|Left)\s+'
    r'(?P<bodypart>Hand|Hallux|Instep|Foot)_'
    r'(?P<Lnum>L\d+)_'
    r'S(?P<scan_number>\d+)__'
    r'(?P<date>\d{2}_\d{2}_\d{4})\.dcm$'
)

file_rows = []
unparsed_files = []

for fname in os.listdir(DATA_PATH):
    if fname.startswith("."):
        continue

    m = filename_pattern.match(fname)
    if not m:
        unparsed_files.append(fname)
        continue

    bodypart = m.group("bodypart")
    file_date = pd.to_datetime(m.group("date"), format="%d_%m_%Y", errors="coerce")

    # map filename bodypart -> Excel protocol area
    if bodypart == "Hand":
        protocol_area = "Hand"
    elif bodypart in {"Hallux", "Instep", "Foot"}:
        protocol_area = "Foot"
    else:
        protocol_area = None

    file_rows.append({
        "filename": fname,
        "initials": m.group("initials"),
        "side": m.group("side"),
        "bodypart": bodypart,
        "protocol_area": protocol_area,
        "Lnum": m.group("Lnum"),
        "scan_number": int(m.group("scan_number")),
        "date": file_date.normalize() if pd.notna(file_date) else pd.NaT,
    })

files_df = pd.DataFrame(file_rows)

print(f"Parsed filenames: {len(files_df)}")
print(f"Unparsed filenames: {len(unparsed_files)}")

if unparsed_files:
    print("\nFirst 20 unparsed files:")
    for f in unparsed_files[:20]:
        print(f)

# =========================================================
# MERGE FILENAMES WITH EXCEL
# Match on:
#   initials + date + scan_number + protocol_area
#
# This means:
#   Hand files only match Hand columns
#   Hallux/Instep files only match Foot columns
# =========================================================
matched_df = files_df.merge(
    excel_long_df,
    on=["initials", "date", "scan_number", "protocol_area"],
    how="left"
)

# =========================================================
# SAVE MATCHED/UNMATCHED FILE VIEW
# =========================================================
output_df = matched_df[[
    "filename",
    "participant_id",
    "DP_or_HP",
    "Lnum",
    "phase",
    "protocol_area",
    "condition",
    "bodypart",
    "side",
    "date",
    "scan_number",
    "excel_column",
]].copy()

output_df = output_df.sort_values(
    by=["date", "participant_id", "protocol_area", "condition", "phase", "scan_number"],
    na_position="last"
).reset_index(drop=True)

output_df.to_excel(OUTPUT_FILE, index=False)

print(f"\nSaved matched file to:\n{OUTPUT_FILE}")
print(output_df.head(30))

unmatched_df = output_df[output_df["participant_id"].isna()].copy()
unmatched_df.to_excel(UNMATCHED_FILE, index=False)

print(f"\nSaved unmatched parsed files to:\n{UNMATCHED_FILE}")
print(f"Unmatched parsed files: {len(unmatched_df)}")

# =========================================================
# ALSO CHECK THE OTHER DIRECTION:
# Excel scans that have no matching file
# =========================================================
reverse_merge = excel_long_df.merge(
    files_df,
    on=["initials", "date", "scan_number", "protocol_area"],
    how="left",
    indicator=True
)

missing_expected_df = reverse_merge[reverse_merge["_merge"] == "left_only"].copy()

missing_expected_df = missing_expected_df[[
    "participant_id",
    "DP_or_HP",
    "initials",
    "date",
    "scan_number",
    "phase",
    "protocol_area",
    "condition",
    "excel_column",
]]

missing_expected_df = missing_expected_df.sort_values(
    by=["date", "participant_id", "protocol_area", "condition", "phase", "scan_number"]
).reset_index(drop=True)

missing_expected_df.to_excel(MISSING_EXPECTED_FILE, index=False)

print(f"\nSaved Excel scans with no matching file to:\n{MISSING_EXPECTED_FILE}")
print(f"Missing expected scans: {len(missing_expected_df)}")

# =========================================================
# SUMMARY
# =========================================================
matched_count = output_df["participant_id"].notna().sum()
total_parsed = len(output_df)

print("\nSummary")
print("-------")
print(f"Total parsed files:   {total_parsed}")
print(f"Matched parsed files: {matched_count}")
print(f"Unmatched files:      {total_parsed - matched_count}")
print(f"Expected scans in Excel: {len(excel_long_df)}")
print(f"Missing expected scans:  {len(missing_expected_df)}")

print("\nMatched counts by protocol area / condition / phase:")
summary = (
    output_df.assign(matched=output_df["participant_id"].notna())
    .groupby(["protocol_area", "condition", "phase", "matched"])
    .size()
    .reset_index(name="count")
)
print(summary)
