"""Build the dataset manifest: patients, scans and frames.

Run from the repo root:

    python src/manifest/build_manifest.py

The acquisition drive is optional. Without it the Excel logs, demographics and
legacy frames are still processed, so the manifest can be built and inspected
before the drive is attached; re-run with the drive mounted to populate scans.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # allow `python src/manifest/build_manifest.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from manifest import cohort_specs as specs
    from manifest.demographics import (
        build_identity_review,
        find_ambiguous_initials,
        load_demographics_sheet,
        load_october_demographics,
        resolve_demographics,
    )
    from manifest.dicom_index import iter_dicom_files, read_dicom_header
    from manifest.excel_utils import parse_dicom_filename
    from manifest.legacy_frames import build_legacy_frames
    from manifest.matching import (
        find_missing_expected_scans,
        find_scan_number_collisions,
        mark_duplicates,
        match_scans,
    )
    from manifest.parse_may import load_may_sheet, parse_may
    from manifest.parse_october import load_october_sheets, parse_october
else:
    from . import cohort_specs as specs
    from .demographics import (
        build_identity_review,
        find_ambiguous_initials,
        load_demographics_sheet,
        load_october_demographics,
        resolve_demographics,
    )
    from .dicom_index import iter_dicom_files, read_dicom_header
    from .excel_utils import parse_dicom_filename
    from .legacy_frames import build_legacy_frames
    from .matching import (
        find_missing_expected_scans,
        find_scan_number_collisions,
        mark_duplicates,
        match_scans,
    )
    from .parse_may import load_may_sheet, parse_may
    from .parse_october import load_october_sheets, parse_october

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DRIVE_ROOTS = {
    specs.MAY_COHORT: Path("/mnt/d/OCTA DICOM files may 2021"),
    specs.OCTOBER_COHORT: Path("/mnt/d/OCTOBER"),
}
COHORT_PREFIX = {specs.MAY_COHORT: "MAY", specs.OCTOBER_COHORT: "OCT"}

# The two sessions are seven months apart, so the acquisition date written into
# every filename identifies the session unambiguously. Files have been found
# filed in the wrong session's folder (e.g. October 2020 scans in the May 2021
# folder), so the folder is recorded but not trusted.
SESSION_BOUNDARY = pd.Timestamp("2021-01-01")


def session_for_date(date, folder_cohort: str) -> str:
    if pd.isna(date):
        return folder_cohort
    return specs.OCTOBER_COHORT if date < SESSION_BOUNDARY else specs.MAY_COHORT


def canonical_patient_id(cohort: str, participant_id: str) -> str:
    """Cohort-scoped patient key.

    Cohorts are kept separate by default: a participant recorded in both is two
    rows until someone confirms they are the same person. Merging on shared
    initials alone is exactly the mistake this manifest exists to prevent.
    """
    return f"{COHORT_PREFIX.get(cohort, cohort.upper())}-{participant_id}"


def index_drive(roots: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Header-index every acquisition. Returns (scans, unparsed_filenames)."""
    records, unparsed = [], []
    for cohort, root in roots.items():
        try:
            root_available = root.exists()
        except OSError as exc:
            # A stale mount point (drive unplugged but /mnt/d still mounted)
            # raises rather than returning False.
            print(f"  ! {cohort}: {root} unavailable ({exc.strerror}) - skipping", file=sys.stderr)
            continue
        if not root_available:
            print(f"  ! {cohort}: {root} not available - skipping", file=sys.stderr)
            continue

        count = 0
        for path in iter_dicom_files(root):
            header = read_dicom_header(path)
            parsed = parse_dicom_filename(path.name)
            if parsed is None:
                unparsed.append({"cohort": cohort, "filepath": str(path), "filename": path.name})
                continue
            records.append(
                {
                    **parsed,
                    **header,
                    "folder_cohort": cohort,
                    "cohort": session_for_date(parsed["date"], cohort),
                    "scan_id": path.stem,
                }
            )
            count += 1
        print(f"  {cohort}: indexed {count} acquisitions from {root}")

    if records:
        misfiled = sum(1 for r in records if r["cohort"] != r["folder_cohort"])
        if misfiled:
            print(
                f"  ! {misfiled} acquisition(s) are in the other session's folder; "
                "assigned by acquisition date (see folder_cohort column)",
                file=sys.stderr,
            )

    return pd.DataFrame(records), pd.DataFrame(unparsed)


def build_patients(participants: pd.DataFrame) -> pd.DataFrame:
    """One row per scanned participant, with identity confidence recorded.

    Participants whose initials also occur in the other session are kept as
    separate patients until a person confirms they are the same individual.
    """
    if participants.empty:
        return participants.assign(patient_id=None, identity_confidence=None)

    by_cohort = participants.groupby("cohort")["initials"].apply(set).to_dict()
    in_both = by_cohort.get(specs.MAY_COHORT, set()) & by_cohort.get(specs.OCTOBER_COHORT, set())

    patients = participants.copy()
    patients["patient_id"] = [
        canonical_patient_id(row.cohort, row.participant_id) for row in patients.itertuples()
    ]
    patients["identity_confidence"] = [
        "unresolved_kept_separate" if initials in in_both else "single_cohort"
        for initials in patients["initials"]
    ]
    columns = [
        "patient_id",
        "cohort",
        "participant_id",
        "initials",
        "group",
        "sex",
        "age",
        "demographics_source",
        "identity_confidence",
    ]
    return patients[columns].sort_values("patient_id").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--may-drive", type=Path, default=DEFAULT_DRIVE_ROOTS[specs.MAY_COHORT])
    parser.add_argument(
        "--october-drive", type=Path, default=DEFAULT_DRIVE_ROOTS[specs.OCTOBER_COHORT]
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo_root
    excel_dir = repo / "excel_filer"
    out_dir = args.out_dir or (repo / "data" / "manifest")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing Excel scan logs...")
    may_long, scan_log_issues = parse_may(load_may_sheet(excel_dir / "MAY.xlsx"))
    missing_columns = scan_log_issues[
        scan_log_issues["issue"] == "configured phase column missing from sheet"
    ]
    if not missing_columns.empty:
        print(
            f"  ! May sheet is missing expected phase columns "
            f"{list(missing_columns['excel_column'])}; those scans are NOT in the manifest",
            file=sys.stderr,
        )
    october_sheet1, october_sheet2 = load_october_sheets(
        excel_dir / "Data_Summary_October_2020.xlsx"
    )
    october_long = parse_october(october_sheet1, october_sheet2)
    long_df = pd.concat([may_long, october_long], ignore_index=True)
    print(f"  may={len(may_long)} october={len(october_long)} expected scan rows")
    if not scan_log_issues.empty:
        print(
            f"  ! {len(scan_log_issues)} scan-log cells rejected as malformed "
            "(see scan_log_issues.csv)",
            file=sys.stderr,
        )

    print("Resolving demographics...")
    participants, demographics_issues = resolve_demographics(
        long_df,
        load_demographics_sheet(excel_dir / "2021-05-OCTAAgeing-hand.xlsx"),
        load_october_demographics(october_sheet1),
    )
    ambiguous = find_ambiguous_initials(participants)
    ambiguous_initials = set(ambiguous["initials"]) if not ambiguous.empty else set()
    identity_review = build_identity_review(participants)
    print(
        f"  {len(participants)} scanned participants | "
        f"{participants['age'].notna().sum()} with demographics | "
        f"{len(demographics_issues)} demographics issues"
    )
    print(
        f"  initials scanned in both sessions: "
        f"{sorted(identity_review['initials'].unique()) if not identity_review.empty else []}"
    )

    print("Indexing acquisition drive...")
    files_df, unparsed = index_drive(
        {specs.MAY_COHORT: args.may_drive, specs.OCTOBER_COHORT: args.october_drive}
    )

    if files_df.empty:
        print("  ! no acquisitions indexed - scans.csv will be empty", file=sys.stderr)
        scans = pd.DataFrame()
    else:
        print("Matching acquisitions to scan logs...")
        matched = []
        for cohort, group in files_df.groupby("cohort"):
            cohort_long = long_df[long_df["cohort"] == cohort]
            matched.append(match_scans(group, cohort_long, ambiguous_initials))
        scans = pd.concat(matched, ignore_index=True)
        scans = mark_duplicates(scans)
        scans["patient_id"] = [
            canonical_patient_id(row.cohort, row.participant_id)
            if pd.notna(row.participant_id)
            else None
            for row in scans.itertuples()
        ]
        # Baseline/Regular acquisitions are the segmentation-relevant ones;
        # Isc/PORH/Flavanol revisit the same anatomy under different perfusion.
        scans["selected_for_segmentation"] = (
            (scans["phase"] == "Baseline")
            & (scans["condition"] == "Regular")
            & scans["is_duplicate_of"].isna()
            & (scans["header_status"] == "ok")
        )
        print(
            "  "
            + " | ".join(
                f"{method}={count}"
                for method, count in scans["match_method"].value_counts().items()
            )
        )

    patients = build_patients(participants)
    legacy = build_legacy_frames(
        repo / "data" / "train" / "images",
        repo / "data" / "train" / "mask",
        repo / "data" / "test" / "images",
        repo / "data" / "test" / "mask",
    )

    print(f"Writing manifest to {out_dir}...")
    patients.to_csv(out_dir / "patients.csv", index=False)
    scans.to_csv(out_dir / "scans.csv", index=False)
    legacy.to_csv(out_dir / "frames.csv", index=False)

    errors = []
    if not unparsed.empty:
        errors.append(unparsed.assign(issue="filename_unparsed"))
    if not scans.empty:
        bad_headers = scans[scans["header_status"] != "ok"]
        if not bad_headers.empty:
            errors.append(
                bad_headers[["cohort", "filepath", "filename", "header_status", "header_detail"]]
                .assign(issue="header_" + bad_headers["header_status"])
            )
        unresolved = scans[scans["match_method"].isin(["unmatched", "ambiguous"])]
        if not unresolved.empty:
            errors.append(
                unresolved[["cohort", "filepath", "filename", "match_method", "match_detail"]]
                .assign(issue="match_" + unresolved["match_method"])
            )
    error_df = pd.concat(errors, ignore_index=True) if errors else pd.DataFrame()
    error_df.to_csv(out_dir / "manifest_build_errors.csv", index=False)

    scan_log_issues.to_csv(out_dir / "scan_log_issues.csv", index=False)
    demographics_issues.to_csv(out_dir / "demographics_issues.csv", index=False)
    ambiguous.to_csv(out_dir / "ambiguous_initials.csv", index=False)
    identity_review.to_csv(out_dir / "identity_review_candidates.csv", index=False)
    find_scan_number_collisions(long_df).to_csv(
        out_dir / "scan_number_collisions.csv", index=False
    )
    if not scans.empty:
        find_missing_expected_scans(long_df, scans).to_csv(
            out_dir / "missing_expected_scans.csv", index=False
        )

    print(
        f"\nDone. patients={len(patients)} scans={len(scans)} legacy_frames={len(legacy)} "
        f"errors={len(error_df)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
