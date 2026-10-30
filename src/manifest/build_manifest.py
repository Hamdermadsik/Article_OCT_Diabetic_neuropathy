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
    from manifest.decisions import (
        apply_aliases,
        apply_file_initials,
        apply_identity_decisions,
        decision_label,
        drop_superseded_issues,
        load_decisions,
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
    from .decisions import (
        apply_aliases,
        apply_file_initials,
        apply_identity_decisions,
        decision_label,
        drop_superseded_issues,
        load_decisions,
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
DECISIONS_FILE = Path("configs") / "manifest_decisions.yaml"
DRIVE_CACHE_NAME = "drive_index.csv"

# The two sessions are seven months apart, so the acquisition date written into
# every filename identifies the session unambiguously. Files have been found
# filed in the wrong session's folder (e.g. October 2020 scans in the May 2021
# folder), so the folder is recorded but not trusted.
SESSION_BOUNDARY = pd.Timestamp("2021-01-01")


def session_for_date(date, folder_cohort: str) -> str:
    if pd.isna(date):
        return folder_cohort
    return specs.OCTOBER_COHORT if date < SESSION_BOUNDARY else specs.MAY_COHORT


def _root_available(cohort: str, root: Path) -> bool:
    try:
        return root.exists()
    except OSError as exc:
        # A stale mount point (drive unplugged, /mnt/d still mounted) raises
        # rather than returning False.
        print(f"  ! {cohort}: {root} unavailable ({exc.strerror})", file=sys.stderr)
        return False


def index_drive(roots: dict[str, Path], cache_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Header-index every acquisition. Returns (files, unparsed_filenames).

    Headers are read from the drive when it is attached and saved to
    ``cache_path``. When it is not, the saved copy is used instead, so the
    spreadsheets and decisions can be revised and the manifest rebuilt without
    the drive. Only drive-derived data is cached; filenames are re-parsed on
    every run so changes to the parsing rules still take effect.
    """
    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()

    headers = []
    for cohort, root in roots.items():
        if _root_available(cohort, root):
            rows = [{**read_dicom_header(path), "folder_cohort": cohort} for path in iter_dicom_files(root)]
            print(f"  {cohort}: read {len(rows)} acquisitions from {root}")
        elif not cache.empty and (cache["folder_cohort"] == cohort).any():
            rows = cache[cache["folder_cohort"] == cohort].to_dict("records")
            print(
                f"  {cohort}: drive not attached - using the {len(rows)} acquisitions saved "
                f"from the last time it was read ({cache_path.name})"
            )
        else:
            print(f"  ! {cohort}: drive not attached and no saved copy - skipping", file=sys.stderr)
            continue
        headers.extend(rows)

    if headers:
        pd.DataFrame(headers).to_csv(cache_path, index=False)

    records, unparsed = [], []
    for header in headers:
        parsed = parse_dicom_filename(header["filename"])
        if parsed is None:
            unparsed.append(
                {
                    "cohort": header["folder_cohort"],
                    "filepath": header["filepath"],
                    "filename": header["filename"],
                }
            )
            continue
        records.append(
            {
                **header,
                **parsed,
                "cohort": session_for_date(parsed["date"], header["folder_cohort"]),
                "scan_id": Path(header["filename"]).stem,
            }
        )

    misfiled = sum(1 for r in records if r["cohort"] != r["folder_cohort"])
    if misfiled:
        print(
            f"  ! {misfiled} acquisition(s) are in the other session's folder; "
            "assigned by acquisition date (see folder_cohort column)",
            file=sys.stderr,
        )
    return pd.DataFrame(records), pd.DataFrame(unparsed)


PATIENT_COLUMNS = [
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--may-drive", type=Path, default=DEFAULT_DRIVE_ROOTS[specs.MAY_COHORT])
    parser.add_argument(
        "--october-drive", type=Path, default=DEFAULT_DRIVE_ROOTS[specs.OCTOBER_COHORT]
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--decisions",
        type=Path,
        default=None,
        help=f"identity decisions file (default: <repo>/{DECISIONS_FILE})",
    )
    args = parser.parse_args()

    repo = args.repo_root
    excel_dir = repo / "excel_filer"
    out_dir = args.out_dir or (repo / "data" / "manifest")
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions = load_decisions(args.decisions or (repo / DECISIONS_FILE))

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
    long_df, file_initials, alias_issues = apply_aliases(long_df, decisions.aliases)
    print(f"  may={len(may_long)} october={len(october_long)} expected scan rows")
    if not scan_log_issues.empty:
        print(
            f"  ! {len(scan_log_issues)} scan-log cells rejected as malformed "
            "(see scan_log_issues.csv)",
            file=sys.stderr,
        )

    print("Resolving demographics...")
    demographics_sheet = load_demographics_sheet(excel_dir / "2021-05-OCTAAgeing-hand.xlsx")
    participants, demographics_issues = resolve_demographics(
        long_df, demographics_sheet, load_october_demographics(october_sheet1)
    )
    participants, identity_issues = apply_identity_decisions(
        participants, decisions.identity, demographics_sheet
    )
    demographics_issues = drop_superseded_issues(
        demographics_issues, alias_issues + identity_issues
    )
    ambiguous = find_ambiguous_initials(participants)
    ambiguous_initials = set(ambiguous["initials"]) if not ambiguous.empty else set()
    identity_review = build_identity_review(participants)
    if not identity_review.empty:
        identity_review["decision"] = decision_label(identity_review, decisions.identity)
    confidence = participants["identity_confidence"].value_counts().to_dict()
    print(
        f"  {len(participants)} scanned participants -> "
        f"{participants['patient_id'].nunique()} patients | "
        f"{participants['age'].notna().sum()} with demographics"
    )
    print(
        f"  identity: {confidence.get('confirmed_same_person', 0) // 2} confirmed same person, "
        f"{confidence.get('confirmed_different_people', 0) // 2} confirmed different, "
        f"{confidence.get('unresolved_kept_separate', 0) // 2} undecided"
    )
    for issue in alias_issues + identity_issues:
        if issue["issue"] in {"decision_unmatched", "alias_initials_shared", "same_person_disagrees"}:
            print(f"  ! {issue['detail']}", file=sys.stderr)

    print("Indexing acquisition drive...")
    files_df, unparsed = index_drive(
        {specs.MAY_COHORT: args.may_drive, specs.OCTOBER_COHORT: args.october_drive},
        out_dir / DRIVE_CACHE_NAME,
    )
    files_df = apply_file_initials(files_df, file_initials)

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
        patient_of = {
            (row.cohort, row.participant_id): row.patient_id for row in participants.itertuples()
        }
        scans["patient_id"] = [
            patient_of.get((row.cohort, row.participant_id)) for row in scans.itertuples()
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

    patients = participants[PATIENT_COLUMNS].sort_values(["patient_id", "cohort"]).reset_index(
        drop=True
    )
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
        f"\nDone. patients={patients['patient_id'].nunique()} scans={len(scans)} "
        f"legacy_frames={len(legacy)} "
        f"errors={len(error_df)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
