"""Report on a built manifest: coverage, confidence and data problems.

Run from the repo root:

    python src/manifest/validate_manifest.py

Exits non-zero if anything is found that would corrupt training data (files
that cannot be read, or scans whose patient identity is ambiguous). Lower-grade
findings -- loose matches, duplicates, missing demographics -- are reported but
do not fail, since they are expected in a hand-maintained dataset and just need
to stay visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

LOOSE_MATCH_METHODS = {"initials_scan", "scan_area_only"}
BLOCKING_HEADER_STATUSES = {"unreadable", "missing_geometry", "truncated", "offset_not_found"}


def _section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # An empty finding table is the good outcome, not a parse failure.
        return pd.DataFrame()


def validate(manifest_dir: Path) -> int:
    patients = _read(manifest_dir / "patients.csv")
    scans = _read(manifest_dir / "scans.csv")
    frames = _read(manifest_dir / "frames.csv")

    failures: list[str] = []
    warnings: list[str] = []

    _section("Manifest contents")
    if not patients.empty:
        print(
            f"patients : {patients['patient_id'].nunique()} "
            f"({len(patients)} participant records across sessions)"
        )
    else:
        print("patients : 0")
    print(f"scans    : {len(scans)}")
    print(f"frames   : {len(frames)}")

    if patients.empty and scans.empty and frames.empty:
        failures.append("manifest is empty - build_manifest.py produced nothing")
    elif scans.empty:
        warnings.append(
            "no acquisitions indexed - attach the acquisition drive and re-run build_manifest.py"
        )

    if not patients.empty:
        _section("Patients")
        print(patients.groupby("cohort").size().to_string())
        missing_demographics = patients["age"].isna().sum()
        print(f"missing demographics: {missing_demographics}")
        if missing_demographics:
            warnings.append(f"{missing_demographics} patients have no age/sex on record")
        print(patients["identity_confidence"].value_counts().to_string())
        undecided = patients[patients["identity_confidence"] == "unresolved_kept_separate"]
        if not undecided.empty:
            warnings.append(
                f"initials {sorted(undecided['initials'].unique())} occur in both sessions with "
                "no recorded decision - kept separate (add one to configs/manifest_decisions.yaml)"
            )

    if not scans.empty:
        _section("Acquisition headers")
        print(scans["header_status"].value_counts().to_string())
        blocking = scans[scans["header_status"].isin(BLOCKING_HEADER_STATUSES)]
        if not blocking.empty:
            failures.append(f"{len(blocking)} acquisitions cannot be read or are truncated")

        _section("Acquisition geometry")
        geometry = (
            scans.groupby(["rows", "cols", "num_frames", "samples_per_pixel"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        print(geometry.to_string(index=False))
        if len(geometry) > 1:
            print(
                "\nnote: geometry varies between acquisitions - preprocessing must not "
                "assume a single input shape"
            )

        _section("Identity matching")
        counts = scans["match_method"].value_counts()
        print(counts.to_string())
        total = len(scans)
        loose = int(counts.reindex(list(LOOSE_MATCH_METHODS)).fillna(0).sum())
        if loose:
            print(f"\nloose matches: {loose}/{total} ({loose / total:.1%}) - identity inferred, verify before trusting")
            warnings.append(f"{loose} scans matched on a loose key")
        ambiguous = int(counts.get("ambiguous", 0))
        if ambiguous:
            failures.append(f"{ambiguous} scans match multiple participants and are unresolved")
        unmatched = int(counts.get("unmatched", 0))
        if unmatched:
            warnings.append(f"{unmatched} scans have no scan-log entry")

        duplicates = scans["is_duplicate_of"].notna().sum()
        if duplicates:
            _section("Duplicates")
            print(f"{duplicates} duplicate acquisitions flagged (excluded from selection)")

        if "selected_for_segmentation" in scans.columns:
            _section("Segmentation candidates")
            selected = scans[scans["selected_for_segmentation"].fillna(False)]
            print(f"selected acquisitions: {len(selected)}")
            if not selected.empty and "patient_id" in selected.columns:
                print(f"covering patients    : {selected['patient_id'].nunique()}")

    if not frames.empty:
        _section("Frames")
        print(frames["annotation_status"].value_counts().to_string())
        print(f"patients represented: {frames['patient_id'].nunique()}")
        if "patient_id_confidence" in frames.columns:
            heuristic = (frames["patient_id_confidence"] == "heuristic_consecutive_run").sum()
            if heuristic:
                warnings.append(
                    f"{heuristic} frames have inferred (not recorded) patient identity"
                )
        unlabelled = frames["mask_path"].isna().sum()
        if unlabelled:
            print(f"frames without a mask: {unlabelled}")

    for name in (
        "manifest_build_errors.csv",
        "demographics_issues.csv",
        "scan_log_issues.csv",
        "scan_number_collisions.csv",
    ):
        table = _read(manifest_dir / name)
        if not table.empty:
            _section(name)
            print(f"{len(table)} rows - see {manifest_dir / name}")
            if name == "scan_number_collisions.csv":
                warnings.append(
                    f"{table['scan_number'].nunique()} scan numbers are claimed by more than one "
                    "participant - matching on scan number alone is unsafe for these"
                )
            if name == "scan_log_issues.csv":
                needs_fix = (
                    table["needs_fix"].astype(str).str.lower() == "true"
                    if "needs_fix" in table.columns
                    else pd.Series(True, index=table.index)
                )
                print(f"needing a fix in the spreadsheet: {int(needs_fix.sum())}")
                if needs_fix.any():
                    warnings.append(
                        f"{int(needs_fix.sum())} scan-log entries need correcting in the source "
                        "spreadsheets (see scan_log_issues.csv, needs_fix=True)"
                    )
            if name == "demographics_issues.csv":
                print(table["issue"].value_counts().to_string())
                open_issues = table[~table["issue"].str.startswith("resolved_")]
                if not open_issues.empty:
                    warnings.append(
                        f"{len(open_issues)} open demographics issues in the source sheets "
                        "(see demographics_issues.csv)"
                    )
                unmatched = table[table["issue"] == "decision_unmatched"]
                if not unmatched.empty:
                    failures.append(
                        f"{len(unmatched)} entries in configs/manifest_decisions.yaml match no "
                        "participant - check them for typos"
                    )

    _section("Result")
    for warning in warnings:
        print(f"  warning: {warning}")
    for failure in failures:
        print(f"  FAIL   : {failure}")
    if not warnings and not failures:
        print("  clean")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=REPO_ROOT / "data" / "manifest")
    args = parser.parse_args()
    if not args.manifest_dir.exists():
        print(f"No manifest at {args.manifest_dir}; run build_manifest.py first", file=sys.stderr)
        return 1
    return validate(args.manifest_dir)


if __name__ == "__main__":
    raise SystemExit(main())
