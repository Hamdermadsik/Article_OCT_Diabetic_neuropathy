"""Scan-level manual QC, joined by the full DICOM filename, never initials."""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

QC_STATUSES = {"use", "discard", "unsure", "unreviewed"}
QC_COLUMNS = ["filename", "qc_status", "qc_note", "qc_reasons",
              "segmentation_review_required", "qc_bscan_index"]


def load_qc_reviews(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=QC_COLUMNS)
    data = json.loads(path.read_text())
    if data.get("schema_version") != 2 or not isinstance(data.get("reviews"), dict):
        raise ValueError("Expected scan-level QC review file")
    records = []
    for record in data["reviews"].values():
        filename, status = record.get("filename"), record.get("status")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError("QC filename must be a full DICOM filename, not a path")
        if status not in QC_STATUSES:
            raise ValueError(f"Unknown QC status for {filename}")
        bscan = record.get("qc_bscan_index")
        if bscan is not None and (isinstance(bscan, bool) or not isinstance(bscan, int) or bscan < 0):
            raise ValueError(f"Invalid QC B-scan index for {filename}")
        records.append({"filename": filename, "qc_status": status, "qc_note": record.get("note", ""),
                        "qc_reasons": ";".join(record.get("reasons", [])),
                        "segmentation_review_required": "segmentation" in record.get("reasons", []),
                        "qc_bscan_index": bscan})
    table = pd.DataFrame(records, columns=QC_COLUMNS)
    if table["filename"].duplicated().any():
        raise ValueError("QC contains multiple decisions for the same scan filename")
    return table


def apply_qc_reviews(scans, review_path):
    """Only explicit use decisions are analysis-eligible; preserve other exclusions."""
    result = scans.copy()
    reviews = load_qc_reviews(review_path).set_index("filename")
    if result.empty:
        if "filename" not in result:
            result["filename"] = pd.Series(dtype=str)
        result["qc_status"] = pd.Series(dtype=str)
        result["qc_note"] = pd.Series(dtype=str)
        result["qc_reasons"] = pd.Series(dtype=str)
        result["segmentation_review_required"] = pd.Series(dtype=bool)
        result["qc_bscan_index"] = pd.Series(dtype="Int64")
        result["analysis_eligible"] = pd.Series(dtype=bool)
        return result
    result["qc_status"] = result["filename"].map(reviews["qc_status"]).fillna("unreviewed")
    result["qc_note"] = result["filename"].map(reviews["qc_note"]).fillna("")
    result["qc_reasons"] = result["filename"].map(reviews["qc_reasons"]).fillna("")
    result["segmentation_review_required"] = result["filename"].map(
        reviews["segmentation_review_required"]).eq(True)
    result["qc_bscan_index"] = result["filename"].map(reviews["qc_bscan_index"]).astype("Int64")
    excluded = result["excluded"].fillna(True).map(
        lambda value: str(value).strip().lower() not in {"false", "0"}
    )
    result["analysis_eligible"] = result["qc_status"].eq("use") & ~excluded
    return result


def _write_csv(table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, suffix=".tmp") as handle:
        temp = Path(handle.name)
        try:
            table.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def sync_manifest_qc(manifest_dir, review_path):
    """Refresh an existing manifest after a decision; never fabricate scan metadata."""
    manifest_dir = Path(manifest_dir)
    reviews = load_qc_reviews(review_path)
    _write_csv(reviews.loc[reviews["qc_status"].eq("use"), ["filename"]],
               manifest_dir / "approved_scan_filenames.csv")
    write_segmentation_candidates(reviews, manifest_dir)
    path = manifest_dir / "scans.csv"
    if not path.exists():
        return False
    try:
        scans = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return False
    scans = apply_qc_reviews(scans, review_path)
    from .generation import apply_generation
    scans = apply_generation(scans, manifest_dir)
    _write_csv(scans, path)
    _write_csv(scans.loc[scans["analysis_eligible"]], manifest_dir / "scans_for_analysis.csv")
    write_segmentation_candidates(scans, manifest_dir)
    return True


def write_segmentation_candidates(scans, manifest_dir):
    """Annotation-review queue, NOT training labels or an automatic training split."""
    flagged = scans["segmentation_review_required"].map(lambda value: str(value).lower() == "true")
    _write_csv(scans.loc[flagged], Path(manifest_dir) / "segmentation_review_candidates.csv")
