"""Persistent MIP generation progress, keyed by full DICOM filename."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .qc import _write_csv, write_segmentation_candidates

GENERATION_COLUMNS = ["filename", "generation_status", "dicom_path", "mip_path",
                      "bscan_path", "generation_error", "generation_updated_at"]
GENERATION_STATUSES = {"pending", "processing", "ready", "failed"}


def load_generation(manifest_dir):
    path = Path(manifest_dir) / "mip_generation.csv"
    if not path.exists():
        return pd.DataFrame(columns=GENERATION_COLUMNS)
    table = pd.read_csv(path, keep_default_na=False)
    if not set(GENERATION_COLUMNS).issubset(table.columns):
        raise ValueError("Invalid MIP generation manifest")
    if table["filename"].duplicated().any() or not table["generation_status"].isin(GENERATION_STATUSES).all():
        raise ValueError("Duplicate filenames or invalid generation status")
    return table


def apply_generation(scans, manifest_dir):
    result = scans.copy()
    generation = load_generation(manifest_dir).set_index("filename")
    for column in GENERATION_COLUMNS[1:]:
        if result.empty:
            result[column] = pd.Series(dtype=str)
        else:
            result[column] = result["filename"].map(generation[column]).fillna(
                "pending" if column == "generation_status" else "")
    return result


def generation_counts(manifest_dir):
    table = load_generation(manifest_dir)
    return {"total": len(table), **{status: int(table["generation_status"].eq(status).sum())
                                   for status in GENERATION_STATUSES}}


def update_generation(manifest_dir, paths, status=None, output_dir=None, error=""):
    """Register a queue or update outcomes; retain all prior scans and decisions."""
    manifest_dir = Path(manifest_dir)
    paths = [Path(path).resolve() for path in paths]
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Duplicate DICOM filenames in generation queue")
    if status is not None and status not in GENERATION_STATUSES:
        raise ValueError("Unknown generation status")
    table = load_generation(manifest_dir)
    records = {row["filename"]: row for row in table.to_dict("records")}
    now = datetime.now(timezone.utc).isoformat()
    for path in paths:
        previous = records.get(path.name, {})
        state = status or previous.get("generation_status", "pending")
        if status is None and state == "processing":
            state = "pending"  # An interrupted previous run is safe to retry.
        records[path.name] = {"filename": path.name, "generation_status": state,
                              "dicom_path": str(path),
                              "mip_path": str(Path(output_dir).resolve() / f"{path.stem}_qc_mip.png") if output_dir else previous.get("mip_path", ""),
                              "bscan_path": str(Path(output_dir).resolve() / f"{path.stem}_qc_bscan.npz") if output_dir else previous.get("bscan_path", ""),
                              "generation_error": str(error) if status else previous.get("generation_error", ""),
                              "generation_updated_at": now}
    _write_csv(pd.DataFrame(records.values(), columns=GENERATION_COLUMNS), manifest_dir / "mip_generation.csv")
    scans_path = manifest_dir / "scans.csv"
    if scans_path.exists():
        try:
            scans = pd.read_csv(scans_path)
        except pd.errors.EmptyDataError:
            return
        scans = apply_generation(scans, manifest_dir)
        _write_csv(scans, scans_path)
        if "segmentation_review_required" in scans.columns:
            write_segmentation_candidates(scans, manifest_dir)
        if "analysis_eligible" in scans.columns:
            eligible = scans["analysis_eligible"].map(lambda value: str(value).lower() == "true")
            _write_csv(scans.loc[eligible], manifest_dir / "scans_for_analysis.csv")
