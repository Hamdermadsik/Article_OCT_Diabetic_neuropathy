"""Per-scan papillary analysis with a shared on-disk result cache.

Analysing one acquisition (segmentation, papillary MIP, blob count, flow
depth profile) takes tens of seconds and needs the acquisition drive, so each
scan's results are saved once per slab setting and reused by every analysis
that needs that scan.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.transform import resize

from model_utils import default_model_path, load_pytorch_model, project_root
from papi import (
    EN_FACE_SIZE,
    SLAB_BOTTOM_PX,
    SLAB_TOP_PX,
    extract_flow,
    identify_and_plot_dots,
    import_dicom,
    papillary_dermis_mip,
    remove_epidermis_from_image,
    test_model_on_sample_image,
)

PROFILE_OFFSETS = np.arange(-20, 101)


def add_slab_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--slab-top",
        type=int,
        default=SLAB_TOP_PX,
        help="slab start in px below the DEJ (negative = above it), e.g. --slab-top=-5",
    )
    parser.add_argument("--slab-bottom", type=int, default=SLAB_BOTTOM_PX)
    parser.add_argument(
        "--recompute", action="store_true", help="re-analyse scans that already have saved results"
    )


def slab_from_args(parser: argparse.ArgumentParser, args) -> tuple[int, int]:
    slab = (args.slab_top, args.slab_bottom)
    if slab[0] >= slab[1]:
        parser.error("--slab-top must be above (smaller than) --slab-bottom")
    return slab


def draw_patients(eligible, patients: pd.DataFrame, n_patients: int, seed: int) -> tuple[list, dict]:
    """Pick patients evenly across diabetic, healthy and unrecorded groups.

    Returns the chosen patient IDs and a patient -> group lookup.
    """
    group = patients.dropna(subset=["group"]).groupby("patient_id")["group"].first()
    by_group = {
        name: sorted(p for p in eligible if group.get(p, "unknown") == name)
        for name in ("DP", "HP", "unknown")
    }
    rng = np.random.default_rng(seed)
    for members in by_group.values():
        rng.shuffle(members)
    chosen = []
    while len(chosen) < n_patients and any(by_group.values()):
        for members in by_group.values():
            if members and len(chosen) < n_patients:
                chosen.append(members.pop())
    return chosen, {p: group.get(p, "unknown") for p in eligible}


def cache_dir(slab: tuple[int, int]) -> Path:
    return Path("outputs") / "scan_cache" / f"slab_{slab[0]}_{slab[1]}"


def resolve_path(row, fallback_dir: Path) -> Path | None:
    """The scan on the drive, or a local copy of it if the drive is absent."""
    for candidate in (Path(row["filepath"]), fallback_dir / row["filename"]):
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def flow_depth_profile(volume, dej) -> np.ndarray:
    """Mean flow at each depth offset from the DEJ, over detected columns."""
    n_slices, height, width = volume.shape[:3]
    totals = np.zeros(len(PROFILE_OFFSETS))
    counts = np.zeros(len(PROFILE_OFFSETS))
    columns = np.arange(width)
    for i in range(n_slices):
        flow = extract_flow(volume[i])
        found = dej[i] >= 0
        for k, offset in enumerate(PROFILE_OFFSETS):
            rows = dej[i, found] + offset
            inside = (rows >= 0) & (rows < height)
            totals[k] += flow[rows[inside], columns[found][inside]].sum()
            counts[k] += inside.sum()
    return totals / np.maximum(counts, 1)


def needs_processing(result_file: Path, slab: tuple[int, int], recompute: bool) -> bool:
    """Whether a scan must be (re-)analysed rather than reusing saved results.

    Results record the slab that produced them, so a different slab never
    silently reuses them. Files from before the slab was recorded used the
    original 0-15 px slab.
    """
    if recompute or not result_file.exists():
        return True
    with np.load(result_file) as saved:
        saved_slab = tuple(saved["slab"]) if "slab" in saved else (0, 15)
    return saved_slab != tuple(slab)


def process_scan(path: Path, model, device, out_file: Path, slab: tuple[int, int]) -> None:
    """Analyse one acquisition and save everything the reports need to disk.

    Only small arrays are kept: holding a full volume (about 200 MB) per scan
    until a report is drawn exhausts memory after a dozen scans.
    """
    volume, _, _ = import_dicom(path)
    mip, dej, _ = papillary_dermis_mip(volume, model, device, slab=slab)
    dots, _ = identify_and_plot_dots(
        resize(mip, EN_FACE_SIZE, anti_aliasing=True, order=1),
        min_size=30,
        intensity_threshold=75,
        circularity_threshold=0.45,
        plot=False,
    )
    profile = flow_depth_profile(volume, dej)

    middle = volume.shape[0] // 2
    bscan = volume[middle].copy()  # a copy, so the volume itself can be freed
    del volume
    mask = test_model_on_sample_image(model, bscan, device)
    _, upper, lower = remove_epidermis_from_image(bscan, mask)
    found = mask.astype(bool).any(axis=0)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        mip=mip,
        profile=profile,
        bscan=bscan,
        dej_line=lower[:, 0],
        found=found,
        blobs=len(dots),
        epidermis_px=int(np.median((lower[:, 0] - upper[:, 0])[found])),
        no_dej_pct=100 * float((dej < 0).mean()),
        slab=np.array(slab),
    )


def scan_metrics(result) -> dict:
    """The per-scan numbers reports tabulate, from a saved result."""
    return {
        "blobs": int(result["blobs"]),
        "mean_slab_flow": round(float(result["mip"].mean()), 2),
        "flow_peak_px_below_dej": int(PROFILE_OFFSETS[np.argmax(result["profile"])]),
        "epidermis_px": int(result["epidermis_px"]),
        "no_dej_pct": round(float(result["no_dej_pct"]), 2),
    }


def analyse_scans(rows, slab: tuple[int, int], recompute: bool = False):
    """Ensure every scan has saved results; return (row, result_file) pairs.

    Scans that cannot be found (drive not attached) are reported and left out.
    The model is only loaded if something actually needs processing.
    """
    repo = project_root()
    folder = cache_dir(slab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None

    done = []
    for row in rows:
        result_file = folder / f"{Path(row['filename']).stem}.npz"
        label = f"{row['patient_id']} {row['bodypart']} {row.get('phase', '')}: {row['filename']}"
        if not needs_processing(result_file, slab, recompute):
            print(f"  {label} (cached)")
        else:
            path = resolve_path(row, repo / "data" / "dicom")
            if path is None:
                print(f"  ! {row['filename']}: not found - is the drive attached?")
                continue
            print(f"  {label}")
            if model is None:
                model = load_pytorch_model(default_model_path(), device)
            process_scan(path, model, device, result_file, slab)
        done.append((row, result_file))
    return done
