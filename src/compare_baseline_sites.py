"""Compare papillary MIPs between hand and foot on matched scans.

Picks patients scanned on both a hand and a foot site in the same phase, takes
one representative acquisition per (patient, site), and writes per-scan MIPs,
a side-by-side sheet and a summary table. Comparing within one phase matters:
blood flow, and so the number of visible loops, differs strongly between
Baseline, occlusion (Isc) and post-occlusion hyperaemia (PORH).

Run from the repo root with the acquisition drive attached:

    python src/compare_baseline_sites.py --patients 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

SITES = ("Hand", "Foot")
PROFILE_OFFSETS = np.arange(-20, 101)


def select_scans(
    scans: pd.DataFrame,
    patients: pd.DataFrame,
    phase: str = "Baseline",
    condition: str = "Regular",
    n_patients: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """One usable acquisition per (patient, site) for patients with both sites.

    Patients are drawn evenly across diabetic, healthy and unrecorded groups.
    Each site is taken from a single visit, and from the repeated acquisitions
    of that visit the middle one is used.
    """
    usable = scans[
        (scans["phase"] == phase)
        & (scans["condition"] == condition)
        & ~scans["excluded"].astype(bool)
        & scans["protocol_area"].isin(SITES)
    ]
    sites = usable.groupby("patient_id")["protocol_area"].apply(set)
    eligible = sorted(sites[sites.apply(lambda s: set(SITES) <= s)].index)
    if not eligible:
        return usable.iloc[0:0]

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

    picks = []
    for patient_id in chosen:
        mine = usable[usable["patient_id"] == patient_id]
        visits = mine.groupby("date")["protocol_area"].apply(set)
        complete = visits[visits.apply(lambda s: set(SITES) <= s)].index
        for site in SITES:
            candidates = mine[mine["protocol_area"] == site]
            if len(complete):
                candidates = candidates[candidates["date"] == sorted(complete)[0]]
            else:
                candidates = candidates[candidates["date"] == sorted(candidates["date"])[0]]
            ordered = candidates.sort_values("scan_number")
            picks.append(ordered.iloc[len(ordered) // 2])
    selected = pd.DataFrame(picks).reset_index(drop=True)
    selected["group"] = [group.get(p, "unknown") for p in selected["patient_id"]]
    return selected


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


def process_scan(row, path: Path, model, device, out_file: Path) -> None:
    """Analyse one acquisition and save everything the report needs to disk.

    Only small arrays are kept: holding a full volume (about 200 MB) per scan
    until the report is drawn exhausts memory after a dozen scans.
    """
    volume, _, _ = import_dicom(path)
    mip, dej, _ = papillary_dermis_mip(volume, model, device)
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
    )


def summarise(row, result) -> dict:
    return {
        "patient_id": row["patient_id"],
        "group": row["group"],
        "site": row["protocol_area"],
        "bodypart": row["bodypart"],
        "filename": row["filename"],
        "blobs": int(result["blobs"]),
        "mean_slab_flow": round(float(result["mip"].mean()), 2),
        "flow_peak_px_below_dej": int(PROFILE_OFFSETS[np.argmax(result["profile"])]),
        "epidermis_px": int(result["epidermis_px"]),
        "no_dej_pct": round(float(result["no_dej_pct"]), 2),
    }


def draw_sheet(rows, result_files, out_path: Path) -> None:
    """One row per scan: cross-section, MIP, and flow-vs-depth profile."""
    fig, axes = plt.subplots(
        len(rows), 3, figsize=(20, 3.4 * len(rows)), gridspec_kw={"width_ratios": [2.4, 1, 1]}
    )
    axes = np.atleast_2d(axes)
    for row, result_file, (ax_b, ax_m, ax_p) in zip(rows, result_files, axes):
        with np.load(result_file) as result:
            bscan, line, found = result["bscan"], result["dej_line"], result["found"]
            mip, profile, blobs = result["mip"], result["profile"], int(result["blobs"])
        cols = np.arange(bscan.shape[1])
        ax_b.imshow(bscan, aspect="auto")
        ax_b.plot(cols[found], line[found] + SLAB_TOP_PX, "lime", lw=1)
        ax_b.plot(cols[found], line[found] + SLAB_BOTTOM_PX, "m", lw=1)
        ax_b.set_ylim(260, 0)
        ax_b.set_title(
            f"{row['patient_id']} ({row['group']}) | {row['bodypart']} | middle slice", fontsize=9
        )
        ax_b.axis("off")
        ax_m.imshow(mip, cmap="gray", aspect="auto", vmax=np.percentile(mip, 99.5))
        ax_m.set_title(f"MIP - {blobs} blobs", fontsize=9)
        ax_m.axis("off")
        ax_p.plot(PROFILE_OFFSETS, profile, "r")
        ax_p.axvspan(SLAB_TOP_PX, SLAB_BOTTOM_PX, color="m", alpha=0.15)
        ax_p.set_title("flow vs depth below DEJ (px); shaded = slab", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=70)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patients", type=int, default=10)
    parser.add_argument("--phase", default="Baseline")
    parser.add_argument("--condition", default="Regular")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "baseline_sites")
    parser.add_argument(
        "--recompute", action="store_true", help="re-analyse scans that already have saved results"
    )
    args = parser.parse_args()

    repo = project_root()
    manifest = repo / "data" / "manifest"
    scans = pd.read_csv(manifest / "scans.csv")
    patients = pd.read_csv(manifest / "patients.csv")
    selected = select_scans(
        scans, patients, args.phase, args.condition, args.patients, args.seed
    )
    if selected.empty:
        print("No patients have usable scans on both a hand and a foot site for that phase.")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{selected['patient_id'].nunique()} patients, {len(selected)} scans selected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pytorch_model(default_model_path(), device)

    done_rows, done_files = [], []
    for _, row in selected.iterrows():
        result_file = args.out_dir / f"{Path(row['filename']).stem}.npz"
        label = f"{row['patient_id']} {row['bodypart']}: {row['filename']}"
        if result_file.exists() and not args.recompute:
            print(f"  {label} (already done)")
        else:
            path = resolve_path(row, repo / "data" / "dicom")
            if path is None:
                print(f"  ! {row['filename']}: not found - is the drive attached?")
                continue
            print(f"  {label}")
            process_scan(row, path, model, device, result_file)
        done_rows.append(row)
        done_files.append(result_file)

    if not done_rows:
        print("Nothing processed.")
        return 1

    summary = []
    for row, result_file in zip(done_rows, done_files):
        with np.load(result_file) as result:
            summary.append(summarise(row, result))
    summary = pd.DataFrame(summary)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    draw_sheet(done_rows, done_files, args.out_dir / "comparison_sheet.png")

    print()
    print(summary.drop(columns=["filename"]).to_string(index=False))
    print()
    print(summary.groupby("site")[["blobs", "mean_slab_flow", "flow_peak_px_below_dej", "epidermis_px"]]
          .median().to_string())
    print(f"\nwritten to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
