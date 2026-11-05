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

from model_utils import project_root
from scan_analysis import (
    PROFILE_OFFSETS,
    add_slab_arguments,
    analyse_scans,
    draw_patients,
    scan_metrics,
    slab_from_args,
)

SITES = ("Hand", "Foot")


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

    chosen, group = draw_patients(eligible, patients, n_patients, seed)

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


def summarise(row, result) -> dict:
    return {
        "patient_id": row["patient_id"],
        "group": row["group"],
        "site": row["protocol_area"],
        "bodypart": row["bodypart"],
        "filename": row["filename"],
        **scan_metrics(result),
    }


def draw_sheet(rows, result_files, out_path: Path, slab: tuple[int, int]) -> None:
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
        ax_b.plot(cols[found], line[found], "lime", lw=1)
        ax_b.plot(cols[found], line[found] + slab[0], "m", lw=0.8, ls="--")
        ax_b.plot(cols[found], line[found] + slab[1], "m", lw=1)
        ax_b.set_ylim(260, 0)
        ax_b.set_title(
            f"{row['patient_id']} ({row['group']}) | {row['bodypart']} | middle slice", fontsize=9
        )
        ax_b.axis("off")
        ax_m.imshow(mip, cmap="gray", aspect="auto", vmax=np.percentile(mip, 99.5))
        ax_m.set_title(f"MIP - {blobs} blobs", fontsize=9)
        ax_m.axis("off")
        ax_p.plot(PROFILE_OFFSETS, profile, "r")
        ax_p.axvspan(slab[0], slab[1], color="m", alpha=0.15)
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
    add_slab_arguments(parser)
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="default: outputs/baseline_sites/slab_<top>_<bottom>"
    )
    args = parser.parse_args()
    slab = slab_from_args(parser, args)
    out_dir = args.out_dir or Path("outputs") / "baseline_sites" / f"slab_{slab[0]}_{slab[1]}"

    manifest = project_root() / "data" / "manifest"
    selected = select_scans(
        pd.read_csv(manifest / "scans.csv"),
        pd.read_csv(manifest / "patients.csv"),
        args.phase,
        args.condition,
        args.patients,
        args.seed,
    )
    if selected.empty:
        print("No patients have usable scans on both a hand and a foot site for that phase.")
        return 1
    print(
        f"{selected['patient_id'].nunique()} patients, {len(selected)} scans selected | "
        f"slab {slab[0]}..{slab[1]} px relative to the DEJ"
    )

    done = analyse_scans([row for _, row in selected.iterrows()], slab, args.recompute)
    if not done:
        print("Nothing processed.")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for row, result_file in done:
        with np.load(result_file) as result:
            summary.append(summarise(row, result))
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "summary.csv", index=False)
    draw_sheet([r for r, _ in done], [f for _, f in done], out_dir / "comparison_sheet.png", slab)

    print()
    print(summary.drop(columns=["filename"]).to_string(index=False))
    print()
    print(summary.groupby("site")[["blobs", "mean_slab_flow", "flow_peak_px_below_dej", "epidermis_px"]]
          .median().to_string())
    print(f"\nwritten to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
