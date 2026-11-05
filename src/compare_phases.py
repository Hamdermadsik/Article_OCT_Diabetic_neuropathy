"""How papillary blob count and slab flow change through the PORH protocol.

Each visit records repeated Baseline scans, one scan during cuff occlusion
(Isc), then a series of scans after release (post-occlusive reactive
hyperaemia, PORH). For every selected visit this analyses all of them and
reports, per site:

* the Baseline level and its spread across the repeats -- the change the
  measure shows when nothing changes physiologically;
* the value during occlusion;
* the PORH peak, its size relative to Baseline, and when it occurs.

Time is taken from each acquisition's recorded time, relative to the
occlusion scan.

Run from the repo root with the acquisition drive attached:

    python src/compare_phases.py --patients 10
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
    add_slab_arguments,
    analyse_scans,
    draw_patients,
    scan_metrics,
    slab_from_args,
)

SITES = ("Hand", "Foot")
PHASES = ("Baseline", "Isc", "PORH")
METRICS = ("blobs", "mean_slab_flow")


def select_visits(
    scans: pd.DataFrame,
    patients: pd.DataFrame,
    n_patients: int = 10,
    seed: int = 0,
    condition: str = "Regular",
) -> pd.DataFrame:
    """Every Baseline/Isc/PORH scan of one visit per (patient, site).

    Only patients with a usable Baseline-and-PORH visit on both a hand and a
    foot site are eligible, drawn evenly across groups as in the hand-vs-foot
    comparison. Each site uses that patient's earliest qualifying visit.
    """
    usable = scans[
        scans["phase"].isin(PHASES)
        & (scans["condition"] == condition)
        & ~scans["excluded"].astype(bool)
        & scans["protocol_area"].isin(SITES)
    ]
    phases_per_visit = usable.groupby(["patient_id", "protocol_area", "date"])["phase"].apply(set)
    complete = phases_per_visit[phases_per_visit.apply(lambda p: {"Baseline", "PORH"} <= p)]
    sites_per_patient = complete.reset_index().groupby("patient_id")["protocol_area"].apply(set)
    eligible = sorted(sites_per_patient[sites_per_patient.apply(lambda s: set(SITES) <= s)].index)
    if not eligible:
        return usable.iloc[0:0]

    chosen, group = draw_patients(eligible, patients, n_patients, seed)
    parts = []
    for patient_id in chosen:
        for site in SITES:
            dates = sorted(complete.loc[patient_id].loc[site].index)
            visit = usable[
                (usable["patient_id"] == patient_id)
                & (usable["protocol_area"] == site)
                & (usable["date"] == dates[0])
            ].sort_values("scan_number")
            parts.append(visit.assign(visit_id=f"{patient_id} | {site} | {dates[0]}"))
    selected = pd.concat(parts, ignore_index=True)
    selected["group"] = [group.get(p, "unknown") for p in selected["patient_id"]]
    return selected


def seconds_from_occlusion(visit: pd.DataFrame) -> pd.Series:
    """Each scan's time relative to the occlusion scan (or first PORH if none)."""
    times = pd.to_datetime(visit["acquisition_datetime"], errors="coerce")
    isc = times[visit["phase"] == "Isc"]
    reference = isc.iloc[0] if len(isc) and pd.notna(isc.iloc[0]) else times[visit["phase"] == "PORH"].min()
    return (times - reference).dt.total_seconds()


def summarise_visit(visit: pd.DataFrame) -> dict:
    """Baseline level and spread, occlusion value, and PORH peak for one visit."""
    baseline = visit[visit["phase"] == "Baseline"]
    isc = visit[visit["phase"] == "Isc"]
    porh = visit[visit["phase"] == "PORH"]
    first = visit.iloc[0]
    out = {
        "visit_id": first["visit_id"],
        "patient_id": first["patient_id"],
        "group": first["group"],
        "site": first["protocol_area"],
        "n_baseline": len(baseline),
        "n_porh": len(porh),
    }
    for metric in METRICS:
        base_mean = baseline[metric].mean()
        base_sd = baseline[metric].std(ddof=1) if len(baseline) > 1 else np.nan
        peak_row = porh.loc[porh[metric].idxmax()] if len(porh) else None
        out[f"{metric}_baseline"] = round(base_mean, 2)
        out[f"{metric}_baseline_sd"] = round(base_sd, 2)
        out[f"{metric}_isc"] = round(isc[metric].mean(), 2) if len(isc) else np.nan
        if peak_row is not None:
            out[f"{metric}_porh_peak"] = round(peak_row[metric], 2)
            out[f"{metric}_peak_ratio"] = round(peak_row[metric] / base_mean, 2) if base_mean else np.nan
            out[f"{metric}_peak_vs_baseline_sd"] = (
                round((peak_row[metric] - base_mean) / base_sd, 1) if base_sd else np.nan
            )
            out[f"{metric}_peak_time_s"] = peak_row["seconds"]
    return out


def draw_time_course(scans: pd.DataFrame, out_path: Path) -> None:
    """Each visit's blobs and slab flow over time, relative to its Baseline."""
    fig, axes = plt.subplots(len(METRICS), len(SITES), figsize=(13, 8), sharex=True)
    colours = {"Baseline": "tab:blue", "Isc": "tab:red", "PORH": "tab:orange"}
    for r, metric in enumerate(METRICS):
        for c, site in enumerate(SITES):
            ax = axes[r, c]
            for _, visit in scans[scans["protocol_area"] == site].groupby("visit_id"):
                base = visit.loc[visit["phase"] == "Baseline", metric].mean()
                if not base:
                    continue
                visit = visit.sort_values("seconds")
                ax.plot(visit["seconds"], visit[metric] / base, color="grey", lw=0.8, alpha=0.6)
                for phase, colour in colours.items():
                    pts = visit[visit["phase"] == phase]
                    ax.scatter(pts["seconds"], pts[metric] / base, s=12, color=colour, zorder=3)
            ax.axhline(1, color="k", lw=0.8, ls="--")
            ax.axvline(0, color="tab:red", lw=0.6, ls=":")
            ax.set_title(f"{site}: {metric.replace('_', ' ')} relative to Baseline mean", fontsize=10)
            if r == len(METRICS) - 1:
                ax.set_xlabel("seconds from the occlusion scan")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=p) for p, c in colours.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_path, dpi=90)
    plt.close(fig)


def draw_mip_strip(scans: pd.DataFrame, files: dict, out_path: Path) -> None:
    """Per visit: middle Baseline, occlusion, and PORH-peak MIPs side by side."""
    visits = list(scans.groupby("visit_id", sort=False))
    fig, axes = plt.subplots(len(visits), 3, figsize=(13, 2.6 * len(visits)))
    axes = np.atleast_2d(axes)
    for (visit_id, visit), row_axes in zip(visits, axes):
        baseline = visit[visit["phase"] == "Baseline"].sort_values("scan_number")
        porh = visit[visit["phase"] == "PORH"]
        picks = [
            ("Baseline (middle repeat)", baseline.iloc[len(baseline) // 2] if len(baseline) else None),
            ("Occlusion (Isc)", visit[visit["phase"] == "Isc"].iloc[0] if (visit["phase"] == "Isc").any() else None),
            ("PORH peak (slab flow)", porh.loc[porh["mean_slab_flow"].idxmax()] if len(porh) else None),
        ]
        vmax = None
        for ax, (label, row) in zip(row_axes, picks):
            ax.axis("off")
            if row is None:
                ax.set_title(f"{visit_id}\n{label}: none", fontsize=8)
                continue
            with np.load(files[row["filename"]]) as saved:
                mip = saved["mip"]
            # One brightness scale per visit, so phases are directly comparable.
            vmax = vmax or np.percentile(mip, 99.5)
            ax.imshow(mip, cmap="gray", aspect="auto", vmin=0, vmax=vmax)
            ax.set_title(f"{visit_id}\n{label} | {int(row['blobs'])} blobs | flow {row['mean_slab_flow']:.1f}",
                         fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=70)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    add_slab_arguments(parser)
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="default: outputs/porh_time_course/slab_<top>_<bottom>"
    )
    args = parser.parse_args()
    slab = slab_from_args(parser, args)
    out_dir = args.out_dir or Path("outputs") / "porh_time_course" / f"slab_{slab[0]}_{slab[1]}"

    manifest = project_root() / "data" / "manifest"
    scans = pd.read_csv(manifest / "scans.csv")
    if "acquisition_datetime" not in scans.columns:
        print("scans.csv has no acquisition times - re-run src/manifest/build_manifest.py with the drive attached.")
        return 1
    selected = select_visits(scans, pd.read_csv(manifest / "patients.csv"), args.patients, args.seed)
    if selected.empty:
        print("No patients have Baseline and PORH visits on both a hand and a foot site.")
        return 1
    print(
        f"{selected['patient_id'].nunique()} patients, {selected['visit_id'].nunique()} visits, "
        f"{len(selected)} scans | slab {slab[0]}..{slab[1]} px relative to the DEJ"
    )

    done = analyse_scans([row for _, row in selected.iterrows()], slab, args.recompute)
    if not done:
        print("Nothing processed.")
        return 1

    records, files = [], {}
    for row, result_file in done:
        with np.load(result_file) as result:
            records.append({**row.to_dict(), **scan_metrics(result)})
        files[row["filename"]] = result_file
    per_scan = pd.DataFrame(records)
    per_scan["seconds"] = pd.concat(
        [seconds_from_occlusion(v) for _, v in per_scan.groupby("visit_id", sort=False)]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["visit_id", "group", "phase", "scan_number", "acquisition_datetime", "seconds",
               "blobs", "mean_slab_flow", "flow_peak_px_below_dej", "filename"]
    per_scan[columns].to_csv(out_dir / "per_scan.csv", index=False)
    visits = pd.DataFrame([summarise_visit(v) for _, v in per_scan.groupby("visit_id", sort=False)])
    visits.to_csv(out_dir / "per_visit.csv", index=False)
    draw_time_course(per_scan, out_dir / "time_course.png")
    draw_mip_strip(per_scan, files, out_dir / "mip_baseline_isc_porh.png")

    pd.set_option("display.width", 200)
    show = ["visit_id", "blobs_baseline", "blobs_baseline_sd", "blobs_isc", "blobs_porh_peak",
            "blobs_peak_ratio", "mean_slab_flow_baseline", "mean_slab_flow_isc",
            "mean_slab_flow_porh_peak", "mean_slab_flow_peak_ratio", "mean_slab_flow_peak_time_s"]
    print()
    print(visits[show].to_string(index=False))
    print()
    for metric in METRICS:
        exceeds = (visits[f"{metric}_peak_vs_baseline_sd"] > 2).sum()
        print(f"{metric}: PORH peak more than 2 Baseline SDs above Baseline in {exceeds} of {len(visits)} visits")
    print(visits.groupby("site")[[f"{m}_peak_ratio" for m in METRICS] + ["mean_slab_flow_peak_time_s"]]
          .median().to_string())
    print(f"\nwritten to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
