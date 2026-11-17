"""Figures and numbers for the T2DM vs control PORH report.

Reads what `compare_phases.py --study-groups` wrote (per_visit.csv,
per_scan.csv) plus the per-scan cache, so it needs neither the drive nor the
model. Run from the repo root:

    python src/report_study_groups.py

Figures go to docs/figures/porh_study_groups/ so the report can show them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from scan_analysis import PROFILE_OFFSETS, cache_dir

GROUPS = ["Control", "T2DM"]
COLOURS = {"Control": "#2a78d6", "T2DM": "#eb6834"}
SITES = ["Hand", "Foot"]
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e4"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    }
)


def load(results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    visits = pd.read_csv(results / "per_visit.csv")
    scans = pd.read_csv(results / "per_scan.csv")
    parts = scans["visit_id"].str.split(" | ", regex=False)
    scans["patient_id"] = parts.str[0]
    scans["site"] = parts.str[1]
    scans["october"] = scans["cohort"].eq("october_2020")
    visits["october"] = visits["session"].eq("october_2020")
    return visits, scans


def p_value(frame: pd.DataFrame, column: str) -> float:
    a = frame.loc[frame["study_group"] == "T2DM", column].dropna()
    b = frame.loc[frame["study_group"] == "Control", column].dropna()
    return float(mannwhitneyu(a, b).pvalue)


def dot_plot(visits: pd.DataFrame, column: str, ylabel: str, title: str, out: Path, ref_line=None) -> None:
    """One dot per participant, split by group, with the group median as a bar."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, site in zip(axes, SITES):
        frame = visits[visits["site"] == site]
        for x, group in enumerate(GROUPS):
            g = frame[frame["study_group"] == group]
            xs = x + rng.uniform(-0.12, 0.12, len(g))
            for october, marker_face in ((False, COLOURS[group]), (True, "white")):
                sel = g["october"].to_numpy() == october
                ax.scatter(xs[sel], g[column].to_numpy()[sel], s=42, facecolor=marker_face,
                           edgecolor=COLOURS[group], linewidth=1.6, zorder=3)
            ax.hlines(g[column].median(), x - 0.28, x + 0.28, color=INK, linewidth=2, zorder=4)
        if ref_line is not None:
            ax.axhline(ref_line, color=MUTED, linestyle=":", linewidth=1)
        ax.set_xticks([0, 1], [f"{g}\n(n={(frame['study_group'] == g).sum()})" for g in GROUPS])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"{site}   (p = {p_value(frame, column):.3f})", color=INK, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.text(0.5, 0.01, "Bar = group median.  Hollow dots = October session.  p = Mann-Whitney U, two-sided.",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=120)
    plt.close(fig)


def protocol_position(scans: pd.DataFrame) -> pd.DataFrame:
    """Label each scan by its place in the protocol: B-n..B-1, Isc, P1..Pn."""
    rows = []
    for _, visit in scans.groupby("visit_id"):
        visit = visit.sort_values("seconds").copy()
        base = visit["mean_slab_flow"][visit["phase"] == "Baseline"].mean()
        visit["relative_flow"] = visit["mean_slab_flow"] / base
        b = visit[visit["phase"] == "Baseline"]
        p = visit[visit["phase"] == "PORH"]
        visit.loc[b.index, "position"] = [f"B{i}" for i in range(-len(b), 0)]
        visit.loc[visit["phase"] == "Isc", "position"] = "Isc"
        visit.loc[p.index, "position"] = [f"P{i}" for i in range(1, len(p) + 1)]
        rows.append(visit)
    return pd.concat(rows)


def time_course(scans: pd.DataFrame, out: Path) -> None:
    """Median (and interquartile band) slab flow relative to each visit's Baseline."""
    scans = protocol_position(scans)
    order = [f"B{i}" for i in range(-5, 0)] + ["Isc"] + [f"P{i}" for i in range(1, 7)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, site in zip(axes, SITES):
        for group in GROUPS:
            g = scans[(scans["site"] == site) & (scans["study_group"] == group)]
            stats = (
                g.groupby("position")
                .agg(t=("seconds", "median"), med=("relative_flow", "median"),
                     q1=("relative_flow", lambda s: s.quantile(0.25)),
                     q3=("relative_flow", lambda s: s.quantile(0.75)), n=("relative_flow", "size"))
                .reindex(order)
                .dropna()
            )
            stats = stats[stats["n"] >= 5]  # a median of fewer visits is noise
            ax.fill_between(stats["t"], stats["q1"], stats["q3"], color=COLOURS[group], alpha=0.15, linewidth=0)
            ax.plot(stats["t"], stats["med"], color=COLOURS[group], linewidth=2, marker="o", markersize=5,
                    label=group)
        ax.axhline(1, color=MUTED, linestyle=":", linewidth=1)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.text(4, 2.35, "cuff\n(Isc)", color=MUTED, fontsize=8, va="top")
        ax.set_title(site, color=INK, fontsize=10)
        ax.set_xlabel("seconds from the occlusion scan")
    axes[0].set_ylabel("slab flow / Baseline mean")
    axes[0].set_ylim(0.4, 2.4)
    axes[0].legend(frameon=False, loc="upper left")
    fig.text(0.5, 0.01, "Line = median over visits, band = interquartile range; each point is one scan position "
             "in the protocol, placed at its median time.", ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=120)
    plt.close(fig)


def cached(folder: Path, filename: str):
    return np.load(folder / f"{Path(filename).stem}.npz")


def middle_baseline(scans: pd.DataFrame, patient: str, site: str) -> pd.Series:
    b = scans[(scans["patient_id"] == patient) & (scans["site"] == site) & (scans["phase"] == "Baseline")]
    b = b.sort_values("scan_number")
    return b.iloc[len(b) // 2]


def foot_extremes(visits: pd.DataFrame, scans: pd.DataFrame, folder: Path, out: Path, n: int = 3) -> list[str]:
    """Foot Baseline MIPs and B-scans: the lowest-flow controls vs the highest-flow T2DM."""
    foot = visits[visits["site"] == "Foot"].sort_values("mean_slab_flow_baseline")
    picks = list(foot[foot["study_group"] == "Control"].head(n)["patient_id"]) + list(
        foot[foot["study_group"] == "T2DM"].tail(n)["patient_id"]
    )
    rows = [middle_baseline(scans, p, "Foot") for p in picks]
    data = []
    for r in rows:
        with cached(folder, r["filename"]) as d:
            data.append({k: d[k].copy() for k in ("mip", "bscan", "dej_line", "found", "no_dej_pct")})
    vmax = np.percentile(np.concatenate([d["mip"].ravel() for d in data]), 99.5)

    fig, axes = plt.subplots(2, len(rows), figsize=(2.6 * len(rows), 5.2))
    for col, (r, d) in enumerate(zip(rows, data)):
        group = r["study_group"]
        top, bottom = axes[0, col], axes[1, col]
        top.imshow(d["mip"], cmap="gray", aspect="auto", vmin=0, vmax=vmax)
        top.set_title(f"{r['patient_id'].replace('MAY-', '')} ({group})\nslab flow {r['mean_slab_flow']:.1f}",
                      fontsize=9, color=COLOURS[group])
        bottom.imshow(d["bscan"], aspect="auto")
        x = np.arange(len(d["dej_line"]))
        dej = np.where(d["found"], d["dej_line"], np.nan)
        bottom.plot(x, dej, color="#00e5ff", linewidth=0.8)
        bottom.set_title(f"middle B-scan, DEJ in cyan\nno DEJ found: {float(d['no_dej_pct']):.0f}% of columns",
                         fontsize=8, color=INK)
        for ax in (top, bottom):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    fig.suptitle("Foot, middle Baseline scan: 3 lowest-flow controls vs 3 highest-flow T2DM "
                 "(MIPs on one shared brightness scale)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return picks


def hand_examples(visits: pd.DataFrame, scans: pd.DataFrame, folder: Path, out: Path) -> list[str]:
    """Baseline, occlusion and PORH peak for one typical participant per group."""
    hand = visits[visits["site"] == "Hand"]
    picks = []
    for group in GROUPS:
        g = hand[hand["study_group"] == group]
        median = g["mean_slab_flow_peak_ratio"].median()
        picks.append(g.loc[(g["mean_slab_flow_peak_ratio"] - median).abs().idxmin(), "patient_id"])

    fig, axes = plt.subplots(len(picks), 3, figsize=(11, 2.9 * len(picks)))
    for row_axes, patient in zip(axes, picks):
        v = scans[(scans["patient_id"] == patient) & (scans["site"] == "Hand")]
        porh = v[v["phase"] == "PORH"]
        chosen = [
            ("Baseline", middle_baseline(scans, patient, "Hand")),
            ("Occlusion (Isc)", v[v["phase"] == "Isc"].iloc[0]),
            ("PORH peak", porh.loc[porh["mean_slab_flow"].idxmax()]),
        ]
        mips = []
        for _, r in chosen:
            with cached(folder, r["filename"]) as d:
                mips.append(d["mip"].copy())
        vmax = np.percentile(np.concatenate([m.ravel() for m in mips]), 99.5)
        group = v["study_group"].iloc[0]
        ratio = hand.loc[hand["patient_id"] == patient, "mean_slab_flow_peak_ratio"].iloc[0]
        for ax, (label, r), mip in zip(row_axes, chosen, mips):
            ax.imshow(mip, cmap="gray", aspect="auto", vmin=0, vmax=vmax)
            ax.set_title(f"{patient.replace('MAY-', '')} ({group}, peak ×{ratio:.2f}) - {label}\n"
                         f"slab flow {r['mean_slab_flow']:.1f}, {int(r['blobs'])} blobs",
                         fontsize=9, color=COLOURS[group] if label == "Baseline" else INK)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    fig.suptitle("Hand: a typical PORH response in each group (participant closest to the group median ratio; "
                 "one brightness scale per row)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return picks


def depth_profiles(scans: pd.DataFrame, folder: Path, out: Path) -> pd.DataFrame:
    """Median Baseline flow-vs-depth profile per group and site."""
    base = scans[scans["phase"] == "Baseline"]
    records = []
    for _, r in base.iterrows():
        with cached(folder, r["filename"]) as d:
            records.append((r["visit_id"], r["site"], r["study_group"], d["profile"].copy(),
                            float(d["epidermis_px"]), float(d["no_dej_pct"])))
    per_visit = (
        pd.DataFrame(records, columns=["visit_id", "site", "study_group", "profile", "epidermis_px", "no_dej_pct"])
        .groupby(["visit_id", "site", "study_group"])
        .agg(profile=("profile", lambda s: np.nanmean(np.stack(s.to_list()), axis=0)),
             epidermis_px=("epidermis_px", "median"), no_dej_pct=("no_dej_pct", "median"))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True, sharex=True)
    depth = np.asarray(PROFILE_OFFSETS)
    for ax, site in zip(axes, SITES):
        for group in GROUPS:
            stack = np.stack(per_visit[(per_visit["site"] == site) & (per_visit["study_group"] == group)]["profile"])
            q1, med, q3 = np.nanpercentile(stack, [25, 50, 75], axis=0)
            ax.fill_betweenx(depth, q1, q3, color=COLOURS[group], alpha=0.15, linewidth=0)
            ax.plot(med, depth, color=COLOURS[group], linewidth=2, label=group)
        ax.axhspan(-5, 15, color=MUTED, alpha=0.12, linewidth=0)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.text(1, 5, "slab", color=MUTED, fontsize=8, va="center")
        ax.set_title(site, color=INK, fontsize=10)
        ax.set_xlabel("mean flow at that depth")
    axes[0].invert_yaxis()
    axes[0].set_ylabel("pixels below the DEJ (negative = above)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Baseline flow vs depth (median over visits, band = interquartile range)", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return per_visit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/porh_time_course/study_groups/slab_-5_15"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures/porh_study_groups"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    folder = cache_dir((-5, 15))

    visits, scans = load(args.results)
    dot_plot(visits, "mean_slab_flow_baseline", "mean slab flow at Baseline",
             "Baseline slab flow per participant", args.out_dir / "baseline_flow.png")
    dot_plot(visits, "mean_slab_flow_peak_ratio", "PORH peak / Baseline (slab flow)",
             "PORH response per participant", args.out_dir / "porh_peak_ratio.png", ref_line=1)
    time_course(scans, args.out_dir / "time_course.png")
    lows_highs = foot_extremes(visits, scans, folder, args.out_dir / "foot_baseline_extremes.png")
    examples = hand_examples(visits, scans, folder, args.out_dir / "hand_porh_examples.png")
    quality = depth_profiles(scans, folder, args.out_dir / "depth_profiles.png")

    print("foot extremes:", lows_highs)
    print("hand examples:", examples)
    print("\nBaseline epidermis thickness and DEJ coverage, medians:")
    print(quality.groupby(["site", "study_group"])[["epidermis_px", "no_dej_pct"]].median().round(1))
    for site in SITES:
        q = quality[quality["site"] == site]
        print(f"  {site} epidermis p = {p_value(q, 'epidermis_px'):.3f}")

    print("\nMann-Whitney p, T2DM vs Control (all | May only):")
    for column in ("mean_slab_flow_baseline", "blobs_baseline", "mean_slab_flow_peak_ratio",
                   "blobs_peak_ratio", "mean_slab_flow_peak_time_s"):
        for site in SITES:
            f = visits[visits["site"] == site]
            m = f[~f["october"]]
            med = f.groupby("study_group")[column].median()
            print(f"  {column:28} {site}: T2DM {med['T2DM']:7.2f} Control {med['Control']:7.2f}  "
                  f"p {p_value(f, column):.3f} | {p_value(m, column):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
