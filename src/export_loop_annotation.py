"""Export papillary MIPs for counting capillary loops by hand in Fiji.

Picks hand visits from the study-group PORH run (a Baseline and a PORH-peak
MIP per participant, balanced over T2DM and controls), adds a few repeats to
measure how consistently loops are marked, and writes them under neutral names
in random order, so the person marking them doesn't know whose scan, which
group or which phase they are looking at. The key that undoes the blinding goes
into a separate file.

Each image is a square tile (default 400 px, about 1.5 mm) cut from exactly
what the loop detector sees: the MIP resized to the square en-face grid and
stretched to 0-255. A whole MIP holds hundreds of loops; a tile keeps marking
manageable. The tile's position is recorded in the key, so marked coordinates
can be compared directly with the detector's output on the same tile. Repeats
show the same tile again.

Run from the repo root (needs only the scan cache, not the drive):

    python src/export_loop_annotation.py

See docs/loop_annotation.md for how to mark the images in Fiji.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize

from papi import EN_FACE_SIZE, identify_and_plot_dots
from scan_analysis import cache_dir

REPO = Path(__file__).resolve().parents[1]
MACRO = REPO / "tools" / "fiji" / "loop_annotation.ijm"
DETECTOR = {"min_size": 30, "intensity_threshold": 75, "circularity_threshold": 0.45}


def detector_view(mip: np.ndarray) -> np.ndarray:
    """The 8-bit en-face image the loop detector thresholds."""
    square = resize(mip, EN_FACE_SIZE, anti_aliasing=True, order=1)
    return cv2.normalize(square, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def detector_loops(image: np.ndarray) -> list[tuple[float, float]]:
    """Loop centres (x, y) the detector finds, with the parameters of the runs."""
    centroids, _ = identify_and_plot_dots(image, plot=False, **DETECTOR)
    return [(float(c), float(r)) for r, c in centroids]


def select_images(scans: pd.DataFrame, per_group: int, seed: int, site: str = "Hand") -> pd.DataFrame:
    """For `per_group` participants per study group: middle Baseline and PORH-peak scan."""
    parts = scans["visit_id"].str.split(" | ", regex=False)
    scans = scans.assign(patient_id=parts.str[0], site=parts.str[1])
    scans = scans[scans["site"] == site]
    rng = np.random.default_rng(seed)
    picks = []
    for group, members in scans.groupby("study_group"):
        patients = sorted(members["patient_id"].unique())
        chosen = rng.choice(patients, size=min(per_group, len(patients)), replace=False)
        for patient in sorted(chosen):
            visit = members[members["patient_id"] == patient]
            baseline = visit[visit["phase"] == "Baseline"].sort_values("scan_number")
            porh = visit[visit["phase"] == "PORH"]
            if baseline.empty or porh.empty:
                continue
            picks.append(baseline.iloc[len(baseline) // 2].to_frame().T.assign(phase_pick="Baseline"))
            picks.append(porh.loc[[porh["mean_slab_flow"].astype(float).idxmax()]].assign(phase_pick="PORH peak"))
    return pd.concat(picks, ignore_index=True)


def choose_tiles(n: int, tile: int, seed: int) -> np.ndarray:
    """Random top-left corners (x, y) of tiles that fit inside the en-face image."""
    rng = np.random.default_rng(seed + 2)
    height, width = EN_FACE_SIZE
    return np.column_stack([rng.integers(0, width - tile + 1, n), rng.integers(0, height - tile + 1, n)])


def cut_tile(view: np.ndarray, x0: int, y0: int, tile: int) -> np.ndarray:
    return view[y0:y0 + tile, x0:x0 + tile]


def blind(selected: pd.DataFrame, repeats: int, seed: int, tile: int = 400) -> pd.DataFrame:
    """Pick a tile per scan, add repeats, shuffle, and give every image a neutral name."""
    corners = choose_tiles(len(selected), tile, seed)
    selected = selected.assign(tile_x0=corners[:, 0], tile_y0=corners[:, 1], tile_size=tile)
    rng = np.random.default_rng(seed + 1)
    repeat_rows = rng.choice(len(selected), size=min(repeats, len(selected)), replace=False)
    copies = selected.iloc[repeat_rows].assign(is_repeat=True)
    images = pd.concat([selected.assign(is_repeat=False), copies], ignore_index=True)
    images = images.iloc[rng.permutation(len(images))].reset_index(drop=True)
    width = max(2, len(str(len(images))))
    images.insert(0, "image", [f"loop_{i + 1:0{width}d}" for i in range(len(images))])
    first = images[~images["is_repeat"]].set_index("filename")["image"]
    images["repeat_of"] = np.where(images["is_repeat"], images["filename"].map(first), "")
    return images


def export(images: pd.DataFrame, folder: Path, out_dir: Path) -> pd.DataFrame:
    """Write the images and the key; return the key."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "key").mkdir(exist_ok=True)
    counts = []
    for _, row in images.iterrows():
        with np.load(folder / f"{Path(row['filename']).stem}.npz") as cached:
            view = detector_view(cached["mip"])
        x0, y0, size = int(row["tile_x0"]), int(row["tile_y0"]), int(row["tile_size"])
        Image.fromarray(cut_tile(view, x0, y0, size)).save(out_dir / f"{row['image']}.tif")
        # Detector run on the whole image (as in the analyses), loops counted inside the tile.
        counts.append(sum(x0 <= x < x0 + size and y0 <= y < y0 + size for x, y in detector_loops(view)))
    key = images.assign(detector_count=counts)[
        ["image", "filename", "patient_id", "study_group", "site", "phase_pick", "mean_slab_flow",
         "detector_count", "tile_x0", "tile_y0", "tile_size", "is_repeat", "repeat_of"]
    ].rename(columns={"phase_pick": "phase"})
    key.to_csv(out_dir / "key" / "key.csv", index=False)
    if MACRO.exists():
        shutil.copy(MACRO, out_dir / MACRO.name)
    return key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path,
                        default=REPO / "outputs/porh_time_course/study_groups/slab_-5_15/per_scan.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO / "data/annotation/loops")
    parser.add_argument("--per-group", type=int, default=5, help="participants per study group")
    parser.add_argument("--repeats", type=int, default=2, help="images shown twice, to measure consistency")
    parser.add_argument("--site", choices=["Hand", "Foot"], default="Hand")
    parser.add_argument("--tile", type=int, default=400, help="tile side in en-face pixels (about 3.8 um each)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite an existing export")
    args = parser.parse_args()

    if any(args.out_dir.glob("*_loops.csv")) and not args.force:
        print(f"{args.out_dir} already holds marked loops; refusing to overwrite (use --force).")
        return 1

    scans = pd.read_csv(args.results)
    images = blind(select_images(scans, args.per_group, args.seed, args.site), args.repeats, args.seed, args.tile)
    key = export(images, REPO / cache_dir((-5, 15)), args.out_dir)
    print(f"{len(key)} images ({key['is_repeat'].sum()} repeats) written to {args.out_dir}")
    print(key.groupby(["study_group", "phase"]).size().to_string())
    print(f"key (don't open while marking): {args.out_dir / 'key' / 'key.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
