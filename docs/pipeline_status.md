# Pipeline status and findings

*Snapshot as of 2026-11-12. Numbers come from the runs described below; re-running
on more patients will change them.*

## What the pipeline does

The goal is to measure the papillary capillary loops in OCTA scans of hands and
feet, so diabetic and healthy participants can be compared. The steps are:

```
external drive (DICOM volumes) ─┐
Excel scan logs + demographics ─┴─> manifest ──> per-scan analysis ──> comparisons
                                    (who/what       (segment, slab,      (hand vs foot,
                                     each scan is)   MIP, flow, loops)    Baseline vs PORH)
```

### 1. The data

- **3,175 scans** on the external drive: May 2021 session (2,620) and October 2020
  session (555).
- Each `.dcm` file is one acquisition: **120 B-scans of 460×1174 pixels** (460×1129
  for October), stored as an RGB *picture*. The structural OCT image is grey, and
  blood flow is painted on as a **red tint**. The red channel is therefore grey plus
  flow; the green channel is the plain grey image.
- The protocol per visit and body site is usually **5 Baseline scans, 1 scan during
  cuff occlusion (Isc), then about 6 scans after release (PORH)**, roughly 47 s
  apart. May participants also repeated it after a flavanol drink.

### 2. The manifest — `src/manifest/`

Builds one source of truth for which patient, site and phase every scan belongs
to. Run `python src/manifest/build_manifest.py`, then
`python src/manifest/validate_manifest.py`.

| Piece | What it does |
|---|---|
| `parse_may.py`, `parse_october.py` | Read the two differently shaped scan logs into one format |
| `demographics.py` | Age/sex from the demographics sheet (May) and the October log |
| `configs/manifest_decisions.yaml` | **Your decisions**: which cross-session records are the same person, and ID corrections (e.g. `DP-12-RS` → `DP-12-SR`) |
| `dicom_index.py` | Reads only file headers (fast): geometry, pixel offset, acquisition time |
| `matching.py` | Links each file to the scan log; refuses to guess when ambiguous |

Outputs in `data/manifest/`:

- **`patients.csv`**: 67 patients (70 records, since 3 people were scanned in both
  sessions), all with age and sex.
  - **12 diabetic (DP), 50 healthy (HP), 5 without a recorded group** (EA, HA and
    MA aged 13–16, PM, and the 26-year-old October RM). These 5 can't be placed
    in the study, so they are **excluded from every analysis** (recorded in
    `configs/manifest_decisions.yaml`).
  - October's log never records the group, so October participants take it from
    the `HP-…`/`DP-…` IDs in the demographics sheet
    (`2021-05-OCTAAgeing-hand.xlsx`), matched on initials, age and sex.
  - A `study_group` column marks the article's comparison groups: **11 T2DM and
    11 controls**, age-matched at group level (mean 60.5 vs 60.4 years). They are
    recorded in `configs/manifest_decisions.yaml`. `DP-11-DC` is diabetic but in
    neither group.
- **`scans.csv`**: all 3,175 scans, with an `excluded` / `exclusion_reason` column.
  2,830 are usable; 223 have no scan-log entry, 98 belong to the 5 excluded
  patients, 14 have an ambiguous phase, and 10 are duplicate copies.
- **Problem reports** for things to check by hand.
- **`drive_index.csv`**: a saved copy of the drive's headers, so the manifest can be
  rebuilt without the drive attached.

Problems the manifest found and fixed in the earlier spreadsheet matching:

- 5 of 12 May columns were silently skipped.
- The demographics sheet's columns are shifted by one.
- Initials are not unique (e.g. two different people called PM).
- Participant IDs are spelled three different ways.
- 19 typo cells (now corrected in `MAY.xlsx`).
- October scans were misfiled in the May folder.

### 3. The segmentation model

- U-Net (ResNet-18 encoder) that outlines the epidermis on each B-scan. Its lower
  edge is taken as the dermal-epidermal junction (**DEJ**).
- Trained on the **63 hand-annotated frames** from the bachelor project.
- **Test Dice 0.88.** This is the first time it was scored on its held-out test
  frames; the earlier 0.909 was a validation score.
- **Known flaw:** its training/validation split mixed frames from the same patients,
  so it should be retrained on a clean, patient-grouped split later. It is good
  enough for the analyses so far.
- Two input details matter a lot:
  - **Resizing:** it was trained on images squashed to 512×512. Fed letterboxed
    images instead, its Dice drops to 0.77. Each checkpoint now records which
    resizing it expects, and the code follows that record.
  - **Colour channel:** it must be fed the **green** channel. Fed the red channel,
    it reads the flow tint as tissue and puts the DEJ about 6 px too deep (up to
    16 px).

### 4. Per-scan analysis — `src/papi.py`, `src/scan_analysis.py`

For each scan:

1. Segment every B-scan and find the DEJ line.
2. Take a **slab from 5 px above to 15 px below the DEJ**.
3. Flow = red minus green, inside the slab.
4. **MIP**: the maximum projection over depth gives an en-face map (120 × width).
5. Measure:
   - **mean slab flow**, the main measure;
   - **loop count** (the thesis detector, earlier called "blob count"): bright, roundish spots on the MIP (threshold 75, minimum size
     30, circularity 0.45). Not yet validated as a loop count.
6. Also record a **flow-vs-depth profile** (mean flow at each depth relative to the
   DEJ) and the epidermis thickness.

Results are cached per scan and slab in `outputs/scan_cache/`, so each scan is
analysed once and reused by every comparison.

### 5. Training code (for later retraining)

`src/train.py` now:

- reads frames from the manifest;
- splits by patient, with a hard check that no patient lands in both train and
  validation;
- letterboxes input and records that in the checkpoint;
- actually evaluates the test set (previously it silently never ran);
- saves each run to its own folder instead of overwriting `models/`.

**128 automated tests** (`pytest tests/`) cover the manifest, the splitting, the
resizing and the analyses.

---

## Comparisons so far

### Model input checks (11 held-out test frames, plus one scan)

| Check | Result |
|---|---|
| Squashed vs letterboxed input | Dice **0.88** vs 0.77 → keep squashing for this model |
| Red vs green channel | Red puts the DEJ **6 px deeper** (median, up to 16 px) → use green |

### First look at 8 sample scans (mixed phases)

- The loop count mostly tracks how much flow is in the slab (rank correlation 0.86).
- The phase matters a lot (Baseline vs occlusion vs PORH), so scans can only be
  compared within one phase.
- No sign of thick callus on the feet: epidermis 26–33 px on every scan.

### Slab depth (Baseline, 10 patients, hand + foot)

| Slab (px relative to DEJ) | Median loop count | Median slab flow |
|---|---|---|
| 0 to +15 | 489 | 20.6 |
| **−5 to +15** (now default) | **612** | **24.6** |
| −5 to +20 (bachelor notebook) | 669 | 29.3 |

- 15 of 20 scans have a small flow peak about **5 px above** the DEJ line. That fits
  loops sitting in dermal papillae that rise above the line.
- The 5 px above the line raise the MIP almost as much as 5 extra px at the bottom,
  despite low *average* flow there. That points to sparse bright spots, like loops.
- Below 15 px, flow climbs towards the deeper vessel plexus (peak about 70 px below
  the DEJ), which isn't what we want to measure.

### Hand vs foot at Baseline (10 patients)

| Median | Hand | Foot |
|---|---|---|
| Slab flow | 23.1 | 26.2 |
| Loop count | 633 | 534 |
| Epidermis thickness | 30.5 px | 32 px |

No consistent difference: within patients it goes both ways.

### Baseline → occlusion → PORH (10 patients, 20 visits, 216 scans)

| | Hands | Feet |
|---|---|---|
| Slab flow at PORH peak vs Baseline | **×1.7** (1.2–3.3) | ×1.2 |
| Loop count at PORH peak vs Baseline | ×1.8 | ×1.3 |
| PORH peak above normal Baseline variation (2 SD) | **9 / 10** | 5 / 10 |
| Flow down during occlusion | 6 / 10 | 3 / 10 (**up** in 5 / 10) |
| Time of peak after the occlusion scan | ~100 s | ~130 s |

- **Hands show a clear PORH curve:** a drop during occlusion, an overshoot, then a
  return over a few minutes.
- **Feet are weak and inconsistent,** and foot flow often *rises* during occlusion.
  The cuff may not fully stop blood flow to the foot.
- **Slab flow is a better measure than the loop count:**
  - it varies less between Baseline repeats (10% vs 14–21%);
  - it detects PORH more often (14 vs 10 of 20 visits).
- The first post-release scan comes about 94 s after the occlusion scan, so the true
  PORH peak may come earlier than any scan and be missed.
- October participants have low Baseline flow but the largest relative PORH
  responses, so their weak Baseline images reflect real low flow.

---

## Limitations and open questions

- **Foot occlusion:** where was the cuff for foot measurements? If occlusion fails
  there, foot PORH can't be compared with hand PORH.
- **Session confound in the study groups:** 3 of the 11 controls (CA, JA, TL) were
  only scanned in October, while every T2DM patient was scanned in May (DH in
  both). October scans have lower flow and a different image width, so a
  T2DM-vs-control difference could partly be a session difference.
- **Why is `DP-11-DC` in neither study group** (e.g. type 1 diabetes, or excluded)?
- **The flow signal is a display overlay,** not the raw OCTA measurement. Raw
  structural/flow exports (`-s`/`-d` files, as used in the bachelor notebook) would
  likely give cleaner loops.
- **Field of view (from the DICOM headers):** about 5.2 × 5.0 mm. Pixels are
  4.4 µm in the B-scan (across and in depth), but B-scans are 42 µm apart. A
  10–20 µm loop therefore appears in only one or two B-scans. The −5 to +15 px slab
  is about 90 µm thick.
- **Loop count is not validated** against loops counted by eye.
- **The segmentation model needs retraining** on a clean, patient-grouped split.
- October scans differ (image width, lower flow); the scanner settings are unknown.

## How to run

```bash
# rebuild the manifest (drive optional; uses the saved copy if absent)
python src/manifest/build_manifest.py
python src/manifest/validate_manifest.py

# papillary MIP for one or more scans
python src/papi.py path/to/scan.dcm --out-dir outputs/papi

# hand vs foot on Baseline scans
python src/compare_baseline_sites.py --patients 10

# Baseline -> occlusion -> PORH time course
python src/compare_phases.py --patients 10

# tests
python -m pytest tests/
```

The comparison scripts accept `--slab-top=-5 --slab-bottom=15` to change the slab,
and resume where they left off if interrupted.

## Next steps

1. Answer the foot-occlusion question.
2. Run the PORH analysis on the 22 study participants (11 T2DM, 11 controls),
   both sites.
3. Compare T2DM vs controls: Baseline slab flow and PORH response, keeping the
   October/May session difference in mind.
4. Validate or drop the loop count (Fiji marking set up: `docs/loop_annotation.md`).
5. Later:
   - try the flow-vs-depth profile as a scan-quality measure;
   - extract and annotate more frames, then retrain the segmentation model.
