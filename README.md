# Article_OCT_Diabetic_neuropathy

Research project on OCT for early diagnostics of diabetic neuropathy.

## Repository Structure

```
├── src/                 # Python source code
├── data/
│   ├── raw/             # Raw input data (not tracked by git)
│   └── processed/       # Processed/cleaned data (not tracked by git)
├── results/
│   ├── figures/         # Generated PNG figures (not tracked by git)
│   └── tables/          # Generated CSV result tables (not tracked by git)
├── docs/
│   └── TODO.md          # Shared todo list
├── pyproject.toml       # Python project config (uv)
└── .gitignore
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) as the Python package manager.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Add a new dependency
uv add <package>

# Add a dev dependency (e.g. jupyter)
uv add --dev <package>
```

## Manual MIP quality review

Run from the repository root, using the existing environment:

```bash
uv run --locked python src/qc_review.py --reviewer Magnus --open
```

Generate the original papillary-dermis QC MIPs and start review:

```bash
uv run --locked python src/qc_review.py --generate-from data/dicom --reviewer Magnus --open
```

The QC projection is the existing `papillary_dermis_mip` with offsets 0 and
15 inclusive (16 rows from the legacy last-epidermal-pixel DEJ convention).
It is resized to 1355 x 1355 and displayed with per-image linear grayscale
scaling, like the original MIP plot. This is a visual QC image, not a new
quantitative intensity metric. No loop counting is performed during generation.
MIPs are saved in `outputs/qc_mips`, scanned recursively for `*_qc_mip.png`.
The middle B-scan is saved in a companion `*_qc_bscan.npz` and displayed below
the MIP. The DEJ overlay fades continuously with a 0-100% slider (default 50%);
the underlying image stays unchanged and the slider does not save a QC decision.
On a fresh generation the line comes
from the same segmentation pass used for the MIP. For an existing MIP without
a preview, only the middle slice is segmented again; the UI labels that line
as recomputed. Existing MIPs and scan decisions remain unchanged. A single
middle slice is a preview, not a validation of every B-scan in the volume.
To review existing exports, explicitly pass a folder and glob:

```bash
uv run --locked python src/qc_review.py /path/to/images --pattern '*_mip.png' --reviewer Magnus --open
```

Each DICOM filename receives one use / discard / unsure decision, independent
of individual depth layers or regenerated image contents. Notes and reasons
are optional. Decisions save to `data/manifest/qc_reviews.json`; CSV export
includes `filename` and `qc_status`. Raw files are never moved or deleted.
Only the latest decision is stored, without review history. Reason checkboxes
and notes are informational; only use / discard / unsure sets the QC status.

The manifest builder joins these decisions by the complete DICOM filename.
`scans.csv` includes `qc_status`, `qc_note`, and `analysis_eligible`. Only
`use` AND no existing exclusion makes a scan analysis-eligible. Discard,
unsure and unreviewed scans are withheld. `scans_for_analysis.csv` contains
only eligible rows. Existing exclusions and segmentation selection remain
separate from manual image QC. The hand/foot comparison respects eligibility
when reading the new manifest. Standalone DICOM scripts do not filter a
manually supplied filepath; use the approved manifest for cohort analysis.

On saving a decision, the review app refreshes an existing `scans.csv` and
`scans_for_analysis.csv`. If no manifest has been built yet, it only writes
`approved_scan_filenames.csv`; build the manifest to join patient metadata.
Builds read `data/manifest/qc_reviews.json` by default, or `--qc-reviews PATH`.
Review decisions survive rebuilding the manifest.

The segmentation reason is also stored as `segmentation_review_required`, with
`qc_bscan_index` for the preview that was displayed when the decision was saved
(zero-based; blank for old decisions without a recorded index). All flagged
scans are exported to `segmentation_review_candidates.csv`, including scans
discarded or withheld from the biological analysis. This is a queue for manual
inspection and corrected annotations, not ground-truth masks or automatic
training eligibility. Preserve patient-level train/validation/test separation
when using these cases for later model improvement.

Generation progress is retained in `data/manifest/mip_generation.csv` and
joined into `scans.csv` on every manifest build or QC refresh. It records
pending / processing / ready / failed, DICOM/MIP/B-scan paths, error and update
time. All queued scans are registered before processing. Interrupted work can
be retried; readable MIPs with readable previews are skipped, and failed scans
do not abort the batch. A fresh generation never changes manual QC decisions.
`scans.csv` is the combined overview; `scans_for_analysis.csv` is its approved
analysis subset. The JSON review log and generation table are persistent
manifest components, not separate ad-hoc exclusion folders.

The original DICOM filename is shown in the image heading; the review is not
clinically blinded. Use a separate results file per independent reviewer
with `--results /path/to/reviewer.json --manifest-dir /path/to/separate/manifest`.
Sparse OCTA signal alone is not a
quality failure. MIP QC does not validate anatomical slab placement.
The server is bound to localhost; do not expose it on a public network.

## Read-only depth viewer

```bash
uv run --locked python src/depth_viewer.py --open
```

This separate local application (default port 8767) browses skin-surface-relative MIPs
with depth, slab thickness, and B-scan sliders. The default scan is the local
AC S2387 volume; pass another DICOM filepath as a positional argument. It uses
the existing `outputs/dej_depth_test/<stem>_depth_slabs.npz` boundaries if found,
or segments the volume once before opening. `--boundaries PATH` selects a cache
explicitly. It does not write QC decisions, manifest entries, or source data.

Depths are nonnegative integer pixels below the local predicted skin surface,
with half-open intervals. At zero, the interval follows the top of the epidermis
column by column; the slider cannot enter air above it. DEJ stays visible as a
reference line but is not the origin of this viewer's depth slider. The default
thickness is 5 pixels; use 1 for thin depth slices. The QC MIP definition remains
unchanged (legacy DEJ +0..+15).
All depths use the same 0-255 RGB-overlay scale (16-bit data divide by 257),
not depth-specific min/max normalization. Invalid intervals are pink, not zero.
The square MIP view is not physically calibrated. Skin surface is white, DEJ
yellow, and the selected interval cyan on the B-scan. Boundaries are model
estimates, not an independent anatomical reference.
