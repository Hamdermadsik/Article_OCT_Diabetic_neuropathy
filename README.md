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

Generate all non-excluded scans in the active manifest while reviewing:

```bash
uv run --locked python src/qc_review.py --reviewer Magnus --generate-manifest --open
```

Keep the drive connected and the server running. Completed images are reused
only when their recorded slab matches -5..+15. New images and generation counts
appear automatically without resetting the current review. Generation and review
manifest writes are serialized within this server; do not run a second generator
or manifest builder concurrently. Restart the same command after interruption.

Generate the original papillary-dermis QC MIPs and start review:

```bash
uv run --locked python src/qc_review.py --generate-from data/dicom --reviewer Magnus --open
```

The QC projection is the existing `papillary_dermis_mip` with offsets -5 and
15 inclusive (21 rows from the legacy last-epidermal-pixel DEJ convention).
The preview stores these offsets; missing or different offsets trigger regeneration.
It is resized to 1355 x 1355 and displayed with per-image linear grayscale
scaling, like the original MIP plot. This is a visual QC image, not a new
quantitative intensity metric. No loop counting is performed during generation.
MIPs are saved in `outputs/qc_mips`, scanned recursively for `*_qc_mip.png`.
The middle B-scan is saved in a companion `*_qc_bscan.npz` and displayed below
the MIP. The DEJ overlay fades continuously with a 0-100% slider (default 50%);
the underlying image stays unchanged and the slider does not save a QC decision.
On generation the line comes from the same segmentation pass used for the MIP.
Existing MIPs without a preview recording the current slab are regenerated.
Scan decisions remain unchanged and should be reconsidered after a slab change. A single
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

Status-filtered review advances to the next scan in that filter, including
already-reviewed scans, without wrapping back to the start. The `Forkert DEJ`
filter shows saved segmentation flags across all decision statuses. Removing a
flag only changes this filter after saving a decision.

The browser navigation regression tests use mocked APIs and never access real
review files. With Node.js and Playwright available, run
`node tests/qc_review_navigation.cjs`. Set
`PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` to use an existing Chromium/Chrome binary.

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
uv run --locked python src/depth_viewer.py "/path/to/scan.dcm" --open
```

This separate local application (default port 8767) displays a depth-dependent
MIP and a selectable B-scan. A searchable scan selector reads filenames from
`data/manifest/scans.csv`; `--manifest PATH` selects another inventory. Search
terms match the filename together, e.g. `JF Instep 244`. Opening a scan loads and
segments it in the background with progress; a missing drive or failed scan
leaves the previous scan available. Duplicate file copies are listed once.
Scan selection changes the active scan for this viewer server, not QC decisions.
Depth is measured in integer pixels below the
predicted skin surface, not the DEJ. Intervals are half-open; thickness defaults
to 5 pixels. The vertical depth slider tracks the selected layer at the right
edge of the B-scan. Depth buttons move 5 pixels and thickness buttons 1 pixel;
holding a button repeats the adjustment. B-scans can be selected by number.

The viewer loads cached boundaries from
`outputs/dej_depth_test/<stem>_depth_slabs.npz` if present, or segments the scan
once. `--boundaries PATH` selects a cache. It does not write source data, QC
decisions or manifest entries. It can run alongside QC on a different port.

MIPs use QC-style interpolation to 1355 x 1355 but a fixed 0-255 OCTA display
scale across depths, not QC's per-image contrast normalization. Invalid areas
are pink. The B-scan is structural grayscale with skin surface in white, DEJ
in yellow and the selected layer in cyan. Physical scaling is not verified.
The QC projection slab is controlled separately by the QC generator.
