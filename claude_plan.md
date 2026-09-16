# Scale up OCTA papillary-dermis dataset + fix pipeline bugs

## Context

This repo trains a U-Net (smp.Unet, resnet18 encoder) to segment the papillary
dermis/plexus in OCT skin B-scans, as a continuation of a bachelor's thesis into
a master's-level article on diabetic small-fiber neuropathy. Investigation this
session found the current small dataset (135 hand-picked, hand-annotated frames
from ~45 patients) has a real train/val leakage problem — the split has no
patient-awareness, and empirically 7/18 candidate same-patient frame groups get
torn across train/val under the current seed. Several other pipeline bugs were
found (a dead test-evaluation code path, a non-uniform aspect-ratio resize baked
into every sample, hardcoded paths, no manifest — pairing is pure filename
convention). Meanwhile, a much larger raw dataset (~3,175 DICOM volumes / ~380k
raw B-scan frames, 568 GB total) sits on an external drive, unorganized.

User's explicit sequencing: **(1) get the big dataset organized, (2) get all the
code working, (3) retrain later once everything is set up** — retraining is
*out of scope* for this plan's execution. Annotation strategy is confirmed:
**semi-automated** — the current model pre-labels new frames, user corrects them
in ImageJ/Fiji (confirmed this session), reusing the existing
upper/lower-boundary-line TIFF convention already in `data/for_segmentation/lines/`.
Frame sampling is scoped to **Baseline + Regular condition scans only**
(confirmed) — Isc/PORH/Flavanol are perfusion-state variants of the same
anatomy, not new segmentation-relevant content.

Key facts gathered this session (verified directly, not assumed):
- External drive mounted at `/mnt/d` (now in `/etc/fstab`, auto-mounts on WSL boot).
  `OCTA DICOM files may 2021/`: 2,620 real `.dcm` (+2,116 macOS `._*` AppleDouble
  junk — ignore), 476GB, 47 patient initials, 94 sessions. `OCTOBER/`: 555 real
  `.dcm`, 92GB, 23 initials, 49 sessions (+1 irrelevant dead-reference `.m` file).
  7 initials overlap between cohorts (`CH, DH, LM, MB, PM, RM, TL`) — ambiguous,
  needs a manual 7-row review, not worth automating.
- Each `.dcm` = one full volume, confirmed via SimpleITK: `(120, 460, 1174, 3)`
  uint8, `Implicit VR Little Endian` (**uncompressed** — frames are seekable by
  computed byte offset, no need to decode a whole volume to read one frame).
  Header-only `pydicom` reads take ~6ms/file (~20s for all 3,175 files); raw
  `open().read()` throughput is ~100-124 MB/s once warm — SimpleITK's decode
  path, not the drive, was the bottleneck in an earlier naive benchmark.
- Existing `src/exl.py` + `notebooks/Excel.ipynb` already prototype
  filename↔Excel-scan-log matching + demographics joining (fragile — fuzzy
  fallback assumes globally-unique scan numbers, demographics join by bare
  initials, October dates are a placeholder) — worth hardening and reusing, not
  discarding.
- `data/for_segmentation/lines/{up,down}` actually contain exactly 63 matched
  pairs each (verified via `pathlib.glob`, not `ls` — `ls` overcounts due to
  Windows `Zone.Identifier` sidecar files), and those 63 stems are exactly the
  union of the current `data/train`+`data/test` indices. No annotation gap to
  close there — this was a miscount in an earlier `ls`-based tally.
- `sklearn` and `pytest` are not currently installed in the project venv.

## Phase 1 — Manifest (`data/manifest/{patients,scans,frames}.csv`)

New `src/manifest/build_manifest.py`. Three-tier schema (patients/scans/frames)
— patient identity needs resolving once and referenced from both scan-level
exclusion decisions and the frame-level `GroupShuffleSplit` group key; a 2-tier
schema would duplicate identity logic. CSV/Parquet, not a DB — this data volume
doesn't need one on a single-user WSL box.

- `patients.csv`: `patient_id` (canonical), `cohorts`, `initials_may`,
  `initials_october`, `age`, `sex`, `group` (DP/HP), `identity_confidence`
  (`confirmed|probable|unresolved_kept_separate|single_cohort`).
- `scans.csv`: one row per `.dcm` — `scan_id`, `patient_id`, `cohort`,
  `filepath`, parsed filename fields (initials/date/Lnum/scan_number/bodypart/
  side/protocol_area), matched Excel fields (phase/condition), `match_method`
  (`exact|fuzzy|unmatched`), `header_status`, `num_frames`, `rows`, `cols`,
  `samples_per_pixel`, `transfer_syntax`, `pixel_data_offset` (computed once
  here, reused by extraction), `is_duplicate_of`, `selected_for_segmentation`.
- `frames.csv`: one row per sampled (scan, frame_index) — `frame_id`, `scan_id`,
  `frame_index`, `patient_id` (denormalized — the literal split group column),
  `extraction_status`, `png_path`, `annotation_status`
  (`none|pre_labeled|corrected|verified`), `upper_line_path`, `lower_line_path`,
  `mask_path`, `source` (`drive_v2|legacy_handpicked`), `patient_id_confidence`.

Builder steps:
1. Walk both drive folders, filter `._*` and `Zone.Identifier` names, keep `*.dcm`.
2. Header-only `pydicom.dcmread(path, stop_before_pixels=True)` per file (~20s
   total) → validate shape/transfer-syntax against the confirmed
   `(120,460,1174,3)` uncompressed pattern; flag (don't crash on) anomalies into
   `manifest_build_errors.csv`. Compute `pixel_data_offset` here once per file.
3. Parse filenames with `exl.py`'s existing `filename_pattern` regex (confirmed
   it matches both May and October naming) — reuse as-is.
4. May and October Excel sheets aren't just differently-configured — they're
   differently *shaped* (May: one row per visit, ~12 phase-columns holding
   scan-number ranges. October: two sheets, forward-filled participant blocks,
   2-letter setting codes like "HB"→Hand+Baseline via `parse_setting()`, a
   freeform "Sheet2" of loose notes, and no real per-scan dates). So this needs
   two separate parser functions, not one shared loop with a config swap:
   `src/manifest/parse_may.py::parse_may(df) -> long_df` (wraps `exl.py`'s
   existing column-lookup logic) and
   `src/manifest/parse_october.py::parse_october(df1, df2) -> long_df`
   (wraps the ffill/two-sheet/code-decoding logic from `Excel.ipynb` cell 6).
   Both must converge on the **same normalized long-format schema**
   (`participant_id, initials, date, scan_number, phase, protocol_area,
   condition, excel_column`) — everything downstream (filename regex matching,
   fuzzy fallback, demographics join, manifest writing) is genuinely
   cohort-agnostic *once* both parsers agree on that shared output shape, and
   only needs to run once against the combined result. Reuse `exl.py`'s
   `expand_scan_field`/`get_initials`/`get_group` in both parsers rather than
   duplicating. `src/manifest/cohort_specs.py` holds only the genuinely
   shared, data-only pieces (`MAY_PHASE_SPECS`, shared phase-name constants) —
   it is not where October's parsing logic belongs, since October isn't a
   config difference, it's a different code path.
5. Run the exact+fuzzy merge (reuse the prototyped logic), but log
   `match_method` per row instead of silently merging — makes the ~5.7% (May)
   / ~14% (October) unmatched rate auditable rather than invisible.
6. Duplicate-scan detection on `(initials, date, scan_number, protocol_area)`
   key (a real example already exists:
   `RS_Right Hand_L1074_S3851__02_06_2021 (1).dcm`) — keep one, set
   `is_duplicate_of` on the other, never double-count in sampling.
7. Patient identity: cohort-scoped by default. Write
   `data/manifest/identity_review_candidates.csv` (the 7 overlapping-initials
   rows, side-by-side age/sex per cohort) for a quick manual call — re-running
   just the identity-merge step (not the full header scan) once resolved.
8. Cross-cohort scan-number-collision sanity check (`groupby(scan_number).initials.nunique()`)
   — the fuzzy-fallback merge assumes global uniqueness; verify it once now that
   both cohorts are combined.
9. Migrate the legacy 135 hand-picked frames into `frames.csv` as
   `source=legacy_handpicked`, `patient_id="legacy-group-{n}"` via the
   consecutive-run-of-≤3 heuristic already validated empirically this session,
   `patient_id_confidence=heuristic_consecutive_run`. **Do not** pursue
   pixel-matching reconciliation against the 3,175 drive volumes to recover
   exact identity — explicitly out of scope; the legacy bucket becomes a minor
   addition once the drive-sourced, properly-grouped dataset exists.

`validate_manifest.py` (or a `--validate` mode): missing-pairs (reuse `exl.py`'s
reverse-merge), corrupt/unreadable files, duplicates, shape mismatches, junk
filtering — all above, runnable standalone against an existing manifest.

## Phase 2 — Sampling & on-demand extraction

**Stage A (candidate generation, header-data only, no pixel I/O):**
`src/manifest/sample_frames.py`. Filter to `phase=="Baseline"`,
`condition=="Regular"` scans only (confirmed). For each selected scan, pick
`N=6` (default, CLI-configurable — not a hardcoded constant) evenly-spaced
frame indices via `np.linspace(0, num_frames-1, N, dtype=int)` (same idiom
already used in `test_mode_dicom.py`). Write `frames.csv` rows with
`extraction_status=pending`.

**Stage B (lazy extraction, only for frames actually queued):**
`src/manifest/extract_frames.py`. Raw-seek extraction per the I/O benchmarking
this session: `open(filepath,'rb')`, `seek(pixel_data_offset + frame_index *
frame_byte_size)`, `read(frame_byte_size)`, `np.frombuffer(...).reshape(...)`
— no SimpleITK, no full-volume decode. Take channel 0 as structural grayscale
(matches `papi.py`'s existing convention). Save to
`data/for_segmentation_v2/images/{frame_id}.png`; update `frames.csv`.

**Correctness gate before trusting this at scale**: validate raw-seek-decoded
frames against a full SimpleITK decode on 5-10 sample files across both
cohorts (byte-identical check) before running Stage B broadly — cheap
insurance against an offset-computation edge case silently corrupting frames.

## Phase 3 — Semi-automated annotation (model pre-labels → ImageJ correction)

**`src/annotate/pre_label.py`:**
1. Extract `load_pytorch_model` (currently duplicated verbatim in `papi.py` and
   `test_mode_dicom.py`) into a shared `src/model_utils.py`; import from there
   in both existing files and this new script instead of a third copy.
2. For each `frames.csv` row with `extraction_status=extracted` and
   `annotation_status=none`: run inference via the existing
   `papi.py::test_model_on_sample_image` path (reused as-is) to get a
   native-resolution predicted mask.
3. Convert predicted raster mask → boundary lines via `papi.py`'s existing
   `remove_epidermis_from_image` (already does this exact raster→line
   inversion). Add one small new function, `coords_to_line_raster(coords,
   shape)`, to rasterize its `(y,x)` coordinate output into the single-pixel-
   per-column TIFF format `lines_to_mask.py` already reads via
   `np.argmax(..., axis=0)`.
4. Save plain 8-bit TIFFs (confirmed compatible with ImageJ/Fiji) to
   `data/for_segmentation_v2/lines/{up,down}/{frame_id}.tif` — same format/
   convention as the existing 63 legacy pairs, zero new tooling for the user.
5. Update `frames.csv`: `annotation_status=pre_labeled`.

Note (non-blocking): the current model's aspect-ratio-distortion bug (see
Phase 4) will make pre-labeled lines somewhat off geometrically, mainly near
lateral edges. Acceptable for draft-quality human-corrected labels — the fix
belongs in Phase 4 and doesn't block starting annotation now.

**`src/annotate/finalize_annotations.py`:** thin manifest-aware wrapper around
`lines_to_mask.py`'s existing `lines_to_mask()` function (reused unchanged) —
takes a list of `frame_id`s the user has just corrected in ImageJ, writes
`mask_{frame_id}.png` to `data/for_segmentation_v2/masks/`, updates
`frames.csv`: `mask_path`, `annotation_status=verified`.

## Phase 4 — Pipeline code fixes

Can run partly in parallel with Phases 1-3 (see Sequencing below).

- **`src/train.py` test-eval bugs**: fix `.glob("*.jpeg")` → `.glob("*.png")`
  (~line 158); remove the invalid `n_samples=10000, noise_sigma=0` kwargs from
  the `OCT_SKIN(...)` call (~lines 168-175) — `OCT_SKIN.__init__` doesn't
  accept them.
- **Centralize `target_size`**: currently duplicated in `dataloaders.py` (x2),
  `train.py`, `papi.py`, `test_mode_dicom.py`. Add to `configs/config.yaml`
  under a new `data:` block for the Hydra-driven `train.py`; for the
  non-Hydra scripts (`papi.py`, `test_mode_dicom.py`), define once in
  `src/model_utils.py` and import — avoids adding config-loading boilerplate
  to throwaway inference scripts.
- **Letterbox (aspect-preserving) resize**: replace the naive
  `cv2.resize(img, target_size)` (squashes 460×1174 → 512×512 non-uniformly)
  with a shared `letterbox_resize`/`letterbox_unresize` pair (new, in
  `src/image_utils.py` or folded into `model_utils.py`) applied consistently
  in `dataloaders.py::OCT_SKIN.__getitem__`, `papi.py::test_model_on_sample_image`
  (both the pre-inference resize and the post-inference resize-back — the
  inverse direction needs to crop padding before resizing to native, not a
  plain resize), and `test_mode_dicom.py::predict_mask` (same pre/post pair).
  This changes numeric behavior of train+inference — land as its own isolated
  commit so any future retraining can attribute its effect cleanly.
- **Manifest-driven `OCT_SKIN`**: change constructor from folder-glob pairing
  to accepting a `frames_df` (pre-filtered to `annotation_status=="verified"`)
  with `png_path`/`mask_path` columns — also fixes the current silent-drop
  of unmatched images (no warning today) by making "not in the DataFrame" an
  explicit, auditable manifest state.
- **Patient-grouped split**: replace `train.py`'s flat
  `torch.randperm` (~lines 69-73) with `sklearn.model_selection.GroupShuffleSplit`
  keyed on `frames_df['patient_id']`. Requires adding `scikit-learn` to
  `pyproject.toml` (confirmed not installed — distinct from `scikit-image`,
  which is already a dependency).
- **Hardcoded absolute paths**: `papi.py` (lines ~167-170), `test_mode_dicom.py`
  (line ~71), `visualize_results.py` (line ~123), `train.sh`/`test.sh`. Use
  `Path(__file__).resolve().parents[1]` as the default repo-root resolution,
  overridable via an `OCT_PROJECT_ROOT` env var for the DTU HPC case.
- **Lightweight pytest suite** (new `tests/`, add `pytest` to dev-dependencies
  — confirmed not installed):
  - `test_dataloaders.py` — synthetic tiny fixtures; shape/dtype correctness;
    once letterbox lands, a known-pixel-position round-trip test.
  - `test_split_integrity.py` — synthetic `frames_df` with known patient
    groups; assert no `patient_id` spans both train and val. This is the
    regression test that would have caught the leakage bug found this session
    — treat as non-negotiable.
  - `test_train_eval_path.py` — regression test for the glob/kwargs fix;
    asserts the test-eval branch actually executes and doesn't raise.
  - `test_manifest_pairing.py` — a frame missing `mask_path` doesn't crash
    filtering and doesn't silently enter a split.

## Sequencing

Phases 1-3 don't need to wait for Phase 4 — pre-labeling deliberately uses the
current (imperfect) model since output is always human-corrected, and
retraining is out of scope here. Recommended order:

**Phase 1 → (Phase 2 + Phase 4's pure bug-fix subset: glob/kwargs fix, path
hardcoding, pytest suite for existing behavior — in parallel) → Phase 3 →
Phase 4's manifest-dependent subset (OCT_SKIN refactor, grouped split,
letterbox resize) → [retraining — explicitly deferred, not in this plan]**

## Verification

- Phase 1: run `build_manifest.py`, inspect `manifest_build_errors.csv` (should
  be empty or fully explained), confirm `scans.csv` row count ≈ 3,175 minus
  duplicates, spot-check a handful of `match_method=fuzzy` rows manually.
- Phase 2: run the correctness gate (raw-seek vs SimpleITK byte-identical
  check) on 5-10 files before trusting Stage B at scale.
- Phase 3: visually spot-check a handful of pre-labeled line TIFFs against
  their source frames before handing the full batch to ImageJ correction.
- Phase 4: `pytest tests/` green, especially `test_split_integrity.py`;
  manually re-run `train.py`'s test-eval branch on the tiny existing
  `data/test` set and confirm it now executes (no `TypeError`, `test_dice`
  populated in `training_history.pt` instead of `None`).
