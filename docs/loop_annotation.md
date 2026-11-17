# Marking capillary loops in Fiji

The loop count (the thesis detector) has never been checked against loops counted
by eye. This sets up 22 small images for you to mark by hand. Comparing your marks
with the detector will tell us whether the count measures loops, and how to tune
it.

## What's in the export

Run `python src/export_loop_annotation.py`. It writes to
`data/annotation/loops/`. From Windows, that folder is

```
\\wsl.localhost\Ubuntu\home\madsl\article\Article_OCT_Diabetic_neuropathy\data\annotation\loops
```

| File | What it is |
|---|---|
| `loop_01.tif` … `loop_22.tif` | The images to mark |
| `loop_annotation.ijm` | Fiji macro: **F5** saves your marks, **F6** loads them back |
| `key/key.csv` | Which scan each image is. **Don't open it until you're done.** |

- **Hands only:** 5 T2DM and 5 controls, each with their Baseline (middle repeat)
  and PORH-peak scan, plus 2 images shown twice to measure how consistent the
  marking is.
- **Blinded:** names and order are random, so you can't tell the group, the phase
  or which images are the repeats. That keeps expectations from steering the
  counts.
- **Each image is a 400 × 400 px tile, about 1.5 mm square,** cut from the papillary
  slab MIP. A whole MIP has hundreds of loops; a tile has roughly 25–100.
- **Each tile is exactly what the detector sees:** the MIP resized to its square
  grid and stretched to 0–255, so your marks can be compared point by point with
  the detector's output.

### Scale (from the DICOM headers)

| | |
|---|---|
| Scan area | about 5.2 × 5.0 mm (1174 × 4.4 µm across, 120 B-scans × 42 µm) |
| One tile pixel | about 3.8 µm |
| B-scan spacing | 42 µm, about 11 tile pixels |

The image is sampled more finely from left to right than from top to bottom. Each
B-scan is one thin horizontal line, so a loop (10–20 µm) shows up in only one or
two B-scans. Expect **short, slightly blurred horizontal dots**, and a faint
horizontal banding every ~11 px.

## Setting up Fiji (once)

1. **Plugins › Macros › Install…** and choose `loop_annotation.ijm` in the export
   folder. Repeat this after every Fiji restart.
2. Right-click the point tool in the toolbar and pick **Multi-point**.
3. Double-click the point tool to set its options:
   - Type: *Hybrid* or *Dot*, size *Small* or *Medium*, colour *Yellow*;
   - untick *Auto-measure* and *Add to ROI Manager*;
   - optionally tick *Show labels* to see a running count.

## Marking an image

1. Open `loop_XX.tif` and zoom to 200% (**+** key). Pan with the space bar and
   drag.
2. **Don't change brightness or contrast.** All tiles are already stretched the
   same way, and keeping the display identical keeps your judgement consistent.
3. Click the centre of each loop once.
   - Alt+click removes a point; dragging moves one.
   - If the points vanish (for example after clicking with another tool),
     **Edit › Selection › Restore Selection** (Ctrl+Shift+E) brings them back.
4. Press **F5** to save. That writes `loop_XX_loops.csv` next to the image. Save
   often; **F6** reloads the saved marks if you come back to an image.

### What counts as a loop

- **Mark:** a small, compact, bright dot, about 3–8 px wide (10–30 µm), standing
  out from its surroundings. A capillary loop seen from above.
- **Don't mark:**
  - elongated streaks or lines: vessels running sideways, usually deeper ones;
  - diffuse grey haze or speckle with no clear centre;
  - stripes that run the full width of the image (motion between B-scans).
- **If unsure, leave it out,** and apply that rule the same way on every image.
  Consistency matters more than catching every faint dot.
- **If an image has no recognisable loops at all,** save it with a single point in
  the top-left corner (0, 0). I'll read that as "no loops", so empty images are
  still recorded.

Expect roughly 1–1.5 hours for all 22 images. Taking breaks is fine, since F5/F6
let you stop anywhere.

## What happens next

Once the `_loops.csv` files exist, a comparison script will:

1. **Match your points to the detector's** within a few pixels, and report how many
   of your loops it finds and how many of its loops are real.
2. **Compare counts per image:** does the detector rank the images the way you do?
3. **Check consistency:** compare your two marks of each repeated image.
4. **Tune the detector:** try thresholds relative to each image's own brightness
   instead of the fixed 75, and choose the setting that best matches your marks.
5. **Test recruitment:** your Baseline vs PORH counts per participant show whether
   loop recruitment differs between groups, at least on these 10 people.

If you find loops can't be told apart from noise in these images, that's a useful
result too. It would mean a loop count on this data (the red display overlay)
can't be trusted, which strengthens the case for getting the raw `_D` exports.
