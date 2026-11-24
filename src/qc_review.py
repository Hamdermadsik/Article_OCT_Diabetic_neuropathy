"""Local, resumable manual review of MIPs. Never modifies source images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import secrets
import tempfile
import threading
import webbrowser
from contextlib import nullcontext
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from image_utils import structural_image

STATUSES = ("use", "discard", "unsure", "unreviewed")
# Keep the removed coverage code readable in historical decisions.
REASONS = ("noise", "motion", "signal_loss", "blur", "clipping", "segmentation", "other", "coverage")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "qc_mips"
DEFAULT_REVIEWS = REPO_ROOT / "data" / "manifest" / "qc_reviews.json"
QC_SLAB = (-5, 15)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(folder, pattern=None):
    """One review item per DICOM filename, independent of image contents."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f"Input folder does not exist: {folder}")
    paths = sorted(folder.rglob(pattern or "*_qc_mip.png"))
    items, warnings, seen = [], [], set()
    for path in paths:
        if not path.is_file():
            continue
        path = path.resolve()
        if not path.is_relative_to(folder):
            warnings.append(f"Skipped external link: {path.name}")
            continue
        try:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("Cannot decode image")
            stem = path.stem
            for suffix in ("_qc_mip", "_papillary_dermis_mip", "_mip"):
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break
            filename = stem + ".dcm"
            if filename in seen:
                warnings.append(f"Duplicate scan filename skipped: {filename}")
                continue
            seen.add(filename)
            relative = path.relative_to(folder).as_posix()
            key = hashlib.sha256(filename.encode()).hexdigest()
            preview = path.with_name(f"{stem}_qc_bscan.npz")
            preview_info = None
            if preview.is_file() and preview.resolve().is_relative_to(folder):
                try:
                    with np.load(preview, allow_pickle=False) as data:
                        preview_info = {"index": int(data["index"]),
                                        "num_frames": int(data["num_frames"]),
                                        "recomputed": ("provenance" in data and
                                            str(data["provenance"]) == "preview_recomputed_for_existing_mip")}
                except (OSError, ValueError, KeyError):
                    warnings.append(f"Unreadable B-scan preview: {preview.name}")
            items.append({"id": key, "path": str(path), "relative_path": relative,
                          "filename": filename,
                          "bscan_path": str(preview) if preview_info else None,
                          "bscan": preview_info,
                          "width": int(image.shape[1]), "height": int(image.shape[0]),
                          "kind": "image"})
        except (ValueError, OSError, cv2.error) as error:
            warnings.append(f"{path.name}: {error}")
    if not items:
        raise ValueError("No readable MIPs found. " + " | ".join(warnings[:5]))
    return items, warnings


def render_image(item):
    path = Path(item["path"])
    display = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if display is None:
        raise ValueError("Cannot decode image")
    ok, encoded = cv2.imencode(".png", display)
    if not ok:
        raise ValueError("Could not encode image")
    return encoded.tobytes()


def render_bscan(item, overlay=True, opacity=1.0, octa=False):
    """Original middle B-scan, optionally with the DEJ used by the QC pipeline."""
    if not item.get("bscan_path"):
        raise FileNotFoundError("No B-scan preview available")
    if not 0 <= opacity <= 1:
        raise ValueError("DEJ opacity must be between 0 and 1")
    with np.load(item["bscan_path"], allow_pickle=False) as data:
        image, dej = data["image"], data["dej"]
    if image.dtype == np.uint16:
        image = (image.astype(float) / 65535 * 255).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    display = (cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if octa and image.ndim == 3
               else cv2.cvtColor(structural_image(image), cv2.COLOR_GRAY2BGR))
    if overlay and opacity > 0:
        original = display.copy()
        valid = np.isfinite(dej) & (dej >= 0) & (dej < image.shape[0])
        columns = np.flatnonzero(valid)
        # Do not draw lines across columns where segmentation is missing.
        for run in np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1):
            if run.size:
                points = np.column_stack((run, dej[run])).astype(np.int32)
                cv2.polylines(display, [points], False, (0, 255, 255), 2)
        if opacity < 1:
            display = cv2.addWeighted(display, opacity, original, 1 - opacity, 0)
    ok, encoded = cv2.imencode(".png", display)
    if not ok:
        raise ValueError("Could not encode B-scan")
    return encoded.tobytes()


class ReviewStore:
    def add_generated(self, path):
        items, _ = discover_images(Path(path).parent, Path(path).name)
        with self.lock:
            for item in items:
                if item["id"] not in self.by_id:
                    self.items.append(item)
                else:
                    self.items = [item if old["id"] == item["id"] else old for old in self.items]
                self.by_id[item["id"]] = item

    def __init__(self, items, path, reviewer, manifest_dir=None):
        self.items = items
        self.by_id = {item["id"]: item for item in items}
        self.path = Path(path).resolve()
        self.reviewer = reviewer
        self.manifest_dir = Path(manifest_dir) if manifest_dir is not None else None
        self.lock = threading.Lock()
        self.disk_digest = file_digest(self.path) if self.path.exists() else None
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {
            "schema_version": 2, "revision": 0, "reviews": {}}
        if self.data.get("schema_version") != 2 or not isinstance(self.data.get("reviews"), dict):
            raise ValueError("Expected scan-level reviews (schema 2), not old layer-level reviews")

    def snapshot(self):
        with self.lock:
            from manifest.generation import generation_counts
            public = [{k: v for k, v in item.items() if k not in ("path", "bscan_path")}
                      for item in self.items]
            return {"items": public, "revision": self.data["revision"],
                    "reviews": {key: value for key, value in self.data["reviews"].items()
                                if key in self.by_id}, "reviewer": self.reviewer,
                    "results_path": str(self.path),
                    "generation": generation_counts(self.manifest_dir) if self.manifest_dir else None}

    def save(self, payload):
        with self.lock:
            if payload.get("revision") != self.data["revision"]:
                raise RuntimeError("Review changed in another window. Reload before saving.")
            current_digest = file_digest(self.path) if self.path.exists() else None
            if current_digest != self.disk_digest:
                raise RuntimeError("Review file changed outside this server. Restart before saving.")
            key, status = payload.get("id"), payload.get("status")
            reasons, note = payload.get("reasons", []), payload.get("note", "")
            if not isinstance(key, str) or key not in self.by_id or status not in STATUSES:
                raise ValueError("Invalid image or status")
            if not isinstance(reasons, list) or any(r not in REASONS for r in reasons):
                raise ValueError("Invalid reasons")
            if not isinstance(note, str) or len(note) > 5000:
                raise ValueError("Note must be text, at most 5000 characters")
            item = self.by_id[key]
            record = {"status": status, "reasons": sorted(set(reasons)), "note": note,
                      "reviewer": self.reviewer, "reviewed_at": datetime.now(timezone.utc).isoformat(),
                      "filename": item["filename"]}
            record["qc_bscan_index"] = item["bscan"]["index"] if item.get("bscan") else None
            reviews = {scan_id: {k: v for k, v in review.items() if k != "history"}
                       for scan_id, review in self.data["reviews"].items()}
            updated = {**self.data, "revision": self.data["revision"] + 1,
                       "reviews": {**reviews, key: record}}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Replace atomically: an interrupted save leaves the previous file intact.
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             delete=False, suffix=".tmp") as handle:
                temp = Path(handle.name)
                try:
                    json.dump(updated, handle, indent=2, ensure_ascii=False)
                    handle.flush()
                    import os
                    os.fsync(handle.fileno())
                except BaseException:
                    temp.unlink(missing_ok=True)
                    raise
            try:
                temp.replace(self.path)
            finally:
                temp.unlink(missing_ok=True)
            self.data = updated
            self.disk_digest = file_digest(self.path)
            warning = None
            if self.manifest_dir is not None:
                try:
                    from manifest.qc import sync_manifest_qc
                    sync_manifest_qc(self.manifest_dir, self.path)
                except (OSError, ValueError) as error:
                    warning = f"Vurdering gemt, men manifest kunne ikke opdateres: {error}"
            return {"revision": updated["revision"], "review": record, "warning": warning}

    def export_csv(self):
        with self.lock:
            output = io.StringIO(newline="")
            fields = ["filename", "qc_status", "reasons", "segmentation_review_required",
                      "qc_bscan_index", "note", "reviewer", "reviewed_at"]
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for item in self.items:
                record = self.data["reviews"].get(item["id"], {})
                row = {"filename": item["filename"],
                       "qc_status": record.get("status", "unreviewed"),
                       "reasons": ";".join(record.get("reasons", [])),
                       "segmentation_review_required": "segmentation" in record.get("reasons", []),
                       "qc_bscan_index": record.get("qc_bscan_index", ""),
                       **{key: record.get(key, "") for key in ("note", "reviewer", "reviewed_at")}}
                # Keep CSV text literal when opened in spreadsheet software.
                for key in ("filename", "note", "reviewer"):
                    if str(row[key]).lstrip().startswith(("=", "+", "-", "@")):
                        row[key] = "'" + str(row[key])
                writer.writerow(row)
            return output.getvalue().encode("utf-8-sig")


def generate_qc_mips(paths, output_dir, model=None, device=None, manifest_dir=None,
                     manifest_lock=None, on_generated=None):
    """Generate DEJ -5..+15 inclusive QC MIPs, without detecting dots."""
    import torch
    from model_utils import default_model_path, load_pytorch_model
    from papi import EN_FACE_SIZE, import_dicom, papillary_dermis_mip, test_model_on_sample_image
    from skimage.transform import resize
    from manifest.generation import generation_counts, update_generation

    def record_generation(*args, **kwargs):
        with manifest_lock if manifest_lock is not None else nullcontext():
            update_generation(*args, **kwargs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = [Path(path) for path in paths]
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Duplicate DICOM filename in generation queue")
    if manifest_dir is not None:
        record_generation(manifest_dir, paths, output_dir=output_dir)
    generated = []
    for number, path in enumerate(paths, 1):
        target = output_dir / f"{path.stem}_qc_mip.png"
        preview = output_dir / f"{path.stem}_qc_bscan.npz"
        try:
            have_mip = target.exists() and cv2.imread(str(target)) is not None
            have_preview = False
            matching_slab = False
            if preview.exists():
                try:
                    with np.load(preview, allow_pickle=False) as data:
                        have_preview = (data["image"].ndim in (2, 3)
                                        and data["dej"].shape == (data["image"].shape[1],)
                                        and 0 <= int(data["index"]) < int(data["num_frames"]))
                        matching_slab = "slab" in data and np.array_equal(data["slab"], QC_SLAB)
                except (OSError, ValueError, KeyError):
                    pass
            have_mip = have_mip and matching_slab
            if have_mip and have_preview:
                print(f"[{number}/{len(paths)}] Already generated: {path.name}", flush=True)
            else:
                if manifest_dir is not None:
                    record_generation(manifest_dir, [path], "processing", output_dir)
                print(f"[{number}/{len(paths)}] Generating: {path.name}", flush=True)
                if model is None:
                    model = load_pytorch_model(default_model_path(), device)
                volume, _, _ = import_dicom(path)
                index = len(volume) // 2
                if not have_mip:
                    mip, dej, _ = papillary_dermis_mip(volume, model, device, slab=QC_SLAB)
                    preview_dej = dej[index]
                    display = resize(mip, EN_FACE_SIZE, anti_aliasing=True, order=1)
                    # Same per-image linear display scaling as the original grayscale plot.
                    low, high = float(display.min()), float(display.max())
                    gray = (np.clip((display - low) / (high - low), 0, 1) * 255).astype(np.uint8) if high > low else np.zeros(display.shape, np.uint8)
                    if not cv2.imwrite(str(target), gray):
                        raise OSError(f"Could not write {target}")
                    provenance = "same_segmentation_as_mip"
                else:
                    mask = test_model_on_sample_image(model, volume[index], device).astype(bool)
                    preview_dej = mask.shape[0] - np.argmax(mask[::-1], axis=0) - 1
                    preview_dej[~mask.any(axis=0)] = -1
                    provenance = "preview_recomputed_for_existing_mip"
                with tempfile.NamedTemporaryFile(dir=output_dir, delete=False, suffix=".npz") as handle:
                    temp = Path(handle.name)
                    try:
                        np.savez_compressed(handle, image=volume[index], dej=preview_dej,
                                            index=index, num_frames=len(volume), provenance=provenance,
                                            slab=np.array(QC_SLAB))
                    except BaseException:
                        temp.unlink(missing_ok=True)
                        raise
                temp.replace(preview)
            if manifest_dir is not None:
                record_generation(manifest_dir, [path], "ready", output_dir)
            generated.append(target)
            if on_generated is not None:
                on_generated(target)
        except Exception as error:
            if manifest_dir is None:
                raise
            record_generation(manifest_dir, [path], "failed", output_dir, str(error))
            print(f"FAILED: {path.name}: {error}", flush=True)
    if manifest_dir is not None:
        print(f"Generation progress: {generation_counts(manifest_dir)}", flush=True)
    return generated


def make_server(store, port=8765, show_names=False, warnings=()):
    token = secrets.token_urlsafe(32)
    page = Path(__file__).with_name("qc_review.html").read_text(encoding="utf-8")
    config = json.dumps({"token": token, "showNames": show_names, "warnings": list(warnings)})
    page = page.replace("/*APP_CONFIG*/", config.replace("<", "\\u003c"))

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body, content_type="application/json", status=200, **headers):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in headers.items():
                self.send_header(key.replace("_", "-"), value)
            try:
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # Navigating to another scan may cancel an image download.
                pass

        def do_GET(self):
            route = urlparse(self.path).path
            try:
                if route == "/":
                    self.respond(page.encode(), "text/html; charset=utf-8")
                elif route == "/api/state":
                    self.respond(json.dumps(store.snapshot()).encode())
                elif route == "/api/export":
                    self.respond(store.export_csv(), "text/csv; charset=utf-8",
                                 Content_Disposition='attachment; filename="qc_reviews.csv"')
                elif route.startswith("/api/image/"):
                    item = store.by_id.get(route.rsplit("/", 1)[-1])
                    if item is None:
                        self.respond(b"Not found", "text/plain", 404)
                    else:
                        self.respond(render_image(item), "image/png")
                elif route.startswith("/api/bscan/"):
                    item = store.by_id.get(route.rsplit("/", 1)[-1])
                    if item is None or not item.get("bscan_path"):
                        self.respond(b"No B-scan preview", "text/plain", 404)
                    else:
                        query = parse_qs(urlparse(self.path).query)
                        overlay = query.get("overlay", ["1"])[0] == "1"
                        opacity = float(query.get("opacity", ["1"])[0])
                        octa = query.get("octa", ["0"])[0] == "1"
                        self.respond(render_bscan(item, overlay, opacity, octa=octa), "image/png")
                else:
                    self.respond(b"Not found", "text/plain", 404)
            except (OSError, ValueError) as error:
                self.respond(json.dumps({"error": str(error)}).encode(), status=400)

        def do_POST(self):
            if self.path != "/api/review" or self.headers.get("X-Review-Token") != token:
                self.respond(b"Forbidden", "text/plain", 403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected an object")
                self.respond(json.dumps(store.save(payload)).encode())
            except RuntimeError as error:
                self.respond(json.dumps({"error": str(error)}).encode(), status=409)
            except (OSError, ValueError, TypeError) as error:
                self.respond(json.dumps({"error": str(error)}).encode(), status=400)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def manifest_generation_paths(manifest_dir):
    import pandas as pd
    scans = pd.read_csv(Path(manifest_dir) / "scans.csv")
    required = {"filename", "filepath", "excluded"}
    if not required.issubset(scans.columns):
        raise ValueError("Manifest must contain filename, filepath and excluded")
    # Unknown exclusion values must not accidentally become approval.
    included = scans["excluded"].map(lambda value: str(value).lower() in {"false", "0"})
    queue = scans.loc[included]
    if queue["filename"].duplicated().any():
        raise ValueError("Duplicate included filenames in manifest")
    paths = [Path(p) for p in queue["filepath"]]
    if any(path.name != name for path, name in zip(paths, queue["filename"])):
        raise ValueError("Manifest filename disagrees with filepath")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pattern", help="Image pattern (default: '*_qc_mip.png')")
    parser.add_argument("--generate-from", type=Path, help="Generate the original QC MIP from a DICOM file or folder")
    parser.add_argument("--generate-manifest", action="store_true",
                        help="Generate non-excluded manifest scans while serving the review UI")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--results", type=Path, default=DEFAULT_REVIEWS, help="Persistent scan-level QC decisions")
    parser.add_argument("--manifest-dir", type=Path, default=REPO_ROOT / "data" / "manifest")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--show-names", action="store_true")
    parser.add_argument("--open", action="store_true", help="Open in the default browser")
    args = parser.parse_args()
    if args.generate_manifest and args.generate_from:
        parser.error("Choose --generate-manifest or --generate-from, not both")
    try:
        if args.generate_from:
            paths = ([args.generate_from] if args.generate_from.is_file() else
                     sorted(p for p in args.generate_from.rglob("*") if p.suffix.lower() == ".dcm"))
            if not paths:
                raise ValueError("No DICOM scans found")
            generate_qc_mips(paths, args.folder, manifest_dir=args.manifest_dir)
        items, warnings = discover_images(args.folder, args.pattern)
        store = ReviewStore(items, args.results, args.reviewer, args.manifest_dir)
        from manifest.qc import sync_manifest_qc
        sync_manifest_qc(args.manifest_dir, args.results)
        queue = manifest_generation_paths(args.manifest_dir) if args.generate_manifest else []
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for port in range(args.port, min(args.port + 20, 65536)):
        try:
            server = make_server(store, port, args.show_names, warnings)
            break
        except OSError as error:
            if error.errno != 48 and error.errno != 98:
                raise
    else:
        parser.error("No free local port found")
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Manual MIP review: {url}\nScans: {len(items)}\nQC MIP: DEJ -5..+15 px (inclusive)\n"
          f"Reviews: {store.path}\nSource images are never moved or deleted.", flush=True)
    for warning in warnings:
        print(f"WARNING: {warning}", flush=True)
    if queue:
        def generate():
            try:
                generate_qc_mips(queue, args.folder, manifest_dir=args.manifest_dir,
                                 manifest_lock=store.lock, on_generated=store.add_generated)
            except Exception as error:
                print(f"Generation stopped: {error}", flush=True)
        threading.Thread(target=generate, name="qc-generation", daemon=True).start()
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
