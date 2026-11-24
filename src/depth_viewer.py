"""Read-only, interactive skin-surface-relative MIP viewer, separate from QC."""

import argparse
import base64
import csv
import hashlib
import json
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from skimage.transform import resize

from depth_slabs import extract_dej
from image_utils import structural_image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN = ROOT / "data/dicom/AC_Right Hallux_L1051_S2387__20_05_2021.dcm"


class DepthVolume:
    def __init__(self, volume, dej, surface, name="scan"):
        if volume.ndim != 4 or volume.shape[-1] != 3 or volume.dtype not in (np.uint8, np.uint16):
            raise ValueError("Expected RGB OCTA volume (N, H, W, 3), uint8 or uint16")
        if 0 in volume.shape:
            raise ValueError("Empty volume")
        self.volume = volume
        self.n, self.h, self.w = volume.shape[:3]
        self.dej, self.surface = np.asarray(dej, dtype=float), np.asarray(surface, dtype=float)
        if self.dej.shape != (self.n, self.w) or self.surface.shape != self.dej.shape:
            raise ValueError("Boundary shape does not match volume")
        self.columns = (np.isfinite(self.dej) & np.isfinite(self.surface)
                        & (self.surface >= 0) & (self.surface < self.dej) & (self.dej <= self.h)
                        & (self.dej == np.floor(self.dej)) & (self.surface == np.floor(self.surface)))
        self.edges = np.where(self.columns, self.surface, 0).astype(int)
        # A single display scale is used for every depth, not min/max per MIP.
        scale = 257.0 if volume.dtype == np.uint16 else 1.0
        self.flow = np.maximum(volume[..., 0].astype(np.float32) - volume[..., 1], 0) / scale
        self.name = name
        self.minimum, self.maximum = 0, min(200, self.h - 1)

    def project(self, start, thickness):
        if (not isinstance(start, int) or isinstance(start, bool) or
                not isinstance(thickness, int) or isinstance(thickness, bool)):
            raise ValueError("Depth and thickness must be integer pixels")
        if not self.minimum <= start <= self.maximum or not 1 <= thickness <= 60:
            raise ValueError("Requested depth or thickness outside viewer range")
        lower = self.edges + start
        valid = self.columns & (lower >= self.surface) & (lower >= 0) & (lower + thickness <= self.h)
        projection = np.full((self.n, self.w), -np.inf, dtype=np.float32)
        frames, columns = np.arange(self.n)[:, None], np.arange(self.w)[None, :]
        for offset in range(thickness):
            # Clip only for safe array indexing; incomplete intervals remain invalid.
            values = self.flow[frames, np.clip(lower + offset, 0, self.h - 1), columns]
            projection = np.maximum(projection, values)
        projection[~valid] = np.nan
        return projection, valid

    def bscan(self, index, start, thickness):
        if not 0 <= index < self.n:
            raise ValueError("B-scan index out of range")
        image = self.volume[index]
        image = (image / 257).astype(np.uint8) if image.dtype == np.uint16 else image.copy()
        image = cv2.cvtColor(structural_image(image), cv2.COLOR_GRAY2BGR)
        top = self.edges[index] + start
        valid = (self.columns[index] & (top >= self.surface[index]) & (top >= 0)
                 & (top + thickness <= self.h))
        rows = np.arange(self.h)[:, None]
        band = valid[None, :] & (rows >= top) & (rows < top + thickness)
        image[band] = (image[band].astype(float) * .75 + np.array([210, 185, 30]) * .25).astype(np.uint8)
        for line, color in [(self.surface[index], (255, 255, 255)),
                            (self.dej[index], (0, 255, 255)),
                            (np.where(valid, top, np.nan), (255, 220, 40)),
                            (np.where(valid, top + thickness, np.nan), (255, 220, 40))]:
            cols = np.flatnonzero(self.columns[index] & np.isfinite(line))
            for run in np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1):
                if run.size:
                    # Boundaries are pixel edges; use row-center coordinates for display.
                    points = np.column_stack((run, np.clip(line[run] - .5, 0, self.h - 1))).round().astype(np.int32)
                    cv2.polylines(image, [points], False, color, 1)
        return image

    def frame(self, start, thickness, index):
        projection, valid = self.project(start, thickness)
        # Match QC's display interpolation without changing quantitative projections.
        display = resize(np.nan_to_num(projection), (1355, 1355), anti_aliasing=True, order=1)
        display_valid = resize(valid, (1355, 1355), order=0, anti_aliasing=False)
        gray = np.clip(display, 0, 255).astype(np.uint8)
        mip = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        mip[~display_valid] = (176, 119, 217)
        columns = np.flatnonzero(self.columns[index])
        return {"start": start, "stop": start + thickness, "thickness": thickness,
                "surface_at_right": float(self.surface[index, columns[-1]]) if columns.size else None,
                "index": index, "reference": "skin_surface", "valid_fraction": float(valid.mean()),
                "mip": encode(mip), "bscan": encode(self.bscan(index, start, thickness))}


def encode(image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Cannot encode image")
    return "data:image/png;base64," + base64.b64encode(data).decode()


def load_volume(path, boundaries=None, progress=None):
    from papi import import_dicom, test_model_on_sample_image
    from model_utils import default_model_path, load_pytorch_model
    import torch

    if progress:
        progress(0, 0)
    volume, _, _ = import_dicom(path)
    candidate = boundaries or ROOT / "outputs/dej_depth_test" / f"{path.stem}_depth_slabs.npz"
    if candidate.exists():
        print(f"Using stored DEJ boundaries: {candidate}", flush=True)
        with np.load(candidate, allow_pickle=False) as data:
            dej, surface = data["dej_px"], data["surface_px"]
    elif boundaries is not None:
        raise ValueError(f"Boundary file not found: {boundaries}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_pytorch_model(default_model_path(), device)
        dej = np.full((len(volume), volume.shape[2]), np.nan)
        surface = np.full_like(dej, np.nan)
        for i, image in enumerate(volume):
            mask = test_model_on_sample_image(model, image, device)
            dej[i], surface[i], _ = extract_dej(mask)
            if progress:
                progress(i + 1, len(volume))
            if i % 10 == 0 or i == len(volume) - 1:
                print(f"Segmenting {i + 1}/{len(volume)} B-scans", flush=True)
    return DepthVolume(volume, dej, surface, path.name)


def scan_catalog(manifest, initial_path=None):
    """Use manifest paths only; duplicate file copies remain one selectable scan."""
    by_name = {}
    manifest = Path(manifest)
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                name, source = row.get("filename"), row.get("filepath")
                if not name or not source:
                    continue
                duplicate = bool(row.get("is_duplicate_of", ""))
                if name in by_name and duplicate:
                    continue
                path = Path(source)
                if not path.is_absolute():
                    path = ROOT / path
                by_name[name] = {"path": path, "name": name}
    if initial_path is not None:
        path = Path(initial_path).resolve()
        by_name[path.name] = {"path": path, "name": path.name}
    return {hashlib.sha256(name.encode()).hexdigest(): entry
            for name, entry in sorted(by_name.items())}


class ScanBrowser:
    def __init__(self, volume, catalog=None, loader=None):
        self.catalog = catalog or {"initial": {"name": volume.name, "path": Path(volume.name)}}
        self.active = next((key for key, entry in self.catalog.items() if entry["name"] == volume.name), "initial")
        if self.active not in self.catalog:
            self.catalog[self.active] = {"name": volume.name, "path": Path(volume.name)}
        self.volume = volume
        self.loader = loader or (lambda path, progress: load_volume(path, progress=progress))
        self.lock = threading.Lock()
        self.loading = None
        self.load_error = None

    def info(self):
        with self.lock:
            v = self.volume
            return {"name": v.name, "scan_id": self.active, "minimum": v.minimum,
                    "maximum": v.maximum, "frames": v.n, "width": v.w, "height": v.h,
                    "reference": "skin_surface", "loading": dict(self.loading) if self.loading else None,
                    "load_error": self.load_error}

    def search(self, query):
        terms = query.casefold().split()
        matches = [{"id": key, "name": entry["name"]} for key, entry in self.catalog.items()
                   if all(term in entry["name"].casefold() for term in terms)]
        return {"items": matches[:100], "total": len(matches)}

    def select(self, key):
        with self.lock:
            if key not in self.catalog:
                raise ValueError("Unknown scan")
            if self.loading:
                if self.loading["id"] == key:
                    return
                raise RuntimeError("Another scan is loading")
            self.load_error = None
            if key == self.active:
                return
            entry = self.catalog[key]
            self.loading = {"id": key, "name": entry["name"], "done": 0, "total": 0}

        def progress(done, total):
            with self.lock:
                self.loading.update(done=done, total=total)

        def load():
            try:
                loaded = self.loader(entry["path"], progress)
                with self.lock:
                    self.volume, self.active = loaded, key
            except Exception as error:
                with self.lock:
                    self.load_error = str(error)
            finally:
                with self.lock:
                    self.loading = None

        threading.Thread(target=load, name="depth-scan-loader", daemon=True).start()

    def frame(self, start, thickness, index, key=None):
        with self.lock:
            if key is not None and key != self.active:
                raise RuntimeError("The active scan changed; reload its metadata")
            result = self.volume.frame(start, thickness, index)
            result["scan_id"] = self.active
            return result


def make_server(volume, port=8767, catalog=None, loader=None):
    browser = ScanBrowser(volume, catalog, loader)
    token = secrets.token_urlsafe(32)
    page = Path(__file__).with_name("depth_viewer.html").read_text()
    page = page.replace("/*SCAN_CONFIG*/", json.dumps({"token": token}))

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body, kind="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            try:
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            request = urlparse(self.path)
            if request.path == "/":
                self.respond(page.encode(), "text/html; charset=utf-8")
            elif request.path == "/api/info":
                self.respond(json.dumps(browser.info()).encode())
            elif request.path == "/api/scans":
                query = parse_qs(request.query).get("q", [""])[0]
                self.respond(json.dumps(browser.search(query)).encode())
            elif request.path == "/api/frame":
                query = parse_qs(request.query)
                try:
                    start = int(query.get("start", ["0"])[0])
                    thickness = int(query.get("thickness", ["5"])[0])
                    index = int(query.get("index", [str(browser.info()["frames"] // 2)])[0])
                    key = query.get("scan_id", [None])[0]
                    frame = browser.frame(start, thickness, index, key)
                    self.respond(json.dumps(frame).encode())
                except RuntimeError as error:
                    self.respond(json.dumps({"error": str(error)}).encode(), status=409)
                except ValueError as error:
                    self.respond(json.dumps({"error": str(error)}).encode(), status=400)
            else:
                self.respond(b"Not found", "text/plain", 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/select":
                self.respond(b"Not found", "text/plain", 404)
                return
            if self.headers.get("X-Viewer-Token") != token:
                self.respond(b"Forbidden", "text/plain", 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict) or not isinstance(data.get("id"), str):
                    raise ValueError("Invalid scan selection")
                browser.select(data["id"])
                self.respond(json.dumps(browser.info()).encode())
            except RuntimeError as error:
                self.respond(json.dumps({"error": str(error)}).encode(), status=409)
            except (ValueError, UnicodeDecodeError) as error:
                self.respond(json.dumps({"error": str(error)}).encode(), status=400)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dicom", nargs="?", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--boundaries", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest/scans.csv")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    try:
        volume = load_volume(args.dicom, args.boundaries)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for port in range(args.port, min(args.port + 20, 65536)):
        try:
            server = make_server(volume, port, catalog=scan_catalog(args.manifest, args.dicom))
            break
        except OSError as error:
            if error.errno not in (48, 98):
                raise
    else:
        parser.error("No free port")
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Depth viewer: {url}\nRead-only; QC and manifest are not modified.\nDepths are pixels, not verified micrometres.", flush=True)
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
