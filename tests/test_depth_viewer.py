"""Surface-relative depth browsing excludes air and uses half-open intervals."""
import json
import base64
import threading
import csv
import re
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen, Request

import numpy as np
import cv2
import pytest

from depth_slabs import project_slab
from depth_viewer import DepthVolume, make_server, ScanBrowser, scan_catalog


def sample():
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, (3, 20, 5, 3), dtype=np.uint8)
    dej = np.array([[6, 7, 8, 9, np.nan]] * 3)
    surface = np.array([[2, 3, 4, 5, np.nan]] * 3)
    return DepthVolume(image, dej, surface)


@pytest.mark.parametrize("start,thickness", [(0, 3), (1, 2), (0, 1), (0, 5), (4, 6), (15, 4)])
def test_projection_matches_existing_slab_algorithm(start, thickness):
    volume = sample()
    result, valid = volume.project(start, thickness)
    for i in range(volume.n):
        reference, columns = project_slab(volume.flow[i], volume.surface[i], volume.columns[i],
                                          start, start + thickness)
        np.testing.assert_array_equal(result[i], reference)
        np.testing.assert_array_equal(valid[i], columns)


def test_intensity_scale_does_not_change_with_depth():
    image = np.zeros((2, 12, 3, 3), dtype=np.uint8)
    image[:, 1, :, 0] = 20
    image[:, 2, :, 0] = 100
    volume = DepthVolume(image, np.full((2, 3), 3), np.ones((2, 3)))
    np.testing.assert_array_equal(volume.project(0, 1)[0], np.full((2, 3), 20))
    np.testing.assert_array_equal(volume.project(1, 1)[0], np.full((2, 3), 100))


def test_16_bit_display_scale():
    image = np.zeros((1, 10, 2, 3), dtype=np.uint16)
    image[..., 0] = 25700
    volume = DepthVolume(image, [[3, 3]], [[1, 1]])
    np.testing.assert_array_equal(volume.project(0, 1)[0], [[100, 100]])


def test_bscan_removes_red_overlay_without_changing_flow():
    image = np.full((1, 20, 5, 3), 40, dtype=np.uint8)
    image[..., 0] = 200
    volume = DepthVolume(image, np.full((1, 5), 5), np.ones((1, 5)))
    np.testing.assert_array_equal(volume.bscan(0, 0, 1)[15], 40)
    np.testing.assert_array_equal(volume.project(0, 1)[0], 160)


def test_display_interpolation_matches_qc_size_and_preserves_fixed_scale():
    image = np.zeros((2, 12, 3, 3), dtype=np.uint8)
    image[..., 0] = 100
    volume = DepthVolume(image, np.full((2, 3), 3), np.ones((2, 3)))
    encoded = volume.frame(0, 1, 0)["mip"].split(",", 1)[1]
    display = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    assert display.shape == (1355, 1355, 3)
    np.testing.assert_array_equal(display, 100)


def test_invalid_regions_are_not_zero_and_source_not_modified():
    volume = sample()
    original = volume.volume.copy()
    mip, valid = volume.project(19, 2)
    assert not valid.any()
    assert np.isnan(mip).all()
    frame = volume.frame(0, 5, 1)
    assert frame["mip"].startswith("data:image/png;base64,")
    assert frame["bscan"].startswith("data:image/png;base64,")
    np.testing.assert_array_equal(volume.volume, original)


@pytest.mark.parametrize("start,thickness", [(0, 0), (0, 61), (0.5, 3), (True, 2), (99, 2), (-1, 2)])
def test_invalid_depth_arguments(start, thickness):
    with pytest.raises(ValueError):
        sample().project(start, thickness)


def test_wrong_cache_shape_rejected():
    with pytest.raises(ValueError, match="Boundary shape"):
        DepthVolume(np.zeros((2, 10, 3, 3), np.uint8), [[3, 3, 3]], [[1, 1, 1]])


def test_topmost_layer_follows_surface_without_sampling_air():
    image = np.zeros((1, 12, 4, 3), dtype=np.uint8)
    surface = np.array([[0, 2, 4, 6]])
    for col, depth in enumerate(surface[0]):
        image[0, :depth, col, 0] = 255  # Air must not contaminate the projection.
        image[0, depth, col, 0] = 20 + col
    volume = DepthVolume(image, surface + 3, surface)
    mip, valid = volume.project(0, 1)
    np.testing.assert_array_equal(mip, [[20, 21, 22, 23]])
    assert valid.all()
    assert volume.minimum == 0


def test_http_and_invalid_parameters():
    server = make_server(sample(), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert b"OCTA-dybdeviser" in urlopen(base).read()
        assert json.loads(urlopen(base + "/api/info").read())["frames"] == 3
        frame = json.loads(urlopen(base + "/api/frame?start=2&thickness=3&index=1").read())
        assert frame["start"] == 2 and frame["stop"] == 5 and frame["index"] == 1
        for query in ("thickness=999", "index=-1", "start=no"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + "/api/frame?" + query)
            assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_catalog_preserves_canonical_paths_and_searches_multiple_terms(tmp_path):
    manifest = tmp_path / "scans.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "filepath", "is_duplicate_of"])
        writer.writeheader()
        writer.writerows([
            {"filename": "JF_Right Instep_S244.dcm", "filepath": "/drive/JF_Right Instep_S244.dcm"},
            {"filename": "JF_Right Instep_S244.dcm", "filepath": "/copy/JF_Right Instep_S244.dcm", "is_duplicate_of": "original"},
            {"filename": "AC_Right Hand_S1.dcm", "filepath": "/drive/AC_Right Hand_S1.dcm"},
        ])
    catalog = scan_catalog(manifest)
    assert len(catalog) == 2
    browser = ScanBrowser(sample(), catalog)
    match = browser.search("jf INSTEP 244")
    assert match["total"] == 1
    assert catalog[match["items"][0]["id"]]["path"] == Path("/drive/JF_Right Instep_S244.dcm")
    assert browser.search("not-a-scan")["total"] == 0


def test_search_caps_results_without_losing_total():
    catalog = {str(i): {"name": f"scan_{i}.dcm", "path": Path(f"/{i}.dcm")} for i in range(150)}
    browser = ScanBrowser(sample(), catalog)
    assert len(browser.search("scan_")["items"]) == 100
    assert browser.search("scan_")["total"] == 150


def test_scan_switch_loads_in_background_and_guards_old_frames():
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    def loader(path, progress):
        progress(3, 120)
        entered.set()
        assert release.wait(5)
        volume = sample()
        volume.name = path.name
        finished.set()
        return volume
    browser = ScanBrowser(sample(), {"old": {"name": "scan", "path": Path("old.dcm")},
                                    "new": {"name": "next.dcm", "path": Path("next.dcm")}}, loader)
    try:
        browser.select("new")
        assert entered.wait(5)
        info = browser.info()
        assert info["scan_id"] == "old"
        assert info["loading"]["done"] == 3
        with pytest.raises(RuntimeError, match="Another"):
            browser.select("old")
        browser.select("new")  # Repeated selection does not spawn another load.
    finally:
        release.set()
    assert finished.wait(5)
    for _ in range(100):
        if not browser.info()["loading"]:
            break
        time.sleep(.01)
    assert browser.info()["scan_id"] == "new"
    with pytest.raises(RuntimeError, match="changed"):
        browser.frame(0, 1, 0, "old")
    assert browser.frame(0, 1, 0, "new")["scan_id"] == "new"


def test_failed_load_preserves_previous_scan():
    def loader(path, progress):
        raise OSError("Drive disconnected")
    browser = ScanBrowser(sample(), {"new": {"name": "missing.dcm", "path": Path("missing.dcm")}}, loader)
    browser.select("new")
    for _ in range(100):
        if not browser.info()["loading"]:
            break
        time.sleep(.01)
    assert browser.info()["name"] == "scan"
    assert browser.info()["load_error"] == "Drive disconnected"
    assert browser.volume is not None


def test_selection_endpoint_requires_token_and_rejects_arbitrary_paths():
    server = make_server(sample(), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        page = urlopen(base).read().decode()
        token = json.loads(re.search(r"const scanConfig=(\{.*?\});", page).group(1))["token"]
        scans = json.loads(urlopen(base + "/api/scans?q=scan").read())
        assert scans["total"] == 1
        key = scans["items"][0]["id"]
        for provided, selection, expected in [(None, key, 403), (token, "/etc/passwd", 400)]:
            request = Request(base + "/api/select", data=json.dumps({"id": selection}).encode(),
                              headers={"X-Viewer-Token": provided or ""})
            with pytest.raises(HTTPError) as error:
                urlopen(request)
            assert error.value.code == expected
        request = Request(base + "/api/select", data=json.dumps({"id": key}).encode(),
                          headers={"X-Viewer-Token": token})
        assert json.loads(urlopen(request).read())["scan_id"] == key
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
