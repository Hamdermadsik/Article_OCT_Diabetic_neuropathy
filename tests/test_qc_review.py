"""Scan-level QC is keyed by filename, not image contents or individual layers."""
import csv
import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pandas as pd
import pytest

from qc_review import ReviewStore, discover_images, generate_qc_mips, make_server, render_image, render_bscan
from manifest.qc import apply_qc_reviews, load_qc_reviews, sync_manifest_qc


@pytest.fixture
def inputs(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    cv2.imwrite(str(folder / "patient_qc_mip.png"), np.array([[0, 255], [100, 50]], dtype=np.uint8))
    return folder


def payload(store, **kwargs):
    return {"id": store.items[0]["id"], "status": "use", "revision": store.data["revision"],
            "reasons": [], "note": "", **kwargs}


def test_scan_identity_and_render(inputs):
    items, warnings = discover_images(inputs)
    assert not warnings
    assert items[0]["filename"] == "patient.dcm"
    assert "source_sha256" not in items[0] and "layer" not in items[0]
    png = cv2.imdecode(np.frombuffer(render_image(items[0]), np.uint8), cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(png, [[0, 255], [100, 50]])


def test_legacy_pattern(inputs):
    cv2.imwrite(str(inputs / "legacy_papillary_dermis_mip.png"), np.zeros((3, 5), np.uint8))
    items, _ = discover_images(inputs, pattern="*_papillary_dermis_mip.png")
    assert items[0]["filename"] == "legacy.dcm"


def test_corrupt_and_duplicate_images_reported(inputs):
    (inputs / "bad_qc_mip.png").write_bytes(b"not an image")
    nested = inputs / "copy"
    nested.mkdir()
    cv2.imwrite(str(nested / "patient_qc_mip.png"), np.zeros((2, 2), np.uint8))
    items, warnings = discover_images(inputs)
    assert len(items) == 1 and len(warnings) == 2


def test_no_images_fails(tmp_path):
    with pytest.raises(ValueError, match="No readable"):
        discover_images(tmp_path)


def test_resume_keeps_only_latest_decision_and_source_preserved(inputs, tmp_path):
    items, _ = discover_images(inputs)
    source = inputs / "patient_qc_mip.png"
    original = source.read_bytes()
    store = ReviewStore(items, tmp_path / "qc_reviews.json", "Magnus")
    assert b"unreviewed" in store.export_csv()
    store.save(payload(store, status="unsure"))
    resumed = ReviewStore(items, store.path, "Magnus")
    resumed.save(payload(resumed, status="discard"))
    record = resumed.snapshot()["reviews"][items[0]["id"]]
    assert record["status"] == "discard"
    assert "history" not in record
    assert "history" not in json.loads(store.path.read_text())["reviews"][items[0]["id"]]
    row = next(csv.DictReader(io.StringIO(resumed.export_csv().decode("utf-8-sig"))))
    assert row["filename"] == "patient.dcm" and row["qc_status"] == "discard"
    assert "source_sha256" not in row
    assert source.read_bytes() == original


def test_csv_formula_is_literal(inputs, tmp_path):
    store = ReviewStore(discover_images(inputs)[0], tmp_path / "reviews.json", "=Reviewer")
    store.save(payload(store, note="=1+1"))
    row = next(csv.DictReader(io.StringIO(store.export_csv().decode("utf-8-sig"))))
    assert row["note"] == "'=1+1" and row["reviewer"] == "'=Reviewer"


@pytest.mark.parametrize("fields", [{"status": "bad"}, {"reasons": ["bad"]},
    {"reasons": "motion"}, {"note": 3}, {"note": "x" * 5001}, {"id": "missing"}])
def test_invalid_not_saved(inputs, tmp_path, fields):
    store = ReviewStore(discover_images(inputs)[0], tmp_path / "reviews.json", "A")
    with pytest.raises(ValueError):
        store.save(payload(store, **fields))
    assert not store.path.exists()


def test_stale_window_or_server_cannot_overwrite(inputs, tmp_path):
    items = discover_images(inputs)[0]
    store = ReviewStore(items, tmp_path / "reviews.json", "A")
    other = ReviewStore(items, store.path, "B")
    original = payload(store)
    store.save(original)
    with pytest.raises(RuntimeError):
        store.save(original)
    with pytest.raises(RuntimeError):
        other.save(payload(other))


def test_regenerated_mip_keeps_decision(inputs, tmp_path):
    items = discover_images(inputs)[0]
    store = ReviewStore(items, tmp_path / "reviews.json", "A")
    store.save(payload(store))
    cv2.imwrite(str(inputs / "patient_qc_mip.png"), np.full((3, 3), 60, np.uint8))
    refreshed = discover_images(inputs)[0]
    assert refreshed[0]["id"] == items[0]["id"]
    assert ReviewStore(refreshed, store.path, "A").snapshot()["reviews"][items[0]["id"]]["status"] == "use"


def test_generate_original_slab(monkeypatch, tmp_path):
    import papi
    seen = []
    monkeypatch.setattr(papi, "EN_FACE_SIZE", (4, 4))
    monkeypatch.setattr(papi, "import_dicom", lambda path: (np.zeros((2, 20, 3, 3)), None, None))
    def mip(volume, model, device, slab):
        seen.append(slab)
        return np.array([[0., 5., 10.], [0., 5., 10.]]), np.full((2, 3), 5), None
    monkeypatch.setattr(papi, "papillary_dermis_mip", mip)
    files = generate_qc_mips([tmp_path / "AC.dcm"], tmp_path / "mips", model=object(), device="cpu")
    assert seen == [(0, 15)]
    assert files[0].name == "AC_qc_mip.png"
    assert cv2.imread(str(files[0])).shape == (4, 4, 3)
    with np.load(tmp_path / "mips" / "AC_qc_bscan.npz") as preview:
        assert int(preview["index"]) == 1
        assert int(preview["num_frames"]) == 2
        np.testing.assert_array_equal(preview["dej"], [5, 5, 5])
        assert str(preview["provenance"]) == "same_segmentation_as_mip"


def test_existing_mip_gets_preview_without_regenerating_it(monkeypatch, tmp_path):
    import papi
    folder = tmp_path / "mips"
    folder.mkdir()
    target = folder / "AC_qc_mip.png"
    cv2.imwrite(str(target), np.full((4, 4), 70, np.uint8))
    original = target.read_bytes()
    monkeypatch.setattr(papi, "import_dicom", lambda path: (np.zeros((3, 20, 4, 3), np.uint8), None, None))
    mask = np.zeros((20, 4), bool)
    mask[2:6, :3] = True
    monkeypatch.setattr(papi, "test_model_on_sample_image", lambda *args: mask)
    monkeypatch.setattr(papi, "papillary_dermis_mip", lambda *a, **kw: pytest.fail("MIP must not be recalculated"))
    generate_qc_mips([tmp_path / "AC.dcm"], folder, model=object(), device="cpu")
    assert target.read_bytes() == original
    items, _ = discover_images(folder)
    assert items[0]["bscan"] == {"index": 1, "num_frames": 3, "recomputed": True}
    with np.load(items[0]["bscan_path"]) as preview:
        np.testing.assert_array_equal(preview["dej"], [5, 5, 5, -1])


def test_bscan_toggle_and_missing_preview(inputs):
    items, _ = discover_images(inputs)
    with pytest.raises(FileNotFoundError):
        render_bscan(items[0])
    image = np.full((20, 20, 3), 40, np.uint8)
    image[..., 0] = 200
    np.savez_compressed(inputs / "patient_qc_bscan.npz", image=image,
                        dej=np.full(20, 5), index=60, num_frames=120)
    items, _ = discover_images(inputs)
    plain = cv2.imdecode(np.frombuffer(render_bscan(items[0], False), np.uint8), cv2.IMREAD_COLOR)
    overlay = cv2.imdecode(np.frombuffer(render_bscan(items[0], True), np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(plain, np.full_like(image, 40))
    assert np.any(overlay != plain)
    np.testing.assert_array_equal(overlay[15], plain[15])
    invisible = cv2.imdecode(np.frombuffer(render_bscan(items[0], opacity=0), np.uint8), cv2.IMREAD_COLOR)
    faded = cv2.imdecode(np.frombuffer(render_bscan(items[0], opacity=.5), np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(invisible, plain)
    np.testing.assert_allclose(faded, (overlay.astype(float) + plain.astype(float)) / 2, atol=1)
    np.testing.assert_array_equal(faded[15], plain[15])
    for opacity in (-1, 2, float("nan")):
        with pytest.raises(ValueError):
            render_bscan(items[0], opacity=opacity)


def test_manifest_only_approved_and_preserve_other_exclusions(inputs, tmp_path):
    store = ReviewStore(discover_images(inputs)[0], tmp_path / "reviews.json", "A")
    scans = pd.DataFrame({"filename": ["patient.dcm", "patient.dcm", "other.dcm"],
                          "excluded": [False, True, False]})
    assert not apply_qc_reviews(scans, store.path)["analysis_eligible"].any()
    store.save(payload(store))
    updated = apply_qc_reviews(scans, store.path)
    assert updated["analysis_eligible"].tolist() == [True, False, False]
    assert updated["excluded"].tolist() == [False, True, False]
    for status in ("discard", "unsure", "unreviewed"):
        store.save(payload(store, status=status))
        assert not apply_qc_reviews(scans, store.path)["analysis_eligible"].any()


@pytest.mark.parametrize("status,eligible", [("use", True), ("discard", False), ("unsure", False)])
def test_reason_checkboxes_never_override_decision(inputs, tmp_path, status, eligible):
    store = ReviewStore(discover_images(inputs)[0], tmp_path / "reviews.json", "A")
    scans = pd.DataFrame({"filename": ["patient.dcm"], "excluded": [False]})
    for reasons in ([], ["segmentation"], ["noise", "motion", "segmentation", "other"]):
        store.save(payload(store, status=status, reasons=reasons, note="Comment only"))
        result = apply_qc_reviews(scans, store.path)
        assert result.loc[0, "qc_status"] == status
        assert bool(result.loc[0, "analysis_eligible"]) is eligible


def test_save_updates_existing_manifest(inputs, tmp_path):
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    pd.DataFrame({"filename": ["patient.dcm", "other.dcm"], "excluded": [False, False]}).to_csv(manifest / "scans.csv", index=False)
    store = ReviewStore(discover_images(inputs)[0], manifest / "qc_reviews.json", "A", manifest)
    store.save(payload(store))
    assert pd.read_csv(manifest / "scans_for_analysis.csv")["filename"].tolist() == ["patient.dcm"]
    store.save(payload(store, status="discard"))
    assert pd.read_csv(manifest / "scans_for_analysis.csv").empty
    assert pd.read_csv(manifest / "approved_scan_filenames.csv").empty


def test_technical_reason_codes_survive_resume_export_and_manifest(inputs, tmp_path):
    reasons = ["noise", "motion", "signal_loss", "blur", "clipping", "segmentation", "other"]
    items = discover_images(inputs)[0]
    store = ReviewStore(items, tmp_path / "qc_reviews.json", "A")
    store.save(payload(store, status="unsure", reasons=reasons))
    resumed = ReviewStore(items, store.path, "A")
    assert resumed.snapshot()["reviews"][items[0]["id"]]["reasons"] == sorted(reasons)
    row = next(csv.DictReader(io.StringIO(resumed.export_csv().decode("utf-8-sig"))))
    assert set(row["reasons"].split(";")) == set(reasons)
    scans = apply_qc_reviews(pd.DataFrame({"filename": ["patient.dcm"], "excluded": [False]}), store.path)
    assert set(scans.loc[0, "qc_reasons"].split(";")) == set(reasons)
    assert not bool(scans.loc[0, "analysis_eligible"])


@pytest.mark.parametrize("status", ["use", "discard", "unsure", "unreviewed"])
def test_segmentation_candidates_independent_of_analysis_status(inputs, tmp_path, status):
    np.savez_compressed(inputs / "patient_qc_bscan.npz", image=np.zeros((20, 20, 3), np.uint8),
                        dej=np.full(20, 5), index=60, num_frames=120)
    items = discover_images(inputs)[0]
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    pd.DataFrame({"filename": ["patient.dcm"], "excluded": [True],
                  "patient_id": ["P-001"], "filepath": ["/raw/patient.dcm"]}).to_csv(manifest / "scans.csv", index=False)
    store = ReviewStore(items, manifest / "qc_reviews.json", "A", manifest)
    store.save(payload(store, status=status, reasons=["segmentation"]))
    scans = pd.read_csv(manifest / "scans.csv")
    assert bool(scans.loc[0, "segmentation_review_required"])
    assert scans.loc[0, "qc_bscan_index"] == 60
    assert not bool(scans.loc[0, "analysis_eligible"])
    candidates = pd.read_csv(manifest / "segmentation_review_candidates.csv")
    assert candidates["filename"].tolist() == ["patient.dcm"]
    assert candidates.loc[0, "patient_id"] == "P-001"
    assert candidates.loc[0, "filepath"] == "/raw/patient.dcm"
    exported = next(csv.DictReader(io.StringIO(store.export_csv().decode("utf-8-sig"))))
    assert exported["qc_bscan_index"] == "60"
    assert exported["segmentation_review_required"] == "True"
    store.save(payload(store, status=status, reasons=[]))
    assert pd.read_csv(manifest / "segmentation_review_candidates.csv").empty


def test_old_segmentation_flag_does_not_invent_bscan_number(tmp_path):
    path = tmp_path / "qc_reviews.json"
    path.write_text(json.dumps({"schema_version": 2, "reviews": {
        "a": {"filename": "patient.dcm", "status": "discard", "reasons": ["segmentation"]}}}))
    scans = apply_qc_reviews(pd.DataFrame({"filename": ["patient.dcm"], "excluded": [False]}), path)
    assert bool(scans.loc[0, "segmentation_review_required"])
    assert pd.isna(scans.loc[0, "qc_bscan_index"])


def test_without_scan_manifest_only_filename_list_is_written(inputs, tmp_path):
    store = ReviewStore(discover_images(inputs)[0], tmp_path / "qc_reviews.json", "A")
    store.save(payload(store))
    assert sync_manifest_qc(tmp_path, store.path) is False
    assert not (tmp_path / "scans.csv").exists()
    assert pd.read_csv(tmp_path / "approved_scan_filenames.csv")["filename"].tolist() == ["patient.dcm"]


def test_invalid_review_format_or_duplicate_filenames(tmp_path):
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps({"schema_version": 1, "reviews": {}}))
    with pytest.raises(ValueError):
        load_qc_reviews(path)
    path.write_text(json.dumps({"schema_version": 2, "reviews": {
        "a": {"filename": "same.dcm", "status": "use"},
        "b": {"filename": "same.dcm", "status": "discard"}}}))
    with pytest.raises(ValueError, match="multiple"):
        load_qc_reviews(path)


def test_http_endpoints(inputs, tmp_path):
    np.savez_compressed(inputs / "patient_qc_bscan.npz", image=np.full((20, 20, 3), 40, np.uint8),
                        dej=np.full(20, 5), index=60, num_frames=120)
    items = discover_images(inputs)[0]
    store = ReviewStore(items, tmp_path / "reviews.json", "Tester")
    server = make_server(store, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        page = urlopen(base).read().decode()
        assert "/*APP_CONFIG*/" not in page
        assert "path" not in json.loads(urlopen(base + "/api/state").read())["items"][0]
        assert urlopen(base + "/api/image/" + items[0]["id"]).read().startswith(b"\x89PNG")
        bscan_route = base + "/api/bscan/" + items[0]["id"]
        plain = urlopen(bscan_route + "?overlay=0").read()
        overlay = urlopen(bscan_route + "?overlay=1").read()
        assert plain.startswith(b"\x89PNG") and plain != overlay
        body = json.dumps(payload(store)).encode()
        with pytest.raises(HTTPError) as rejected:
            urlopen(Request(base + "/api/review", data=body, method="POST"))
        assert rejected.value.code == 403
        config = json.loads(page.split("const config=", 1)[1].split(";", 1)[0])
        response = urlopen(Request(base + "/api/review", data=body, method="POST",
                                  headers={"X-Review-Token": config["token"]}))
        assert json.loads(response.read())["revision"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
