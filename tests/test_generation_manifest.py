"""Generation progress and manual QC survive separate runs and manifest refresh."""
from pathlib import Path

import numpy as np
import pandas as pd

from manifest.generation import apply_generation, generation_counts, load_generation, update_generation
from manifest.qc import sync_manifest_qc
from qc_review import ReviewStore, discover_images, generate_qc_mips


def test_persistent_queue_and_interrupted_run(tmp_path):
    paths = [tmp_path / "a.dcm", tmp_path / "b.dcm"]
    update_generation(tmp_path, paths)
    update_generation(tmp_path, [paths[0]], "processing")
    update_generation(tmp_path, paths)
    assert generation_counts(tmp_path) == {"total": 2, "pending": 2, "processing": 0, "ready": 0, "failed": 0}
    update_generation(tmp_path, [paths[0]], "ready")
    update_generation(tmp_path, [paths[1]], "failed", error="Read failure")
    assert load_generation(tmp_path).set_index("filename").loc["b.dcm", "generation_error"] == "Read failure"
    scans = apply_generation(pd.DataFrame({"filename": ["b.dcm", "a.dcm", "c.dcm"]}), tmp_path)
    assert scans["generation_status"].tolist() == ["failed", "ready", "pending"]


def test_failed_scan_does_not_abort_batch_and_restart_skips_ready(monkeypatch, tmp_path):
    import papi
    output, manifest = tmp_path / "mips", tmp_path / "manifest"
    bad, good = tmp_path / "bad.dcm", tmp_path / "good.dcm"
    called = []
    def read(path):
        called.append(Path(path).name)
        if Path(path).name == "bad.dcm":
            raise OSError("Disconnected")
        return np.zeros((2, 10, 4, 3), np.uint8), None, None
    monkeypatch.setattr(papi, "import_dicom", read)
    monkeypatch.setattr(papi, "EN_FACE_SIZE", (4, 4))
    monkeypatch.setattr(papi, "papillary_dermis_mip", lambda *a, **kw: (np.zeros((2, 4)), np.full((2, 4), 5), None))
    generate_qc_mips([bad, good], output, model=object(), device="cpu", manifest_dir=manifest)
    assert generation_counts(manifest)["failed"] == 1
    assert generation_counts(manifest)["ready"] == 1
    called.clear()
    generate_qc_mips([bad, good], output, model=object(), device="cpu", manifest_dir=manifest)
    assert called == ["bad.dcm"]
    assert generation_counts(manifest)["ready"] == 1


def test_generation_updates_do_not_change_qc_decision(monkeypatch, tmp_path):
    import cv2
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    cv2.imwrite(str(image_folder / "a_qc_mip.png"), np.zeros((4, 4), np.uint8))
    items = discover_images(image_folder)[0]
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    pd.DataFrame({"filename": ["a.dcm"], "excluded": [False]}).to_csv(manifest / "scans.csv", index=False)
    store = ReviewStore(items, manifest / "qc_reviews.json", "Reviewer", manifest)
    store.save({"id": items[0]["id"], "revision": 0, "status": "use"})
    original = store.path.read_bytes()
    update_generation(manifest, [tmp_path / "a.dcm"], "ready", image_folder)
    scans = pd.read_csv(manifest / "scans.csv")
    assert scans.loc[0, "qc_status"] == "use"
    assert bool(scans.loc[0, "analysis_eligible"])
    assert scans.loc[0, "generation_status"] == "ready"
    assert store.path.read_bytes() == original
    sync_manifest_qc(manifest, store.path)
    assert pd.read_csv(manifest / "scans.csv").loc[0, "generation_status"] == "ready"
