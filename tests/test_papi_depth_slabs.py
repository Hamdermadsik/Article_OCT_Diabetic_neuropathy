"""Synthetic integration tests; no checkpoint or patient scans required."""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

import papi


def synthetic_scan(monkeypatch):
    volume = np.zeros((3, 14, 4, 3), dtype=np.uint8)
    volume[..., 0] = np.arange(14, dtype=np.uint8)[None, :, None]
    mask = np.zeros((14, 4), dtype=bool)
    mask[2:6, :3] = True
    calls = []

    def segment(model, image, device):
        calls.append(image)
        return mask

    monkeypatch.setattr(papi, "test_model_on_sample_image", segment)
    return volume, calls


def test_four_layers_use_one_segmentation_per_bscan(monkeypatch):
    volume, calls = synthetic_scan(monkeypatch)
    result = papi.depth_slab_mips(volume, None, "cpu", 0, 2)

    assert len(calls) == len(volume)
    assert result["mips"].shape == (4, 3, 4)
    np.testing.assert_array_equal(result["offsets_px"], [[-2, 0], [0, 2], [2, 4], [4, 6]])
    for layer, expected in enumerate([5, 7, 9, 11]):
        np.testing.assert_array_equal(result["mips"][layer, :, :3], np.full((3, 3), expected))
    assert np.isnan(result["mips"][:, :, 3]).all()
    np.testing.assert_array_equal(result["common_valid"], result["valid"].all(axis=0))
    assert not result["common_valid"][:, 3].any()


def test_margin_and_control_validity_are_independent_of_dermal_roi(monkeypatch):
    volume, _ = synthetic_scan(monkeypatch)
    result = papi.depth_slab_mips(volume, None, "cpu", 3, 2)
    np.testing.assert_array_equal(result["offsets_px"], [[-5, -3], [3, 5], [5, 7], [7, 9]])
    assert not result["valid"][0].any()  # Control crosses above surface.
    assert not result["valid"][3].any()  # Deep layer extends past the image.
    assert result["valid"][1:3, :, :3].all()

    result = papi.depth_slab_mips(volume, None, "cpu", 0, 5)
    assert not result["valid"][0].any()
    assert result["valid"][1, :, :3].all()


def test_invalid_control_does_not_remove_valid_dermal_area(monkeypatch):
    volume, _ = synthetic_scan(monkeypatch)
    mask = np.zeros((14, 4), dtype=bool)
    mask[5:6] = True
    monkeypatch.setattr(papi, "test_model_on_sample_image", lambda *args: mask)
    result = papi.depth_slab_mips(volume, None, "cpu", 0, 2)
    assert not result["common_valid"].any()
    assert result["common_dermal_valid"].all()


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_output_arrays_metadata_and_figures(monkeypatch, tmp_path, dtype):
    volume, _ = synthetic_scan(monkeypatch)
    volume = volume.astype(dtype)
    result = papi.depth_slab_mips(volume, None, "cpu", 0, 2)
    papi.save_depth_slab_outputs(result, volume, tmp_path / "scan.dcm", tmp_path, (1, 2, 3))
    with np.load(tmp_path / "scan_depth_slabs.npz") as saved:
        for key, value in result.items():
            np.testing.assert_array_equal(saved[key], value)
    metadata = json.loads((tmp_path / "scan_depth_slabs.json").read_text())
    assert metadata["physical_spacing_verified"] is False
    assert metadata["slabs_px"]["superficial"] == [0, 2]
    assert metadata["valid_fraction"]["superficial"] == 0.75
    for suffix in ("mips", "boundaries"):
        assert (tmp_path / f"scan_depth_slabs_{suffix}.png").stat().st_size > 1000


def test_empty_segmentation_saves_without_inventing_signal(monkeypatch, tmp_path):
    volume, _ = synthetic_scan(monkeypatch)
    monkeypatch.setattr(papi, "test_model_on_sample_image", lambda *args: np.zeros((14, 4)))
    result = papi.depth_slab_mips(volume, None, "cpu", 0, 2)
    assert np.isnan(result["mips"]).all()
    assert not result["valid"].any()
    papi.save_depth_slab_outputs(result, volume, "empty.dcm", tmp_path, (1, 1, 1))


@pytest.mark.parametrize("n_slices", [1, 3])
def test_boundaries_show_short_markers_and_full_overlays(monkeypatch, tmp_path, n_slices):
    from matplotlib.figure import Figure

    volume, _ = synthetic_scan(monkeypatch)
    volume = volume[:n_slices]
    result = papi.depth_slab_mips(volume, None, "cpu", 0, 2)
    captured = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *args, **kwargs: captured.append(self))
    papi.save_depth_slab_boundaries(result, volume, tmp_path / "boundaries.png")
    axes = captured[0].axes
    assert len(axes) == 2 * n_slices
    for left, right in zip(axes[::2], axes[1::2]):
        assert len(left.lines) == 10
        for line in left.lines:
            np.testing.assert_array_equal(line.get_xdata(), [0, 1])
        assert len(left.collections) == 0
        assert len(right.lines) == 2
        assert len(right.collections) == 4
        np.testing.assert_array_equal(left.images[0].get_array(), right.images[0].get_array())
        assert left.images[0].get_extent() == right.images[0].get_extent()
        assert left.get_xlim() == right.get_xlim()
        assert left.get_ylim() == right.get_ylim()


@pytest.mark.parametrize("margin,thickness", [(-1, 2), (0, 0), (0, -2), (0.5, 2), (True, 2)])
def test_invalid_parameters_fail_before_model_call(margin, thickness):
    with pytest.raises(ValueError):
        papi.depth_slab_mips(np.zeros((2, 14, 4, 3)), None, "cpu", margin, thickness)


def test_grayscale_is_not_assumed_to_be_flow():
    with pytest.raises(ValueError, match="RGB OCTA"):
        papi.depth_slab_mips(np.zeros((2, 14, 4)), None, "cpu", 0, 2)


@pytest.mark.parametrize("options", [
    ["--depth-slabs"],
    ["--margin-px", "0"],
    ["--depth-slabs", "--margin-px", "-1", "--thickness-px", "2"],
])
def test_cli_rejects_incomplete_or_invalid_configuration(monkeypatch, options):
    monkeypatch.setattr(sys, "argv", ["papi.py", *options])
    with pytest.raises(SystemExit) as error:
        papi.main()
    assert error.value.code == 2


def test_cli_depth_mode_does_not_run_legacy_loop_count(monkeypatch, tmp_path):
    volume, _ = synthetic_scan(monkeypatch)
    monkeypatch.setattr(papi, "load_pytorch_model", lambda *args: None)
    monkeypatch.setattr(papi, "import_dicom", lambda *args: (volume, (1, 1, 1), {}))

    def legacy_not_expected(*args, **kwargs):
        pytest.fail("Depth mode must not run legacy loop counting")

    monkeypatch.setattr(papi, "identify_and_plot_dots", legacy_not_expected)
    monkeypatch.setattr(papi, "papillary_dermis_mip", legacy_not_expected)
    monkeypatch.setattr(sys, "argv", ["papi.py", "synthetic.dcm", "--depth-slabs",
                        "--margin-px", "0", "--thickness-px", "2", "--out-dir", str(tmp_path)])
    papi.main()
    assert (tmp_path / "synthetic_depth_slabs.npz").exists()
