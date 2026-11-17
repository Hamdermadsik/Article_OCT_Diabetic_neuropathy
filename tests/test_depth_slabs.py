import numpy as np
import pytest

from depth_slabs import extract_dej, project_slab


@pytest.mark.parametrize("dtype", [bool, np.uint8, np.float32])
def test_flat_band_uses_exclusive_lower_boundary(dtype):
    mask = np.zeros((9, 4), dtype=dtype)
    mask[2:6] = 1
    original = mask.copy()

    dej, surface, valid = extract_dej(mask)

    np.testing.assert_array_equal(dej, [6, 6, 6, 6])
    np.testing.assert_array_equal(surface, [2, 2, 2, 2])
    np.testing.assert_array_equal(valid, [True] * 4)
    np.testing.assert_array_equal(mask, original)
    assert valid.dtype == np.bool_
    assert not np.any(mask[dej.astype(int), np.arange(mask.shape[1])])


def test_sloping_band_preserves_local_boundaries():
    mask = np.zeros((10, 4), dtype=bool)
    for column in range(4):
        mask[column + 1 : column + 4, column] = True

    dej, surface, valid = extract_dej(mask)

    np.testing.assert_array_equal(dej, [4, 5, 6, 7])
    np.testing.assert_array_equal(surface, [1, 2, 3, 4])
    assert valid.all()


def test_empty_and_disconnected_columns_are_invalid():
    mask = np.zeros((10, 3), dtype=bool)
    mask[2:5, 0] = True
    mask[2:4, 2] = True
    mask[6:8, 2] = True

    dej, surface, valid = extract_dej(mask)

    np.testing.assert_array_equal(valid, [True, False, False])
    np.testing.assert_array_equal(dej, [5, np.nan, np.nan])
    np.testing.assert_array_equal(surface, [2, np.nan, np.nan])


def test_empty_mask_does_not_generate_top_of_image_boundaries():
    dej, surface, valid = extract_dej(np.zeros((7, 3), dtype=bool))

    assert not valid.any()
    assert np.isnan(dej).all()
    assert np.isnan(surface).all()


def test_image_edges_and_single_pixel_runs():
    mask = np.zeros((5, 3), dtype=bool)
    mask[:2, 0] = True
    mask[3:, 1] = True
    mask[2, 2] = True

    dej, surface, valid = extract_dej(mask)

    np.testing.assert_array_equal(dej, [2, 5, 3])
    np.testing.assert_array_equal(surface, [0, 3, 2])
    assert valid.all()


@pytest.mark.parametrize(
    "mask",
    [np.zeros(5), np.zeros((2, 3, 1)), np.zeros((0, 3)), np.zeros((3, 0))],
)
def test_rejects_wrong_dimensions_or_empty_array(mask):
    with pytest.raises(ValueError, match="nonempty 2D"):
        extract_dej(mask)


@pytest.mark.parametrize("value", [0.5, -1, 255, np.nan, np.inf, "1"])
def test_rejects_nonbinary_input(value):
    with pytest.raises(ValueError, match="binary values"):
        extract_dej(np.array([[value]]))


def test_projection_follows_dej_and_excludes_interval_end():
    flow = np.full((10, 3), 999.0)
    dej = np.array([3.0, 4.0, 5.0])
    for col, depth in enumerate(dej.astype(int)):
        flow[depth : depth + 2, col] = [col + 1, col + 4]
    original = flow.copy()

    row, valid = project_slab(flow, dej, np.ones(3, dtype=bool), 0, 2)

    np.testing.assert_array_equal(row, [4, 5, 6])
    assert valid.all()
    np.testing.assert_array_equal(flow, original)


def test_adjacent_slabs_do_not_overlap():
    flow = np.arange(10.0)[:, None]
    rows = [project_slab(flow, [2.0], [True], start, start + 2)[0][0]
            for start in (0, 2, 4)]
    assert rows == [3, 5, 7]


def test_out_of_bounds_is_invalid_not_clipped():
    row, valid = project_slab(np.ones((8, 3)), [5, 6, 8], [True] * 3, 0, 3)
    np.testing.assert_array_equal(row, [1, np.nan, np.nan])
    np.testing.assert_array_equal(valid, [True, False, False])


def test_negative_slab_excludes_air_and_accepts_surface_boundary():
    flow = np.arange(8.0)[:, None].repeat(3, axis=1)
    row, valid = project_slab(
        flow, [5, 5, 2], [True] * 3, -3, -1, surface_px=[2, 3, 0]
    )
    np.testing.assert_array_equal(row, [3, np.nan, np.nan])
    np.testing.assert_array_equal(valid, [True, False, False])


def test_zero_flow_is_valid_but_missing_segmentation_is_not():
    mask = np.zeros((8, 2), dtype=bool)
    mask[1:3, 0] = True
    dej, surface, columns = extract_dej(mask)
    row, valid = project_slab(np.zeros((8, 2)), dej, columns, 0, 2, surface_px=surface)
    np.testing.assert_array_equal(row, [0, np.nan])
    np.testing.assert_array_equal(valid, [True, False])


def test_nonfinite_signal_only_invalidates_affected_slab_columns():
    flow = np.ones((8, 3))
    flow[3, 0] = np.nan
    flow[4, 1] = np.inf
    flow[7, 2] = np.nan
    row, valid = project_slab(flow, [3] * 3, [True] * 3, 0, 2)
    np.testing.assert_array_equal(row, [np.nan, np.nan, 1])
    np.testing.assert_array_equal(valid, [False, False, True])


@pytest.mark.parametrize("dej", [np.nan, np.inf, -1, 0, 9, 2.5])
def test_invalid_dej_is_not_rounded_or_used(dej):
    row, valid = project_slab(np.ones((8, 1)), [dej], [True], 0, 1)
    assert np.isnan(row[0])
    assert not valid[0]


@pytest.mark.parametrize("surface", [np.nan, np.inf, -1, 5, 6, 1.5])
def test_invalid_surface_is_not_used(surface):
    row, valid = project_slab(
        np.ones((8, 1)), [5], [True], -2, 0, surface_px=[surface]
    )
    assert np.isnan(row[0])
    assert not valid[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"flow": np.zeros((2, 3, 1))},
        {"flow": np.zeros((0, 2))},
        {"dej_px": [3]},
        {"valid_columns": [True]},
        {"valid_columns": [1, 1]},
        {"surface_px": [1]},
        {"start_px": 0.5},
        {"stop_px": True},
        {"start_px": 2, "stop_px": 2},
        {"start_px": 3, "stop_px": 2},
        {"start_px": -2},
    ],
)
def test_projection_rejects_malformed_inputs(kwargs):
    arguments = dict(flow=np.ones((8, 2)), dej_px=[3, 3],
                     valid_columns=[True, True], start_px=0, stop_px=2)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        project_slab(**arguments)
