"""Tests for scripts/prepare_ismip6_projection.py (the ISMIP6 2300 ensemble-mean packer)."""

from __future__ import annotations

import json

import numpy as np
import pytest

GRID = {"nx": 4, "ny": 3, "x0_m": -15000.0, "y0_m": 10000.0, "dx_m": 10000.0, "dy_m": -10000.0}


@pytest.fixture(scope="module")
def mod(ismip6_projection_module):
    return ismip6_projection_module


# ------------------------------------------------------------------ present-day ice and frames


@pytest.mark.unit
def test_present_day_ice_needs_thickness_and_an_ice_flag(mod):
    thickness = np.array([[0.0, 10.0, 10.0, np.nan]])
    mask = np.array([[2, 1, 3, 2]], dtype=np.uint8)  # ice flag but no ice, ice on land flag, shelf, fill
    np.testing.assert_array_equal(mod.present_day_ice(thickness, mask), [[False, False, True, False]])
    assert mod.present_day_ice(np.array([[5.0]]), np.array([[4]], dtype=np.uint8))[0, 0]  # Lake Vostok


@pytest.mark.unit
def test_thickness_frames_add_thickening_and_scale_thinning_to_todays_ice(mod):
    h_now = np.array([[400.0, 300.0, 50.0], [0.0, 200.0, 80.0]])
    domain = np.array([[True, True, True], [False, True, True]])
    # The models' mean 2015 thickness: thicker than today in the first cell, thinner in the
    # fourth, no ice at all in the third.
    start = np.array([[500.0, 300.0, 0.0], [0.0, 100.0, 80.0]])
    dh = np.zeros((3, 2, 3))
    dh[1] = [[-250.0, -150.0, -1e-6], [40.0, 20.0, -40.0]]
    dh[2] = [[-500.0, -10.0, 0.0], [999.0, -100.0, -120.0]]
    frames = mod.thickness_frames(h_now, domain, dh, start)

    assert frames.shape == (3, 5)  # three frames x five domain cells, row-major
    np.testing.assert_allclose(frames[0], [400.0, 300.0, 50.0, 200.0, 80.0])
    # Half of the models' ice gone -> half of today's; thickening is added as it is; a cell the
    # models never had ice in keeps today's ice.
    np.testing.assert_allclose(frames[1], [200.0, 150.0, 50.0, 220.0, 40.0])
    # All of the models' ice gone -> all of today's, whether today's ice is thinner (first
    # cell: adding -500 m would also empty it) or thicker (fourth: adding -100 m would leave
    # 100 m). A loss beyond the models' own ice (float rounding) empties the cell, no more.
    np.testing.assert_allclose(frames[2], [0.0, 290.0, 50.0, 0.0, 0.0])
    assert frames.min() >= 0.0


@pytest.mark.unit
def test_thickness_frames_keep_shelves_whole_where_the_models_start_thicker(mod):
    # A shelf whose thickness varies from cell to cell, and models that start 25 % thicker and
    # lose 90 % of their ice: adding their mean change empties the thin cells and leaves the
    # thick ones (holes); scaling keeps 10 % everywhere.
    h_now = np.array([[200.0, 400.0, 250.0, 380.0]])
    domain = np.ones_like(h_now, dtype=bool)
    start = 1.25 * h_now.mean() * np.ones_like(h_now)
    dh = np.stack([np.zeros_like(h_now), -0.9 * start])
    frames = mod.thickness_frames(h_now, domain, dh, start)
    np.testing.assert_allclose(frames[1], 0.1 * h_now.ravel())
    additive = np.maximum(0.0, h_now + dh[1]).ravel()
    assert (additive == 0).sum() == 2 and (frames[1] > 0).all()


@pytest.mark.unit
def test_thickness_frames_drop_ice_the_models_have_all_but_lost(mod):
    h_now = np.array([[300.0, 300.0, 50.0, 2.0]])
    domain = np.ones_like(h_now, dtype=bool)
    # The models go from 300 m to 8 m (below the 10 m threshold), from 300 m to 12 m (above
    # it), from a 5 m film to 4 m (never above it), and from 100 m to 15 m over a cell that
    # BedMachine gives only 2 m.
    start = np.array([[300.0, 300.0, 5.0, 100.0]])
    dh = np.stack([np.zeros_like(h_now), np.array([[-292.0, -288.0, -1.0, -85.0]])])
    frames = mod.thickness_frames(h_now, domain, dh, start)
    np.testing.assert_allclose(frames[0], h_now.ravel())  # the first frame is today's ice
    # Gone; 4 % kept; the film rule does not apply where the models never had 10 m, so
    # 80 % is kept; and 15 % of 2 m is held at 1 m, the package's resolution, so rounding
    # to whole metres cannot open a hole.
    np.testing.assert_allclose(frames[1], [0.0, 12.0, 40.0, 1.0])
    assert mod.LOST_ICE_THICKNESS_M == 10.0


@pytest.mark.unit
def test_lost_ice_returns_only_once_the_models_regrow_it_clearly(mod):
    # The models' mean wobbles around the 10 m threshold before regrowing to 25 m: the cell
    # must not blink on and off with it.
    h_now = np.array([[300.0]])
    domain = np.ones_like(h_now, dtype=bool)
    start = np.array([[300.0]])
    models = np.array([300.0, 9.0, 12.0, 8.0, 15.0, 25.0])
    dh = (models - 300.0)[:, None, None]
    frames = mod.thickness_frames(h_now, domain, dh, start)
    np.testing.assert_allclose(frames[:, 0], [300.0, 0.0, 0.0, 0.0, 0.0, 25.0])
    assert mod.REGROWN_ICE_THICKNESS_M == 20.0


@pytest.mark.unit
def test_speed_change_needs_enough_models_and_remaining_ice(mod):
    domain = np.ones((1, 4), dtype=bool)
    dspeed = np.array([[[10.0, 20.0, np.nan, 40.0]]])
    n_speed = np.array([[[4, 3, 8, 8]]])
    frames = np.array([[100.0, 100.0, 100.0, 0.0]])
    out = mod.speed_change_frames(dspeed, n_speed, domain, frames, min_models=4)
    np.testing.assert_allclose(out, [[10.0, 0.0, 0.0, 0.0]])


# ------------------------------------------------------------------ sea level


@pytest.mark.unit
def test_vaf_counts_only_ice_above_flotation(mod):
    ratio = 917.0 / 1028.0
    bed = np.array([100.0, -500.0, -1000.0])
    # Grounded on land, grounded below sea level, and floating (thinner than flotation).
    frames = np.array([[1000.0, 1000.0, 500.0]])
    flotation_deep = 500.0 / ratio
    expected = (1000.0 + (1000.0 - flotation_deep)) * 1e8
    np.testing.assert_allclose(mod.flotation_vaf_m3(bed, frames, ratio, 1e8), [expected])


@pytest.mark.unit
def test_sle_from_vaf_is_relative_and_positive_for_loss(mod):
    vaf = np.array([1e15, 1e15 - 362.5e9 / 0.917])  # lose 362.5 Gt of ice above flotation
    sle = mod.sle_from_vaf(vaf, 917.0)
    np.testing.assert_allclose(sle, [0.0, 1e-3], rtol=1e-9)


# ------------------------------------------------------------------ grid and inputs


@pytest.mark.unit
def test_check_same_grid_rejects_a_shifted_grid(mod):
    x = GRID["x0_m"] + GRID["dx_m"] * np.arange(GRID["nx"])
    y = GRID["y0_m"] + GRID["dy_m"] * np.arange(GRID["ny"])
    mod.check_same_grid(GRID, x, y)
    with pytest.raises(ValueError):
        mod.check_same_grid(GRID, x + 5000.0, y)
    with pytest.raises(ValueError):
        mod.check_same_grid(GRID, x[:-1], y)


@pytest.mark.unit
def test_read_terrain_decodes_fill_and_mask(mod, tmp_path):
    shape = (GRID["ny"], GRID["nx"])
    thickness = np.arange(12, dtype="<i2").reshape(shape)
    thickness[0, 0] = -32768
    mask = np.full(shape, 2, dtype=np.uint8)
    blobs = [thickness.tobytes(), mask.tobytes()]
    (tmp_path / "t.bin").write_bytes(b"".join(blobs))
    meta = {
        "grid": GRID,
        "quantization": {"int16_fill_value": -32768, "scale": 1.0, "offset": 0.0},
        "fields": [
            {"name": "thickness", "dtype": "int16", "byte_offset": 0, "byte_length": len(blobs[0])},
            {"name": "mask", "dtype": "uint8", "byte_offset": len(blobs[0]), "byte_length": len(blobs[1])},
        ],
    }
    (tmp_path / "t.meta.json").write_text(json.dumps(meta))
    _, fields = mod.read_terrain(tmp_path / "t.meta.json")
    assert np.isnan(fields["thickness"][0, 0])
    assert fields["thickness"][2, 3] == 11.0
    assert fields["mask"].dtype == np.uint8 and fields["mask"].shape == shape


@pytest.mark.unit
def test_published_mean_sle_drops_the_initial_state_of_a_287_record_run(mod, tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    exp_dir = tmp_path / "expAE05" / "sle"
    exp_dir.mkdir(parents=True)
    series = {"A": np.linspace(0.0, 1.0, 286), "B": np.concatenate([[9.0], np.linspace(1.0, 3.0, 286)])}
    for model, values in series.items():
        with netCDF4.Dataset(exp_dir / f"computed_sle_AIS_{model}_expAE05.nc", "w") as ds:
            ds.createDimension("time", values.size)
            ds.createVariable("sle", "f4", ("time",))[:] = values
    out = mod.published_mean_sle(tmp_path, ["A", "B"], "expAE05")

    assert out["years"][0] == 2015 and out["years"][-1] == 2300
    np.testing.assert_allclose(out["mean"][[0, -1]], [0.0, 1.5], atol=1e-6)
    np.testing.assert_allclose(out["max"][-1], 2.0, atol=1e-6)
    assert out["per_model_end"]["B"] == pytest.approx(2.0, abs=1e-6)


# ------------------------------------------------------------------ package layout


@pytest.mark.unit
def test_write_package_round_trips_every_field(mod, tmp_path):
    domain = np.zeros((GRID["ny"], GRID["nx"]), dtype=bool)
    domain[1, 1:3] = True
    domain[2, 0] = True
    thickness = np.array([[100.4, 200.6, 0.0], [90.0, 0.0, 0.0]])
    speed = np.array([[0.0, 0.0, 0.0], [12.3, -4.6, 0.0]])
    bed = np.array([-100.0, 50.0, -2000.0])
    meta = mod.write_package(
        tmp_path, "pkg", grid=GRID, domain=domain, bed=bed, thickness=thickness,
        speed_change=speed, years=[2015, 2020], extra_meta={"title": "t"},
    )
    raw = (tmp_path / "pkg.bin").read_bytes()
    fields = {f["name"]: f for f in meta["fields"]}
    assert [f["name"] for f in meta["fields"]] == ["thickness", "speed_change", "bed", "domain_mask"]
    for f in meta["fields"]:
        if f["dtype"] == "int16":
            assert f["byte_offset"] % 2 == 0

    def view(name, dtype):
        f = fields[name]
        return np.frombuffer(raw, dtype=dtype, count=f["byte_length"] // np.dtype(dtype).itemsize, offset=f["byte_offset"])

    np.testing.assert_array_equal(view("thickness", "<i2").reshape(2, 3), [[100, 201, 0], [90, 0, 0]])
    np.testing.assert_array_equal(view("speed_change", "<i2").reshape(2, 3), [[0, 0, 0], [12, -5, 0]])
    np.testing.assert_array_equal(view("bed", "<i2"), [-100, 50, -2000])
    np.testing.assert_array_equal(view("domain_mask", np.uint8).reshape(domain.shape), domain.astype(np.uint8))
    assert meta["domain"]["cell_count"] == 3
    assert meta["frames"]["years"] == [2015, 2020]
    assert meta["geometry_type"] == "sparse_grid_time_series"


@pytest.mark.unit
def test_write_package_rejects_mismatched_frames(mod, tmp_path):
    domain = np.zeros((GRID["ny"], GRID["nx"]), dtype=bool)
    domain[0, 0] = True
    with pytest.raises(ValueError):
        mod.write_package(
            tmp_path, "bad", grid=GRID, domain=domain, bed=np.zeros(1), thickness=np.zeros((2, 1)),
            speed_change=np.zeros((3, 1)), years=[2015, 2020], extra_meta={},
        )
