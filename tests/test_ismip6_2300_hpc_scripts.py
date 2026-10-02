"""Tests for the HPC half of the ISMIP6 2300 pipeline.

scripts/regrid_ismip6_2300_run.py resamples one model run onto the explorer grid, and
scripts/combine_ismip6_2300_mean.py averages the change of several runs. Both run next to the
ISMIP6 archive; these tests exercise them on synthetic files.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 not installed")


@pytest.fixture(scope="module")
def regrid(ismip6_regrid_module):
    return ismip6_regrid_module


@pytest.fixture(scope="module")
def combine(ismip6_combine_module):
    return ismip6_combine_module


# ------------------------------------------------------------------ regridding


@pytest.mark.unit
def test_record_index_counts_from_2015_and_skips_an_initial_state(regrid):
    assert regrid.record_index(2015, 286) == 0
    assert regrid.record_index(2300, 286) == 285
    assert regrid.record_index(2015, 287) == 1
    assert regrid.record_index(2300, 287) == 286
    with pytest.raises(ValueError):
        regrid.record_index(2300, 285)
    with pytest.raises(ValueError):
        regrid.record_index(2100, 100)


@pytest.mark.unit
def test_usable_axis_rejects_unwritten_coordinates(regrid):
    good = np.linspace(-3_040_000.0, 3_040_000.0, 191)
    assert regrid.usable_axis(good, 191)
    assert not regrid.usable_axis(np.full(191, 9.96921e36), 191)  # netCDF default fill (IMAU)
    assert not regrid.usable_axis(good[:-1], 191)
    uneven = good.copy()
    uneven[10] += 5000.0
    assert not regrid.usable_axis(uneven, 191)


@pytest.mark.unit
def test_resample_record_conserves_volume_and_weights_speed_by_ice(regrid):
    # One 10-unit target cell over a 5 x 5 block of 2-unit source cells.
    xs = np.arange(1.0, 10.0, 2.0)
    w = regrid.overlap_weights(xs, 2.0, np.array([5.0]), 10.0)
    thickness = np.full((5, 5), 100.0)
    fraction = np.zeros((5, 5))
    fraction[:, :3] = 1.0  # three of five columns covered
    u = np.where(fraction > 0, 1e-5, np.nan)
    u[:, 0] = 2e-5
    v = np.zeros((5, 5))
    h10, f10, s10, volume = regrid.resample_record(thickness, fraction, u, v, w, w, min_cover=0.5)

    assert h10[0, 0] == pytest.approx(100.0 * 15 / 25)  # cell-mean thickness, volume-conserving
    assert f10[0, 0] == pytest.approx(0.6)
    expected = (2e-5 + 2 * 1e-5) / 3 * regrid.SECONDS_PER_YEAR  # mean over the iced columns
    assert s10[0, 0] == pytest.approx(expected)
    assert volume == pytest.approx(100.0 * 15)

    _, _, sparse, _ = regrid.resample_record(thickness, fraction, u, v, w, w, min_cover=0.7)
    assert np.isnan(sparse[0, 0])  # only 60 % of the cell has ice with a speed


@pytest.mark.unit
def test_resample_record_ignores_ice_fraction_without_thickness(regrid):
    xs = np.arange(1.0, 10.0, 2.0)
    w = regrid.overlap_weights(xs, 2.0, np.array([5.0]), 10.0)
    fraction = np.ones((5, 5))
    thickness = np.zeros((5, 5))
    speed = np.full((5, 5), 1e-5)
    h10, f10, s10, _ = regrid.resample_record(thickness, fraction, speed, speed, w, w, min_cover=0.5)
    assert h10[0, 0] == 0.0 and f10[0, 0] == 0.0 and np.isnan(s10[0, 0])


# ------------------------------------------------------------------ ensemble mean


def _write_run(path, years, thickness, speed):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("year", len(years))
        ds.createDimension("y", thickness.shape[1])
        ds.createDimension("x", thickness.shape[2])
        ds.createVariable("year", "i4", ("year",))[:] = years
        ds.createVariable("x", "f8", ("x",))[:] = np.arange(thickness.shape[2]) * 10000.0
        ds.createVariable("y", "f8", ("y",))[:] = -np.arange(thickness.shape[1]) * 10000.0
        ds.createVariable("thickness", "f4", ("year", "y", "x"))[:] = thickness
        ds.createVariable("ice_fraction", "f4", ("year", "y", "x"))[:] = (thickness > 0).astype(float)
        ds.createVariable("speed", "f4", ("year", "y", "x"), fill_value=np.float32(np.nan))[:] = speed


@pytest.mark.integration
def test_combine_averages_change_and_drops_thin_or_runaway_speeds(combine, tmp_path, monkeypatch):
    years = [2015, 2020]
    # Model A thins a strip of three cells and speeds up; model B's third cell thins to 2 m and
    # runs away. A second, ice-free row gives the grid a y spacing.
    def strip(first, second, empty):
        return np.array([[first, [empty] * 3], [second, [empty] * 3]], dtype=float)

    h_a = strip([500.0, 500.0, 500.0], [400.0, 450.0, 500.0], 0.0)
    s_a = strip([100.0, 100.0, 100.0], [300.0, 150.0, 100.0], np.nan)
    h_b = strip([500.0, 500.0, 500.0], [500.0, 500.0, 2.0], 0.0)
    s_b = strip([100.0, 100.0, 100.0], [200.0, 50000.0, 1e7], np.nan)
    (tmp_path / "regridded").mkdir()
    _write_run(tmp_path / "regridded" / "A_expAE05.nc", years, h_a, s_a)
    _write_run(tmp_path / "regridded" / "B_expAE05.nc", years, h_b, s_b)
    out = tmp_path / "mean.nc"
    monkeypatch.setattr(
        sys,
        "argv",
        ["combine", "--experiment", "expAE05", "--models", "A", "B", "--in-dir", str(tmp_path / "regridded"), "--out", str(out)],
    )
    combine.main()

    with netCDF4.Dataset(out) as ds:
        np.testing.assert_allclose(ds["dthickness_mean"][1, 0], [-50.0, -25.0, -249.0])
        speed = np.ma.filled(ds["dspeed_mean"][1, 0].astype(float), np.nan)
        np.testing.assert_allclose(speed[:2], [150.0, 50.0])  # B's 50 km/yr is rejected
        assert speed[2] == pytest.approx(0.0)  # only A counts where B's ice is 2 m thick
        np.testing.assert_array_equal(ds["n_speed"][1, 0], [2, 1, 1])
        np.testing.assert_array_equal(ds["n_speed_rejected"][1, 0], [0, 1, 1])
        assert json.loads(ds.getncattr("models")) == ["A", "B"]
        assert json.loads(ds.getncattr("speed_filter"))["max_speed_m_per_yr"] == 20000.0
        np.testing.assert_allclose(ds["volume_change_mean_km3"][1], (-150.0 + -498.0) / 2 * 1e8 / 1e9)


def _write_variable(path, name, records, ny=4, nx=5):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", records)
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        ds.createVariable("time", "f8", ("time",))[:] = np.arange(records) * 365.0
        ds.createVariable("x", "f8", ("x",))[:] = np.arange(nx) * 8000.0 - 16000.0
        ds.createVariable("y", "f8", ("y",))[:] = np.arange(ny) * 8000.0 - 12000.0
        ds.createVariable(name, "f4", ("time", "y", "x"))[:] = np.ones((records, ny, nx))


@pytest.mark.integration
def test_regrid_refuses_variables_whose_records_do_not_line_up(regrid, tmp_path, monkeypatch):
    run = tmp_path / "expAE05_08"
    run.mkdir()
    for name, records in (("lithk", 286), ("sftgif", 286), ("xvelmean", 287), ("yvelmean", 287)):
        _write_variable(run / f"{name}_AIS_TEST_MODEL_expAE05.nc", name, records)
    grid = {"nx": 6, "ny": 6, "x0_m": -25000.0, "y0_m": 25000.0, "dx_m": 10000.0, "dy_m": -10000.0}
    (tmp_path / "grid.meta.json").write_text(json.dumps({"grid": grid}))
    monkeypatch.setattr(
        sys,
        "argv",
        ["regrid", "--run-dir", str(run), "--model", "TEST", "--experiment", "expAE05",
         "--grid-meta", str(tmp_path / "grid.meta.json"), "--out", str(tmp_path / "out.nc"), "--years", "2015"],
    )
    with pytest.raises(ValueError, match="would not line up"):
        regrid.main()
