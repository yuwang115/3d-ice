"""Tests for scripts/prepare_nsidc_sea_ice.py (NOAA/NSIDC monthly sea-ice concentration)."""

from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mod(nsidc_sea_ice_module):
    return nsidc_sea_ice_module


# Cell centres and NSIDC's own latitude and longitude for them, from the G02202 v6 ancillary
# files (G02202-ancillary-ps{n,s}25-v06r00.nc).
NSIDC_CELLS = {
    "EPSG:3411": [
        (-3837500.0, 5837500.0, 31.10267175, 168.32042246),
        (3737500.0, 5837500.0, 31.48750008, 102.37031356),
        (-3837500.0, -5337500.0, 34.05145898, -80.71498512),
        (-37500.0, 237500.0, 87.78072248, 143.97262661),
        (-87500.0, 3337500.0, 59.86692015, 136.50179252),
    ],
    "EPSG:3412": [
        (-3937500.0, 4337500.0, -39.36486911, -42.23256961),
        (3937500.0, 4337500.0, -39.36486911, 42.23256961),
        (-3937500.0, -3937500.0, -41.58344924, -135.0),
        (12500.0, 187500.0, -88.26545629, 3.81407483),
        (-187500.0, 1837500.0, -73.06910524, -5.82634203),
    ],
}


# ------------------------------------------------------------------ projections


@pytest.mark.unit
@pytest.mark.parametrize("code", sorted(NSIDC_CELLS))
def test_projection_reproduces_nsidc_cell_positions(mod, code):
    projection = mod.PROJECTIONS[code]
    for x, y, lat, lon in NSIDC_CELLS[code]:
        got_lon, got_lat = projection.inverse(x, y)
        assert got_lat == pytest.approx(lat, abs=1e-7)
        assert ((got_lon - lon + 180) % 360) - 180 == pytest.approx(0, abs=1e-7)
        fx, fy = projection.forward(lon, lat)
        assert fx == pytest.approx(x, abs=0.01)
        assert fy == pytest.approx(y, abs=0.01)


@pytest.mark.unit
@pytest.mark.parametrize("code", ["EPSG:3411", "EPSG:3412", "EPSG:3413", "EPSG:3031"])
def test_projections_round_trip_and_are_true_to_scale_at_their_standard_parallel(mod, code):
    projection = mod.PROJECTIONS[code]
    sign = -1 if projection.is_south else 1
    lon = np.array([-170.0, -45.0, 0.0, 33.3, 120.0, 179.0])
    lat = sign * np.array([50.0, 60.0, 71.0, 80.0, 85.0, 89.9])
    x, y = projection.forward(lon, lat)
    back_lon, back_lat = projection.inverse(x, y)
    assert np.allclose(back_lat, lat, atol=1e-9)
    assert np.allclose(((back_lon - lon + 180) % 360) - 180, 0, atol=1e-9)
    assert projection.scale_factor(projection.true_scale_lat_deg) == pytest.approx(1.0, abs=1e-12)
    assert projection.scale_factor(sign * 89.0) < 1.0 < projection.scale_factor(sign * 50.0)


@pytest.mark.unit
def test_projections_point_their_meridians_as_epsg_defines_them(mod):
    south = mod.PROJECTIONS["EPSG:3031"]
    x, y = south.forward(0.0, -71.0)
    assert x == pytest.approx(0, abs=1e-6) and y > 0, "Greenwich points up in EPSG:3031"
    x, y = south.forward(90.0, -71.0)
    assert x > 0 and y == pytest.approx(0, abs=1e-6), "90 E points right"
    north = mod.PROJECTIONS["EPSG:3413"]
    x, y = north.forward(-45.0, 75.0)
    assert x == pytest.approx(0, abs=1e-6) and y < 0, "45 W points down in EPSG:3413"
    x, y = north.forward(45.0, 75.0)
    assert x > 0 and y == pytest.approx(0, abs=1e-6)


@pytest.mark.unit
def test_wgs84_south_polar_projection_matches_the_cpom_packers_circles(mod, cpom_land_ice_module):
    south = mod.PROJECTIONS["EPSG:3031"]
    for latitude in (-81.5, -88.0, -60.0):
        x, y = south.forward(0.0, latitude)
        assert np.hypot(x, y) == pytest.approx(cpom_land_ice_module.pole_hole_radius_m(latitude), rel=1e-12)


# ------------------------------------------------------------------ grids and resampling


@pytest.mark.unit
def test_sea_ice_grid_covers_every_terrain_grid_with_nsidc_style_centres(mod):
    terrains = [
        {"nx": 511, "ny": 918, "x0_m": -652925.0, "y0_m": -632675.0, "dx_m": 3000.0, "dy_m": -3000.0},
        {"nx": 1533, "ny": 2752, "x0_m": -652925.0, "y0_m": -632675.0, "dx_m": 1000.0, "dy_m": -1000.0},
    ]
    grid = mod.sea_ice_grid(terrains)
    assert grid["dx_m"] == 25000.0 and grid["dy_m"] == -25000.0
    for key in ("x0_m", "y0_m"):
        assert (grid[key] - 12500.0) % 25000.0 == pytest.approx(0)
    x_last = grid["x0_m"] + grid["dx_m"] * (grid["nx"] - 1)
    y_last = grid["y0_m"] + grid["dy_m"] * (grid["ny"] - 1)
    for terrain in terrains:
        assert grid["x0_m"] <= terrain["x0_m"]
        assert x_last >= terrain["x0_m"] + terrain["dx_m"] * (terrain["nx"] - 1)
        assert grid["y0_m"] >= terrain["y0_m"]
        assert y_last <= terrain["y0_m"] + terrain["dy_m"] * (terrain["ny"] - 1)
    assert (grid["nx"], grid["ny"]) == (63, 112)


@pytest.mark.unit
def test_bilinear_lookup_is_exact_at_nodes_and_renormalises_over_valid_cells(mod):
    src_x = np.array([0.0, 10.0, 20.0])
    src_y = np.array([20.0, 10.0, 0.0])  # rows north to south, as NSIDC's
    field = np.array([[0.0, 10.0, 20.0], [30.0, 40.0, 50.0], [60.0, 70.0, 80.0]])
    valid = np.ones_like(field, dtype=bool)
    lookup = mod.BilinearLookup.build(src_x, src_y, np.array([10.0, 5.0, 15.0, -30.0]), np.array([10.0, 15.0, 5.0, 10.0]))
    values = lookup.mean(field, valid)
    assert values[0] == pytest.approx(40.0)
    assert values[1] == pytest.approx((0 + 10 + 30 + 40) / 4)
    assert values[2] == pytest.approx((40 + 50 + 70 + 80) / 4)
    assert np.isnan(values[3]), "a point outside the source grid has no value"
    # Only the valid corners count, renormalised.
    valid[0, 0] = False
    assert lookup.mean(field, valid)[1] == pytest.approx((10 + 30 + 40) / 3)
    valid[:] = False
    assert np.isnan(lookup.mean(field, valid)).all()


@pytest.mark.unit
def test_terrain_reach_marks_the_cells_around_ocean_nodes(mod):
    grid = {"nx": 6, "ny": 6, "x0_m": 12500.0, "y0_m": 137500.0, "dx_m": 25000.0, "dy_m": -25000.0}
    terrain = {"nx": 2, "ny": 1, "x0_m": 70000.0, "y0_m": 80000.0, "dx_m": 10000.0, "dy_m": -10000.0}
    ocean = np.array([[True, False]])
    reach = mod.terrain_reach(grid, [(terrain, ocean)])
    # The ocean node at x = 70 km, y = 80 km sits between columns 2-3 and rows 2-3; one more ring.
    rows, cols = np.nonzero(reach)
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (1, 4, 1, 4)
    assert reach.sum() == 16


@pytest.mark.unit
def test_fill_from_neighbours_reaches_every_needed_cell_and_stops(mod):
    values = np.full((7, 7), np.nan)
    values[3, 3] = 80.0
    values[3, 2] = 40.0
    needed = np.zeros((7, 7), dtype=bool)
    needed[3, 4] = True
    needed[5, 5] = True  # two rings out, across a cell nobody needs
    filled, count = mod.fill_from_neighbours(values, needed)
    assert filled[3, 4] == pytest.approx(80.0)
    assert filled[5, 5] == pytest.approx(80.0)
    assert count == 2
    assert np.isnan(filled[0, 0]), "three rings out, after every needed cell is filled, stays empty"
    assert filled[3, 2] == 40.0, "values already there are kept"
    unchanged, none = mod.fill_from_neighbours(np.full((3, 3), np.nan), np.ones((3, 3), dtype=bool))
    assert none == 0 and np.isnan(unchanged).all()


# ------------------------------------------------------------------ hemisphere totals


@pytest.mark.unit
def test_cell_areas_follow_the_scale_factor(mod):
    region = mod.REGIONS["greenland"]
    source = {"x": np.array([-37500.0, -12500.0, 12500.0]), "y": np.array([-1837500.0, -1862500.0])}
    areas = mod.cell_areas_km2(source, region)
    assert areas.shape == (2, 3)
    assert np.all((areas > 600) & (areas < 660))
    projection = mod.PROJECTIONS["EPSG:3411"]
    _, lat = projection.inverse(-12500.0, -1837500.0)
    assert areas[0, 1] == pytest.approx(625.0 / projection.scale_factor(lat) ** 2)


@pytest.mark.unit
def test_extent_counts_cells_from_15_percent_and_area_weights_by_concentration(mod):
    codes = np.array([[0, 14, 15], [50, 100, mod.NO_DATA]], dtype=np.uint8)
    areas = np.full(codes.shape, 1e6)
    extent, area = mod.extent_and_area(codes, areas)
    assert extent == pytest.approx(3.0)
    assert area == pytest.approx(0.15 + 0.5 + 1.0)
    # Counting the cells with ice on at least half of the days instead: a cell below 15 % in the
    # monthly mean can count, one above can drop out, and cells without data never count.
    counted = np.array([[False, True, False], [True, True, True]])
    extent, area = mod.extent_and_area(codes, areas, counted)
    assert extent == pytest.approx(3.0)
    assert area == pytest.approx(0.14 + 0.5 + 1.0)
    empty = np.full((2, 2), mod.NO_DATA, dtype=np.uint8)
    assert all(np.isnan(v) for v in mod.extent_and_area(empty, np.ones((2, 2))))


@pytest.mark.unit
def test_reference_frequency_is_the_share_of_1981_2010_years_with_ice(mod):
    years = np.array([1980, 1981, 1982, 1983, 2010, 2011])
    months = np.array([9, 9, 9, 9, 9, 9])
    stack = np.zeros((6, 1, 3), dtype=np.uint8)
    stack[:, 0, 0] = [100, 100, 0, 20, 10, 100]  # ice in 2 of the 4 reference years
    stack[:, 0, 1] = mod.NO_DATA
    stack[3, 0, 2] = mod.NO_DATA
    stack[[1, 2, 4], 0, 2] = 15
    frequency = mod.reference_frequency(stack, years, months)
    assert frequency.shape == (12, 1, 3)
    assert frequency[8, 0, 0] == pytest.approx(50.0)
    assert np.isnan(frequency[8, 0, 1]), "a cell without data in the period has no frequency"
    assert frequency[8, 0, 2] == pytest.approx(100.0), "years without data are left out of the share"
    assert np.isnan(frequency[0]).all(), "no January in the reference years"


@pytest.mark.unit
def test_monthly_statistics_give_the_reference_mean_and_the_trend(mod):
    years = np.repeat(np.arange(1978, 2027), 12)
    months = np.tile(np.arange(1, 13), 2027 - 1978)
    extent = 10.0 + months - 0.05 * (years - 1979)
    extent[(years == 1987) & (months == 12)] = np.nan
    stats = mod.monthly_statistics(years, months, extent)
    september = stats[8]
    assert september["month"] == 9
    assert september["trend_first_year"] == 1979 and september["trend_last_year"] == 2026
    assert september["trend_years"] == 48
    assert september["slope_mkm2_per_year"] == pytest.approx(-0.05, abs=1e-9)
    assert september["reference_mean_mkm2"] == pytest.approx(19.0 - 0.05 * (1995.5 - 1979), abs=1e-4)
    assert september["percent_per_decade"] == pytest.approx(-0.5 / september["reference_mean_mkm2"] * 100, abs=0.01)
    assert september["intercept_mkm2_at_year_0"] + september["slope_mkm2_per_year"] * 1979 == pytest.approx(19.0, abs=1e-3)
    assert stats[11]["reference_years"] == 29, "December 1987 is missing"


# ------------------------------------------------------------------ inputs


QA_MEANINGS = (
    "average_concentration_exceeds_0.15 average_concentration_exceeds_0.30 "
    "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15 at_least_half_the_days_have_sea_ice_conc_exceeds_0.30 "
    "invalid_ice_mask_applied at_least_one_day_during_month_has_spatial_interpolation "
    "at_least_one_day_during_month_has_temporal_interpolation"
)


def write_month(path, x, y, codes, *, majority=None, sensor="AMSR2 > Advanced Microwave Scanning Radiometer 2", qa_meanings=QA_MEANINGS):
    """A CDR-style monthly file. `majority` marks the cells with ice on at least half of the
    days; by default those whose concentration is 15-100 %."""
    netCDF4 = pytest.importorskip("netCDF4")
    if majority is None:
        majority = (codes >= 15) & (codes <= 100)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("y", y.size)
        ds.createDimension("x", x.size)
        ds.createVariable("x", "f8", ("x",))[:] = x
        ds.createVariable("y", "f8", ("y",))[:] = y
        var = ds.createVariable("cdr_seaice_conc_monthly", "u1", ("time", "y", "x"), fill_value=np.uint8(255))
        var.setncattr("valid_range", np.array([0, 100], dtype=np.uint8))
        var.setncattr("scale_factor", 0.01)
        var.set_auto_maskandscale(False)
        var[0] = codes
        qa = ds.createVariable("cdr_seaice_conc_monthly_qa_flag", "u1", ("time", "y", "x"))
        qa.setncattr("flag_meanings", qa_meanings)
        qa.setncattr("flag_masks", np.array([1, 2, 4, 8, 16, 32, 64][: len(qa_meanings.split())], dtype=np.uint8))
        qa.set_auto_maskandscale(False)
        qa[0] = np.where(majority, 4 | 1, 0).astype(np.uint8)
        ds.setncattr("date_created", "2026-01-13T16:58:11Z")
        ds.setncattr("product_version", "v06r00")
        ds.setncattr("sensor", sensor)


def write_ancillary(path, x, y, surface_type, projection):
    netCDF4 = pytest.importorskip("netCDF4")
    gx, gy = np.meshgrid(x, y)
    lon, lat = projection.inverse(gx, gy)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("y", y.size)
        ds.createDimension("x", x.size)
        ds.createVariable("x", "f8", ("x",))[:] = x
        ds.createVariable("y", "f8", ("y",))[:] = y
        ds.createVariable("surface_type", "u1", ("y", "x"))[:] = surface_type
        ds.createVariable("latitude", "f8", ("y", "x"))[:] = lat
        ds.createVariable("longitude", "f8", ("y", "x"))[:] = lon


@pytest.mark.unit
def test_read_month_codes_fill_and_out_of_range_values_as_no_data(mod, tmp_path):
    x = np.array([-12500.0, 12500.0])
    y = np.array([12500.0, -12500.0])
    path = tmp_path / "sic_pss25_202509_am2_v06r00.nc"
    majority = np.array([[True, True], [True, False]])
    write_month(path, x, y, np.array([[0, 100], [255, 120]], dtype=np.uint8), majority=majority)
    month = mod.read_month(path)
    assert month["codes"].tolist() == [[0, 100], [mod.NO_DATA, mod.NO_DATA]]
    # The QA flag's cells, never where there is no data.
    assert month["majority"].tolist() == [[True, True], [False, False]]
    assert month["attrs"]["product_version"] == "v06r00"
    assert month["x"].tolist() == x.tolist()


@pytest.mark.unit
def test_read_month_needs_the_half_the_days_flag(mod, tmp_path):
    x = np.array([-12500.0, 12500.0])
    y = np.array([12500.0, -12500.0])
    path = tmp_path / "sic_pss25_202509_am2_v06r00.nc"
    write_month(path, x, y, np.zeros((2, 2), dtype=np.uint8), qa_meanings="average_concentration_exceeds_0.15")
    with pytest.raises(ValueError, match="at_least_half_the_days"):
        mod.read_month(path)


@pytest.mark.unit
def test_list_months_orders_the_files_and_refuses_a_missing_month(mod, tmp_path):
    for name in ("sic_psn25_197812_n07_v06r00.nc", "sic_psn25_197811_n07_v06r00.nc", "sic_psn25_197901_n07_v06r00.nc"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sic_pss25_197811_n07_v06r00.nc").write_bytes(b"")
    months = mod.list_months(tmp_path, "n")
    assert [(m["year"], m["month"], m["sensor"]) for m in months] == [(1978, 11, "n07"), (1978, 12, "n07"), (1979, 1, "n07")]
    (tmp_path / "sic_psn25_197903_F08_v06r00.nc").write_bytes(b"")
    with pytest.raises(ValueError, match="without a gap"):
        mod.list_months(tmp_path, "n")
    with pytest.raises(FileNotFoundError):
        mod.list_months(tmp_path / "missing", "n")


@pytest.mark.unit
def test_manifests_and_sea_ice_index_files_are_read(mod, tmp_path):
    manifest = tmp_path / "a.nc.mnf"
    manifest.write_text("sic_psn25_197812_n07_v06r00.nc,c00bf3d7de121e83861ad3dd7f2cf4e1,438206\n", encoding="utf-8")
    assert mod.read_manifest(manifest) == {
        "name": "sic_psn25_197812_n07_v06r00.nc",
        "md5": "c00bf3d7de121e83861ad3dd7f2cf4e1",
        "bytes": 438206,
    }
    (tmp_path / "N_09_extent_v4.0.csv").write_text(
        "year, mo,source_dataset, region, extent,   area\n"
        "2012,  9,    NSIDC-0051,      N,   3.57,   2.41\n"
        "1987, 12,         -9999,      N,  -9999,  -9999\n",
        encoding="utf-8",
    )
    assert mod.read_sea_ice_index(tmp_path, "N") == {(2012, 9): 3.57}


# ------------------------------------------------------------------ packaging


@pytest.mark.unit
def test_percent_codes_round_clip_and_mark_missing(mod):
    codes = mod.to_percent_codes(np.array([0.4, 0.6, 99.5, 101.0, -3.0, np.nan]))
    assert codes.tolist() == [0, 1, 100, 100, 0, mod.NO_DATA]
    assert mod.percent_stats(codes) == {"min": 0.0, "max": 100.0, "mean": round(201 / 5, 4)}


@pytest.mark.unit
def test_sensor_runs_group_consecutive_months(mod):
    labels = ["1987-07", "1987-08", "1987-09", "2025-01"]
    runs = mod.sensor_runs(labels, ["n07", "F08", "F08", "am2"])
    assert [(r["sensor"], r["first"], r["last"]) for r in runs] == [
        ("n07", "1987-07", "1987-07"),
        ("F08", "1987-08", "1987-09"),
        ("am2", "2025-01", "2025-01"),
    ]
    assert runs[2]["name"] == "AMSR2 (GCOM-W1)"


@pytest.mark.unit
def test_comparison_with_the_sea_ice_index_reports_relative_differences(mod):
    years = np.array([2012, 2012, 2013])
    months = np.array([9, 10, 9])
    extent = np.array([3.75, np.nan, 5.5])
    index = {(2012, 9): 3.57, (2013, 9): 5.0, (2012, 10): 5.0}
    assert "mean_field_mean_relative_difference_percent" not in mod.compare_with_index(years, months, extent, index)
    result = mod.compare_with_index(years, months, extent, index, np.array([3.57 * 1.1, np.nan, 5.5]))
    assert result["months_compared"] == 2
    assert result["mean_difference_mkm2"] == pytest.approx(((3.75 - 3.57) + 0.5) / 2, abs=1e-3)
    assert result["mean_relative_difference_percent_by_month"][8] == pytest.approx(
        ((3.75 - 3.57) / 3.57 * 100 + 10) / 2, abs=0.01
    )
    assert result["mean_relative_difference_percent_by_month"][0] is None
    assert result["mean_field_mean_relative_difference_percent_by_month"][8] == pytest.approx((10 + 10) / 2, abs=0.01)
    assert mod.compare_with_index(years, months, extent, {}) is None


def synthetic_frames(mod, *, empty=(1,)):
    """Three months on a 3 x 2 package grid whose domain is three cells."""
    frames = np.zeros((3, 2, 3))
    frames[:, 0, 1] = [80, np.nan, 90]
    frames[:, 1, 0] = [10.4, np.nan, 0.0]
    frames[:, 1, 2] = [55, np.nan, 100]
    has_data = np.array([index not in empty for index in range(3)])
    domain = np.zeros((2, 3), dtype=bool)
    domain[0, 1] = domain[1, 0] = domain[1, 2] = True
    return {
        "grid": {"nx": 3, "ny": 2, "x0_m": 12500.0, "y0_m": -12500.0, "dx_m": 25000.0, "dy_m": -25000.0},
        "source": {"x": np.array([-12500.0, 12500.0]), "y": np.array([12500.0, -12500.0])},
        "domain": domain,
        "frames": frames,
        "reference": np.full((12, 2, 3), 60.0),
        "has_data": has_data,
        "years": np.array([1987, 1987, 1988]),
        "months": np.array([11, 12, 1]),
        "sensors": ["F08", "F08", "F08"],
        "extent": np.array([12.0, np.nan, 14.0]),
        "area": np.array([10.0, np.nan, 12.5]),
        "mean_field_extent": np.array([12.5, np.nan, 14.5]),
        "files": [{"name": f"f{i}.nc", "md5": "0" * 32} for i in range(3)],
        "filled_cells": np.array([1, 0, 2]),
    }


@pytest.mark.unit
def test_build_package_lays_out_frames_reference_and_domain(mod, monkeypatch):
    out = synthetic_frames(mod)
    # Three months cannot fit a trend per calendar month; the statistics are tested on their own.
    monkeypatch.setattr(mod, "monthly_statistics", lambda years, months, extent: [{"month": m} for m in range(1, 13)])
    payload, meta = mod.build_package(out, "greenland", {"max_latitude_error_deg": 0.0}, None)
    fields = {field["name"]: field for field in meta["fields"]}
    assert [field["name"] for field in meta["fields"]] == ["concentration", "reference_ice_frequency", "domain_mask"]
    assert sum(field["byte_length"] for field in meta["fields"]) == len(payload)
    count = meta["domain"]["cell_count"]
    assert count == 3
    concentration = np.frombuffer(payload, np.uint8, fields["concentration"]["byte_length"], 0).reshape(3, count)
    assert concentration.tolist() == [[80, 10, 55], [255, 255, 255], [90, 0, 100]]
    reference = np.frombuffer(payload, np.uint8, fields["reference_ice_frequency"]["byte_length"], fields["reference_ice_frequency"]["byte_offset"])
    assert reference.size == 12 * count and set(reference.tolist()) == {60}
    mask = np.frombuffer(payload, np.uint8, fields["domain_mask"]["byte_length"], fields["domain_mask"]["byte_offset"])
    assert mask.tolist() == [0, 1, 0, 1, 0, 1]
    assert meta["geometry_type"] == "sparse_grid_time_series"
    assert meta["time_series_kind"] == "observed_sea_ice_concentration"
    assert meta["crs"] == "EPSG:3413" and meta["source_grid"]["crs"] == "EPSG:3411"
    assert meta["frames"]["months"] == ["1987-11", "1987-12", "1988-01"]
    assert meta["frames"]["empty_months"] == ["1987-12"]
    assert meta["frames"]["years"][0] == pytest.approx(1987 + 10.5 / 12, abs=1e-4)
    assert meta["series"]["extent_mkm2"] == [12.0, None, 14.0]
    assert meta["series"]["mean_field_extent_mkm2"] == [12.5, None, 14.5]
    assert meta["series"]["hemisphere"] == "north"
    assert meta["validation"]["filled_cells"] == {"per_month_max": 2, "per_month_median": 1.5, "share_of_domain_max": round(2 / 3, 4)}
    assert "sea_ice_index" not in meta["validation"]
    json.dumps(meta)


@pytest.mark.unit
def test_build_package_refuses_an_empty_domain_or_missing_values(mod, monkeypatch):
    monkeypatch.setattr(mod, "monthly_statistics", lambda years, months, extent: [])
    out = synthetic_frames(mod)
    out["domain"] = np.zeros((2, 3), dtype=bool)
    with pytest.raises(ValueError, match="no package cell"):
        mod.build_package(out, "greenland", {}, None)
    out = synthetic_frames(mod)
    out["reference"][3, 0, 1] = np.nan
    with pytest.raises(ValueError, match="reference frequency"):
        mod.build_package(out, "greenland", {}, None)
    out = synthetic_frames(mod, empty=())
    with pytest.raises(ValueError, match="empty in a month with data"):
        mod.build_package(out, "greenland", {}, None)


@pytest.mark.unit
def test_build_frames_resamples_fills_the_coast_and_totals_the_hemisphere(mod, tmp_path):
    """Three Southern Hemisphere months on a 6 x 6 NSIDC-style grid, one of them empty."""
    region = mod.REGIONS["antarctica"]
    x = np.arange(-62500.0, 62501.0, 25000.0)
    y = x[::-1].copy()
    surface = np.full((6, 6), 50, dtype=np.uint8)
    surface[2:4, 2:4] = 250  # land around the pole
    anc = tmp_path / "anc.nc"
    write_ancillary(anc, x, y, surface, mod.PROJECTIONS["EPSG:3412"])
    monthly = tmp_path / "south" / "monthly"
    monthly.mkdir(parents=True)
    ice = np.where(surface == 50, 80, 255).astype(np.uint8)
    ice[0, :] = 0
    write_month(monthly / "sic_pss25_198711_F08_v06r00.nc", x, y, ice)
    write_month(monthly / "sic_pss25_198712_F08_v06r00.nc", x, y, np.full((6, 6), 255, dtype=np.uint8))
    write_month(monthly / "sic_pss25_198801_F08_v06r00.nc", x, y, np.where(surface == 50, 20, 255).astype(np.uint8))
    terrain = {"nx": 9, "ny": 9, "x0_m": -40000.0, "y0_m": 40000.0, "dx_m": 10000.0, "dy_m": -10000.0}
    ocean = np.ones((9, 9), dtype=bool)
    ocean[4, 4] = False
    out = mod.build_frames(mod.list_months(monthly, "s"), region, [(terrain, ocean)], mod.read_ancillary(anc))
    assert out["has_data"].tolist() == [True, False, True]
    assert np.isnan(out["extent"][1])
    # 32 ocean cells, of which 26 hold 80 % ice in November, and all 32 hold 20 % in January.
    areas = mod.cell_areas_km2({"x": x, "y": y}, region)
    assert out["extent"][0] == pytest.approx(areas[1:][surface[1:] == 50].sum() / 1e6)
    assert out["extent"][2] == pytest.approx(areas[surface == 50].sum() / 1e6)
    assert out["area"][2] == pytest.approx(0.2 * out["extent"][2])
    assert out["mean_field_extent"][0] == pytest.approx(out["extent"][0])
    # The land cells around the pole carry the concentration of their ocean neighbours.
    frame = out["frames"][2]
    assert np.isfinite(frame[out["reach"]]).all()
    assert np.allclose(frame[out["reach"]], 20.0)
    assert out["filled_cells"][2] > 0
    assert out["domain"].any() and not (out["domain"] & ~out["reach"]).any()
    assert len(out["files"]) == 3 and all(len(f["md5"]) == 32 for f in out["files"])


# ------------------------------------------------------------------ committed packages


def load_package(data_dir, name):
    meta = json.loads((data_dir / f"{name}.meta.json").read_text(encoding="utf-8"))
    payload = (data_dir / f"{name}.bin").read_bytes()
    return meta, payload


@pytest.mark.integration
@pytest.mark.parametrize("region", ["antarctica", "greenland"])
def test_committed_packages_cover_their_terrains_and_the_whole_record(mod, data_dir, region):
    config = mod.REGIONS[region]
    meta, payload = load_package(data_dir, config["basename"])
    assert meta["crs"] == config["explorer_crs"]
    grid = meta["grid"]
    terrain_grids = [mod.read_terrain_ocean(data_dir / f"{name}.meta.json")[0] for name in config["terrains"]]
    assert grid == mod.sea_ice_grid(terrain_grids)
    frames = meta["frames"]
    assert frames["months"][0] == "1978-11"
    assert len(frames["months"]) == frames["count"] == len(frames["years"])
    assert {"1987-12", "1988-01"} <= set(frames["empty_months"])
    series = meta["series"]
    for key in ("extent_mkm2", "area_mkm2"):
        assert len(series[key]) == frames["count"]
    empty = set(frames["empty_months"])
    for label, extent in zip(frames["months"], series["extent_mkm2"]):
        assert (extent is None) == (label in empty), label
    assert len(series["monthly"]) == 12
    assert meta["validation"]["projection"]["max_latitude_error_deg"] < 1e-6
    assert len(meta["source_files"]) == frames["count"]
    assert all(entry.get("md5_matches_manifest") for entry in meta["source_files"])
    fields = {field["name"]: field for field in meta["fields"]}
    count = meta["domain"]["cell_count"]
    concentration = np.frombuffer(payload, np.uint8, fields["concentration"]["byte_length"], 0).reshape(frames["count"], count)
    for index, label in enumerate(frames["months"]):
        row = concentration[index]
        if label in empty:
            assert (row == mod.NO_DATA).all(), label
        else:
            assert row.max() <= 100, label


@pytest.mark.integration
def test_committed_arctic_record_holds_the_2012_minimum(mod, data_dir):
    meta, _ = load_package(data_dir, mod.REGIONS["greenland"]["basename"])
    months = meta["frames"]["months"]
    extents = meta["series"]["extent_mkm2"]
    septembers = {label: value for label, value in zip(months, extents) if label.endswith("-09") and value is not None}
    assert min(septembers, key=septembers.get) == "2012-09"
    assert meta["series"]["monthly"][8]["percent_per_decade"] < -10
