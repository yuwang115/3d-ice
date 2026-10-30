"""Tests for scripts/prepare_cpom_land_ice.py (CPOM Antarctic surface elevation change)."""

from __future__ import annotations

import json
from itertools import pairwise

import numpy as np
import pytest

GRID = {"nx": 3, "ny": 2, "x0_m": -10000.0, "y0_m": 5000.0, "dx_m": 10000.0, "dy_m": -10000.0}


@pytest.fixture(scope="module")
def mod(cpom_land_ice_module):
    return cpom_land_ice_module


def cpom_axes():
    """A 5 km CPOM-style grid covering GRID with a 5 km margin: centres at odd multiples of 2.5 km."""
    x = np.arange(-17500.0, 17501.0, 5000.0)  # 8 cells
    y = np.arange(-17500.0, 12501.0, 5000.0)  # 7 cells, south to north as in the CPOM files
    return x, y


def write_cpom_file(path, x, y, fields, *, start=None, end=None, basin=None, basin_flags=None):
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("nx", x.size)
        ds.createDimension("ny", y.size)
        ds.createVariable("x", "f4", ("nx",))[:] = x
        ds.createVariable("y", "f4", ("ny",))[:] = y
        for name, values in fields.items():
            if values.ndim == 3:
                if "time_period" not in ds.dimensions:
                    ds.createDimension("time_period", 1)
                var = ds.createVariable(name, "f4", ("time_period", "ny", "nx"), fill_value=np.float32(np.nan))
            else:
                var = ds.createVariable(name, "f4", ("ny", "nx"), fill_value=np.float32(np.nan))
            var[:] = values
        if basin is not None:
            var = ds.createVariable("basin_id", "i1", ("ny", "nx"), fill_value=np.int8(-128))
            var[:] = basin
            if basin_flags is not None:
                # As CPOM writes them: an array of values and a list of strings.
                var.setncattr("flag_values", np.array(list(basin_flags), dtype=np.int8))
                var.setncattr("flag_meanings", list(basin_flags.values()))
        if start is not None:
            dims = ("time_period",) if "time_period" in ds.dimensions else ()
            ds.createVariable("start_time", "f4", dims)[:] = start
            ds.createVariable("end_time", "f4", dims)[:] = end
        ds.setncattr("tracking_id", "test-id")
        ds.setncattr("license", "Creative Commons Attribution 4.0 International (CC BY 4.0)")


# ------------------------------------------------------------------ resampling


@pytest.mark.unit
def test_area_mean_keeps_a_constant_field_and_conserves_area_integrals(mod):
    x, y = cpom_axes()
    yy = y[::-1]  # rows north to south, as read_cpom_field returns them
    remap = mod.AreaMean(x, yy, GRID)
    rng = np.random.default_rng(1)
    field = rng.normal(size=(yy.size, x.size))
    valid = np.ones(field.shape, dtype=bool)
    mean, cover = remap.mean(np.full(field.shape, 2.5), valid)
    np.testing.assert_allclose(cover, 1.0)
    np.testing.assert_allclose(mean, 2.5)
    # The explorer cells span x in [-15, 15] km and y in [-10, 10] km: the integral of the
    # field over that window equals the integral of the means over the explorer cells.
    mean, cover = remap.mean(field, valid)
    # Cells straddling the window edge count by the share of them inside it.
    wx = np.clip(np.minimum(x + 2500, 15000) - np.maximum(x - 2500, -15000), 0, None) / 5000
    wy = np.clip(np.minimum(yy + 2500, 10000) - np.maximum(yy - 2500, -10000), 0, None) / 5000
    source = float((field * wy[:, None] * wx[None, :]).sum() * 25e6)
    packed = float((mean * cover).sum() * 1e8)
    assert packed == pytest.approx(source, rel=1e-12)


@pytest.mark.unit
def test_area_mean_reports_partial_cover_and_drawn_cells_need_half_and_ice(mod):
    x, y = cpom_axes()
    yy = y[::-1]
    remap = mod.AreaMean(x, yy, GRID)
    field = np.ones((yy.size, x.size))
    valid = np.zeros(field.shape, dtype=bool)
    valid[:, x < 0] = True  # the western half of the 5 km cells only
    mean, cover = remap.mean(field, valid)
    # Explorer columns centre on -10, 0, +10 km: fully, half and not covered.
    np.testing.assert_allclose(cover[:, 0], 1.0)
    np.testing.assert_allclose(cover[:, 1], 0.5)
    np.testing.assert_allclose(cover[:, 2], 0.0)
    np.testing.assert_allclose(mean[:, :2], 1.0)
    assert np.all(np.isnan(mean[:, 2]))
    mask = np.array([[2, 3, 2], [0, 4, 1]], dtype=np.uint8)
    drawn = mod.drawn_cells(cover, mask)
    np.testing.assert_array_equal(drawn, [[True, True, False], [False, True, False]])


@pytest.mark.unit
def test_sample_basins_takes_the_cpom_cell_under_each_node(mod):
    x, y = cpom_axes()
    yy = y[::-1]
    basin = np.arange(yy.size * x.size, dtype=np.int16).reshape(yy.size, x.size) % 19
    sampled = mod.sample_basins(basin, x, yy, GRID)
    tx, ty = mod.explorer_axes(GRID)
    for r, ty_m in enumerate(ty):
        for c, tx_m in enumerate(tx):
            # The node at an even 5 km multiple sits on a cell edge; rint picks the next cell.
            i = int(np.rint((ty_m - yy[0]) / (yy[1] - yy[0])))
            j = int(np.rint((tx_m - x[0]) / (x[1] - x[0])))
            assert sampled[r, c] == basin[i, j]


@pytest.mark.unit
def test_pole_hole_radius_matches_known_latitudes(mod):
    # The explorer's square Antarctic grid (half-width 3,333 km) reaches 60 S along its axes.
    assert mod.pole_hole_radius_m(-60.0) == pytest.approx(3.333e6, rel=2e-3)
    assert mod.pole_hole_radius_m(-81.5) == pytest.approx(925175.0, abs=5.0)
    assert mod.pole_hole_radius_m(-88.0) == pytest.approx(217326.0, abs=5.0)


# ------------------------------------------------------------------ inputs


@pytest.mark.unit
def test_read_cpom_field_flips_rows_to_run_north_to_south(mod, tmp_path):
    x, y = cpom_axes()
    values = np.tile(y[:, None], (1, x.size))  # each row holds its own y
    path = tmp_path / "rate.nc"
    write_cpom_file(path, x, y, {"sec": values}, basin=np.ones((y.size, x.size), dtype=np.int8))
    data = mod.read_cpom_field(path, ("sec",))
    assert data["y"][0] > data["y"][-1]
    np.testing.assert_allclose(data["fields"]["sec"][:, 0], data["y"])
    assert data["basin_id"].shape == (y.size, x.size)
    assert data["attrs"]["tracking_id"] == "test-id"


@pytest.mark.unit
def test_read_cpom_field_rejects_axes_that_are_not_5_km(mod, tmp_path):
    x, y = cpom_axes()
    path = tmp_path / "bad.nc"
    write_cpom_file(path, x * 2, y, {"sec": np.zeros((y.size, x.size))})
    with pytest.raises(ValueError, match="5 km"):
        mod.read_cpom_field(path, ("sec",))


@pytest.mark.unit
def test_list_epochs_orders_by_end_date_and_needs_the_1995_reference(mod, tmp_path):
    for end in ("20260407", "19950808", "20101130"):
        (tmp_path / f"CPOM-AIS-L3C-DH-MULTIMISSION-5KM-19950321-{end}-fv1.nc").write_bytes(b"")
    (tmp_path / "CPOM-AIS-L3C-DH-MULTIMISSION-5KM-ALL-fv1.tar.gz").write_bytes(b"")
    epochs = mod.list_epochs(tmp_path)
    assert [epoch["end_date"] for epoch in epochs] == ["19950808", "20101130", "20260407"]
    (tmp_path / "CPOM-AIS-L3C-DH-MULTIMISSION-5KM-19960101-20260407-fv1.nc").write_bytes(b"")
    with pytest.raises(ValueError, match="21 March 1995"):
        mod.list_epochs(tmp_path)


@pytest.mark.unit
def test_read_series_reads_the_four_regions(mod, tmp_path):
    for region in mod.REGIONS:
        rows = [{"time": 1995.4, "dm": 0.0, "uncertainty": 1.0}, {"time": 1995.8, "dm": -5.0, "uncertainty": 2.0}]
        (tmp_path / f"dmdt_{region}.json").write_text(json.dumps(rows))
    series = mod.read_series(tmp_path)
    assert set(series) == set(mod.REGIONS)
    np.testing.assert_allclose(series["AIS"]["mass_change_gt"], [0.0, -5.0])
    (tmp_path / "dmdt_WAIS.json").write_text("{}")
    with pytest.raises(ValueError, match="dmdt_WAIS"):
        mod.read_series(tmp_path)


@pytest.mark.unit
def test_read_series_reads_the_basins_by_id(mod, tmp_path):
    rows = [{"time": 1995.4, "dm": 0.0, "uncertainty": 0.1}, {"time": 1995.8, "dm": -2.0, "uncertainty": 0.5}]
    for basin in mod.BASINS:
        (tmp_path / f"dmdt_{basin}.json").write_text(json.dumps(rows))
    series = mod.read_series(tmp_path, mod.BASINS)
    assert list(series) == [str(i) for i in range(1, 19)]
    np.testing.assert_allclose(series["18"]["uncertainty_gt"], [0.1, 0.5])


@pytest.mark.unit
def test_basin_names_reads_the_cf_flags_in_each_form(mod):
    values = np.array([0, 9, 7])
    expected = {"9": "West G-H", "7": "East A-Ap"}  # 0 (islands, or outside the mask) is not a basin
    assert mod.basin_names(values, ["Islands", "West G-H", "East A-Ap"]) == expected
    assert mod.basin_names(values, "['Islands', 'West G-H', 'East A-Ap']") == expected
    assert mod.basin_names(values, "Islands West_G-H East_A-Ap") == expected
    with pytest.raises(ValueError, match="3 flag values but 2 meanings"):
        mod.basin_names(values, ["Islands", "West G-H"])
    with pytest.raises(ValueError, match="flag_meanings is not a list of names"):
        mod.basin_names(values, "['Islands', 'West G-H'")
    with pytest.raises(ValueError, match="repeats flag values"):
        mod.basin_names(np.array([0, 9, 9]), ["Islands", "West G-H", "East A-Ap"])


@pytest.mark.unit
def test_read_cpom_field_names_the_basins_from_their_flags(mod, tmp_path):
    x, y = cpom_axes()
    sec = np.full((y.size, x.size), -0.5)
    basin = np.full(sec.shape, 9, dtype=np.int8)
    path = tmp_path / "rate.nc"
    flags = {0: "Islands", 9: "West G-H", 18: "West Ep-F"}
    write_cpom_file(path, x, y, {"sec": sec, "sec_uncertainty": sec * 0}, basin=basin, basin_flags=flags)
    rate = mod.read_cpom_field(path, ("sec", "sec_uncertainty"))
    assert rate["basin_names"] == {"9": "West G-H", "18": "West Ep-F"}
    mask = np.full((2, 3), 2, dtype=np.uint8)
    _, meta = mod.rate_package(rate, path, GRID, mask, "terrain")
    basin_field = next(f for f in meta["fields"] if f["name"] == "basin_id")
    assert basin_field["flags"] == {"0": "none", "9": "West G-H", "18": "West Ep-F"}


# ------------------------------------------------------------------ packaging


@pytest.mark.unit
def test_quantize_rounds_to_the_step_keeps_fill_and_refuses_overflow(mod):
    codes = mod.quantize(np.array([0.004, -0.006, np.nan, 1.234]), 0.01)
    np.testing.assert_array_equal(codes, [0, -1, mod.INT16_FILL, 123])
    with pytest.raises(ValueError, match="int16"):
        mod.quantize(np.array([400.0]), 0.01)


@pytest.mark.unit
def test_pack_fields_tiles_the_payload(mod):
    payload, fields = mod.pack_fields([
        {"name": "a", "dtype": "int16", "blob": b"\x01\x00\x02\x00"},
        {"name": "b", "dtype": "uint8", "blob": b"\x07"},
    ])
    assert payload == b"\x01\x00\x02\x00\x07"
    assert [(f["byte_offset"], f["byte_length"]) for f in fields] == [(0, 4), (4, 1)]
    assert all("blob" not in field for field in fields)


def synthetic_frames(mod, observed_from: int, count: int = 12):
    """Frames over a 20 x 20 grid around the pole whose increments become structured at one epoch."""
    grid = {"nx": 20, "ny": 20, "x0_m": -950000.0, "y0_m": 950000.0, "dx_m": 100000.0, "dy_m": -100000.0}
    domain = np.ones((20, 20), dtype=bool)
    rng = np.random.default_rng(3)
    steps = np.zeros((count, int(domain.sum())))
    steps[1:observed_from] = 0.01  # uniform, extrapolated increments
    steps[observed_from:] = rng.normal(0.0, 0.3, size=(count - observed_from, steps.shape[1]))
    change = np.cumsum(steps, axis=0)
    starts = 1995.2 + 0.384 * np.arange(count)
    return grid, {
        "domain": domain,
        "change": change,
        "final_sigma": np.full(change.shape[1], 0.2),
        "start_years": starts,
        "end_years": starts + 0.384,
        "source_volume_km3": change.sum(axis=1),
        "source_gross_km3": {"thickening": 1.0, "thinning": -1.0},
        "files": [{"name": f"f{i}.nc", "sha256": "0" * 64} for i in range(count)],
    }


@pytest.mark.unit
def test_first_observed_epoch_finds_where_increments_gain_structure(mod):
    grid, frames = synthetic_frames(mod, observed_from=7)
    assert mod.first_observed_epoch(frames, grid) == 7


@pytest.mark.unit
def test_change_package_lays_out_frames_and_checks_the_series(mod):
    grid, frames = synthetic_frames(mod, observed_from=5, count=8)
    centres = (frames["start_years"] + frames["end_years"]) / 2
    series = {
        region: {"years": centres.copy(), "mass_change_gt": np.linspace(0, -10, 8), "uncertainty_gt": np.linspace(1, 5, 8)}
        for region in mod.REGIONS
    }
    payload, meta = mod.change_package(frames, series, grid, "bedmachine_antarctica_v4_480")
    assert meta["geometry_type"] == "sparse_grid_time_series"
    assert meta["time_series_kind"] == "observed_surface_change"
    assert [f["name"] for f in meta["fields"]] == ["surface_change", "final_uncertainty", "domain_mask"]
    count = meta["domain"]["cell_count"]
    assert meta["fields"][0]["byte_length"] == 8 * count * 2
    assert meta["fields"][1]["byte_length"] == count * 2
    assert meta["fields"][2]["byte_length"] == 400
    assert len(payload) == sum(f["byte_length"] for f in meta["fields"])
    assert meta["frames"]["count"] == 8 and len(meta["frames"]["years"]) == 8
    assert meta["series"]["regions"]["AIS"]["mass_change_gt"][-1] == -10.0
    assert meta["pole_hole"]["observed_from_year"] == pytest.approx(frames["start_years"][5], abs=1e-4)

    series["WAIS"]["years"] = centres + 0.5
    with pytest.raises(ValueError, match="WAIS"):
        mod.change_package(frames, series, grid, "bedmachine_antarctica_v4_480")


# CPOM's numbering and names of the 18 IMBIE basins, from the rate file's basin_id flags.
CPOM_BASIN_NAMES = {
    "1": "West H-Hp", "2": "West F-G", "3": "East E-Ep", "4": "East D-Dp", "5": "East Cp-D", "6": "East B-C",
    "7": "East A-Ap", "8": "East Jpp-K", "9": "West G-H", "10": "East Dp-E", "11": "East Ap-B", "12": "East C-Cp",
    "13": "East K-A", "14": "West J-Jpp", "15": "Peninsula Ipp-J", "16": "Peninsula I-Ipp", "17": "Peninsula Hp-I",
    "18": "West Ep-F",
}


def basin_and_region_series(mod, centres):
    """A series per basin, and regional series that are their sums (the AIS the sum of all)."""
    basins = {
        key: {"years": centres.copy(), "mass_change_gt": -float(key) * np.arange(centres.size),
              "uncertainty_gt": np.full(centres.size, 1.0)}
        for key in mod.BASINS
    }
    regions = {}
    for region, prefix in (("WAIS", "West"), ("EAIS", "East"), ("APIS", "Peninsula")):
        members = [key for key, name in CPOM_BASIN_NAMES.items() if name.startswith(prefix + " ")]
        regions[region] = {"years": centres.copy(),
                           "mass_change_gt": sum(basins[key]["mass_change_gt"] for key in members),
                           "uncertainty_gt": np.full(centres.size, 2.0)}
    regions["AIS"] = {"years": centres.copy(),
                      "mass_change_gt": sum(regions[r]["mass_change_gt"] for r in ("WAIS", "EAIS", "APIS")),
                      "uncertainty_gt": np.full(centres.size, 3.0)}
    return basins, regions


@pytest.mark.unit
def test_change_package_carries_the_basins_and_refuses_ones_that_do_not_sum_to_their_regions(mod):
    grid, frames = synthetic_frames(mod, observed_from=5, count=8)
    centres = (frames["start_years"] + frames["end_years"]) / 2
    basins, regions = basin_and_region_series(mod, centres)
    payload, meta = mod.change_package(frames, regions, grid, "terrain", basin_series=basins,
                                       basin_names_by_id=CPOM_BASIN_NAMES)
    entries = meta["series"]["basins"]
    assert list(entries) == list(mod.BASINS)
    assert entries["9"] == {"name": "West G-H", "code": "G-H", "region": "WAIS",
                            "mass_change_gt": [-9.0 * i for i in range(8)], "uncertainty_gt": [1.0] * 8}
    assert [entry["region"] for entry in entries.values()].count("EAIS") == 10
    assert meta["validation"]["basin_sum_residual_gt"] == {"WAIS": 0.0, "EAIS": 0.0, "APIS": 0.0}
    # The payload is the same with or without the basins: they live in the metadata.
    assert payload == mod.change_package(frames, regions, grid, "terrain")[0]

    # Within rounding, the gap is recorded; beyond the tolerance, a basin is in the wrong region.
    regions["APIS"]["mass_change_gt"] = regions["APIS"]["mass_change_gt"] + 0.5
    _, meta = mod.change_package(frames, regions, grid, "terrain", basin_series=basins,
                                 basin_names_by_id=CPOM_BASIN_NAMES)
    assert meta["validation"]["basin_sum_residual_gt"]["APIS"] == 0.5
    with pytest.raises(ValueError, match="named into WAIS do not sum"):
        mod.change_package(frames, regions, grid, "terrain", basin_series=basins,
                           basin_names_by_id={**CPOM_BASIN_NAMES, "3": "West E-Ep"})

    with pytest.raises(ValueError, match="basin 9 is named 'G-H'"):
        mod.change_package(frames, regions, grid, "terrain", basin_series=basins,
                           basin_names_by_id={**CPOM_BASIN_NAMES, "9": "G-H"})
    with pytest.raises(ValueError, match="basin 18 has no name"):
        names = {key: name for key, name in CPOM_BASIN_NAMES.items() if key != "18"}
        mod.change_package(frames, regions, grid, "terrain", basin_series=basins, basin_names_by_id=names)
    basins["3"]["years"] = centres + 0.5
    with pytest.raises(ValueError, match="the 3 series"):
        mod.change_package(frames, regions, grid, "terrain", basin_series=basins, basin_names_by_id=CPOM_BASIN_NAMES)


@pytest.mark.unit
def test_rate_package_draws_explorer_ice_and_records_the_polar_caps(mod, tmp_path):
    x, y = cpom_axes()
    sec = np.full((y.size, x.size), -0.5)
    sec[:, x > 2500] = np.nan  # nothing under the eastern explorer column
    sigma = np.full(sec.shape, 0.01)
    basin = np.tile(np.arange(1, x.size + 1, dtype=np.int8), (y.size, 1))
    path = tmp_path / "rate.nc"
    write_cpom_file(path, x, y, {"sec": sec, "sec_uncertainty": sigma}, start=1995.41, end=2026.07, basin=basin)
    rate = mod.read_cpom_field(path, ("sec", "sec_uncertainty"))
    # Grounded, floating, grounded / land, grounded, grounded.
    mask = np.array([[2, 3, 2], [1, 2, 2]], dtype=np.uint8)
    payload, meta = mod.rate_package(rate, path, GRID, mask, "terrain")

    assert [f["name"] for f in meta["fields"]] == ["dhdt", "dhdt_uncertainty", "basin_id"]
    assert len(payload) == 6 * 2 + 6 * 2 + 6
    codes = np.frombuffer(payload, dtype="<i2", count=6).reshape(2, 3)
    drawn = codes != mod.INT16_FILL
    # Floating cells next to the grounding line are kept; land and uncovered cells are not.
    np.testing.assert_array_equal(drawn, [[True, True, False], [False, True, False]])
    np.testing.assert_array_equal(codes[drawn], -500)
    basins = np.frombuffer(payload, dtype=np.uint8, count=6, offset=24).reshape(2, 3)
    assert basins[0, 2] == 0 and basins[1, 0] == 0 and basins[0, 0] > 0
    assert meta["coverage"]["drawn_cells"] == 3
    assert meta["pole_hole"]["radius_m"]["cryosat2"] == pytest.approx(217326.1, abs=0.2)
    assert meta["period"] == {"start_year": pytest.approx(1995.41, abs=1e-4), "end_year": pytest.approx(2026.07, abs=1e-4),
                              "note": meta["period"]["note"]}
    assert meta["source_file"]["name"] == "rate.nc" and len(meta["source_file"]["sha256"]) == 64
    assert set(meta["validation"]["packed_km3_per_yr"]) == {"net", "thickening", "thinning"}


@pytest.mark.unit
def test_epoch_frames_read_the_time_period_layout_and_take_the_domain_from_the_last_epoch(mod, tmp_path):
    x, y = cpom_axes()
    ends = ["19950808", "19951226", "19960514"]
    for k, end in enumerate(ends):
        dh = np.full((1, y.size, x.size), 0.1 * k)
        if k == len(ends) - 1:
            dh[0][:, x > 2500] = np.nan  # the last epoch loses the eastern cells
        sigma = np.full(dh.shape, 0.05 * (k + 1))
        write_cpom_file(
            tmp_path / f"CPOM-AIS-L3C-DH-MULTIMISSION-5KM-19950321-{end}-fv1.nc",
            x, y, {"dH": dh, "dH_uncertainty": sigma},
            start=np.array([1995.2 + 0.38 * k]), end=np.array([1995.6 + 0.38 * k]),
        )
    mask = np.array([[2, 3, 2], [1, 2, 2]], dtype=np.uint8)
    frames = mod.epoch_frames(mod.list_epochs(tmp_path), GRID, mask)

    np.testing.assert_array_equal(frames["domain"], [[True, True, False], [False, True, False]])
    assert frames["change"].shape == (3, 3)
    np.testing.assert_allclose(frames["change"][:, 0], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(frames["final_sigma"], 0.15)
    np.testing.assert_allclose(frames["start_years"], [1995.2, 1995.58, 1995.96], atol=1e-5)
    assert [f["name"].endswith(f"-{end}-fv1.nc") for f, end in zip(frames["files"], ends)] == [True, True, True]
    assert all(len(f["sha256"]) == 64 for f in frames["files"])


# ------------------------------------------------------------------ committed packages


@pytest.mark.integration
def test_the_record_thins_most_at_smith_glacier_in_the_last_epoch(mod, data_dir):
    meta = json.loads((data_dir / f"{mod.CHANGE_BASENAME}.meta.json").read_text())
    raw = (data_dir / f"{mod.CHANGE_BASENAME}.bin").read_bytes()
    fields = {f["name"]: f for f in meta["fields"]}
    count, frames = meta["domain"]["cell_count"], meta["frames"]["count"]
    change = fields["surface_change"]
    codes = np.frombuffer(raw, dtype="<i2", count=frames * count, offset=change["byte_offset"]).reshape(frames, count)
    mask_field = fields["domain_mask"]
    domain = np.frombuffer(raw, dtype=np.uint8, count=mask_field["byte_length"], offset=mask_field["byte_offset"])
    grid = meta["grid"]
    rows, cols = np.divmod(np.flatnonzero(domain), grid["nx"])
    last = np.where(codes[-1] == mod.INT16_FILL, np.nan, codes[-1] * change["scale"])
    deepest = int(np.nanargmin(last))
    x = grid["x0_m"] + cols[deepest] * grid["dx_m"]
    y = grid["y0_m"] + rows[deepest] * grid["dy_m"]
    # Smith Glacier, Amundsen Sea Embayment: more than 140 m of lowering since 1995.
    assert last[deepest] == pytest.approx(-141.02, abs=0.05)
    assert (x, y) == (-1483000, -637000)


@pytest.mark.integration
def test_committed_packages_share_the_terrain_grids(mod, data_dir):
    for suffix, terrain_name in mod.TERRAIN_PACKAGES.items():
        rate = json.loads((data_dir / f"{mod.RATE_BASENAME}_{suffix}.meta.json").read_text())
        terrain = json.loads((data_dir / f"{terrain_name}.meta.json").read_text())
        assert rate["grid"] == {key: terrain["grid"][key] for key in rate["grid"]}
        assert rate["license"].startswith("CC BY 4.0")
    change = json.loads((data_dir / f"{mod.CHANGE_BASENAME}.meta.json").read_text())
    years = change["frames"]["years"]
    assert change["frames"]["count"] == len(years) == len(change["series"]["years"])
    assert all(later > earlier for earlier, later in pairwise(years))
    for region in mod.REGIONS:
        assert len(change["series"]["regions"][region]["mass_change_gt"]) == len(years)
    assert 2010.5 < change["pole_hole"]["observed_from_year"] < 2011.5


@pytest.mark.integration
def test_committed_record_carries_cpoms_18_basins_and_they_sum_to_the_regions(mod, data_dir):
    change = json.loads((data_dir / f"{mod.CHANGE_BASENAME}.meta.json").read_text())
    basins = change["series"]["basins"]
    years = change["series"]["years"]
    assert {key: entry["name"] for key, entry in basins.items()} == CPOM_BASIN_NAMES
    assert all(len(entry["mass_change_gt"]) == len(entry["uncertainty_gt"]) == len(years) for entry in basins.values())
    assert max(change["validation"]["basin_sum_residual_gt"].values()) <= 0.1
    # The Amundsen Sea sector alone loses more than West Antarctica as a whole.
    assert basins["9"]["mass_change_gt"][-1] < change["series"]["regions"]["WAIS"]["mass_change_gt"][-1] < -2000
    for suffix in mod.TERRAIN_PACKAGES:
        rate = json.loads((data_dir / f"{mod.RATE_BASENAME}_{suffix}.meta.json").read_text())
        flags = next(f for f in rate["fields"] if f["name"] == "basin_id")["flags"]
        assert flags == {"0": "none", **CPOM_BASIN_NAMES}
