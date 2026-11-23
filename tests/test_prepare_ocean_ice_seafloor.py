"""Tests for scripts/prepare_ocean_ice_seafloor.py (OCEAN ICE climatology, seafloor water)."""

from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mod(ocean_ice_seafloor_module):
    return ocean_ice_seafloor_module


PRESSURE = np.arange(5.0, 1005.0, 10.0)  # 100 levels, 5 to 995 dbar


def column(bottom_dbar: float, top_dbar: float = 5.0, value=None) -> np.ndarray:
    """A column with data from top_dbar to bottom_dbar (inclusive), NaN elsewhere."""
    values = np.full(PRESSURE.size, np.nan)
    inside = (PRESSURE >= top_dbar) & (PRESSURE <= bottom_dbar)
    values[inside] = PRESSURE[inside] if value is None else value
    return values


# --------------------------------------------------------------------- seafloor window


@pytest.mark.unit
def test_deepest_levels_reads_across_chunks(mod):
    stack = np.stack([column(835.0), column(95.0), np.full(PRESSURE.size, np.nan), column(995.0)], axis=1)
    valid = np.isfinite(stack)
    chunks = [(k, valid[k:k + 7]) for k in range(0, PRESSURE.size, 7)]
    bottom = mod.deepest_levels(chunks)
    assert list(PRESSURE[bottom[[0, 1, 3]]]) == [835.0, 95.0, 995.0]
    assert bottom[2] == -1


@pytest.mark.unit
def test_window_holds_fifteen_levels_above_the_seafloor(mod):
    bottom = np.array([int(np.flatnonzero(PRESSURE == 835.0)[0]), -1])
    inside = mod.window_levels(PRESSURE, mod.bottom_pressures(PRESSURE, bottom))
    chosen = PRESSURE[inside[:, 0]]
    assert chosen.size == 15
    assert chosen.min() == 695.0 and chosen.max() == 835.0
    assert not inside[:, 1].any()


@pytest.mark.unit
def test_seafloor_means_average_the_window_and_count_support(mod):
    ct = np.stack([column(835.0), column(95.0, value=-1.9)], axis=1)
    sa = np.stack([column(835.0, value=34.9), column(95.0, value=34.5)], axis=1)
    mse = np.full_like(ct, 0.01)
    mse[PRESSURE >= 795.0, 0] = 1e10  # the five deepest levels of column 0 have no profile nearby
    valid = np.isfinite(ct)
    bottom = mod.deepest_levels([(0, valid)])
    out = mod.seafloor_means(PRESSURE, bottom, [(k, ct[k:k + 9], sa[k:k + 9], mse[k:k + 9]) for k in range(0, PRESSURE.size, 9)])
    assert out["ct"][0] == pytest.approx(np.mean(np.arange(695.0, 836.0, 10.0)))
    assert out["sa"][0] == pytest.approx(34.9)
    assert out["support"][0] == pytest.approx(10 / 15)
    assert out["levels"][0] == 15
    # A column shallower than the window is averaged whole.
    assert out["ct"][1] == pytest.approx(-1.9)
    assert out["levels"][1] == 10
    assert out["bottom_pressure"].tolist() == [835.0, 95.0]


@pytest.mark.unit
def test_seafloor_means_skip_gaps_under_an_ice_shelf(mod):
    ct = column(835.0, top_dbar=405.0, value=0.5)[:, None]  # the upper 400 dbar are ice
    sa = column(835.0, top_dbar=405.0, value=34.7)[:, None]
    bottom = mod.deepest_levels([(0, np.isfinite(ct))])
    out = mod.seafloor_means(PRESSURE, bottom, [(0, ct, sa, np.zeros_like(ct))])
    assert out["ct"][0] == pytest.approx(0.5)
    assert out["support"][0] == pytest.approx(1.0)


# ------------------------------------------------------------------------- resampling


def synthetic_columns(mod) -> dict:
    lat = np.arange(-80.0, -44.95, 0.1)
    lon = np.arange(-180.0, 180.01, 0.2)
    glon, glat = np.meshgrid(lon, lat)
    ct = 0.01 * (glat + 70.0)  # varies with latitude only
    sa = np.full(ct.shape, 34.6)
    support = np.ones(ct.shape)
    pressure = np.full(ct.shape, 500.0)
    land = glat < -75.0
    for array in (ct, sa, support, pressure):
        array[land] = np.nan
    return {"lat": lat, "lon": lon, "ct": ct, "sa": sa, "support": support, "bottom_pressure": pressure}


@pytest.mark.unit
def test_resample_follows_latitude_and_leaves_land_empty(mod):
    columns = synthetic_columns(mod)
    grid = {"nx": 61, "ny": 61, "x0_m": -3000000.0, "y0_m": 3000000.0, "dx_m": 100000.0, "dy_m": -100000.0}
    lookup = mod.explorer_lookup(columns, grid)
    fields = mod.resample(columns, lookup, grid)
    x, y = mod.axes(grid)
    gx, gy = np.meshgrid(x, y)
    lon, lat = mod.PROJECTIONS["EPSG:3031"].inverse(gx, gy)
    open_water = lat > -74.8
    expected = 0.01 * (lat + 70.0)
    assert np.nanmax(np.abs(fields["ct"][open_water] - expected[open_water])) < 1e-6
    assert np.all(np.isnan(fields["ct"][lat < -75.3]))
    assert np.allclose(fields["sa"][open_water], 34.6)


@pytest.mark.unit
def test_resample_crosses_the_antimeridian(mod):
    columns = synthetic_columns(mod)
    columns["ct"] = np.where(np.isfinite(columns["ct"]), 1.0, np.nan)
    # One cell on 180 deg longitude at 65 S: x = 0, y = +rho (EPSG:3031 puts 180 at +y).
    x, y = mod.PROJECTIONS["EPSG:3031"].forward(179.95, -65.0)
    grid = {"nx": 1, "ny": 1, "x0_m": float(x), "y0_m": float(y), "dx_m": 10000.0, "dy_m": -10000.0}
    fields = mod.resample(columns, mod.explorer_lookup(columns, grid), grid)
    assert fields["ct"][0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------- packing


@pytest.mark.unit
def test_quantize_round_trips_and_marks_missing(mod):
    values = np.array([-1.9, 0.0, 1.0349, np.nan])
    codes = mod.quantize(values, mod.CT_SCALE)
    assert codes[-1] == mod.INT16_FILL
    assert np.allclose(codes[:3] * mod.CT_SCALE, values[:3], atol=mod.CT_SCALE / 2)
    sa = mod.quantize(np.array([34.2, 34.95]), mod.SA_SCALE, mod.SA_OFFSET)
    assert np.allclose(sa * mod.SA_SCALE + mod.SA_OFFSET, [34.2, 34.95], atol=mod.SA_SCALE / 2)
    with pytest.raises(ValueError):
        mod.quantize(np.array([40.0]), mod.SA_SCALE, mod.SA_OFFSET)


@pytest.mark.unit
def test_percent_codes(mod):
    assert mod.percent_codes(np.array([0.0, 0.666, 1.0, np.nan])).tolist() == [0, 67, 100, 255]


def synthetic_periods(grid):
    shape = (grid["ny"], grid["nx"])
    ct = np.linspace(-2.0, 1.2, shape[0] * shape[1]).reshape(shape)
    full = {"ct": ct, "sa": np.full(shape, 34.7), "support": np.full(shape, 0.4), "bottom_pressure": np.full(shape, 612.0)}
    post = {key: value.copy() for key, value in full.items()}
    post["ct"] = ct + 0.1
    post["ct"][0, 0] = np.nan
    post["support"] = np.full(shape, 0.9)
    return {"full": full, "post2000": post}


@pytest.mark.unit
def test_build_package_tiles_the_payload_and_decodes(mod):
    grid = {"nx": 5, "ny": 4, "x0_m": 0.0, "y0_m": 0.0, "dx_m": 10000.0, "dy_m": -10000.0}
    periods = synthetic_periods(grid)
    domain = np.zeros((4, 5), dtype=bool)
    domain[1:, 1:] = True
    domain[0, 0] = True
    payload, meta = mod.build_package(periods, domain, grid, [], [])
    assert meta["geometry_type"] == "sparse_grid"
    assert meta["domain"]["cell_count"] == 13
    names = [field["name"] for field in meta["fields"]]
    assert names == ["ct_full", "sa_full", "ct_post2000", "sa_post2000", "bottom_pressure",
                     "support", "domain_mask"]
    cursor = 0
    for field in meta["fields"]:
        assert field["byte_offset"] == cursor
        cursor += field["byte_length"]
    assert cursor == len(payload)

    def decode(name):
        field = next(f for f in meta["fields"] if f["name"] == name)
        dtype = {"int16": "<i2", "uint8": "u1"}[field["dtype"]]
        raw = np.frombuffer(payload, dtype=dtype, count=field["byte_length"] // np.dtype(dtype).itemsize, offset=field["byte_offset"])
        if field["dtype"] != "int16":
            return raw
        out = raw * field["scale"] + field["offset"]
        return np.where(raw == field["fill_value"], np.nan, out)

    mask = decode("domain_mask").reshape(4, 5).astype(bool)
    assert np.array_equal(mask, domain)
    assert np.allclose(decode("ct_full"), periods["full"]["ct"][domain], atol=mod.CT_SCALE / 2)
    assert np.allclose(decode("sa_full"), 34.7, atol=mod.SA_SCALE / 2)
    post = decode("ct_post2000")
    assert np.isnan(post[0])  # the first domain cell, (0, 0), has no post-2000 value
    # Both periods are faded by the full record's support (the post-2000 one's is ignored).
    assert decode("support").tolist() == [40] * 13
    assert meta["support"]["share_of_domain_below_fade"] == 1.0
    assert [period["key"] for period in meta["periods"]] == ["full", "post2000"]
    assert meta["period_difference"]["share_abs_ct_above_0_05_degc"] == pytest.approx(1.0)
    json.dumps(meta)  # serialisable


@pytest.mark.unit
def test_build_package_refuses_an_empty_domain(mod):
    grid = {"nx": 2, "ny": 2, "x0_m": 0.0, "y0_m": 0.0, "dx_m": 1.0, "dy_m": -1.0}
    with pytest.raises(ValueError):
        mod.build_package(synthetic_periods(grid), np.zeros((2, 2), dtype=bool), grid, [], [])


# ------------------------------------------------------------------ committed package


@pytest.mark.integration
def test_committed_package_matches_the_climatology_at_known_sites():
    from pathlib import Path

    meta_path = Path(__file__).resolve().parent.parent / "static" / "tools" / "data" / "ocean_ice_seafloor_antarctica.meta.json"
    if not meta_path.exists():
        pytest.skip("package not built")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sites = {site["name"]: site for site in meta["validation"]["sites"]}
    # Warm Circumpolar Deep Water floods the Amundsen Sea shelf; the Ross and Weddell shelves
    # hold water near the surface freezing point, and Ice Shelf Water below it in Filchner Trough.
    assert sites["Pine Island Bay"]["packed_ct_full_degc"] > 0.8
    assert sites["Belgica Trough"]["packed_ct_full_degc"] > 0.8
    assert sites["Ross Sea shelf"]["packed_ct_full_degc"] < -1.7
    assert sites["Filchner Trough"]["packed_ct_full_degc"] < -1.9
    for site in sites.values():
        if site["in_domain"]:
            assert abs(site["packed_ct_full_degc"] - site["column_ct_degc"]) < 0.35, site["name"]
