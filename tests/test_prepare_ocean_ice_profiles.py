"""Tests for scripts/prepare_ocean_ice_profiles.py (OCEAN ICE hydrography profile compilation)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parent.parent / "static" / "tools" / "data"


@pytest.fixture(scope="module")
def mod(ocean_ice_profiles_module):
    return ocean_ice_profiles_module


LEVELS = np.arange(0.0, 1010.0, 10.0)  # 0 to 1000 m


def profiles_block(bottoms, value=lambda depth: depth / 1000.0):
    """(levels, profiles) with data from 10 m to each profile's bottom, NaN elsewhere."""
    block = np.full((LEVELS.size, len(bottoms)), np.nan)
    for i, bottom in enumerate(bottoms):
        if bottom is None:
            continue
        inside = (LEVELS >= 10) & (LEVELS <= bottom)
        block[inside, i] = value(LEVELS[inside])
    return block


@pytest.mark.unit
def test_deepest_reads_across_chunks(mod):
    block = profiles_block([400.0, None, 990.0])
    index, value = mod.deepest(((k, block[k:k + 13]) for k in range(0, LEVELS.size, 13)), 3)
    assert list(index) == [40, -1, 99]
    assert value[0] == pytest.approx(0.4) and np.isnan(value[1]) and value[2] == pytest.approx(0.99)


@pytest.mark.unit
def test_nearest_with_data_prefers_the_centre_then_the_shallower_neighbour(mod):
    rows = np.array([[1.0, np.nan, np.nan], [2.0, 2.0, np.nan], [np.nan, np.nan, np.nan], [4.0, 4.0, np.nan], [5.0, np.nan, 5.0]])
    out = mod.nearest_with_data(rows, 2)
    assert out[0] == 2.0  # one bin up and one down are as near: the shallower wins
    assert out[1] == 2.0
    assert out[2] == 5.0


@pytest.mark.unit
def test_standard_values_search_twenty_metres(mod):
    block = profiles_block([1000.0])
    block[LEVELS == 500.0, 0] = np.nan  # a gap at 500 m is filled from 490 m
    block[(LEVELS >= 780.0) & (LEVELS <= 820.0), 0] = np.nan  # a 50 m gap at 800 m is not
    out = mod.standard_values(block, LEVELS, 1)
    depths = list(mod.STANDARD_DEPTHS_M)
    assert out[depths.index(500), 0] == pytest.approx(0.49)
    assert np.isnan(out[depths.index(800), 0])
    assert out[depths.index(1000), 0] == pytest.approx(1.0)
    assert np.isnan(out[depths.index(1500), 0])


@pytest.mark.unit
def test_curves_hold_the_standard_depths_above_the_deepest_measurement_then_it(mod):
    max_depth = np.array([400.0, 95.0, 5000.0])
    lengths = mod.curve_lengths(max_depth)
    # 400 m: 10, 50, 100, 150, 200, 300 and the bottom; 95 m: 10, 50 and the bottom; 5000 m:
    # every standard depth shallower than 5000 m (15) and the bottom.
    assert lengths.tolist() == [7, 3, 16]
    standard = np.tile(np.arange(len(mod.STANDARD_DEPTHS_M), dtype=float)[:, None], (1, 3))
    curves = mod.build_curves(standard, np.array([-1.0, -2.0, -3.0]), max_depth)
    assert curves.size == lengths.sum()
    assert curves[:7].tolist() == [0, 1, 2, 3, 4, 5, -1]
    assert curves[7:10].tolist() == [0, 1, -2]
    assert curves[-1] == -3


@pytest.mark.unit
def test_select_profiles_keeps_good_dated_profiles_in_the_domain_by_date(mod):
    t = {
        "lat": np.array([-70.0, -40.0, -75.0, -65.0, -72.0, -71.0]),
        "lon": np.array([0.0, 0.0, 100.0, -60.0, 10.0, 20.0]),
        "days": np.array([30000.0, 30000.0, 20000.0, 25000.0, 1000.0, 26000.0]),
        "CODC_QCflag": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        "instr_flag": np.array([1.0, 2.0, 3.0, 1.0, 1.0, 9.0]),
        "deepest": np.array([10, 10, 10, 10, 10, 10]),
    }
    # 1: north of the domain; 3: flagged bad; 4: before 1970; 5: unknown platform.
    assert mod.select_profiles(t).tolist() == [2, 0]
    # A profile the salinity file flags bad is left out too.
    s = {"CODC_QCflag": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])}
    assert mod.select_profiles(t, s).tolist() == [0]


@pytest.mark.unit
def test_year_index_and_counts(mod):
    days = np.array([0, 10, 400, 400, 1200])  # 1970, 1970, 1971, 1971, 1973
    index = mod.year_index(days)
    assert (index["first_year"], index["last_year"]) == (1970, 1973)
    assert index["starts"] == [0, 2, 4, 4, 5]
    counts = mod.counts_by_year(index["years"], np.array([1, 2, 3, 3, 1]), 1970, 1973)
    assert counts == {"1": [1, 0, 0, 1], "2": [1, 0, 0, 0], "3": [0, 2, 0, 0]}


@pytest.mark.unit
def test_quantize(mod):
    codes = mod.quantize(np.array([34.1234, np.nan]), mod.SA_SCALE, mod.SA_OFFSET)
    assert codes[1] == mod.INT16_FILL
    assert codes[0] * mod.SA_SCALE + mod.SA_OFFSET == pytest.approx(34.1234, abs=mod.SA_SCALE / 2)
    with pytest.raises(ValueError):
        mod.quantize(np.array([60.0]), mod.SA_SCALE, mod.SA_OFFSET)


@pytest.mark.integration
def test_committed_packages_agree():
    points_meta = DATA / "ocean_ice_profiles_antarctica.meta.json"
    curves_meta = DATA / "ocean_ice_profiles_antarctica_curves.meta.json"
    if not points_meta.exists():
        pytest.skip("packages not built")
    points = json.loads(points_meta.read_text(encoding="utf-8"))
    curves = json.loads(curves_meta.read_text(encoding="utf-8"))
    count = points["points"]["count"]
    assert count > 400_000
    assert curves["points"]["count"] == count
    assert points["years"]["starts"][-1] == count
    starts = points["years"]["starts"]
    assert all(b >= a for a, b in zip(starts, starts[1:]))
    totals = sum(sum(values) for values in points["series"]["counts_by_platform_and_year"].values())
    assert totals == count
    field = next(f for f in points["fields"] if f["name"] == "max_depth")
    raw = (DATA / "ocean_ice_profiles_antarctica.bin").read_bytes()
    max_depth = np.frombuffer(raw, dtype="<u2", count=count, offset=field["byte_offset"]).astype(float)
    standard = np.asarray(curves["curves"]["standard_depths_m"], dtype=float)
    lengths = (standard[None, :] < max_depth[:, None]).sum(axis=1) + 1
    assert int(lengths.sum()) == curves["points"]["value_count"]


@pytest.mark.unit
def test_implausible_bins_read_as_missing(mod):
    block = np.array([[1.0, 71.7], [-12.0, 0.5]])  # 71.7 degC would not even fit the int16 codes
    wrapped = mod.PlausibleBins(block, mod.PLAUSIBLE["ct"])
    out = wrapped[0:2, :]
    assert out[0, 0] == 1.0 and np.isnan(out[0, 1]) and np.isnan(out[1, 0]) and out[1, 1] == 0.5
    assert wrapped.masked == 2
    assert wrapped.shape == (2, 2)
