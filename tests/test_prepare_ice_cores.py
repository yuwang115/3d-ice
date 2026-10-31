"""Tests for scripts/prepare_ice_cores.py and the ice-core site catalogues it writes."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE_TABLE = ROOT / "scripts" / "data" / "ice_core_sites.json"
CATALOGUES = {"antarctica": "antarctica_ice_cores.json", "greenland": "greenland_ice_cores.json"}

SITE = {
    "name": "WAIS Divide",
    "aliases": ["WDC06A", "WAIS"],
    "region": "antarctica",
    "core_type": "deep",
    "project": "WAIS Divide Ice Core Project",
    "operator": "United States",
    "drilled": "2006–2011",
    "depth_m": 3405,
    "reached_bed": False,
    "oldest_ka": 68,
    "elevation_m": 1766,
    "lat": -79.468,
    "lon": -112.086,
    "coordinate_sources": [
        {"url": "https://example.org/a", "lat": -79.467, "lon": -112.085},
        {"url": "https://example.org/b", "lat": -79.47, "lon": -112.09},
    ],
    "reference": {"citation": "WAIS Divide Project Members 2013, Nature 500, 440-444", "doi": "10.1038/nature12376"},
}


def site(**changes):
    return {**copy.deepcopy(SITE), **changes}


@pytest.fixture(scope="module")
def mod(ice_cores_module):
    return ice_cores_module


def test_a_checked_site_has_no_errors(mod) -> None:
    assert mod.site_errors(site()) == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": ""}, "missing name"),
        ({"region": "arctic"}, "unknown region"),
        ({"core_type": "medium"}, "unknown core_type"),
        ({"lat": 79.468}, "not in antarctica"),
        ({"region": "greenland"}, "not in greenland"),
        ({"lat": float("nan")}, "not in antarctica"),
        ({"aliases": "WDC06A"}, "aliases"),
        ({"drilled": "2006-2011"}, "en dash"),
        ({"drilled": "2011–2006"}, "rising"),
        ({"drilled": "2006\n"}, "en dash"),
        ({"drilled": 2006}, "en dash"),
        ({"name": "   "}, "name must be text"),
        ({"name": "冰芯"}, "Latin letters"),
        ({"name": "W" * 181}, "name must be text"),
        ({"project": 7}, "project must be text"),
        ({"operator": "  "}, "operator must be text"),
        ({"reached_bed": 1}, "reached_bed"),
        ({"depth_m": 9000}, "depth_m"),
        ({"depth_m": True}, "depth_m"),
        ({"oldest_ka": -1}, "oldest_ka"),
        ({"elevation_m": 5000}, "elevation_m"),
        ({"reached_bed": "yes"}, "reached_bed"),
        ({"reached_bed": True, "depth_m": None}, "needs its depth"),
        ({"reference": {"citation": "", "doi": "10.1038/nature12376"}}, "citation and a DOI"),
        ({"reference": {"citation": "X", "doi": "doi:10.1038/x"}}, "citation and a DOI"),
        ({"reference": "10.1038/nature12376"}, "citation and a DOI"),
    ],
)
def test_site_errors_name_each_problem(mod, changes, message) -> None:
    errors = mod.site_errors(site(**changes))
    assert any(message in error for error in errors), errors


def test_a_site_without_aliases_is_reported_not_raised(mod) -> None:
    missing = site()
    del missing["aliases"]
    assert "aliases must be a list of strings" in mod.site_errors(missing)


def test_a_site_needs_two_agreeing_coordinate_sources(mod) -> None:
    one_source = site(coordinate_sources=SITE["coordinate_sources"][:1])
    assert "two coordinate sources" in mod.site_errors(one_source)[0]
    same_url = site(coordinate_sources=[SITE["coordinate_sources"][0]] * 2)
    assert "two coordinate sources" in mod.site_errors(same_url)[0]
    fragments = site(coordinate_sources=[
        {**SITE["coordinate_sources"][0], "url": "https://example.org/paper#table1"},
        {**SITE["coordinate_sources"][1], "url": "https://example.org/paper#fig2"},
    ])
    assert "two coordinate sources" in mod.site_errors(fragments)[0]
    http_only = site(coordinate_sources=[{**source, "url": source["url"].replace("https", "http")} for source in SITE["coordinate_sources"]])
    assert "two coordinate sources" in mod.site_errors(http_only)[0]
    # Siple Dome and WAIS Divide are ~300 km apart; a source that far off is refused.
    far = site(coordinate_sources=[SITE["coordinate_sources"][0], {"url": "https://example.org/c", "lat": -81.66, "lon": -148.81}])
    assert any("km away" in error for error in mod.site_errors(far))


def test_haversine_matches_a_known_distance(mod) -> None:
    # One degree of latitude is ~111.2 km on the mean sphere; longitude shrinks with cos(lat).
    assert mod.haversine_km(-80, 0, -79, 0) == pytest.approx(111.2, abs=0.1)
    assert mod.haversine_km(-80, 0, -80, 1) == pytest.approx(111.2 * 0.17365, abs=0.1)


def test_build_item_projects_and_keeps_the_shown_facts(mod, polar_features_module) -> None:
    item = mod.build_item(site(aliases=["WDC06A", "wais divide", "WAIS"]))
    assert item["id"] == "antarctica-ice-core-wais-divide"
    assert (item["layer"], item["kind"], item["feature_type"], item["display_priority"]) == (
        "ice_cores",
        "ice_core",
        "Deep ice core",
        1,
    )
    assert (item["x_m"], item["y_m"]) == polar_features_module.project_polar(-79.468, -112.086, 3031)
    # The name is not repeated as an alias, and the Chinese search term is added once.
    assert item["aliases"] == ["WDC06A", "WAIS", "冰芯"]
    assert item["source_url"] == "https://doi.org/10.1038/nature12376"
    assert item["citation"].startswith("WAIS Divide Project Members 2013")
    assert {"coordinate_sources", "notes", "reference"}.isdisjoint(item)


def test_build_catalogues_splits_regions_and_refuses_bad_or_duplicate_sites(mod) -> None:
    greenland = site(
        name="GRIP",
        aliases=[],
        region="greenland",
        core_type="deep",
        lat=72.58,
        lon=-37.64,
        coordinate_sources=[
            {"url": "https://example.org/g1", "lat": 72.58, "lon": -37.64},
            {"url": "https://example.org/g2", "lat": 72.5833, "lon": -37.6333},
        ],
    )
    shallow = site(name="Gomez", core_type="shallow", depth_m=136, lat=-73.59, lon=-70.36, coordinate_sources=[
        {"url": "https://example.org/s1", "lat": -73.59, "lon": -70.36},
        {"url": "https://example.org/s2", "lat": -73.5904, "lon": -70.3608},
    ])
    payloads = mod.build_catalogues([shallow, site(), greenland], "2026-10-30")
    assert set(payloads) == set(CATALOGUES.values())
    antarctica = payloads["antarctica_ice_cores.json"]
    assert (antarctica["region"], antarctica["layer"], antarctica["projection"]) == ("antarctica", "ice_cores", "EPSG:3031")
    assert [item["name"] for item in antarctica["items"]] == ["WAIS Divide", "Gomez"]
    assert payloads["greenland_ice_cores.json"]["projection"] == "EPSG:3413"
    assert payloads["greenland_ice_cores.json"]["feature_count"] == 1

    with pytest.raises(ValueError, match="WAIS Divide: unknown core_type"):
        mod.build_catalogues([site(core_type="medium")], "2026-10-30")
    with pytest.raises(ValueError, match="Duplicate"):
        mod.build_catalogues([site(), site(aliases=[])], "2026-10-30")


def test_load_site_table_rejects_other_files(mod, tmp_path) -> None:
    path = tmp_path / "sites.json"
    path.write_text(json.dumps({"schema_version": 2, "as_of": "2026-10-30", "sites": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="version-1"):
        mod.load_site_table(path)


# --- The committed table and catalogues ------------------------------------------------


@pytest.fixture(scope="module")
def table(mod) -> dict:
    return mod.load_site_table(SITE_TABLE)


@pytest.fixture(scope="module")
def catalogues(data_dir: Path) -> dict[str, dict]:
    return {region: json.loads((data_dir / name).read_text(encoding="utf-8")) for region, name in CATALOGUES.items()}


def test_committed_catalogues_are_the_build_of_the_committed_table(mod, table, catalogues) -> None:
    rebuilt = mod.build_catalogues(table["sites"], table["as_of"])
    for region, filename in CATALOGUES.items():
        assert catalogues[region] == rebuilt[filename]


def test_committed_catalogues_cover_the_landmark_cores(catalogues) -> None:
    names = {
        region: {item["name"] for item in catalogue["items"]} | {alias for item in catalogue["items"] for alias in item["aliases"]}
        for region, catalogue in catalogues.items()
    }
    assert {"Vostok", "EPICA Dome C", "EDML", "Dome Fuji", "Talos Dome", "Law Dome", "Byrd", "Siple Dome", "WAIS Divide", "RICE"} <= names["antarctica"]
    assert {"Camp Century", "Dye-3", "GRIP", "GISP2", "NGRIP", "NEEM", "EGRIP", "Renland"} <= names["greenland"]
    assert 30 <= catalogues["antarctica"]["feature_count"] <= 50
    assert 12 <= catalogues["greenland"]["feature_count"] <= 25


def test_committed_items_link_a_publication_and_sit_on_their_grid(catalogues, polar_features_module) -> None:
    for region, catalogue in catalogues.items():
        epsg = 3031 if region == "antarctica" else 3413
        ids = [item["id"] for item in catalogue["items"]]
        assert len(ids) == len(set(ids))
        assert catalogue["feature_count"] == len(catalogue["items"])
        for item in catalogue["items"]:
            assert item["layer"] == "ice_cores" and item["kind"] == "ice_core"
            assert item["source_url"].startswith("https://doi.org/10.")
            assert item["citation"]
            assert (item["x_m"], item["y_m"]) == polar_features_module.project_polar(item["lat"], item["lon"], epsg)
            if item["core_type"] == "deep":
                assert item["depth_m"] and item["depth_m"] >= 1000, item["name"]


TERRAIN = {
    "antarctica": ("bedmachine_antarctica_v4_741.meta.json", "bedmachine_antarctica_v4_741.bin"),
    "greenland": ("bedmachine_greenland_v6_1km.meta.json", "bedmachine_greenland_v6_1km.bin"),
}
ICE_MASK_VALUES = {2, 3, 4}  # grounded ice, floating ice, Lake Vostok
# BedMachine's HD grids are 4 km (Antarctica) and 1 km (Greenland); in mountains a cell's mean
# surface can sit a few hundred metres from a drill site's own elevation (Dominion Range: 310 m).
MAX_ELEVATION_MISMATCH_M = 350


def test_every_site_lies_on_ice_at_its_published_elevation(catalogues, data_dir: Path) -> None:
    """A wrong coordinate shows up as a site off the ice or far from its own elevation."""

    np = pytest.importorskip("numpy")
    for region, (meta_name, bin_name) in TERRAIN.items():
        meta = json.loads((data_dir / meta_name).read_text(encoding="utf-8"))
        grid = meta["grid"]
        fields = {field["name"]: field for field in meta["fields"]}
        raw = (data_dir / bin_name).read_bytes()
        shape = (grid["ny"], grid["nx"])
        surface = np.frombuffer(raw, "<i2", grid["nx"] * grid["ny"], fields["surface"]["byte_offset"]).reshape(shape)
        mask = np.frombuffer(raw, "u1", grid["nx"] * grid["ny"], fields["mask"]["byte_offset"]).reshape(shape)
        for item in catalogues[region]["items"]:
            col = round((item["x_m"] - grid["x0_m"]) / grid["dx_m"])
            row = round((item["y_m"] - grid["y0_m"]) / grid["dy_m"])
            assert 0 <= row < grid["ny"] and 0 <= col < grid["nx"], item["name"]
            assert int(mask[row, col]) in ICE_MASK_VALUES, item["name"]
            if item["elevation_m"] is not None:
                mismatch = abs(int(surface[row, col]) - item["elevation_m"])
                assert mismatch <= MAX_ELEVATION_MISMATCH_M, f"{item['name']}: {mismatch:.0f} m"
