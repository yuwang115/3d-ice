"""Schema and source checks for the static polar feature catalogues."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest


CATALOGUE_PATHS = (
    "antarctica_research_stations.json",
    "greenland_research_stations.json",
    "antarctica_geographic_names.json",
    "greenland_geographic_names.json",
)


@pytest.fixture(scope="module")
def catalogues(data_dir: Path) -> list[dict]:
    return [json.loads((data_dir / name).read_text(encoding="utf-8")) for name in CATALOGUE_PATHS]


def test_catalogues_have_consistent_metadata_and_unique_ids(catalogues: list[dict]) -> None:
    for catalogue in catalogues:
        assert catalogue["schema_version"] == 1
        assert catalogue["region"] in {"antarctica", "greenland"}
        assert catalogue["layer"] in {"research_stations", "geographic_names"}
        assert catalogue["projection"] in {"EPSG:3031", "EPSG:3413"}
        assert catalogue["as_of"]
        assert catalogue["sources"]
        assert catalogue["feature_count"] == len(catalogue["items"])
        ids = [item["id"] for item in catalogue["items"]]
        assert len(ids) == len(set(ids))


def test_features_have_searchable_names_sources_and_finite_positions(catalogues: list[dict]) -> None:
    for catalogue in catalogues:
        catalogue_source_urls = {source["url"] for source in catalogue["sources"]}
        for item in catalogue["items"]:
            assert item["region"] == catalogue["region"]
            assert item["layer"] == catalogue["layer"]
            assert item["name"].strip()
            assert isinstance(item["aliases"], list)
            assert -90 <= item["lat"] <= 90
            assert -180 <= item["lon"] <= 180
            assert math.isfinite(item["x_m"])
            assert math.isfinite(item["y_m"])
            assert item["source_url"].startswith("https://")
            assert item["source_url"] in catalogue_source_urls


def test_station_snapshot_covers_national_programmes_and_landmark_sites(catalogues: list[dict]) -> None:
    stations = [item for catalogue in catalogues if catalogue["layer"] == "research_stations" for item in catalogue["items"]]
    antarctic_countries = {item["country"] for item in stations if item["region"] == "antarctica"}
    assert len(antarctic_countries) >= 25
    assert {"Argentina", "Australia", "China", "United Kingdom", "United States"} <= antarctic_countries
    station_names = {item["name"] for item in stations}
    assert "Amundsen-Scott South Pole Station" in station_names
    assert "Summit Station" in station_names


def test_geographic_names_cover_requested_feature_types(catalogues: list[dict]) -> None:
    geographic_catalogues = [catalogue for catalogue in catalogues if catalogue["layer"] == "geographic_names"]
    expected_kinds = {
        "antarctica": {"ocean", "sea", "basin", "mountain_range", "plateau", "ice_shelf"},
        "greenland": {"ocean", "sea", "basin", "mountain_range", "mountain", "plateau", "fjord", "strait", "ice_sheet", "nunatak"},
    }
    for catalogue in geographic_catalogues:
        assert 100 <= catalogue["feature_count"] <= 200
        assert catalogue["feature_count"] == 160
        assert catalogue["selection"]["strategy"] == "balanced_key_features"
        assert catalogue["selection"]["limit"] == 160
        assert catalogue["selection"]["source_feature_count"] >= catalogue["feature_count"]
        assert sum(catalogue["selection"]["kind_quotas"].values()) == catalogue["feature_count"]
        assert Counter(item["kind"] for item in catalogue["items"]) == catalogue["selection"]["kind_quotas"]
        assert expected_kinds[catalogue["region"]] <= {item["kind"] for item in catalogue["items"]}


def test_geographic_name_selection_preserves_landmarks(catalogues: list[dict]) -> None:
    searchable_by_region: dict[str, set[str]] = {"antarctica": set(), "greenland": set()}
    for catalogue in catalogues:
        if catalogue["layer"] != "geographic_names":
            continue
        for item in catalogue["items"]:
            searchable_by_region[catalogue["region"]].add(item["name"].casefold())
            searchable_by_region[catalogue["region"]].update(alias.casefold() for alias in item["aliases"])

    assert {
        "southern ocean",
        "ross sea",
        "weddell sea",
        "transantarctic mountains",
        "ellsworth mountains",
        "wilkes subglacial basin",
        "ross ice shelf",
        "south polar plateau",
    } <= searchable_by_region["antarctica"]
    assert {
        "greenland sea",
        "baffin bay",
        "davis strait",
        "kane basin",
        "gunnbjørn fjeld",
        "watkins range",
        "greenland ice sheet",
    } <= searchable_by_region["greenland"]


def test_balanced_selection_is_deterministic_and_spatially_diverse(polar_features_module) -> None:
    items = [
        {
            "id": f"sea-{index}",
            "kind": "sea",
            "name": f"Sea {index:02d}",
            "display_priority": 1 if index == 11 else 3,
            "lat": -70 + index % 3,
            "lon": -175 + index * 30,
        }
        for index in range(12)
    ]
    items.extend(
        {
            "id": f"basin-{index}",
            "kind": "basin",
            "name": f"Basin {index:02d}",
            "display_priority": 2,
            "lat": -80,
            "lon": -150 + index * 60,
        }
        for index in range(6)
    )
    original = [dict(item) for item in items]

    selected = polar_features_module.select_balanced_features(items, {"sea": 4, "basin": 3})
    reversed_selected = polar_features_module.select_balanced_features(list(reversed(items)), {"sea": 4, "basin": 3})

    assert len(selected) == 7
    assert {item["kind"] for item in selected} == {"sea", "basin"}
    assert "sea-11" in {item["id"] for item in selected}
    assert len({polar_features_module.longitude_sector(item["lon"]) for item in selected if item["kind"] == "sea"}) == 4
    assert [item["id"] for item in selected] == [item["id"] for item in reversed_selected]
    assert items == original


def test_balanced_selection_spreads_across_latitude_bands(polar_features_module) -> None:
    items = [
        {
            "id": f"mountain-{index}",
            "kind": "mountain",
            "name": f"Mountain {index}",
            "display_priority": 4,
            "lat": 57 + index * 4,
            "lon": -42,
        }
        for index in range(7)
    ]

    selected = polar_features_module.select_balanced_features(items, {"mountain": 4})

    assert len({polar_features_module.spatial_cell(item["lat"], item["lon"]) for item in selected}) == 4


def test_balanced_selection_removes_same_kind_name_and_alias_duplicates(polar_features_module) -> None:
    items = [
        {
            "id": "greenland-place-nunagis-kane",
            "kind": "basin",
            "name": "Ikersuaq",
            "aliases": ["Kane Basin"],
            "display_priority": 1,
            "lat": 78.5,
            "lon": -70,
        },
        {
            "id": "greenland-place-gebco-kane",
            "kind": "basin",
            "name": "Kane Basin",
            "aliases": [],
            "display_priority": 1,
            "lat": 79,
            "lon": -69,
        },
        {
            "id": "greenland-place-gebco-labrador",
            "kind": "basin",
            "name": "Labrador Basin",
            "aliases": [],
            "display_priority": 2,
            "lat": 58,
            "lon": -48,
        },
    ]

    selected = polar_features_module.select_balanced_features(items, {"basin": 2})

    assert [item["id"] for item in selected] == [
        "greenland-place-nunagis-kane",
        "greenland-place-gebco-labrador",
    ]


@pytest.mark.parametrize(
    ("items", "quotas", "message"),
    [
        ([{"id": "", "kind": "sea", "name": "Sea", "lat": 0, "lon": 0}], {"sea": 1}, "id"),
        ([{"id": "same", "kind": "sea", "name": "A", "lat": 0, "lon": 0}, {"id": "same", "kind": "sea", "name": "B", "lat": 1, "lon": 1}], {"sea": 1}, "duplicate"),
        ([{"id": "bad", "kind": "sea", "name": "Sea", "lat": math.nan, "lon": 0}], {"sea": 1}, "coordinate"),
        ([{"id": "ridge", "kind": "ridge", "name": "Ridge", "lat": 0, "lon": 0}], {"sea": 1}, "unconfigured"),
        ([{"id": "sea", "kind": "sea", "name": "Sea", "lat": 0, "lon": 0}], {"sea": 0}, "positive"),
    ],
)
def test_balanced_selection_rejects_invalid_inputs(polar_features_module, items, quotas, message) -> None:
    with pytest.raises(ValueError, match=message):
        polar_features_module.select_balanced_features(items, quotas)


def test_refined_basin_search_catalogues_match_boundary_sources(data_dir: Path) -> None:
    pairs = (
        ("antarctica", "imbie_refined_basins_v2.json", "antarctica_refined_basins_search.json"),
        ("greenland", "greenland_basins_ps_v1_4_2.json", "greenland_refined_basins_search.json"),
    )

    for region, boundary_filename, search_filename in pairs:
        boundary = json.loads((data_dir / boundary_filename).read_text(encoding="utf-8"))
        search = json.loads((data_dir / search_filename).read_text(encoding="utf-8"))
        assert search["schema_version"] == 1
        assert search["region"] == region
        assert search["layer"] == "refined_basins"
        assert search["feature_count"] == boundary["basin_count"] == len(search["items"])
        assert all(item["region"] == region and item["layer"] == "refined_basins" for item in search["items"])
        assert all("segments_xy_m" not in item for item in search["items"])
