"""Static runtime wiring checks for polar feature layers and search."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPLORERS = (
    ROOT / "static/tools/3D-interactive-cryosphere-explorer.html",
    ROOT / "static/zh/tools/3D-interactive-cryosphere-explorer.html",
)


def test_both_locales_wire_search_and_independent_feature_layers() -> None:
    for explorer in EXPLORERS:
        html = explorer.read_text(encoding="utf-8")
        for control_id in ("polarSearchInput", "showResearchStations", "showGeographicNames", "polarSearchResults"):
            assert f'id="{control_id}"' in html
        assert "js/polar-features.js" in html


def test_runtime_references_all_four_catalogues() -> None:
    runtime = (ROOT / "static/tools/3D-interactive-cryosphere-explorer.html").read_text(encoding="utf-8")
    for filename in (
        "antarctica_research_stations.json",
        "greenland_research_stations.json",
        "antarctica_geographic_names.json",
        "greenland_geographic_names.json",
    ):
        assert filename in runtime
