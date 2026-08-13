"""Static runtime wiring checks for polar feature layers and search."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPLORERS = (
    ROOT / "static/tools/3D-interactive-cryosphere-explorer.html",
    ROOT / "static/zh/tools/3D-interactive-cryosphere-explorer.html",
)


class _IdStructureParser(HTMLParser):
    """Record ID order and ancestry without adding an HTML dependency."""

    _VOID_ELEMENTS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.ancestors: dict[str, tuple[str, ...]] = {}
        self.counts: dict[str, int] = {}
        self.positions: dict[str, int] = {}
        self._stack: list[tuple[str, str | None]] = []
        self._position = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element_id = dict(attrs).get("id")
        if element_id:
            self.counts[element_id] = self.counts.get(element_id, 0) + 1
            self.positions.setdefault(element_id, self._position)
            self.ancestors.setdefault(
                element_id,
                tuple(ancestor_id for _, ancestor_id in self._stack if ancestor_id),
            )
            self._position += 1
        if tag not in self._VOID_ELEMENTS:
            self._stack.append((tag, element_id))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index][0] == tag:
                del self._stack[index:]
                return


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
        "antarctica_refined_basins_search.json",
        "greenland_refined_basins_search.json",
    ):
        assert filename in runtime


def test_both_locales_wire_refined_basins_into_search_without_replacing_the_boundary_layer() -> None:
    for explorer in EXPLORERS:
        html = explorer.read_text(encoding="utf-8")
        for runtime_hook in (
            "activateRefinedBasinFeature",
            "ensureRefinedBasinsLoaded",
            "buildRefinedBasinOverlays",
        ):
            assert runtime_hook in html

    label_runtime = (ROOT / "static/tools/js/polar-features.js").read_text(encoding="utf-8")
    assert 'from "./polar-feature-label-style.js"' in label_runtime
    assert (ROOT / "static/tools/js/polar-refined-basins.js").is_file()


def test_places_and_geographic_features_heading_is_localized() -> None:
    expected_headings = (
        "Places &amp; Geographic Features",
        "地点与地理特征",
    )

    for explorer, expected_heading in zip(EXPLORERS, expected_headings, strict=True):
        html = explorer.read_text(encoding="utf-8")
        assert (
            f'<h3 id="polarFeatureHeading" class="section-title">{expected_heading}</h3>'
            in html
        )


def test_places_and_geographic_features_sits_between_ice_opacity_and_bed_in_both_locales() -> None:
    ordered_ids = (
        "viewControlsSection",
        "regionPreset",
        "resolutionPreset",
        "exaggeration",
        "iceOpacity",
        "polarFeatureSection",
        "polarSearchInput",
        "showResearchStations",
        "showGeographicNames",
        "showRefinedBasins",
        "showBed",
    )

    for explorer in EXPLORERS:
        parser = _IdStructureParser()
        parser.feed(explorer.read_text(encoding="utf-8"))

        assert all(parser.counts.get(element_id) == 1 for element_id in ordered_ids)
        assert "viewControlsSection" in parser.ancestors["polarFeatureSection"]
        for element_id in (
            "polarSearchInput",
            "showResearchStations",
            "showGeographicNames",
            "showRefinedBasins",
        ):
            assert "polarFeatureSection" in parser.ancestors[element_id]
        assert "viewControlsSection" in parser.ancestors["showBed"]
        assert "polarFeatureSection" not in parser.ancestors["showBed"]
        assert [parser.positions[element_id] for element_id in ordered_ids] == sorted(
            parser.positions[element_id] for element_id in ordered_ids
        )
