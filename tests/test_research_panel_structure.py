"""The research edition's panel: its groups, what each holds, and the two locale shells.

The panel is one scrolling column of collapsible groups (docs/research-panel-redesign.md):
the masthead, where to look, then Scene, Ice flow, Beneath the ice, Ocean and ice
shelves, Change over time, Places and labels, Display, and Data and sources. Every
control keeps the id the runtime, its presets and the tests address it by, and each
legend sits under the layer that draws it.

The two locale pages are hand-written copies, so these checks also hold them to the
same ids, in the same order and the same groups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPLORERS = {
    "en": ROOT / "static/tools/3D-interactive-cryosphere-explorer.html",
    "zh": ROOT / "static/zh/tools/3D-interactive-cryosphere-explorer.html",
}
PUBLIC_PAGES = (ROOT / "static/explore/index.html", ROOT / "static/zh/explore/index.html")

GROUP_ORDER = ("scene", "flow", "bed", "ocean", "time", "places", "display", "data")
OPEN_BY_DEFAULT = ("scene", "flow", "time")

# Where every control goes. "where" is the block above the groups.
GROUP_CONTROLS = {
    "where": (
        "regionPreset",
        "polarSearchInput",
        "polarSearchResults",
        "polarSearchStatus",
        "polarFeatureDetails",
        "terrainPreset",
        "gridPreset",
        "resolutionPreset",
        "terrainHint",
    ),
    "scene": (
        "showIce",
        "showIceBottom",
        "showBed",
        "showSea",
        "showGroundingLine",
        "bedLegendSection",
        "iceOpacity",
        "exaggeration",
    ),
    "flow": ("showVelocity", "showFlowline", "flowlineProfileCardMount", "velocityLegendSection"),
    "bed": (
        "showBasalFriction",
        "basalFrictionLegendSection",
        "showEffectivePressure",
        "effectivePressureLegendSection",
        "showSubglacialChannels",
        "channelLegendSection",
    ),
    "ocean": (
        "showOceanCurrents",
        "oceanCurrentLayerControls",
        "oceanLegendSection",
        "showBasalMelt",
        "basalMeltLegendSection",
        "showThermalDriving",
        "thermalDrivingLegendSection",
    ),
    "time": (
        "showSurfaceChange",
        "surfaceChangeControls",
        "surfaceChangeLegendSection",
        "showSeaIce",
        "seaIceControls",
        "seaIceLegendSection",
        "showIceProjection",
        "iceProjectionControls",
        "projectionLegendSection",
        "showIsostaticRebound",
        "isostaticReboundControls",
        "reboundLegendNote",
    ),
    "places": ("showResearchStations", "showGeographicNames", "showIceCores", "showRefinedBasins"),
    "display": ("animateFlow", "wireframe"),
    "data": ("metaList",),
}

# Each legend follows the toggle that draws it, in the same group.
LEGEND_OF = {
    "bedLegendSection": "showBed",
    "velocityLegendSection": "showVelocity",
    "basalFrictionLegendSection": "showBasalFriction",
    "effectivePressureLegendSection": "showEffectivePressure",
    "channelLegendSection": "showSubglacialChannels",
    "oceanLegendSection": "showOceanCurrents",
    "basalMeltLegendSection": "showBasalMelt",
    "thermalDrivingLegendSection": "showThermalDriving",
    "surfaceChangeLegendSection": "showSurfaceChange",
    "seaIceLegendSection": "showSeaIce",
    "projectionLegendSection": "showIceProjection",
    "reboundLegendNote": "showIsostaticRebound",
}

# The layers that start off: a group counts them, and opens when one is switched on.
LAYERS = (
    "showSea",
    "showGroundingLine",
    "showVelocity",
    "showFlowline",
    "showBasalFriction",
    "showEffectivePressure",
    "showSubglacialChannels",
    "showOceanCurrents",
    "showBasalMelt",
    "showThermalDriving",
    "showSurfaceChange",
    "showSeaIce",
    "showIceProjection",
    "showIsostaticRebound",
    "showResearchStations",
    "showGeographicNames",
    "showIceCores",
    "showRefinedBasins",
    "wireframe",
)

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


@dataclass
class _Element:
    tag: str
    attrs: dict[str, str]
    ancestors: tuple[str, ...]
    group: str | None
    position: int
    # Every enclosing element, outermost first, as (tag, attributes).
    chain: tuple[tuple[str, dict[str, str]], ...] = ()

    def inside(self, tag: str | None = None, **attrs: str) -> bool:
        """Whether an enclosing element has the tag and attributes (class matches one class)."""
        for ancestor_tag, ancestor_attrs in self.chain:
            if tag and ancestor_tag != tag:
                continue
            if all(
                value in ancestor_attrs.get(key, "").split() if key == "class" else ancestor_attrs.get(key) == value
                for key, value in attrs.items()
            ):
                return True
        return False


@dataclass
class _Page:
    elements: dict[str, _Element] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    groups: list[tuple[str, bool]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    h1_count: int = 0


class _PanelParser(HTMLParser):
    """Ids with their ancestry and group, the groups in order, and the text of each label."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.page = _Page()
        self._stack: list[tuple[str, dict[str, str]]] = []
        self._label_text: list[str] | None = None

    def _group(self) -> str | None:
        for tag, attrs in reversed(self._stack):
            if tag == "details" and attrs.get("data-group"):
                return attrs["data-group"]
            if attrs.get("id") == "viewControlsSection":
                return "where"
        return None

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {key: value or "" for key, value in attrs_list}
        if tag == "h1":
            self.page.h1_count += 1
        if tag == "details" and attrs.get("data-group"):
            self.page.groups.append((attrs["data-group"], "open" in attrs))
        element_id = attrs.get("id")
        if element_id:
            assert element_id not in self.page.elements, f"duplicate id {element_id}"
            self.page.elements[element_id] = _Element(
                tag=tag,
                attrs=attrs,
                ancestors=tuple(a["id"] for _, a in self._stack if a.get("id")),
                group=self._group(),
                position=len(self.page.order),
                chain=tuple((ancestor_tag, dict(ancestor_attrs)) for ancestor_tag, ancestor_attrs in self._stack),
            )
            self.page.order.append(element_id)
        if tag == "label":
            self._label_text = []
        if tag not in _VOID:
            self._stack.append((tag, attrs))

    def handle_endtag(self, tag: str) -> None:
        if tag == "label" and self._label_text is not None:
            self.page.labels.append(" ".join("".join(self._label_text).split()))
            self._label_text = None
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index][0] == tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if self._label_text is not None:
            self._label_text.append(data)


def _parse(path: Path) -> _Page:
    parser = _PanelParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.page


@pytest.fixture(scope="module", params=sorted(EXPLORERS), ids=str)
def locale(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def pages() -> dict[str, _Page]:
    return {key: _parse(path) for key, path in EXPLORERS.items()}


def test_the_page_names_the_research_edition_and_loads_the_panel_styles_after_the_shared_ones(locale) -> None:
    html = EXPLORERS[locale].read_text(encoding="utf-8")
    assert re.search(r'<html lang="[^"]+" data-locale="[^"]+" data-edition="research">', html)
    stylesheets = re.findall(r'<link rel="stylesheet" href="([^"]+)" />', html)
    assert [Path(href).name for href in stylesheets] == ["explorer.css", "panel.css"]


def test_the_groups_come_in_the_agreed_order_with_scene_flow_and_time_open(locale, pages) -> None:
    groups = pages[locale].groups
    assert [key for key, _ in groups] == list(GROUP_ORDER)
    assert [key for key, is_open in groups if is_open] == list(OPEN_BY_DEFAULT)


@pytest.mark.parametrize("group", sorted(GROUP_CONTROLS))
def test_every_control_is_in_its_group(locale, pages, group) -> None:
    elements = pages[locale].elements
    for control_id in GROUP_CONTROLS[group]:
        assert control_id in elements, f"{control_id} is missing"
        assert elements[control_id].group == group, f"{control_id} is in {elements[control_id].group}"


def test_each_legend_follows_the_layer_that_draws_it(locale, pages) -> None:
    elements = pages[locale].elements
    for legend_id, toggle_id in LEGEND_OF.items():
        legend, toggle = elements[legend_id], elements[toggle_id]
        assert legend.group == toggle.group, legend_id
        assert legend.position > toggle.position, f"{legend_id} comes before {toggle_id}"


def test_both_locales_carry_the_same_ids_in_the_same_order_and_groups(pages) -> None:
    english, chinese = pages["en"], pages["zh"]
    assert english.order == chinese.order
    assert {key: element.group for key, element in english.elements.items()} == {
        key: element.group for key, element in chinese.elements.items()
    }


def test_layers_that_start_off_are_marked_for_their_group_to_count(locale, pages) -> None:
    elements = pages[locale].elements
    marked = sorted(key for key, element in elements.items() if "data-layer" in element.attrs)
    assert marked == sorted(LAYERS)
    for layer_id in LAYERS:
        assert "checked" not in elements[layer_id].attrs, f"{layer_id} should start off"


def test_the_masthead_is_one_wordmark_with_the_intro_and_the_tour_in_its_menu(locale, pages) -> None:
    html = EXPLORERS[locale].read_text(encoding="utf-8")
    elements = pages[locale].elements
    assert pages[locale].h1_count == 1
    assert re.search(
        r'<h1 class="rp-wordmark"><img class="rp-wordmark__icon" src="[^"]+" alt="" width="28" height="28" />'
        r'<span class="rp-wordmark__name"><span class="rp-wordmark__3d">3D</span> ICE</span></h1>',
        html,
    )
    # The edition switch names the edition now, so the tag beside the wordmark has gone.
    assert "rp-edition-tag" not in html
    # The logo images are no longer fetched on this page.
    assert "panel-brand-logo" not in html
    button = elements["panelInfoButton"].attrs
    assert button["aria-controls"] == "panelInfoMenu" and button["aria-expanded"] == "false"
    assert "hidden" in elements["panelInfoMenu"].attrs
    assert "panelInfoMenu" in elements["panelSubtitle"].ancestors
    menu = html[html.index('id="panelInfoMenu"') : html.index("</header>")]
    assert menu.count("tour=1") == 1


def test_the_page_opens_the_groups_as_stored_before_the_runtime_loads(locale) -> None:
    """An inline script applies the reader's stored choice before the first paint; it must
    read the key the runtime writes."""
    html = EXPLORERS[locale].read_text(encoding="utf-8")
    module = (ROOT / "static/tools/js/research-panel.js").read_text(encoding="utf-8")
    key = re.search(r'GROUP_STATE_STORAGE_KEY = "([^"]+)"', module).group(1)
    assert f'window.localStorage.getItem("{key}")' in html
    assert html.index(f'getItem("{key}")') > html.index("</aside>")


def test_the_viewer_holds_the_view_actions_and_the_panel_none(locale, pages) -> None:
    elements = pages[locale].elements
    assert "viewerShell" in elements["resetView"].ancestors
    assert "viewerShell" in elements["viewerFullscreenToggle"].ancestors
    # One fullscreen button, the viewer's; Ice flow > Flowlines is the one flowline switch.
    assert "fullscreenToggle" not in elements
    assert "projectionFlowline" not in elements


def test_the_region_and_dataset_selects_stay_as_the_runtime_s_source_of_truth(locale, pages) -> None:
    elements = pages[locale].elements
    for select_id in ("regionPreset", "resolutionPreset"):
        assert elements[select_id].tag == "select"
        assert "hidden" in elements[select_id].attrs, f"{select_id} is shown by its own controls"
    html = EXPLORERS[locale].read_text(encoding="utf-8")
    assert 'data-segmented-for="regionPreset"' in html
    assert re.findall(r'data-segmented-for="regionPreset"[\s\S]*?</div>', html)[0].count("data-value=") == 2


def test_no_layer_label_starts_with_show(locale, pages) -> None:
    prefix = {"en": "Show ", "zh": "显示"}[locale]
    labels = [label for label in pages[locale].labels if label]
    assert labels, "the page should have labels"
    assert [label for label in labels if label.startswith(prefix)] == []


def test_the_public_pages_share_the_panel_styles_but_not_the_collapsible_groups(locale) -> None:
    """The public pages have the same masthead and look, and groups that stay open: with no
    <details> group, js/research-panel.js is never loaded there and stays inert."""
    for page in PUBLIC_PAGES:
        html = page.read_text(encoding="utf-8")
        stylesheets = re.findall(r'<link rel="stylesheet" href="([^"]+)" />', html)
        assert [Path(href).name for href in stylesheets] == ["explorer.css", "panel.css", "explore.css"]
        assert '<details class="rp-group"' not in html
        assert "data-group=" not in html


# ------------------------------------------------ phase 2: choices, time, data

PANEL_STATE = (ROOT / "static/tools/js/panel-state.js").read_text(encoding="utf-8")


def _js_list(name: str) -> list[str]:
    """The ids of a frozen list in panel-state.js, such as `bed: Object.freeze([...])`."""
    match = re.search(rf"\b{name}\s*[:=]\s*Object\.freeze\(\[([^\]]*)\]\)", PANEL_STATE)
    assert match, name
    return re.findall(r'"([^"]+)"', match.group(1))


@pytest.mark.parametrize("choice", ["bed", "shelf"])
def test_each_one_of_choice_holds_its_layers_and_a_first_pill_for_none(locale, choice, pages) -> None:
    elements = pages[locale].elements
    html = EXPLORERS[locale].read_text(encoding="utf-8")
    for layer_id in _js_list(choice):
        assert elements[layer_id].inside("div", **{"data-choice": choice}), layer_id
        assert elements[layer_id].inside("label", **{"class": "rp-choice__opt"}), layer_id
    field = html[html.index(f'data-choice="{choice}"') :]
    field = field[: field.index("</div>\n          </div>")]
    assert field.count("data-choice-none") == 1
    assert 'aria-pressed="true"' in field


def test_change_over_time_is_one_at_a_time_from_the_present_day(locale, pages) -> None:
    elements = pages[locale].elements
    html = EXPLORERS[locale].read_text(encoding="utf-8")
    toggles = _js_list("TIME_FEATURE_TOGGLES")
    assert [element_id for element_id in pages[locale].order if element_id in toggles] == toggles
    keys = [elements[toggle].attrs.get("data-time") for toggle in toggles]
    assert keys == ["surfaceChange", "seaIce", "projection", "rebound"]
    time_group = html[html.index('data-group="time"') : html.index('data-group="places"')]
    assert time_group.count("data-time-none") == 1
    assert time_group.index("data-time-none") < time_group.index('id="showSurfaceChange"')
    assert 'class="rp-group__hint"' in time_group


@pytest.mark.parametrize(
    ("controls", "notes"),
    [
        ("surfaceChangeControls", ("surfaceChangeLegendNote", "surfaceChangeNote")),
        ("seaIceControls", ("seaIceLegendNote", "seaIceNote")),
        ("iceProjectionControls", ("projectionLegendNote", "projectionNote")),
    ],
)
def test_a_record_s_longer_notes_wait_in_its_about_box(locale, controls, notes, pages) -> None:
    elements = pages[locale].elements
    for note in notes:
        assert controls in elements[note].ancestors, note
        assert elements[note].inside("details", **{"class": "rp-about"}), note


def test_data_and_sources_offers_every_dataset_from_a_switch_under_the_list(locale, pages) -> None:
    elements = pages[locale].elements
    switch = elements["metaShowAll"]
    assert switch.group == "data" and switch.tag == "button"
    assert switch.position > elements["metaList"].position
    assert switch.attrs.get("aria-expanded") == "false" and "hidden" in switch.attrs
