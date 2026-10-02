"""Integration checks for the isostatic-rebound layer's wiring in both locale explorers.

The repository has no generic en/zh parity test: parity is enforced per feature by
asserting the same control ids and module references exist in both HTML files. These
tests do that for the rebound layer, and additionally pin the control ordering so the
layer cannot drift out of the interactive-scenarios section in one locale only.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from tests.explorer_sources import explorer_source

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "static" / "tools" / "data"
EXPLORERS = (
    REPO_ROOT / "static" / "tools" / "3D-interactive-cryosphere-explorer.html",
    REPO_ROOT / "static" / "zh" / "tools" / "3D-interactive-cryosphere-explorer.html",
)
SOLVER_MODULES = (
    REPO_ROOT / "static" / "tools" / "js" / "gia-rebound.js",
    REPO_ROOT / "static" / "tools" / "js" / "gia-grid.js",
    REPO_ROOT / "static" / "tools" / "js" / "fft2d.js",
)
REBOUND_WORKER = REPO_ROOT / "static" / "tools" / "gia-rebound-worker.js"

# The section the rebound shares with the future projection: both are interactive scenarios.
SCENARIOS_SECTION = "interactiveScenariosSection"

CONTROL_IDS = (
    "showIsostaticRebound",
    "isostaticReboundControls",
    "reboundProgress",
    "reboundProgressValue",
    "reboundProgressNote",
    "reboundModel",
    "reboundModelNote",
    "reboundSeaLevel",
    "reboundSeaLevelValue",
    "reboundSeaLevelNote",
    "highlightEmergentLand",
    "reboundLegendNote",
)


class _IdCollector(HTMLParser):
    """Records element ids in document order, plus the open-element stack for each."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.order: list[str] = []
        self.counts: dict[str, int] = {}
        self.ancestors: dict[str, list[str]] = {}
        self._stack: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element_id = dict(attrs).get("id")
        if element_id:
            self.order.append(element_id)
            self.counts[element_id] = self.counts.get(element_id, 0) + 1
            self.ancestors[element_id] = [item for item in self._stack if item]
        if tag not in ("input", "br", "img", "meta", "link", "hr", "source"):
            self._stack.append(element_id)

    def handle_endtag(self, tag: str) -> None:
        if tag not in ("input", "br", "img", "meta", "link", "hr", "source") and self._stack:
            self._stack.pop()


@pytest.fixture(scope="module", params=EXPLORERS, ids=lambda path: path.parent.parent.name)
def explorer(request) -> str:
    return explorer_source(request.param)


@pytest.fixture(scope="module", params=EXPLORERS, ids=lambda path: path.parent.parent.name)
def explorer_ids(request) -> _IdCollector:
    collector = _IdCollector()
    collector.feed(request.param.read_text(encoding="utf-8"))
    return collector


def test_solver_modules_and_worker_exist() -> None:
    for path in (*SOLVER_MODULES, REBOUND_WORKER):
        assert path.is_file(), f"missing {path.relative_to(REPO_ROOT)}"
        assert path.stat().st_size > 0


def test_solver_logic_lives_outside_the_explorer_html() -> None:
    """The JOSS audit asks for testable domain logic in modules, not the page or its runtime."""
    solver = (REPO_ROOT / "static" / "tools" / "js" / "gia-rebound.js").read_text(encoding="utf-8")
    assert "export function solveIsostaticRebound" in solver
    for explorer_path in EXPLORERS:
        html = explorer_source(explorer_path)
        # The HTML may orchestrate the solver but must not reimplement it.
        assert "solveIsostaticRebound" in html
        assert "D del^4" not in html
        assert "function solveFlexuralRebound" not in html


def test_both_locales_expose_every_rebound_control(explorer: str) -> None:
    for control_id in CONTROL_IDS:
        assert f'id="{control_id}"' in explorer, f"{control_id} is missing"
    assert "js/gia-rebound.js" in explorer
    assert "gia-rebound-worker.js" in explorer


def test_rebound_control_ids_are_unique(explorer_ids: _IdCollector) -> None:
    duplicates = [key for key in CONTROL_IDS if explorer_ids.counts.get(key, 0) != 1]
    assert duplicates == [], f"ids appearing other than exactly once: {duplicates}"


def test_rebound_controls_sit_in_the_interactive_scenarios_section(explorer_ids: _IdCollector) -> None:
    for control_id in CONTROL_IDS:
        if control_id == "reboundLegendNote":
            assert "bedLegendSection" in explorer_ids.ancestors[control_id]
            continue
        assert SCENARIOS_SECTION in explorer_ids.ancestors[control_id], control_id
        assert "viewControlsSection" not in explorer_ids.ancestors[control_id], control_id
    for control_id in CONTROL_IDS[2:-1]:
        assert "isostaticReboundControls" in explorer_ids.ancestors[control_id], control_id


def test_the_scenarios_section_follows_the_view_controls(explorer_ids: _IdCollector) -> None:
    """The future projection and the ice-free rebound share a section of their own, after the
    layer and display toggles: the projection first, then the rebound, then the legends."""
    order = explorer_ids.order
    assert order.index("showBed") < order.index("showSubglacialChannels") < order.index("showOceanCurrents")
    assert order.index("showSea") < order.index("animateFlow") < order.index("wireframe")
    assert order.index("wireframe") < order.index(SCENARIOS_SECTION)
    assert order.index(SCENARIOS_SECTION) < order.index("iceProjectionRow") < order.index("showIceProjection")
    assert order.index("iceProjectionControls") < order.index("showIsostaticRebound")
    assert order.index("showIsostaticRebound") < order.index("isostaticReboundControls")
    assert order.index("isostaticReboundControls") < order.index("bedLegendSection")
    for control_id in ("showIceProjection", "iceProjectionControls", "projectionFlowline"):
        assert SCENARIOS_SECTION in explorer_ids.ancestors[control_id], control_id


def test_the_scenarios_section_is_titled_in_each_locale() -> None:
    english, chinese = (path.read_text(encoding="utf-8") for path in EXPLORERS)
    assert '<h2 id="interactiveScenariosHeading" class="section-title">Interactive Scenarios</h2>' in english
    assert '<h2 id="interactiveScenariosHeading" class="section-title">交互式情景</h2>' in chinese


def test_rebound_controls_start_in_the_headline_scenario(explorer: str) -> None:
    """Enabling the layer should land on 'ice gone, rebound complete' with no extra clicks."""
    assert 'id="showIsostaticRebound" type="checkbox" />' in explorer, "must default to off"
    assert 'id="reboundProgress" type="range" min="0" max="100" step="1" value="100"' in explorer
    assert 'id="reboundSeaLevel" type="range" min="0" max="70" step="1" value="0"' in explorer
    assert 'id="highlightEmergentLand" type="checkbox" checked />' in explorer
    # The published response of Paxman et al. (2022) is the default; the idealised
    # responses stay selectable for what-ifs.
    assert 'value="paxman2022" selected' in explorer
    assert '<option value="flexural">' in explorer
    assert '<option value="local">' in explorer
    assert 'reboundModel.value = REBOUND_MODEL_KEYS.published' in explorer


def _registered_rebound_packages(explorer: str) -> list[tuple[str, str]]:
    metas = re.findall(r'reboundMetaUrl: assetUrl\("data/([a-z0-9_]+)\.meta\.json"\)', explorer)
    bins = re.findall(r'reboundBinUrl: assetUrl\("data/([a-z0-9_]+)\.bin"\)', explorer)
    return list(zip(metas, bins, strict=True))


def test_every_terrain_dataset_registers_a_committed_response_package(explorer: str) -> None:
    datasets = re.findall(r'^\s+metaUrl: assetUrl\("data/([a-z0-9_]+)\.meta\.json"\)', explorer, flags=re.M)
    packages = _registered_rebound_packages(explorer)
    assert len(datasets) == 8
    assert len(packages) == len(datasets), "every terrain dataset needs a published response"
    for meta_name, bin_name in packages:
        assert meta_name == bin_name
        assert (DATA_DIR / f"{meta_name}.meta.json").is_file(), meta_name
        assert (DATA_DIR / f"{bin_name}.bin").is_file(), bin_name


def test_each_response_package_matches_its_terrain_grid(explorer: str) -> None:
    datasets = re.findall(r'^\s+metaUrl: assetUrl\("data/([a-z0-9_]+)\.meta\.json"\)', explorer, flags=re.M)
    for terrain, (response, _) in zip(datasets, _registered_rebound_packages(explorer), strict=True):
        terrain_grid = json.loads((DATA_DIR / f"{terrain}.meta.json").read_text(encoding="utf-8"))["grid"]
        response_meta = json.loads((DATA_DIR / f"{response}.meta.json").read_text(encoding="utf-8"))
        assert response_meta["grid"] == terrain_grid, f"{response} is not on the {terrain} grid"
        # Only QRF borrows another product's response; everything else samples its own.
        if terrain.startswith("greenland_qrf"):
            assert response_meta["source_package"]["metadata"].startswith("bedmachine_greenland_v6")
        else:
            assert response_meta["source_package"]["metadata"] == f"{terrain}.meta.json"


def test_qrf_datasets_flag_the_borrowed_bedmachine_load(explorer: str) -> None:
    assert explorer.count("reboundBorrowsBedMachineLoad: true") == 2
    assert "explorer.meta.reboundQrfLoadNote" in explorer
    assert "explorer.rebound.modelNotePublishedQrf" in explorer


def test_published_mode_is_wired_through_the_rebound_module(explorer: str) -> None:
    assert "summarisePublishedResponse" in explorer
    assert "explorer.meta.reboundMethodTextPublished" in explorer
    assert "explorer.meta.reboundAssumptionsTextPublished" in explorer
    assert "explorer.meta.sourceIsostaticResponse" in explorer
    assert "https://doi.org/10.18739/A22Z12R8C" in explorer
    # The datum slider is disabled while the published response fixes the sea surface.
    assert "controlsUI.reboundSeaLevel.disabled = published" in explorer


def test_rebound_capability_is_declared_for_both_regions(explorer: str) -> None:
    # getDatasetConfig merges region capabilities under each dataset's own, so one flag
    # per region reaches every dataset.
    assert explorer.count("isostaticRebound: true") == 2


def test_rebound_state_is_observable_for_e2e(explorer: str) -> None:
    assert "isostaticRebound: {" in explorer
    for field in ("emergentAreaKm2", "maxUpliftMeters", "seaLevelEquivalentMeters", "progressPercent"):
        assert field in explorer, f"{field} is missing from the exported state"
    assert "showIsostaticRebound: Boolean(controlsUI.showIsostaticRebound?.checked)" in explorer
    # The exact-equality assertion on flowAnimation must stay untouched.
    assert "flowAnimation: {" in explorer
    assert "isostaticRebound" not in explorer.split("flowAnimation: {")[1].split("},")[0]


def test_solver_cites_its_sources(explorer: str) -> None:
    """Every scientific layer carries its provenance into the runtime (paper.md:133)."""
    solver = (REPO_ROOT / "static" / "tools" / "js" / "gia-rebound.js").read_text(encoding="utf-8")
    for citation in ("Le Meur", "Lingle", "Brotchie", "Bueler", "Whitehouse", "Paxman"):
        assert citation in solver, f"{citation} is not cited in the solver"
    assert "explorer.meta.reboundMethod" in explorer
    assert "explorer.meta.reboundAssumptions" in explorer


def test_solver_uses_bedmachine_hydrostatic_constants() -> None:
    solver = (REPO_ROOT / "static" / "tools" / "js" / "gia-rebound.js").read_text(encoding="utf-8")
    assert "export const ICE_DENSITY_KG_M3 = 917;" in solver
    assert "export const SEAWATER_DENSITY_KG_M3 = 1027;" in solver
    assert "export const MANTLE_DENSITY_KG_M3 = 3300;" in solver
    assert "export const DEFAULT_FLEXURAL_RIGIDITY_N_M = 1e25;" in solver
    assert "export const REBOUND_RELAXATION_TIME_YEARS = 3000;" in solver
