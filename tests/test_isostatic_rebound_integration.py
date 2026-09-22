"""Integration checks for the isostatic-rebound layer's wiring in both locale explorers.

The repository has no generic en/zh parity test: parity is enforced per feature by
asserting the same control ids and module references exist in both HTML files. These
tests do that for the rebound layer, and additionally pin the control ordering so the
layer cannot drift out of the view-controls section in one locale only.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
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
    return request.param.read_text(encoding="utf-8")


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
    """The JOSS audit asks for testable domain logic in modules, not the HTML entry point."""
    solver = (REPO_ROOT / "static" / "tools" / "js" / "gia-rebound.js").read_text(encoding="utf-8")
    assert "export function solveIsostaticRebound" in solver
    for explorer_path in EXPLORERS:
        html = explorer_path.read_text(encoding="utf-8")
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


def test_rebound_controls_sit_inside_the_view_controls_section(explorer_ids: _IdCollector) -> None:
    for control_id in CONTROL_IDS:
        if control_id == "reboundLegendNote":
            assert "bedLegendSection" in explorer_ids.ancestors[control_id]
            continue
        assert "viewControlsSection" in explorer_ids.ancestors[control_id], control_id
    for control_id in CONTROL_IDS[2:-1]:
        assert "isostaticReboundControls" in explorer_ids.ancestors[control_id], control_id


def test_rebound_toggle_follows_the_existing_layer_toggles(explorer_ids: _IdCollector) -> None:
    """Placed after showBed and showSea so the polar-feature ordering contract holds."""
    order = explorer_ids.order
    assert order.index("showBed") < order.index("showSea") < order.index("showIsostaticRebound")
    assert order.index("showIsostaticRebound") < order.index("isostaticReboundControls")
    assert order.index("isostaticReboundControls") < order.index("wireframe")


def test_rebound_controls_start_in_the_headline_scenario(explorer: str) -> None:
    """Enabling the layer should land on 'ice gone, rebound complete' with no extra clicks."""
    assert 'id="showIsostaticRebound" type="checkbox" />' in explorer, "must default to off"
    assert 'id="reboundProgress" type="range" min="0" max="100" step="1" value="100"' in explorer
    assert 'id="reboundSeaLevel" type="range" min="0" max="70" step="1" value="0"' in explorer
    assert 'id="highlightEmergentLand" type="checkbox" checked />' in explorer
    assert 'value="flexural" selected' in explorer


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
    for citation in ("Le Meur", "Lingle", "Brotchie", "Bueler", "Whitehouse"):
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
