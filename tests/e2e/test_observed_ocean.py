"""E2E tests: the OCEAN ICE observations (seafloor water, ocean profiles) in the research edition.

The committed packages are loaded as they are: the seafloor water (3.9 MB) and the profile points
(5.8 MB); the profile curves (16 MB) only when a profile is clicked, which the unit tests cover. Expected counts come from the packages' own metadata, so a
rebuild does not break these tests. Playback is driven with `window.advanceTime`, because a
headless browser may not schedule animation frames on its own. Besides uncaught exceptions, the
tests fail on the layers' own console errors and on shader compile errors, which three.js only
logs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.panel import open_group

pytestmark = pytest.mark.e2e

READY_TIMEOUT_MS = 60_000
LAYER_TIMEOUT_MS = 90_000
DATA_DIR = Path(__file__).resolve().parents[2] / "static/tools/data"
SEAFLOOR = json.loads((DATA_DIR / "ocean_ice_seafloor_antarctica.meta.json").read_text())
PROFILES = json.loads((DATA_DIR / "ocean_ice_profiles_antarctica.meta.json").read_text())
LAYER_CONSOLE_ERRORS = ("Observed ocean", "Ocean profile", "Shader Error", "THREE.WebGLProgram")
LAYERS = {"seafloor": "showOceanSeafloor", "profiles": "showOceanProfiles"}


@pytest.fixture
def page(playwright_browser, explorer_url):
    context = playwright_browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def on_console(message):
        if message.type == "error" and any(marker in message.text for marker in LAYER_CONSOLE_ERRORS):
            errors.append(message.text)

    page.on("console", on_console)
    page.goto(explorer_url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_function(
        "() => window.render_game_to_text && JSON.parse(window.render_game_to_text()).ready",
        timeout=READY_TIMEOUT_MS,
    )
    open_group(page, "ocean")
    try:
        yield page, errors
    finally:
        context.close()


def _state(page) -> dict:
    return json.loads(page.evaluate("window.render_game_to_text()"))["observedOcean"]


def _show(page, key: str) -> dict:
    page.locator(f"#{LAYERS[key]}").check()
    page.wait_for_function(
        f"() => JSON.parse(window.render_game_to_text()).observedOcean.{key}.shown",
        timeout=LAYER_TIMEOUT_MS,
    )
    return _state(page)[key]


def _year_count(year: int) -> int:
    first = PROFILES["years"]["first"]
    starts = PROFILES["years"]["starts"]
    return starts[year - first + 1] - starts[year - first]


def test_the_layers_load_and_draw(page):
    page, errors = page
    assert _state(page)["available"] is True
    seafloor = _show(page, "seafloor")
    assert seafloor["drawnCells"] > 100_000
    assert (seafloor["variable"], seafloor["period"], seafloor["fade"]) == ("temperature", "full", True)
    profiles = _show(page, "profiles")
    assert profiles["count"] == PROFILES["points"]["count"]
    assert profiles["drawn"] == _year_count(profiles["year"])
    assert profiles["chartShown"] is True
    for section in ("oceanSeafloorLegendSection", "oceanProfilesLegendSection"):
        assert page.locator(f"#{section}").is_visible(), section
    assert "°C" in page.locator("#oceanSeafloorLegendTitle").inner_text()
    assert errors == []


def test_the_seafloor_view_follows_its_controls(page):
    page, errors = page
    _show(page, "seafloor")
    page.select_option("#oceanSeafloorVariable", "salinity")
    page.select_option("#oceanSeafloorPeriod", "post2000")
    page.locator("#oceanSeafloorFade").uncheck()
    state = _state(page)["seafloor"]
    assert (state["variable"], state["period"], state["fade"]) == ("salinity", "post2000", False)
    assert "g/kg" in page.locator("#oceanSeafloorLegendTitle").inner_text()
    page.locator("#showOceanSeafloor").uncheck()
    state = _state(page)["seafloor"]
    assert state["enabled"] is False and state["shown"] is False
    assert page.locator("#oceanSeafloorControls").is_hidden()
    assert errors == []


def test_profiles_show_a_year_or_every_year_and_play(page):
    page, errors = page
    profiles = _show(page, "profiles")
    last = PROFILES["years"]["last"]
    # The record's last year is still filling up, so the layer opens on the year before it.
    assert profiles["year"] in (last, last - 1)
    page.evaluate("(year) => { const s = document.getElementById('oceanProfilesYear'); s.value = String(year); s.dispatchEvent(new Event('input')); }", 1994)
    state = _state(page)["profiles"]
    assert state["year"] == 1994 and state["drawn"] == _year_count(1994)
    assert "1994" in page.locator("#oceanProfilesReadout").inner_text()
    page.locator("#oceanProfilesAllYears").check()
    state = _state(page)["profiles"]
    assert state["allYears"] is True and state["drawn"] == PROFILES["points"]["count"]
    assert page.locator("#oceanProfilesYear").is_disabled()
    page.locator("#oceanProfilesAllYears").uncheck()
    page.locator("#oceanProfilesPlay").click()
    page.evaluate("() => { for (let i = 0; i < 20; i += 1) window.advanceTime(100); }")
    state = _state(page)["profiles"]
    assert state["playing"] is True and state["year"] > 1994
    page.locator("#oceanProfilesPlay").click()
    assert _state(page)["profiles"]["playing"] is False
    assert errors == []


def test_the_rebound_switches_the_layers_off(page):
    page, errors = page
    _show(page, "seafloor")
    _show(page, "profiles")
    open_group(page, "time")
    page.locator("#showIsostaticRebound").check()
    page.wait_for_function(
        "() => { const s = JSON.parse(window.render_game_to_text()).observedOcean;"
        " return !s.seafloor.enabled && !s.profiles.enabled && !s.seafloor.shown && !s.profiles.shown; }",
        timeout=LAYER_TIMEOUT_MS,
    )
    assert page.locator("#oceanSeafloorControls").is_hidden()
    assert errors == []


def test_the_seafloor_water_view_opens_with_the_layer_on(playwright_browser, explorer_url):
    # The home page's What's new card links to this view.
    context = playwright_browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = context.new_page()
        page.goto(f"{explorer_url}?region=antarctica&preset=antarctica-seafloor-water", wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => window.render_game_to_text && JSON.parse(window.render_game_to_text()).observedOcean.seafloor.shown",
            timeout=LAYER_TIMEOUT_MS,
        )
        assert page.locator("#showOceanSeafloor").is_checked()
        assert _state(page)["profiles"]["enabled"] is False
    finally:
        context.close()
