"""E2E tests: the observed sea-ice layer (NOAA/NSIDC CDR, 1978-2026) in the research edition.

The committed packages are loaded as they are (24 MB for Antarctica, 2 MB for Greenland). The
expected readout is computed from the packages' own metadata, so a rebuild with newer months
does not break these tests. Playback is driven with `window.advanceTime`, because a headless
browser may not schedule animation frames on its own. Besides uncaught exceptions, the tests
fail on the layer's own console errors and on shader compile errors, which three.js only logs.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageStat

pytestmark = pytest.mark.e2e

READY_TIMEOUT_MS = 60_000
LAYER_TIMEOUT_MS = 90_000
DATA_DIR = Path(__file__).resolve().parents[2] / "static/tools/data"
ANTARCTICA = json.loads((DATA_DIR / "nsidc_sea_ice_antarctica.meta.json").read_text())
GREENLAND = json.loads((DATA_DIR / "nsidc_sea_ice_greenland.meta.json").read_text())

# Console errors that mean this layer failed: its own, and a shader that does not compile.
LAYER_CONSOLE_ERRORS = ("Sea-ice", "Shader Error", "THREE.WebGLProgram")


def _open(playwright_browser, url, routes=()):
    context = playwright_browser.new_context(viewport={"width": 1280, "height": 800})
    for pattern, handler in routes:
        context.route(pattern, handler)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def on_console(message):
        if message.type == "error" and any(marker in message.text for marker in LAYER_CONSOLE_ERRORS):
            errors.append(message.text)

    page.on("console", on_console)
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_function(
        "() => window.render_game_to_text && JSON.parse(window.render_game_to_text()).ready",
        timeout=READY_TIMEOUT_MS,
    )
    return context, page, errors


@pytest.fixture
def page(playwright_browser, explorer_url):
    context, page, errors = _open(playwright_browser, explorer_url)
    try:
        yield page, errors
    finally:
        context.close()


def _state(page) -> dict:
    return json.loads(page.evaluate("window.render_game_to_text()"))


def _wait_for_layer(page, region: str = "antarctica") -> dict:
    page.wait_for_function(
        f"""() => {{
            const s = JSON.parse(window.render_game_to_text()).seaIce;
            return s.shown && s.region === '{region}';
        }}""",
        timeout=LAYER_TIMEOUT_MS,
    )
    return _state(page)["seaIce"]


def _show_layer(page, region: str = "antarctica") -> dict:
    page.locator("#showSeaIce").check()
    return _wait_for_layer(page, region)


def _pick(page, year: int, month: int) -> dict:
    """Set the year and then the month, as a visitor dragging the two sliders would."""
    page.evaluate(
        """([year, month]) => {
            const set = (id, value) => {
                const el = document.getElementById(id);
                el.value = String(value);
                el.dispatchEvent(new Event('input', { bubbles: true }));
            };
            set('seaIceYear', year);
            set('seaIceMonth', month);
        }""",
        [year, month],
    )
    return _state(page)["seaIce"]


def _readout(page) -> str:
    return page.locator("#seaIceReadout").inner_text().replace("\xa0", " ")


def _extent(meta: dict, label: str) -> float:
    return meta["series"]["extent_mkm2"][meta["frames"]["months"].index(label)]


def _canvas_brightness(page) -> float:
    """Mean grey level of the rendered scene, after drawing a fresh frame."""
    page.evaluate("window.advanceTime(32)")
    page.wait_for_timeout(300)
    png = page.locator("#viewer canvas").screenshot()
    return ImageStat.Stat(Image.open(io.BytesIO(png)).convert("L")).mean[0]


def test_the_layer_opens_on_the_latest_month_with_the_hemisphere_extent(page):
    page, errors = page
    assert not page.locator("#showSea").is_checked()
    state = _show_layer(page)
    latest = ANTARCTICA["frames"]["months"][-1]
    year, month = (int(part) for part in latest.split("-"))
    assert state["frameCount"] == ANTARCTICA["frames"]["count"]
    assert (state["year"], state["month"]) == (year, month)
    assert state["position"] == ANTARCTICA["frames"]["count"] - 1
    assert state["iceDrawn"] and state["medianEdge"]
    assert state["extentMkm2"] == pytest.approx(_extent(ANTARCTICA, latest), abs=1e-4)
    readout = _readout(page)
    assert "Antarctic extent" in readout
    assert f"{_extent(ANTARCTICA, latest):.2f} million km²" in readout
    assert "vs 1981–2010" in readout
    assert page.locator("#seaIceYearValue").inner_text() == str(year)
    # The open water shows: the layer raised the sea plane.
    assert page.locator("#showSea").is_checked() and state["raisedSea"]
    assert state["chartShown"]
    assert "Southern Ocean" in page.locator("#seaIceChartNote").inner_text()
    assert not page.locator("#seaIceLegendSection").is_hidden()
    assert page.locator("#seaIceLegendLabels span").all_inner_texts() == ["15%", "40", "70", "100"]
    page.wait_for_function(
        "() => [...document.querySelectorAll('#seaIceLegendLabels span')].every((s) => s.style.top)", timeout=10_000
    )
    rows = page.locator("#seaIceLegendLabels span").evaluate_all("(spans) => spans.map((s) => s.style.top)")
    assert set(rows) == {"0px"}, "the legend labels fit on one row"
    assert _state(page)["meshes"]["seaIce"]
    # Switching the layer off lowers the sea plane it raised.
    page.locator("#showSeaIce").uncheck()
    assert not page.locator("#showSea").is_checked()
    assert not _state(page)["seaIce"]["shown"]
    assert errors == []


def test_the_sliders_pick_a_month_and_the_chart_follows_the_calendar_month(page):
    page, errors = page
    _show_layer(page)
    state = _pick(page, 2023, 2)
    assert (state["year"], state["month"]) == (2023, 2)
    assert state["extentMkm2"] == pytest.approx(_extent(ANTARCTICA, "2023-02"), abs=1e-4)
    assert state["rank"] == 1
    assert "Lowest February of" in _readout(page)
    assert page.locator("#seaIceChart .label--series").text_content() == "February"
    assert page.locator("#seaIceChart .cursor-dot").get_attribute("visibility") == "visible"
    trend = ANTARCTICA["series"]["monthly"][1]["percent_per_decade"]
    assert f"{abs(trend):.1f}% per decade" in page.locator("#seaIceChart").text_content()
    # The Data Snapshot follows once the slider is let go.
    page.locator("#seaIceMonth").dispatch_event("change")
    snapshot = page.locator("#metaList").inner_text()
    assert "February 2023" in snapshot
    assert "whole Southern Ocean" in snapshot
    assert errors == []


def test_months_without_data_draw_no_ice_and_say_why(page):
    page, errors = page
    _show_layer(page)
    gap = _pick(page, 1987, 12)
    assert gap["position"] is not None and not gap["iceDrawn"]
    assert "gap in the satellite record" in _readout(page)
    assert gap["medianEdge"], "the month's median edge still shows"
    before = _pick(page, 1978, 9)
    assert before["position"] is None and not before["iceDrawn"]
    assert "the record runs from November 1978" in _readout(page)
    assert page.locator("#seaIceChart .cursor-dot").get_attribute("visibility") == "hidden"
    first = _pick(page, 1978, 11)
    assert first["position"] == 0 and first["iceDrawn"]
    assert errors == []


def test_playback_runs_month_by_month_across_the_gap_and_stops_on_a_month(page):
    page, errors = page
    _show_layer(page)
    _pick(page, 1987, 10)
    page.locator("#seaIcePlay").click()
    assert _state(page)["seaIce"]["playing"]
    months = page.evaluate(
        """() => {
            const seen = [];
            for (let i = 0; i < 12; i += 1) {
                window.advanceTime(100);
                const s = JSON.parse(window.render_game_to_text()).seaIce;
                const label = `${s.year}-${String(s.month).padStart(2, '0')}`;
                if (seen[seen.length - 1] !== label) seen.push(label);
            }
            return seen;
        }"""
    )
    # A tenth of a second is six tenths of a month: the labels name the nearer month.
    assert months == sorted(months)
    assert "1987-11" in months and "1988-02" in months
    assert "1987-12" not in months and "1988-01" not in months, "playback names no empty month"
    page.locator("#seaIcePlay").click()
    state = _state(page)["seaIce"]
    assert not state["playing"]
    assert state["position"] == int(state["position"]), "playback stops on a whole month"
    assert errors == []


def test_a_slider_moved_during_playback_takes_the_visitors_month_at_once(page):
    page, errors = page
    _show_layer(page)
    _pick(page, 2003, 3)
    page.locator("#seaIcePlay").click()
    page.evaluate("window.advanceTime(500)")
    # One input event, as a click on the track or an arrow key gives.
    page.evaluate(
        """() => {
            const el = document.getElementById('seaIceYear');
            el.value = '1995';
            el.dispatchEvent(new Event('input', { bubbles: true }));
        }"""
    )
    state = _state(page)["seaIce"]
    assert not state["playing"]
    assert state["year"] == 1995
    assert page.locator("#seaIceYear").input_value() == "1995"
    page.evaluate(
        """() => {
            const el = document.getElementById('seaIceMonth');
            el.value = '8';
            el.dispatchEvent(new Event('input', { bubbles: true }));
        }"""
    )
    assert (_state(page)["seaIce"]["year"], _state(page)["seaIce"]["month"]) == (1995, 8)
    page.locator("#seaIcePlay").click()
    page.evaluate("window.advanceTime(300)")
    page.evaluate(
        """() => {
            const el = document.getElementById('seaIceMonth');
            el.value = '2';
            el.dispatchEvent(new Event('input', { bubbles: true }));
        }"""
    )
    state = _state(page)["seaIce"]
    assert not state["playing"] and state["month"] == 2
    assert errors == []


def test_the_sliders_wait_for_the_record(playwright_browser, explorer_url):
    held = []
    routes = [("**/data/nsidc_sea_ice_antarctica.bin*", lambda route: held.append(route))]
    context, page, errors = _open(playwright_browser, explorer_url, routes)
    try:
        page.locator("#showSeaIce").check()
        for _ in range(300):
            if held:
                break
            page.wait_for_timeout(100)
        assert held, "the package was requested"
        assert page.locator("#seaIceYear").is_disabled() and page.locator("#seaIceMonth").is_disabled()
        held[0].continue_()
        state = _wait_for_layer(page)
        assert page.locator("#seaIceYear").is_enabled() and page.locator("#seaIceMonth").is_enabled()
        latest = ANTARCTICA["frames"]["months"][-1]
        assert f"{state['year']}-{state['month']:02d}" == latest
        assert errors == []
    finally:
        context.close()


def test_switching_the_layer_off_abandons_its_download(playwright_browser, explorer_url):
    held = []
    routes = [("**/data/nsidc_sea_ice_antarctica.bin*", lambda route: held.append(route))]
    context, page, errors = _open(playwright_browser, explorer_url, routes)
    try:
        page.locator("#showSeaIce").check()
        for _ in range(300):
            if held:
                break
            page.wait_for_timeout(100)
        assert held
        page.locator("#showSeaIce").uncheck()
        page.wait_for_function(
            "() => document.getElementById('loadingOverlay').classList.contains('hidden')", timeout=10_000
        )
        assert not _state(page)["seaIce"]["shown"]
        assert errors == []
    finally:
        context.close()


def test_more_ice_is_drawn_in_the_antarctic_winter_than_in_summer(page):
    page, errors = page
    _show_layer(page)
    _pick(page, 2014, 9)
    winter = _canvas_brightness(page)
    _pick(page, 2023, 2)
    summer = _canvas_brightness(page)
    page.locator("#showSeaIce").uncheck()
    without = _canvas_brightness(page)
    assert winter > summer > without - 0.5
    assert winter - without > 2, "the September 2014 maximum brightens the scene"
    assert errors == []


def test_the_median_edge_can_be_switched_off(page):
    page, errors = page
    _show_layer(page)
    page.locator("#seaIceMedianEdge").uncheck()
    assert not _state(page)["seaIce"]["medianEdge"]
    _pick(page, 2000, 3)
    assert not _state(page)["seaIce"]["medianEdge"], "a new month keeps the choice"
    page.locator("#seaIceMedianEdge").check()
    assert _state(page)["seaIce"]["medianEdge"]
    assert errors == []


def test_greenland_reads_the_whole_arctic_and_keeps_the_month(page):
    page, errors = page
    _show_layer(page)
    _pick(page, 2012, 9)
    page.locator("#regionPreset").select_option("greenland")
    state = _wait_for_layer(page, "greenland")
    assert (state["year"], state["month"]) == (2012, 9)
    assert state["frameCount"] == GREENLAND["frames"]["count"]
    assert state["extentMkm2"] == pytest.approx(_extent(GREENLAND, "2012-09"), abs=1e-4)
    assert state["rank"] == 1
    readout = _readout(page)
    assert "Arctic extent" in readout and "Lowest September of" in readout
    assert "Greenland view holds only part of it" in page.locator("#seaIceChartNote").inner_text()
    # July 1984 is empty in the Arctic file.
    assert not _pick(page, 1984, 7)["iceDrawn"]
    assert errors == []


def test_another_terrain_of_the_region_draws_the_same_record(page):
    page, errors = page
    _show_layer(page)
    _pick(page, 2010, 9)
    page.locator("#resolutionPreset").select_option("bedmap3")
    page.wait_for_function(
        "() => { const s = JSON.parse(window.render_game_to_text()); return s.ready && s.dataset === 'bedmap3' && s.seaIce.shown; }",
        timeout=LAYER_TIMEOUT_MS,
    )
    state = _state(page)["seaIce"]
    assert (state["year"], state["month"]) == (2010, 9)
    assert state["domainCells"] == ANTARCTICA["domain"]["cell_count"]
    assert errors == []


def test_the_layer_gives_way_to_the_rebound_and_takes_over_from_it(page):
    page, _errors = page
    _show_layer(page)
    page.locator("#showIsostaticRebound").check()
    assert not page.locator("#showSeaIce").is_checked()
    assert not _state(page)["seaIce"]["shown"]
    # The rebound shows its own waterline, so the sea plane stays.
    assert page.locator("#showSea").is_checked()
    _show_layer(page)
    assert not page.locator("#showIsostaticRebound").is_checked()


def test_the_layer_gives_way_to_the_projection_and_takes_over_from_it(page):
    page, _errors = page
    page.wait_for_function("() => !document.getElementById('showIceProjection').disabled", timeout=LAYER_TIMEOUT_MS)
    _show_layer(page)
    page.locator("#showIceProjection").check()
    # The projection takes the scene once its package is in.
    page.wait_for_function(
        "() => { const s = JSON.parse(window.render_game_to_text()); return s.iceProjection.active; }",
        timeout=LAYER_TIMEOUT_MS,
    )
    assert not page.locator("#showSeaIce").is_checked()
    assert not _state(page)["seaIce"]["shown"]
    _show_layer(page)
    assert not page.locator("#showIceProjection").is_checked()
    assert not _state(page)["iceProjection"]["active"]


def test_a_package_that_fails_to_load_switches_the_layer_off(playwright_browser, explorer_url):
    routes = [("**/data/nsidc_sea_ice_antarctica.bin*", lambda route: route.fulfill(status=404, body="gone"))]
    context, page, errors = _open(playwright_browser, explorer_url, routes)
    try:
        page.locator("#showSeaIce").check()
        page.wait_for_function("() => !document.getElementById('showSeaIce').checked", timeout=LAYER_TIMEOUT_MS)
        state = _state(page)["seaIce"]
        assert not state["shown"]
        assert not page.locator("#showSea").is_checked(), "the sea plane it raised goes again"
        assert len(errors) == 1 and errors[0].startswith("Sea-ice layer load failed")
    finally:
        context.close()


def test_the_chinese_page_reads_the_record_in_chinese(playwright_browser, server):
    context, page, errors = _open(playwright_browser, f"{server}/zh/tools/3D-interactive-cryosphere-explorer.html")
    try:
        _show_layer(page)
        _pick(page, 2023, 2)
        readout = _readout(page)
        assert "南极海冰范围" in readout
        assert "百万 km²" in readout
        assert "个二月中最低" in readout
        assert page.locator("#seaIceChart .label--series").text_content() == "二月"
        assert "整个南大洋" in page.locator("#seaIceChartNote").inner_text()
        assert errors == []
    finally:
        context.close()
