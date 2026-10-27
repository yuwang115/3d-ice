"""E2E tests: the observed elevation change layer (CPOM, 1995-2026) in the research edition.

The committed packages are loaded as they are: the record is 20 MB, served from disk. The
expected readout is computed from the record's own metadata, so a rebuild with a newer CPOM
epoch does not break these tests. Playback is driven with `window.advanceTime`, because a
headless browser may not schedule animation frames on its own. Controls are located by `#id`,
so the Chinese page needs no other locators.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

READY_TIMEOUT_MS = 60_000
LAYER_TIMEOUT_MS = 90_000
DATA_DIR = Path(__file__).resolve().parents[2] / "static/tools/data"
RECORD_META = json.loads((DATA_DIR / "cpom_antarctica_dh_480.meta.json").read_text())
RATE_META = json.loads((DATA_DIR / "cpom_antarctica_dhdt_480.meta.json").read_text())


def _signed(value: float, step: float) -> str:
    """The runtime's readout format: rounded to `step`, thousands separators, a real minus sign."""
    rounded = round(value / step) * step
    text = f"{abs(rounded):,.1f}" if step < 1 else f"{abs(rounded):,.0f}"
    if rounded == 0:
        return text
    return ("+" if rounded > 0 else "−") + text


def _expected_ais_readout() -> tuple[str, str]:
    """CPOM's ice-sheet mass change and its sea-level equivalent at the last epoch."""
    ais = RECORD_META["series"]["regions"]["AIS"]
    mass, sigma = ais["mass_change_gt"][-1], ais["uncertainty_gt"][-1]
    step = 10 if sigma >= 10 else 1
    mass_text = f"{_signed(mass, step)} ± {max(1, round(sigma / step)) * step:,} Gt"
    sea_level = -mass / RECORD_META["series"]["gt_per_mm_sle"]
    return mass_text, f"{_signed(sea_level, 0.1)} mm"


def _open(playwright_browser, url, routes=()):
    context = playwright_browser.new_context(viewport={"width": 1280, "height": 800})
    for pattern, handler in routes:
        context.route(pattern, handler)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
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


def _wait_for_view(page, view: str) -> dict:
    page.wait_for_function(
        f"""() => {{
            const s = JSON.parse(window.render_game_to_text()).surfaceChange;
            return s.shown && s.view === '{view}';
        }}""",
        timeout=LAYER_TIMEOUT_MS,
    )
    return _state(page)["surfaceChange"]


def _show_layer(page, view: str) -> dict:
    page.locator("#showSurfaceChange").check()
    if page.locator("#surfaceChangeView").input_value() != view:
        page.locator("#surfaceChangeView").select_option(view)
    return _wait_for_view(page, view)


def _readout(page) -> str:
    # The readout keeps each value on one line with no-break spaces.
    return page.locator("#surfaceChangeReadout").inner_text().replace("\xa0", " ")


def test_the_record_shows_cpoms_totals_at_the_last_epoch(page):
    page, errors = page
    state = _show_layer(page, "cumulative")
    epochs = RECORD_META["frames"]["count"]
    assert state["epochCount"] == epochs
    assert state["position"] == epochs - 1
    assert state["year"] == pytest.approx(RECORD_META["frames"]["years"][-1], abs=1e-3)
    assert state["extrapolatedRadiusKm"] == 217  # CryoSat-2's polar cap, 88 S
    mass_text, sea_level_text = _expected_ais_readout()
    readout = _readout(page)
    assert "Antarctic Ice Sheet since 1995" in readout
    assert mass_text in readout
    assert sea_level_text in readout
    assert not page.locator("#surfaceChangeLegendSection").is_hidden()
    labels = page.locator("#surfaceChangeLegendLabels").inner_text()
    assert "−150 m" in labels and "+150 m" in labels
    assert _state(page)["meshes"]["surfaceChange"]
    assert errors == []


def test_the_slider_and_playback_move_through_the_epochs(page):
    page, errors = page
    _show_layer(page, "cumulative")
    slider = page.locator("#surfaceChangeTime")
    slider.evaluate("(el) => { el.value = '10'; el.dispatchEvent(new Event('input', { bubbles: true })); }")
    state = _state(page)["surfaceChange"]
    assert state["position"] == 10
    assert state["year"] == pytest.approx(RECORD_META["frames"]["years"][10], abs=1e-3)
    # Before CryoSat-2, everything south of 81.5 S is extrapolated.
    assert state["year"] < RECORD_META["pole_hole"]["observed_from_year"]
    assert state["extrapolatedRadiusKm"] == 925
    assert str(int(state["year"])) in page.locator("#surfaceChangeTimeValue").inner_text()

    page.locator("#surfaceChangePlay").click()
    assert _state(page)["surfaceChange"]["playing"]
    # Real animation frames may also advance it, so compare across the simulated second only.
    before, after = page.evaluate(
        """() => {
            const read = () => JSON.parse(window.render_game_to_text()).surfaceChange.position;
            const before = read();
            window.advanceTime(1000);
            return [before, read()];
        }"""
    )
    assert after - before >= 5.9  # six epochs a second
    page.evaluate("window.advanceTime(30000)")
    state = _state(page)["surfaceChange"]
    assert state["position"] == RECORD_META["frames"]["count"] - 1 and not state["playing"]
    assert page.locator("#surfaceChangePlay").inner_text() == "Play"
    assert errors == []


def test_the_fade_can_be_switched_off(page):
    page, _errors = page
    assert _show_layer(page, "cumulative")["fade"]
    page.locator("#surfaceChangeFade").uncheck()
    assert not _state(page)["surfaceChange"]["fade"]


def test_the_rate_view_fades_the_polar_cap_and_runs_on_the_hd_dataset(page):
    page, errors = page
    state = _show_layer(page, "rate")
    assert page.locator("#surfaceChangeTimeControl").is_hidden()
    assert "Period" in _readout(page)
    assert state["epochCount"] is None
    assert state["extrapolatedRadiusKm"] == round(RATE_META["pole_hole"]["radius_m"]["cryosat2"] / 1000)

    page.locator("#resolutionPreset").select_option("hd")
    page.wait_for_function(
        """() => {
            const s = JSON.parse(window.render_game_to_text());
            return s.ready && s.dataset === 'hd' && s.surfaceChange.shown;
        }""",
        timeout=LAYER_TIMEOUT_MS,
    )
    # The record is packed on the 10 km grid only.
    assert page.locator("#surfaceChangeView option[value='cumulative']").is_disabled()
    state = _state(page)["surfaceChange"]
    assert state["view"] == "rate" and state["extrapolatedRadiusKm"] == 217
    assert errors == []


def test_a_view_that_fails_to_load_leaves_the_working_one_on(playwright_browser, explorer_url):
    routes = [("**/data/cpom_antarctica_dhdt_480.meta.json*", lambda route: route.fulfill(status=404, body="gone"))]
    context, page, errors = _open(playwright_browser, explorer_url, routes)
    try:
        _show_layer(page, "cumulative")
        page.locator("#surfaceChangeView").select_option("rate")
        page.wait_for_function(
            "() => document.getElementById('surfaceChangeView').value === 'cumulative'",
            timeout=LAYER_TIMEOUT_MS,
        )
        state = _state(page)["surfaceChange"]
        assert state["enabled"] and state["shown"] and state["view"] == "cumulative"
        assert errors == []
    finally:
        context.close()


def test_a_stalled_download_does_not_hold_up_another_view(playwright_browser, explorer_url):
    # The record's payload never arrives; the rate view must load regardless.
    routes = [("**/data/cpom_antarctica_dh_480.bin*", lambda route: None)]
    context, page, errors = _open(playwright_browser, explorer_url, routes)
    try:
        page.locator("#showSurfaceChange").check()
        page.wait_for_function(
            "() => document.getElementById('status').textContent.length > 0", timeout=10_000
        )
        page.locator("#surfaceChangeView").select_option("rate")
        state = _wait_for_view(page, "rate")
        assert state["enabled"]
        assert page.locator("#loadingOverlay").get_attribute("aria-busy") == "false"
        assert errors == []
    finally:
        context.close()


def test_the_layer_gives_way_to_velocity_and_the_rebound_and_takes_over_from_them(page):
    page, _errors = page
    _show_layer(page, "cumulative")
    page.locator("#showVelocity").check()
    state = _state(page)
    assert not page.locator("#showSurfaceChange").is_checked()
    assert not state["surfaceChange"]["shown"]

    _show_layer(page, "cumulative")
    assert not page.locator("#showVelocity").is_checked()
    assert not _state(page)["meshes"]["velocity"]

    page.locator("#showIsostaticRebound").check()
    assert not page.locator("#showSurfaceChange").is_checked()
    assert not _state(page)["surfaceChange"]["shown"]
    _show_layer(page, "cumulative")
    assert not page.locator("#showIsostaticRebound").is_checked()


def test_other_terrains_and_greenland_do_not_offer_the_layer(page):
    page, _errors = page
    page.locator("#resolutionPreset").select_option("bedmap3")
    page.wait_for_function(
        "() => { const s = JSON.parse(window.render_game_to_text()); return s.ready && s.dataset === 'bedmap3'; }",
        timeout=READY_TIMEOUT_MS,
    )
    assert page.locator("#surfaceChangeRow").is_hidden()
    assert not _state(page)["surfaceChange"]["available"]
    page.locator("#regionPreset").select_option("greenland")
    page.wait_for_function(
        "() => { const s = JSON.parse(window.render_game_to_text()); return s.ready && s.region === 'greenland'; }",
        timeout=READY_TIMEOUT_MS,
    )
    assert page.locator("#surfaceChangeRow").is_hidden()


def test_the_chinese_page_reads_the_record_in_chinese(playwright_browser, server):
    context, page, errors = _open(playwright_browser, f"{server}/zh/tools/3D-interactive-cryosphere-explorer.html")
    try:
        _show_layer(page, "cumulative")
        readout = _readout(page)
        assert "1995 年以来南极冰盖" in readout
        assert "南极对海平面的贡献" in readout
        assert "年" in page.locator("#surfaceChangeTimeValue").inner_text()
        assert errors == []
    finally:
        context.close()
