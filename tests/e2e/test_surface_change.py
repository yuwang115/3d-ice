"""E2E tests: the observed elevation change layer (CPOM, 1995-2026) in the research edition.

The committed packages are loaded as they are: the record is 20 MB, served from disk. The
expected readout is computed from the record's own metadata, so a rebuild with a newer CPOM
epoch does not break these tests. Playback is driven with `window.advanceTime`, because a
headless browser may not schedule animation frames on its own. Controls are located by `#id`,
so the Chinese page needs no other locators. Besides uncaught exceptions, the tests fail on
the layer's own console errors and on shader compile errors, which three.js only logs.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

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


def _mass_text(mass: float, sigma: float) -> str:
    """The readout's "−1,890 ± 320 Gt": tens of gigatonnes once the uncertainty reaches them."""
    step = 10 if sigma >= 10 else 1
    return f"{_signed(mass, step)} ± {max(1, round(sigma / step)) * step:,} Gt"


def _expected_ais_readout() -> tuple[str, str]:
    """CPOM's ice-sheet mass change and its sea-level equivalent at the last epoch."""
    ais = RECORD_META["series"]["regions"]["AIS"]
    mass, sigma = ais["mass_change_gt"][-1], ais["uncertainty_gt"][-1]
    sea_level = -mass / RECORD_META["series"]["gt_per_mm_sle"]
    return _mass_text(mass, sigma), f"{_signed(sea_level, 0.1)} mm"


AMUNDSEN = "basin:9"  # CPOM's "West G-H": Pine Island, Thwaites and Smith


def _basin_mass_text(basin_id: str, epoch: int = -1) -> str:
    basin = RECORD_META["series"]["basins"][basin_id]
    return _mass_text(basin["mass_change_gt"][epoch], basin["uncertainty_gt"][epoch])


# Console errors that mean this layer failed: its own, and a shader that does not compile.
LAYER_CONSOLE_ERRORS = ("Surface-change", "Shader Error", "THREE.WebGLProgram")


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


def _show_basins(page, view: str) -> dict:
    """The layer in `view`, once its basin list is in."""
    _show_layer(page, view)
    page.wait_for_function(
        "() => JSON.parse(window.render_game_to_text()).surfaceChange.basinCount === 18",
        timeout=LAYER_TIMEOUT_MS,
    )
    return _state(page)["surfaceChange"]


def _wait_for_selection(page, key: str | None = None) -> dict:
    """The selection `key` (any basin if None), outlined on the drape unless it is the whole sheet."""
    match = f"s.selection === '{key}'" if key else "s.selection.startsWith('basin:')"
    page.wait_for_function(
        f"""() => {{
            const s = JSON.parse(window.render_game_to_text()).surfaceChange;
            return {match} && (s.selection === 'AIS' ? s.selectedCells === 0 : s.selectedCells > 0);
        }}""",
        timeout=LAYER_TIMEOUT_MS,
    )
    return _state(page)["surfaceChange"]


def _chart_title(page) -> str:
    # An SVG <text> has no inner text for Playwright; its text content is the label.
    return page.locator("#surfaceChangeChart .label--series").text_content()


def _canvas_centre(page) -> tuple[float, float]:
    box = page.locator("#viewer canvas").bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def _pixels_at(page, x: float, y: float, size: int = 5) -> list[tuple[int, int, int]]:
    """The rendered colours of a small square around a point, after drawing a fresh frame."""
    page.evaluate("window.advanceTime(32)")
    page.wait_for_timeout(300)
    png = page.screenshot(clip={"x": x - size // 2, "y": y - size // 2, "width": size, "height": size})
    return list(Image.open(io.BytesIO(png)).convert("RGB").getdata())


def _mean_difference(a, b) -> float:
    return sum(abs(p - q) for pa, pb in zip(a, b) for p, q in zip(pa, pb)) / (3 * len(a))


def _hold_until_requested(page, held: list, timeout_ms: int = 30_000) -> None:
    """Wait until a held route has caught its request."""
    for _ in range(timeout_ms // 100):
        if held:
            return
        page.wait_for_timeout(100)
    raise AssertionError("the request was never made")


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
    assert _legend_labels(page) == ["−150", "−20", "−5", "0 m", "+5", "+20", "+150"]
    assert _state(page)["meshes"]["surfaceChange"]
    assert errors == []


def _legend_labels(page) -> list[str]:
    return page.locator("#surfaceChangeLegendLabels span").all_inner_texts()


def _legend_rows(page) -> set[str]:
    """The `top` of every legend label, once the layout has placed them: one value is one row."""
    page.wait_for_function(
        "() => [...document.querySelectorAll('#surfaceChangeLegendLabels span')].every((s) => s.style.top)",
        timeout=10_000,
    )
    return set(page.locator("#surfaceChangeLegendLabels span").evaluate_all("(spans) => spans.map((s) => s.style.top)"))


@pytest.mark.parametrize("viewport", [(1280, 800), (390, 844)])
def test_the_legend_labels_of_both_views_fit_on_one_row(playwright_browser, explorer_url, viewport):
    context = playwright_browser.new_context(viewport={"width": viewport[0], "height": viewport[1]})
    try:
        page = context.new_page()
        page.goto(explorer_url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => window.render_game_to_text && JSON.parse(window.render_game_to_text()).ready",
            timeout=READY_TIMEOUT_MS,
        )
        # The phone layout keeps the controls in a drawer; check the box through the page.
        page.evaluate(
            "() => { const el = document.getElementById('showSurfaceChange'); el.checked = true;"
            " el.dispatchEvent(new Event('change', { bubbles: true })); }"
        )
        _wait_for_view(page, "cumulative")
        assert _legend_rows(page) == {"0px"}
        page.evaluate(
            "() => { const el = document.getElementById('surfaceChangeView'); el.value = 'rate';"
            " el.dispatchEvent(new Event('change', { bubbles: true })); }"
        )
        _wait_for_view(page, "rate")
        page.wait_for_function(
            "() => document.getElementById('surfaceChangeLegendLabels').textContent.includes('m/yr')", timeout=10_000
        )
        assert _legend_labels(page) == ["−5", "−1", "−0.2", "0 m/yr", "+0.2", "+1", "+5"]
        assert _legend_rows(page) == {"0px"}
    finally:
        context.close()


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


def test_a_basin_from_the_list_is_outlined_charted_and_read_out(page):
    page, errors = page
    _show_basins(page, "cumulative")
    assert not page.locator("#surfaceChangeBasinControl").is_hidden()
    # The whole sheet, then each region and its basins: 1 + 3 + 18 entries.
    assert page.locator("#surfaceChangeBasin option").count() == 22
    assert page.locator("#surfaceChangeBasin optgroup").evaluate_all("(groups) => groups.map((g) => g.label)") == [
        "West Antarctica",
        "East Antarctica",
        "Antarctic Peninsula",
    ]
    assert _chart_title(page) == "Antarctic Ice Sheet"
    assert _state(page)["surfaceChange"]["chartShown"]

    page.locator("#surfaceChangeBasin").select_option(AMUNDSEN)
    state = _wait_for_selection(page, AMUNDSEN)
    assert state["selectedCells"] > 1000
    assert _chart_title(page) == "Basin G-H, West Antarctica"
    readout = _readout(page)
    assert "Basin G-H, West Antarctica" in readout and _basin_mass_text("9") in readout
    snapshot = page.locator("#metaList").inner_text().replace("\xa0", " ")
    assert f"Selected: Basin G-H, West Antarctica: {_basin_mass_text('9')}" in snapshot
    assert "Mass change since 1995 (CPOM): Antarctic Ice Sheet" in snapshot

    # The chart's cursor and the readout follow the epoch.
    cursor = page.locator("#surfaceChangeChart .cursor")
    at_end = float(cursor.get_attribute("x1"))
    page.locator("#surfaceChangeTime").evaluate(
        "(el) => { el.value = '10'; el.dispatchEvent(new Event('input', { bubbles: true })); }"
    )
    assert float(cursor.get_attribute("x1")) < at_end
    assert _basin_mass_text("9", 10) in _readout(page)

    page.locator("#surfaceChangeBasin").select_option("WAIS")
    region = _wait_for_selection(page, "WAIS")
    assert region["selectedCells"] > state["selectedCells"], "West Antarctica holds the Amundsen basin and four more"
    page.locator("#surfaceChangeBasin").select_option("AIS")
    assert _wait_for_selection(page, "AIS")["selectedCells"] == 0
    assert _chart_title(page) == "Antarctic Ice Sheet"
    assert errors == []


def test_clicking_the_ice_picks_the_basin_under_the_pointer_and_again_lets_it_go(page):
    page, errors = page
    _show_basins(page, "cumulative")
    centre = _canvas_centre(page)
    page.mouse.click(*centre)
    picked = _wait_for_selection(page)
    assert page.locator("#surfaceChangeBasin").input_value() == picked["selection"]
    assert _chart_title(page).startswith("Basin ")
    selected = _pixels_at(page, *centre)
    page.mouse.click(*centre)
    assert _wait_for_selection(page, "AIS")["selectedCells"] == 0
    # The picked basin keeps its colour and the rest recedes: under the pointer nothing
    # changed, so the basin picked is the one drawn there, not a neighbour or a mirror image.
    assert _mean_difference(selected, _pixels_at(page, *centre)) < 3
    assert errors == []


def test_a_click_waiting_for_the_basins_gives_way_to_a_later_choice(playwright_browser, explorer_url):
    # The record's first pick fetches the rate map for its basins; hold that download.
    held = []
    routes = [("**/data/cpom_antarctica_dhdt_480.bin*", lambda route: held.append(route))]
    context, page, errors = _open(playwright_browser, explorer_url, routes)
    try:
        _show_basins(page, "cumulative")
        page.mouse.click(*_canvas_centre(page))
        _hold_until_requested(page, held)
        page.locator("#surfaceChangeBasin").select_option("WAIS")
        held[0].continue_()
        _wait_for_selection(page, "WAIS")
        page.evaluate("window.advanceTime(32)")
        page.wait_for_timeout(500)
        assert _state(page)["surfaceChange"]["selection"] == "WAIS"
        assert page.locator("#surfaceChangeBasin").input_value() == "WAIS"
        assert errors == []
    finally:
        context.close()


def test_a_second_click_while_the_basins_load_does_not_undo_the_first(playwright_browser, explorer_url):
    held = []
    routes = [("**/data/cpom_antarctica_dhdt_480.bin*", lambda route: held.append(route))]
    context, page, errors = _open(playwright_browser, explorer_url, routes)
    try:
        _show_basins(page, "cumulative")
        centre = _canvas_centre(page)
        page.mouse.click(*centre)
        _hold_until_requested(page, held)
        page.mouse.click(*centre)
        held[0].continue_()
        picked = _wait_for_selection(page)
        page.wait_for_timeout(500)
        assert _state(page)["surfaceChange"]["selection"] == picked["selection"]
        assert errors == []
    finally:
        context.close()


def test_the_rate_view_and_the_hd_grid_keep_the_selection(page):
    page, errors = page
    _show_basins(page, "cumulative")
    page.locator("#surfaceChangeBasin").select_option(AMUNDSEN)
    cumulative = _wait_for_selection(page, AMUNDSEN)

    page.locator("#surfaceChangeView").select_option("rate")
    _wait_for_view(page, "rate")
    rate = _wait_for_selection(page, AMUNDSEN)
    # The same basin ids on the same grid; the two views draw nearly the same cells.
    assert abs(rate["selectedCells"] - cumulative["selectedCells"]) < 0.05 * cumulative["selectedCells"]
    assert page.locator("#surfaceChangeChart .cursor").get_attribute("visibility") == "hidden"
    first, last = int(RECORD_META["series"]["years"][0]), int(RECORD_META["series"]["years"][-1])
    assert f"Basin G-H, West Antarctica, {first}–{last}" in _readout(page)

    page.locator("#resolutionPreset").select_option("hd")
    page.wait_for_function(
        f"""() => {{
            const s = JSON.parse(window.render_game_to_text());
            return s.ready && s.dataset === 'hd' && s.surfaceChange.shown
              && s.surfaceChange.selection === '{AMUNDSEN}' && s.surfaceChange.selectedCells > 0;
        }}""",
        timeout=LAYER_TIMEOUT_MS,
    )
    hd = _state(page)["surfaceChange"]
    # A 4 km cell is a sixth of a 10 km one.
    assert 5 * rate["selectedCells"] < hd["selectedCells"] < 7.5 * rate["selectedCells"]
    assert _chart_title(page) == "Basin G-H, West Antarctica"
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
        # The failure is reported once, as the layer's own; nothing else goes wrong.
        assert len(errors) == 1 and errors[0].startswith("Surface-change layer load failed")
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
        _show_basins(page, "cumulative")
        readout = _readout(page)
        assert "1995 年以来南极冰盖" in readout
        assert "南极对海平面的贡献" in readout
        assert "年" in page.locator("#surfaceChangeTimeValue").inner_text()
        page.locator("#surfaceChangeBasin").select_option(AMUNDSEN)
        _wait_for_selection(page, AMUNDSEN)
        assert _chart_title(page) == "西南极 G-H 流域"
        assert "所选范围: 西南极 G-H 流域：" in page.locator("#metaList").inner_text()
        assert errors == []
    finally:
        context.close()
