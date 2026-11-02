"""E2E tests: the research edition's grouped panel (docs/research-panel-redesign.md, phase 1).

The panel fits its height budget, its groups open as the page marks them, remember what
the reader chose and open by themselves when the runtime switches a layer on in them; the
region, terrain and grid controls drive the runtime's selects; a layer the terrain lacks
says why; every label and legend tick fits on one line; and the view actions sit in the
viewer.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.e2e.panel import choose_region, open_group

pytestmark = pytest.mark.e2e

READY_TIMEOUT_MS = 90_000
DESKTOP = {"width": 1440, "height": 900}
GROUPS = ["scene", "flow", "bed", "ocean", "time", "places", "display", "data"]
GROUP_STATE_KEY = "3d-ice-research-panel-groups"

# The overlapping-band test for one line: every piece of the element's text shares a row.
ONE_LINE = """(element) => {
    const range = document.createRange();
    range.selectNodeContents(element);
    const rects = [...range.getClientRects()].filter((r) => r.width > 0 && r.height > 0);
    return !rects.length || Math.max(...rects.map((r) => r.top)) < Math.min(...rects.map((r) => r.bottom));
}"""
LABELS = (
    ".rp-masthead, .rp-segmented__option, .rp-field__label, .rp-group__title, .rp-group__hint, .rp-chip,"
    " .rp-choice__opt, .rp-row__label, .rp-row__meta, .rp-reason, .rp-legend__caption, .rp-group .control > label,"
    " .rp-group .toggle, .rp-about > summary"
)


def _open(playwright_browser, url, viewport=DESKTOP, **context_args):
    context = playwright_browser.new_context(viewport=viewport, **context_args)
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    _wait_ready(page)
    return context, page, errors


def _state(page) -> dict:
    return json.loads(page.evaluate("window.render_game_to_text()"))


def _wait_ready(page, condition: str = "true") -> dict:
    page.wait_for_function(
        f"""() => {{
            if (!window.render_game_to_text) return false;
            const s = JSON.parse(window.render_game_to_text());
            return s.ready && ({condition});
        }}""",
        timeout=READY_TIMEOUT_MS,
        polling=250,
    )
    return _state(page)


def _open_groups(page) -> dict:
    return page.evaluate("() => Object.fromEntries([...document.querySelectorAll('.rp-group')].map((g) => [g.dataset.group, g.open]))")


def _reason(page, control_id: str) -> str:
    """Why a layer is off: on its row, or on the note line of the one-of choice it is in."""
    return page.evaluate(
        """(id) => {
            const control = document.getElementById(id);
            const choice = control.closest('[data-choice]');
            const note = choice
                ? choice.querySelector('.rp-choice__note:not([hidden])')
                : control.closest('label').querySelector('.rp-reason:not([hidden])');
            return note?.textContent || '';
        }""",
        control_id,
    )


@pytest.fixture
def page(playwright_browser, explorer_url):
    context, page, errors = _open(playwright_browser, explorer_url)
    try:
        yield page, errors
    finally:
        context.close()


@pytest.mark.parametrize("viewport", [DESKTOP, {"width": 1280, "height": 720}], ids=["1440x900", "1280x720"])
def test_the_panel_is_at_most_1000_px_tall_with_nothing_on(playwright_browser, explorer_url, viewport):
    context, page, errors = _open(playwright_browser, explorer_url, viewport)
    try:
        height = page.evaluate("() => document.querySelector('.panel').scrollHeight")
        assert height <= 1000, f"the panel is {height} px tall"
        # Today's panel was 2,135 px tall; the first layer toggle now sits in the first screen.
        first_layer = page.evaluate("() => document.getElementById('showSea').getBoundingClientRect().bottom")
        assert first_layer < viewport["height"]
        assert errors == []
    finally:
        context.close()


def test_groups_open_as_marked_and_remember_what_the_reader_chose(page):
    page, errors = page
    assert _open_groups(page) == {group: group in ("scene", "flow", "time") for group in GROUPS}
    page.locator('details[data-group="bed"] > summary').click()
    page.locator('details[data-group="flow"] > summary').click()
    page.wait_for_function(
        f"() => JSON.parse(localStorage.getItem('{GROUP_STATE_KEY}') || '{{}}').flow === false", timeout=5_000
    )
    # Only the groups the reader touched are stored, so a later default is never pinned.
    assert json.loads(page.evaluate(f"() => localStorage.getItem('{GROUP_STATE_KEY}')")) == {"bed": True, "flow": False}
    page.reload(wait_until="domcontentloaded")
    _wait_ready(page)
    groups = _open_groups(page)
    assert groups["bed"] is True and groups["flow"] is False and groups["scene"] is True
    assert errors == []


def test_a_closed_group_opens_when_the_runtime_switches_a_layer_on_in_it(page):
    page, errors = page
    page.locator('details[data-group="scene"] > summary').click()
    assert _open_groups(page)["scene"] is False
    count = page.locator('details[data-group="scene"] .rp-group__count')
    assert count.is_hidden()
    # Observed sea ice raises the sea-level plane, which lives in Scene.
    page.locator("#showSeaIce").check()
    _wait_ready(page, "s.seaIce.shown")
    page.wait_for_function("() => document.querySelector('details[data-group=\"scene\"]').open", timeout=5_000)
    assert count.inner_text() == "1 on"
    assert page.locator('details[data-group="time"] .rp-group__count').inner_text() == "1 on"
    # Opened by the panel, not the reader: nothing is remembered.
    assert page.evaluate(f"() => localStorage.getItem('{GROUP_STATE_KEY}')") == '{"scene":false}'
    assert errors == []


def test_place_search_opens_places_and_counts_the_layer_it_switched_on(page):
    page, errors = page
    places = page.locator('details[data-group="places"]')
    assert places.evaluate("(details) => details.open") is False
    page.get_by_role("combobox", name="Search places and features").fill("Amundsen-Scott")
    page.get_by_role("option", name=re.compile("Amundsen", re.IGNORECASE)).click()
    # The search switches the station layer on without an event; the panel still sees it.
    page.wait_for_function("() => document.getElementById('showResearchStations').checked", timeout=30_000)
    page.wait_for_function("() => document.querySelector('details[data-group=\"places\"]').open", timeout=5_000)
    assert places.locator(".rp-group__count").inner_text() == "1 on"
    assert errors == []


def test_a_group_the_reader_closed_stays_closed_when_a_layer_elsewhere_comes_on(page):
    page, errors = page
    open_group(page, "places")
    page.get_by_label("Research stations").check()
    page.locator('details[data-group="places"] > summary').click()
    page.get_by_label("Surface speed").check()
    page.get_by_label("Sea level (0 m)").check()
    assert _open_groups(page)["places"] is False
    assert page.locator('details[data-group="places"] .rp-group__count').inner_text() == "1 on"
    assert page.locator('details[data-group="scene"] .rp-group__count').inner_text() == "1 on"
    assert page.locator('details[data-group="flow"] .rp-group__count').inner_text() == "1 on"
    assert errors == []


def test_region_terrain_and_grid_drive_the_runtime_and_a_missing_layer_says_why(page):
    page, errors = page
    terrain = page.locator("#terrainPreset")
    assert terrain.evaluate("(select) => [...select.options].map((o) => o.textContent)") == ["BedMachine v4", "Bedmap3"]
    assert page.locator('#gridPreset [aria-pressed="true"]').inner_text() == "10 km"
    assert page.locator("#terrainHint").is_hidden()

    terrain.select_option("bedmap3")
    _wait_ready(page, "s.dataset === 'bedmap3'")
    assert page.locator("#resolutionPreset").input_value() == "bedmap3"
    assert _reason(page, "showSurfaceChange") == "BedMachine v4 only"
    assert _reason(page, "showIceProjection") == "BedMachine v4 10 km only"
    open_group(page, "ocean")
    assert page.locator("#showBasalMelt").is_disabled() and _reason(page, "showBasalMelt") == "BedMachine v4 only"
    assert _reason(page, "showVelocity") == ""

    choose_region(page, "greenland")
    state = _wait_ready(page, "s.region === 'greenland'")
    assert state["dataset"] == "3km"
    assert page.locator('[data-segmented-for="regionPreset"] [aria-pressed="true"]').inner_text() == "Greenland"
    assert terrain.evaluate("(select) => [...select.options].map((o) => o.textContent)") == ["BedMachine v6", "QRF 2025"]
    assert page.locator("#gridPreset button").all_inner_texts() == ["3 km", "1 km"]
    open_group(page, "bed")
    # Basal friction stays on offer, so the bed's choice names the colouring that is not.
    assert _reason(page, "showEffectivePressure") == "Effective pressure: Antarctica only"
    open_group(page, "ocean")
    assert _reason(page, "showBasalMelt") == "Antarctica only"
    assert _reason(page, "showIceProjection") == "Antarctica only"
    # Rows are no longer hidden per region: they stay, switched off.
    for row in ("showBasalMeltRow", "showThermalDrivingRow", "surfaceChangeRow", "iceProjectionRow"):
        assert page.locator(f"#{row}").is_visible(), row
    assert page.locator("#showRefinedBasinsLabel").text_content() == "Basins"

    # Back in Antarctica the terrain chosen there is kept. The 4 km grid comes last: while
    # headless WebGL draws it, the page is too busy for Playwright to finish a click.
    choose_region(page, "antarctica")
    _wait_ready(page, "s.region === 'antarctica' && s.dataset === 'bedmap3'")
    page.locator('#gridPreset [data-value="4km"]').click()
    _wait_ready(page, "s.dataset === 'bedmap3-hd'")
    assert page.locator("#terrainHint").text_content() == "4 km grid; ~18.6 MB, desktop recommended"
    assert errors == []


# Each tick legend with the layer that draws it and the mesh that layer builds. One at a
# time: some of these exclude each other, as basal friction and effective pressure do.
TICK_LEGENDS = (
    ("#showVelocity", "velocity", "velocityLegendLabels"),
    ("#showBasalFriction", "basalFriction", "basalFrictionLegendLabels"),
    ("#showEffectivePressure", "hydrology", "effectivePressureLegendLabels"),
    ("#showSubglacialChannels", "hydrology", "channelLegendLabels"),
    ("#showBasalMelt", "basalMelt", "basalMeltLegendLabels"),
)


def test_every_label_and_legend_tick_fits_on_one_line(page):
    page, errors = page
    for group in GROUPS:
        open_group(page, group)
    for control, mesh, legend in TICK_LEGENDS:
        page.locator(control).check()
        _wait_ready(page, f"s.meshes.{mesh}")
        spans = f"[...document.querySelectorAll('#{legend} span')]"
        page.wait_for_function(
            f"() => {spans}.length > 0 && {spans}.every((s) => s.style.left.endsWith('px'))", timeout=30_000
        )
        rows = page.evaluate(f"() => [...new Set({spans}.map((s) => s.style.top))]")
        assert rows == ["0px"], f"{legend} wraps onto {len(rows)} rows"
        # The unit is in the caption, so no tick carries one.
        assert not any(" " in text for text in page.locator(f"#{legend} span").all_inner_texts()), legend
    wrapped = page.evaluate(
        f"""() => [...document.querySelectorAll({json.dumps(LABELS)})]
            .filter((element) => element.offsetParent && !({ONE_LINE})(element))
            .map((element) => element.textContent.trim().replace(/\\s+/g, " "))"""
    )
    assert wrapped == []
    assert page.evaluate("() => document.querySelector('.panel').getBoundingClientRect().width") == 340
    assert errors == []


def test_the_flowline_row_says_how_to_pick_a_flowline_in_one_line(page):
    page, errors = page
    page.locator("#showFlowline").check()
    hint = page.locator("#flowlineProfileCardMount .flowline-hint")
    hint.wait_for(state="visible", timeout=30_000)
    assert hint.inner_text() == "Click a flowline for its profile."
    assert page.locator("#flowlineProfileCardMount").bounding_box()["height"] < 30
    assert errors == []


def test_the_info_menu_holds_the_intro_and_the_tour(page):
    page, errors = page
    menu = page.locator("#panelInfoMenu")
    assert menu.is_hidden()
    page.locator("#panelInfoButton").click()
    assert menu.is_visible()
    assert page.locator("#panelInfoButton").get_attribute("aria-expanded") == "true"
    assert "Explore Antarctica" in page.locator("#panelSubtitle").inner_text()
    assert menu.locator("a[href='../explore/?tour=1']").is_visible()
    page.keyboard.press("Escape")
    assert menu.is_hidden()
    # A click anywhere outside the open menu closes it; the menu itself covers the panel
    # below the masthead, so click the 3D view.
    page.locator("#panelInfoButton").click()
    page.locator("#viewer").click(position={"x": 40, "y": 300})
    assert menu.is_hidden()
    assert errors == []


def test_reset_view_sits_in_the_viewer_and_puts_the_camera_back(page):
    page, errors = page
    assert page.locator(".viewer-toolbar #resetView").is_visible()
    assert page.locator("#fullscreenToggle").count() == 0
    start = _state(page)["camera"]["position"]
    box = page.locator("#viewer canvas").bounding_box()
    centre = (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.move(*centre)
    page.mouse.down()
    page.mouse.move(centre[0] + 220, centre[1] + 60, steps=8)
    page.mouse.up()
    page.wait_for_function(
        f"() => JSON.parse(window.render_game_to_text()).camera.position.x !== {start['x']}", timeout=10_000
    )
    page.locator("#resetView").click()
    assert _state(page)["camera"]["position"] == start
    assert errors == []


def test_the_chinese_page_gives_the_same_panel_in_chinese(playwright_browser, server):
    context, page, errors = _open(playwright_browser, f"{server}/zh/tools/3D-interactive-cryosphere-explorer.html")
    try:
        assert page.evaluate("() => document.querySelector('.panel').scrollHeight") <= 1000
        assert _open_groups(page) == {group: group in ("scene", "flow", "time") for group in GROUPS}
        choose_region(page, "greenland")
        _wait_ready(page, "s.region === 'greenland'")
        open_group(page, "bed")
        assert _reason(page, "showEffectivePressure") == "有效压力：仅限南极洲"
        page.locator("#showVelocity").check()
        page.wait_for_function("() => document.querySelector('[data-group=\"flow\"] .rp-group__count').textContent", timeout=10_000)
        assert page.locator('details[data-group="flow"] .rp-group__count').inner_text() == "已开 1"
        assert errors == []
    finally:
        context.close()


def test_the_phone_drawer_opens_on_the_masthead_where_and_scene(playwright_browser, explorer_url):
    context, page, errors = _open(
        playwright_browser, explorer_url, {"width": 375, "height": 812}, is_mobile=True, has_touch=True
    )
    try:
        page.locator("#panelCloseButton").click()
        page.wait_for_function("() => document.body.classList.contains('mobile-panel-open')", timeout=10_000)
        page.wait_for_timeout(400)  # the drawer slides up
        bottom = page.evaluate("() => document.querySelector('.panel').getBoundingClientRect().bottom")
        for selector in (".rp-masthead", ".rp-where", 'details[data-group="scene"] > summary'):
            box = page.locator(selector).bounding_box()
            assert box and box["y"] >= 0 and box["y"] + box["height"] <= bottom, selector
        assert errors == []
    finally:
        context.close()
