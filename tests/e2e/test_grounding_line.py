"""E2E tests: the grounding line on the research page.

The line is traced in the browser from the grounded and floating ice the scene draws
(static/tools/js/grounding-line.js), so these tests check that it follows the scene: the
surface it rides on, the region on screen, the projection's frame, and the rebound, which
takes today's ice away and so switches the line off until the present day comes back. The
public edition's toggle and tour stop are covered in test_public_edition.py.
"""

from __future__ import annotations

import json

import pytest

from tests.e2e.panel import choose_region, open_group

pytestmark = pytest.mark.e2e

READY_TIMEOUT_MS = 150_000
REBOUND_OFF = "Off while the rebound is on"


@pytest.fixture
def page(playwright_browser, explorer_url):
    context = playwright_browser.new_context(viewport={"width": 1440, "height": 900})
    try:
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.goto(explorer_url, wait_until="domcontentloaded", timeout=60_000)
        _wait(page, "s.iceProjection.probed")
        open_group(page, "time")
        yield page, errors
    finally:
        context.close()


def _state(page) -> dict:
    return json.loads(page.evaluate("window.render_game_to_text()"))


def _wait(page, condition: str) -> dict:
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


def _set_projection_year(page, year: int) -> None:
    page.evaluate(
        """(year) => {
            const slider = document.getElementById("projectionYear");
            slider.value = String(year);
            slider.dispatchEvent(new Event("input", { bubbles: true }));
        }""",
        year,
    )


def test_the_line_rides_on_the_highest_surface_on_screen(page):
    page, errors = page
    assert _state(page)["groundingLine"]["visible"] is False, "the line starts off"
    page.locator("#showGroundingLine").check()
    line = _wait(page, "s.groundingLine.visible")["groundingLine"]
    assert (line["source"], line["carrier"]) == ("terrain", "surface")
    assert line["lines"] > 100 and line["triangles"] > 10_000
    assert "Grounding Line:" in page.locator("#metaList").text_content()

    page.locator("#showIce").uncheck()
    assert _wait(page, "s.groundingLine.carrier === 'base'")["groundingLine"]["visible"]
    page.locator("#showIceBottom").uncheck()
    assert _wait(page, "s.groundingLine.carrier === 'bed'")["groundingLine"]["visible"]
    page.locator("#showBed").uncheck()
    assert _wait(page, "!s.groundingLine.visible")["groundingLine"]["enabled"], "nothing left to ride on"
    page.locator("#showIce").check()
    assert _wait(page, "s.groundingLine.visible")["groundingLine"]["carrier"] == "surface"

    page.locator("#showGroundingLine").uncheck()
    assert _wait(page, "!s.meshes.groundingLine")["toggles"]["showGroundingLine"] is False
    assert "Grounding Line:" not in page.locator("#metaList").text_content()
    assert errors == []


def test_the_line_is_traced_again_for_another_region(page):
    page, errors = page
    page.locator("#showGroundingLine").check()
    antarctica = _wait(page, "s.groundingLine.visible")["groundingLine"]
    choose_region(page, "greenland")
    greenland = _wait(page, "s.region === 'greenland' && s.groundingLine.visible")["groundingLine"]
    # Greenland's grounding line is its tidewater fronts and a few floating tongues.
    assert greenland["source"] == "terrain" and greenland["lines"] > 100
    assert greenland["points"] < antarctica["points"]
    assert errors == []


def test_the_line_follows_the_projection_and_the_present_day_brings_todays_back(page):
    page, errors = page
    page.locator("#showGroundingLine").check()
    today = _wait(page, "s.groundingLine.visible")["groundingLine"]
    page.locator("#showIceProjection").check()
    first = _wait(page, "s.iceProjection.active && s.groundingLine.source === 'projection'")["groundingLine"]
    assert first["visible"] and first["year"] == 2015
    _set_projection_year(page, 2300)
    last = _wait(page, "s.groundingLine.year === 2300")["groundingLine"]
    assert last["visible"] and last["points"] != first["points"], "the line moved with the projected ice"
    # Without the ice surface it rides on the projected ice base, from the same frame.
    page.locator("#showIce").uncheck()
    base = _wait(page, "s.groundingLine.carrier === 'base'")["groundingLine"]
    assert (base["source"], base["year"], base["lines"]) == ("projection", 2300, last["lines"])
    page.locator("#showIce").check()
    _wait(page, "s.groundingLine.carrier === 'surface'")

    page.locator("[data-time-none]").click()
    back = _wait(page, "!s.iceProjection.active && s.groundingLine.source === 'terrain'")["groundingLine"]
    assert back["visible"] and back["points"] == today["points"]
    assert errors == []


def test_the_rebound_switches_the_line_off_and_the_present_day_puts_it_back(page):
    page, errors = page
    page.locator("#showGroundingLine").check()
    _wait(page, "s.groundingLine.visible")
    page.locator("#showIsostaticRebound").check()
    state = _wait(page, "s.meshes.isostaticRebound && s.isostaticRebound.solved && !s.isostaticRebound.pendingResolve")
    assert state["toggles"]["showGroundingLine"] is False
    assert state["meshes"]["groundingLine"] is False
    assert page.locator("#showGroundingLine").locator("xpath=..").locator(".rp-takeover").inner_text() == REBOUND_OFF

    page.locator("[data-time-none]").click()
    state = _wait(page, "!s.isostaticRebound.enabled && s.groundingLine.visible")
    assert state["toggles"]["showGroundingLine"] is True
    assert page.locator(".rp-takeover:not([hidden])").count() == 0
    assert errors == []
