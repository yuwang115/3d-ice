"""E2E tests: which layers exclude each other on the research page, and what comes back.

One case per row of the exclusivity table in docs/research-panel-redesign.md, with the
restore rules of phase 2: a time feature puts back what it switched off when the reader
returns to the present day or moves straight to another feature, and a layer it replaced,
switched on by the reader, ends it with nothing else restored. The cases use the real data
packages, all but the ocean streamlines, which at 62 MB add nothing the other layers do not
already test.
"""

from __future__ import annotations

import json

import pytest

from tests.e2e.panel import open_group

pytestmark = pytest.mark.e2e

READY_TIMEOUT_MS = 150_000
GROUPS = ("scene", "flow", "bed", "ocean", "time", "places")
PROJECTION_OFF = "Off while the projection is on"


@pytest.fixture
def page(playwright_browser, explorer_url):
    context = playwright_browser.new_context(viewport={"width": 1440, "height": 900})
    try:
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.goto(explorer_url, wait_until="domcontentloaded", timeout=60_000)
        _wait(page, "s.iceProjection.probed")
        for group in GROUPS:
            open_group(page, group)
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


def _on(page, *ids: str) -> dict[str, bool]:
    return page.evaluate("(ids) => Object.fromEntries(ids.map((id) => [id, document.getElementById(id).checked]))", list(ids))


def _switch_on(page, *ids: str) -> None:
    for control_id in ids:
        page.locator(f"#{control_id}").check()


def _notes(page) -> list[str]:
    """What the panel says about the layers a time feature switched off."""
    return page.locator(".rp-takeover:not([hidden]), .rp-choice__note:not([hidden])").all_inner_texts()


def _present_day(page) -> None:
    page.locator("[data-time-none]").click()


def _rebound_solved(page) -> dict:
    return _wait(page, "s.meshes.isostaticRebound && s.isostaticRebound.solved && !s.isostaticRebound.pendingResolve")


def test_the_projection_switches_its_layers_off_and_the_present_day_puts_them_back(page):
    page, errors = page
    layers = ("showVelocity", "showBasalFriction", "showSubglacialChannels", "showBasalMelt", "showRefinedBasins")
    _switch_on(page, *layers)
    _wait(page, "s.meshes.velocity && s.meshes.basalFriction && s.meshes.hydrology && s.meshes.basalMelt && s.meshes.basins")
    assert _on(page, "showIce", "showIceBottom") == {"showIce": False, "showIceBottom": False}, "basal melt hides the ice"

    page.locator("#showIceProjection").check()
    _wait(page, "s.iceProjection.active")
    assert _on(page, *layers) == dict.fromkeys(layers, False)
    # The projection brings the flowlines and the sea plane, and its ice shows again.
    assert _on(page, "showFlowline", "showSea", "showIce", "showIceBottom") == {
        "showFlowline": True,
        "showSea": True,
        "showIce": True,
        "showIceBottom": True,
    }
    notes = _notes(page)
    assert notes.count(PROJECTION_OFF) == 3  # surface speed, channels, refined basins
    assert f"Basal friction: {PROJECTION_OFF}" in notes and f"Basal melt: {PROJECTION_OFF}" in notes

    _present_day(page)
    _wait(page, "!s.iceProjection.active && s.meshes.velocity && s.meshes.basalFriction && s.meshes.basalMelt")
    assert _on(page, *layers) == dict.fromkeys(layers, True)
    assert _on(page, "showFlowline", "showSea", "showIce", "showIceBottom") == {
        "showFlowline": False,
        "showSea": False,
        "showIce": False,
        "showIceBottom": False,
    }
    assert _notes(page) == []
    assert page.locator("[data-time-none]").get_attribute("aria-pressed") == "true"
    assert errors == []


def test_the_elevation_change_switches_velocity_off_and_takes_turns_with_the_sea_ice(page):
    page, errors = page
    _switch_on(page, "showVelocity")
    _wait(page, "s.meshes.velocity")
    page.locator("#showSurfaceChange").check()
    _wait(page, "s.surfaceChange.shown")
    assert _on(page, "showVelocity") == {"showVelocity": False}
    assert _notes(page) == ["Off while the elevation change is on"]

    # One at a time: the sea ice ends the elevation change, which gives velocity back.
    page.locator("#showSeaIce").check()
    _wait(page, "s.seaIce.shown && !s.surfaceChange.shown")
    assert _on(page, "showSurfaceChange", "showVelocity", "showSea") == {
        "showSurfaceChange": False,
        "showVelocity": True,
        "showSea": True,
    }
    _present_day(page)
    _wait(page, "!s.seaIce.shown")
    assert _on(page, "showSeaIce", "showSea", "showVelocity") == {"showSeaIce": False, "showSea": False, "showVelocity": True}
    assert errors == []


def test_the_sea_ice_takes_over_from_the_projection_and_the_rebound_and_keeps_the_sea_plane(page):
    page, errors = page
    page.locator("#showIceProjection").check()
    _wait(page, "s.iceProjection.active")
    page.locator("#showSeaIce").check()
    _wait(page, "s.seaIce.shown && !s.iceProjection.active")
    # The projection's flowlines go with it; the sea plane stays, for the sea ice.
    assert _on(page, "showIceProjection", "showFlowline", "showSea") == {
        "showIceProjection": False,
        "showFlowline": False,
        "showSea": True,
    }
    page.locator("#showIsostaticRebound").check()
    _rebound_solved(page)
    assert _on(page, "showSeaIce", "showSea") == {"showSeaIce": False, "showSea": True}
    _present_day(page)
    _wait(page, "!s.isostaticRebound.enabled")
    assert _on(page, "showIsostaticRebound", "showSea") == {"showIsostaticRebound": False, "showSea": False}
    assert errors == []


def test_basal_friction_and_effective_pressure_exclude_each_other_and_end_the_rebound(page):
    page, errors = page
    page.locator("#showIsostaticRebound").check()
    _rebound_solved(page)
    page.locator("#showBasalFriction").check()
    _wait(page, "s.meshes.basalFriction")
    # The new choice stands, and nothing else changes: the sea plane the rebound raised stays.
    assert _on(page, "showIsostaticRebound", "showBasalFriction", "showSea") == {
        "showIsostaticRebound": False,
        "showBasalFriction": True,
        "showSea": True,
    }
    page.locator("#showEffectivePressure").check()
    _wait(page, "s.meshes.hydrology")
    assert _on(page, "showBasalFriction", "showEffectivePressure") == {"showBasalFriction": False, "showEffectivePressure": True}
    page.locator('[data-choice="bed"] [data-choice-none]').click()
    assert _on(page, "showBasalFriction", "showEffectivePressure") == {"showBasalFriction": False, "showEffectivePressure": False}
    assert page.locator('[data-choice="bed"] [data-choice-none]').get_attribute("aria-pressed") == "true"
    assert errors == []


def test_basal_melt_and_thermal_driving_share_the_ice_shelf_base_and_off_brings_the_ice_back(page):
    page, errors = page
    page.locator("#showBasalMelt").check()
    _wait(page, "s.meshes.basalMelt")
    assert _on(page, "showIce", "showIceBottom") == {"showIce": False, "showIceBottom": False}
    assert _state(page)["meshes"]["iceBottom"] is False
    page.locator("#showThermalDriving").check()
    _wait(page, "s.meshes.thermalDriving")
    assert _on(page, "showBasalMelt", "showThermalDriving", "showIce", "showIceBottom") == {
        "showBasalMelt": False,
        "showThermalDriving": True,
        "showIce": False,
        "showIceBottom": False,
    }
    page.locator('[data-choice="shelf"] [data-choice-none]').click()
    assert _on(page, "showThermalDriving", "showIce", "showIceBottom") == {
        "showThermalDriving": False,
        "showIce": True,
        "showIceBottom": True,
    }
    meshes = _state(page)["meshes"]
    assert meshes["ice"] is True and meshes["iceBottom"] is True
    assert errors == []


def test_thermal_driving_leaves_alone_an_ice_base_the_reader_chose(page):
    page, errors = page
    # Already off before the colouring: Off must not switch it on.
    page.locator("#showIceBottom").uncheck()
    page.locator("#showThermalDriving").check()
    _wait(page, "s.meshes.thermalDriving")
    assert _on(page, "showIce", "showIceBottom") == {"showIce": False, "showIceBottom": False}
    page.locator('[data-choice="shelf"] [data-choice-none]').click()
    assert _on(page, "showIce", "showIceBottom") == {"showIce": True, "showIceBottom": False}

    # Ticked back by the reader while the colouring is on: Off leaves it on, and keeps the
    # ice surface the colouring hid coming back.
    page.locator("#showIceBottom").check()
    page.locator("#showBasalMelt").check()
    _wait(page, "s.meshes.basalMelt")
    page.locator("#showIceBottom").check()
    page.locator('[data-choice="shelf"] [data-choice-none]').click()
    assert _on(page, "showIce", "showIceBottom") == {"showIce": True, "showIceBottom": True}
    assert errors == []


def test_velocity_colour_ends_the_elevation_change(page):
    page, errors = page
    page.locator("#showSurfaceChange").check()
    _wait(page, "s.surfaceChange.shown")
    page.locator("#showVelocity").check()
    _wait(page, "s.meshes.velocity && !s.surfaceChange.shown")
    assert _on(page, "showSurfaceChange", "showVelocity") == {"showSurfaceChange": False, "showVelocity": True}
    assert page.locator("[data-time-none]").get_attribute("aria-pressed") == "true"
    assert errors == []


def test_a_layer_the_projection_replaced_ends_it_and_nothing_else_comes_back(page):
    page, errors = page
    _switch_on(page, "showVelocity", "showBasalFriction")
    _wait(page, "s.meshes.velocity && s.meshes.basalFriction")
    page.locator("#showIceProjection").check()
    _wait(page, "s.iceProjection.active")
    page.locator("#showVelocity").check()
    _wait(page, "!s.iceProjection.active")
    assert _on(page, "showIceProjection", "showVelocity", "showBasalFriction") == {
        "showIceProjection": False,
        "showVelocity": True,
        "showBasalFriction": False,
    }
    assert _notes(page) == []
    assert errors == []


def test_going_straight_from_one_feature_to_another_keeps_what_both_replace(page):
    page, errors = page
    layers = ("showVelocity", "showBasalFriction", "showFlowline")
    _switch_on(page, *layers)
    _wait(page, "s.meshes.velocity && s.meshes.basalFriction && s.meshes.flowline")
    page.locator("#showIsostaticRebound").check()
    _rebound_solved(page)
    assert _on(page, *layers, "showSea") == {**dict.fromkeys(layers, False), "showSea": True}

    # The projection replaces velocity and friction too, so they stay off; it shows the
    # flowlines, which come back.
    page.locator("#showIceProjection").check()
    _wait(page, "s.iceProjection.active && !s.isostaticRebound.enabled")
    assert _on(page, *layers, "showSea") == {
        "showVelocity": False,
        "showBasalFriction": False,
        "showFlowline": True,
        "showSea": True,
    }
    _present_day(page)
    _wait(page, "!s.iceProjection.active && s.meshes.velocity && s.meshes.basalFriction")
    assert _on(page, *layers, "showSea") == {**dict.fromkeys(layers, True), "showSea": False}
    assert errors == []
