"""E2E tests: the ISMIP6 ice-sheet projection layer plays back in the browser.

The committed packages are tens of megabytes, so these tests serve a small synthetic package
in place of the high-emissions one, written by the packer's own `write_package`, through
Playwright request routing. That keeps the assertions exact while exercising the same
runtime path the real packages take. The other two scenarios load their real metadata, so
the picker lists all three. Controls are located by `#id`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.e2e

READY_TIMEOUT_MS = 60_000
REBOUND_TIMEOUT_MS = 90_000
TERRAIN_META = Path(__file__).resolve().parents[2] / "static/tools/data/bedmachine_antarctica_v4_480.meta.json"
SYNTHETIC_STEM = "ismip6_2300_mean8_ae05_480"
RHO_RATIO = 917.0 / 1028.0

# A 70 x 100 block over Pine Island and Thwaites glaciers, where the observed flowlines run,
# on a flat bed 600 m below sea level. The ice starts 1000 m thick (grounded: flotation needs
# 673 m), thins to 700 m, and ends with its western half gone and its eastern half 500 m thick
# and afloat. Its speed rises by 500 m/yr by 2100 and 1500 m/yr by 2300.
BLOCK_ROWS = slice(330, 400)
BLOCK_COLS = slice(130, 230)
BED_M = -600.0
FRAME_YEARS = [2015, 2100, 2300]


@pytest.fixture(scope="session")
def synthetic_projection(ismip6_projection_module, tmp_path_factory) -> dict:
    grid = json.loads(TERRAIN_META.read_text())["grid"]
    domain = np.zeros((grid["ny"], grid["nx"]), dtype=bool)
    domain[BLOCK_ROWS, BLOCK_COLS] = True
    count = int(domain.sum())
    cols = np.nonzero(domain)[1]
    west = cols < (BLOCK_COLS.start + BLOCK_COLS.stop) // 2
    frames = np.stack([np.full(count, 1000.0), np.full(count, 700.0), np.where(west, 0.0, 500.0)])
    speed = np.stack([np.zeros(count), np.full(count, 500.0), np.where(west, 0.0, 1500.0)])

    years = list(range(2015, 2301))
    ramp = np.linspace(0.0, 1.0, len(years))
    out_dir = tmp_path_factory.mktemp("ice-projection")
    ismip6_projection_module.write_package(
        out_dir,
        SYNTHETIC_STEM,
        grid=grid,
        domain=domain,
        bed=np.full(count, BED_M),
        thickness=frames,
        speed_change=speed,
        years=FRAME_YEARS,
        extra_meta={
            "title": "Synthetic projection for the E2E suite",
            "experiment": {"id": "expAE05", "climate_model": "SYNTHETIC", "scenario": "SSP5-8.5"},
            "models": ["A", "B"],
            "physical_constants": {"density_ratio": RHO_RATIO},
            "keyframes": {"interval_years": 5},
            "validation": {
                "sea_level_of_packed_geometry_m": {"end_packed": 1.3},
                "volume_change_captured": {"end_ratio": 0.96},
            },
            "series": {
                "years": years,
                "sea_level_contribution_m": ramp.round(5).tolist(),
                "sea_level_contribution_min_m": (-0.5 * ramp).round(5).tolist(),
                "sea_level_contribution_max_m": (3.0 * ramp).round(5).tolist(),
            },
        },
    )
    stem = out_dir / SYNTHETIC_STEM
    return {
        "meta": stem.with_suffix(".meta.json"),
        "bin": stem.with_suffix(".bin"),
        "cells": count,
        "west_cells": int(west.sum()),
    }


def _synthetic_routes(synthetic_projection, scenario: str = "ae05"):
    stem = SYNTHETIC_STEM.replace("_ae05_", f"_{scenario}_")
    return [
        # The runtime appends a cache-busting query to the package URLs.
        (f"**/data/{stem}.meta.json*",
         lambda route: route.fulfill(path=str(synthetic_projection["meta"]), content_type="application/json")),
        (f"**/data/{stem}.bin*",
         lambda route: route.fulfill(path=str(synthetic_projection["bin"]), content_type="application/octet-stream")),
    ]


def _open(playwright_browser, url, routes):
    context = playwright_browser.new_context(viewport={"width": 1280, "height": 800})
    for pattern, handler in routes:
        context.route(pattern, handler)
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_function(
        """() => {
            if (!window.render_game_to_text) return false;
            const s = JSON.parse(window.render_game_to_text());
            return s.ready && s.iceProjection.probed;
        }""",
        timeout=READY_TIMEOUT_MS,
    )
    return context, page, errors


def _not_found(route):
    route.fulfill(status=404, body="not found")


@pytest.fixture
def projection_page(playwright_browser, explorer_url, synthetic_projection):
    """The English research page, with the synthetic package as the high-emissions scenario."""
    context, page, errors = _open(playwright_browser, explorer_url, _synthetic_routes(synthetic_projection))
    try:
        yield page, errors
    finally:
        context.close()


def _state(page) -> dict:
    return json.loads(page.evaluate("window.render_game_to_text()"))


def _enable(page) -> dict:
    page.locator("#showIceProjection").check()
    page.wait_for_function(
        "() => { const s = JSON.parse(window.render_game_to_text()); return s.ready && s.iceProjection.active; }",
        timeout=READY_TIMEOUT_MS,
    )
    return _state(page)["iceProjection"]


def _next_frames(page) -> None:
    page.evaluate("() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))")


def _set_year(page, year: int) -> dict:
    page.locator("#projectionYear").fill(str(year))
    page.wait_for_function(
        f"() => JSON.parse(window.render_game_to_text()).iceProjection.year === {year}",
        timeout=10_000,
    )
    # The geometry follows on the next animation frame.
    _next_frames(page)
    return _state(page)["iceProjection"]


def _wait_for_flowlines(page) -> dict:
    page.wait_for_function(
        "() => JSON.parse(window.render_game_to_text()).iceProjection.flowlines.built", timeout=READY_TIMEOUT_MS
    )
    return _state(page)["iceProjection"]["flowlines"]


def _show_flowlines(page) -> dict:
    page.locator("#showFlowline").check()
    return _wait_for_flowlines(page)


def _disable(page) -> None:
    page.locator("#showIceProjection").uncheck()
    page.wait_for_function("() => !JSON.parse(window.render_game_to_text()).iceProjection.active", timeout=10_000)


class TestIceProjectionAbsent:
    def test_the_layer_stays_hidden_without_packages(self, playwright_browser, explorer_url):
        context, page, errors = _open(playwright_browser, explorer_url, [("**/data/ismip6_2300_*", _not_found)])
        try:
            projection = _state(page)["iceProjection"]
            assert projection["probed"] is True
            assert projection["available"] is False
            assert projection["scenarios"] == []
            assert page.locator("#iceProjectionRow").is_hidden()
            assert errors == []
        finally:
            context.close()


class TestIceProjection:
    def test_the_three_scenarios_are_offered_high_emissions_selected(self, projection_page):
        page, errors = projection_page
        projection = _state(page)["iceProjection"]
        assert projection["available"] is True
        assert projection["enabled"] is False
        assert projection["scenarios"] == ["ae10", "ae05", "ae14"]
        assert page.locator("#projectionScenario").input_value() == "ae05"
        assert page.locator("#iceProjectionRow").is_visible()
        assert page.locator("#iceProjectionControls").is_hidden()
        assert errors == []

    def test_enabling_draws_the_first_keyframe_on_the_bed(self, projection_page, synthetic_projection):
        page, errors = projection_page
        projection = _enable(page)
        cells = synthetic_projection["cells"]

        assert projection["scenario"] == "ae05"
        assert projection["year"] == 2015
        assert projection["keyframes"] == 3
        assert projection["domainCells"] == cells
        assert projection["iceCells"] == cells
        # Grounded 1000 m on a -600 m bed: the surface sits at +400 m.
        assert projection["meanIceSurfaceMeters"] == pytest.approx(BED_M + 1000.0, abs=0.5)
        assert projection["seaLevelContributionMeters"] == pytest.approx(0.0, abs=1e-6)
        assert projection["bedMatchesDatasetOutsideDomain"] is True

        state = _state(page)
        assert state["meshes"]["iceProjection"] is True
        assert state["legends"]["iceProjection"] is True
        readout = page.locator("#projectionReadout").inner_text()
        assert "0.00 m" in readout
        assert "Range of the models" in readout
        assert errors == []

    def test_the_year_slider_interpolates_and_refloats_the_ice(self, projection_page, synthetic_projection):
        page, _ = projection_page
        _enable(page)
        assert "(0.0%)" in page.locator("#projectionReadout").inner_text()

        middle = _set_year(page, 2200)
        # Halfway from 700 m to (0 | 500 m): the west is 350 m (afloat), the east 600 m (afloat).
        assert middle["iceCells"] == synthetic_projection["cells"]
        assert middle["seaLevelContributionMeters"] == pytest.approx(185 / 285, abs=1e-3)

        end = _set_year(page, 2300)
        assert end["iceCells"] == synthetic_projection["cells"] - synthetic_projection["west_cells"]
        # 500 m of floating ice stands 500 * (1 - rho_i/rho_w) above the water.
        assert end["meanIceSurfaceMeters"] == pytest.approx(500.0 * (1 - RHO_RATIO), abs=0.5)
        assert end["seaLevelContributionMeters"] == pytest.approx(1.0, abs=1e-4)
        assert page.locator("#projectionYearValue").inner_text() == "2300"
        readout = page.locator("#projectionReadout").inner_text()
        assert "+3.00 m" in readout  # top of the model range
        # The grounded area is that of the year shown, not of the previous update: by 2300 the
        # block is all afloat or gone, so its grounded area has fallen to zero.
        assert "(−100.0%)" in readout

    def test_the_flowlines_stay_on_and_follow_the_projection(self, projection_page):
        page, errors = projection_page
        today = _show_flowlines(page)
        assert today["following"] is False
        assert today["visibleSegments"] == today["segments"]

        _enable(page)
        assert _state(page)["toggles"]["showFlowline"] is True
        start = _state(page)["iceProjection"]["flowlines"]
        assert start["following"] is True
        # Only the segments over the synthetic block carry projected ice.
        assert 0 < start["visibleSegments"] < start["segments"]

        end = _set_year(page, 2300)["flowlines"]
        assert end["followedYear"] == 2300
        assert end["visibleSegments"] < start["visibleSegments"]  # the western half is gone
        assert end["meanRate"] > start["meanRate"]  # the rest flows faster

        page.locator("#showIceProjection").uncheck()
        page.wait_for_function(
            "() => !JSON.parse(window.render_game_to_text()).iceProjection.active", timeout=10_000
        )
        assert _state(page)["toggles"]["showSea"] is False  # raised by the projection, lowered with it
        restored = _state(page)["iceProjection"]["flowlines"]
        assert restored["following"] is False
        assert restored["visibleSegments"] == restored["segments"]
        assert restored["meanRate"] == pytest.approx(today["meanRate"], abs=1e-3)
        assert errors == []

    def test_the_projection_has_its_own_flowline_switch(self, projection_page):
        page, errors = projection_page
        switch = page.locator("#projectionFlowline")
        assert switch.is_hidden()  # it lives in the projection's controls
        assert _state(page)["toggles"]["showFlowline"] is False  # the page opens without flowlines
        _enable(page)
        # The projection brings them with it.
        assert switch.is_visible() and switch.is_checked()
        _wait_for_flowlines(page)
        state = _state(page)
        # One layer, one state: the switch is the flowline toggle itself.
        assert state["toggles"]["showFlowline"] is True
        assert state["iceProjection"]["flowlines"]["visible"] is True
        assert state["iceProjection"]["flowlines"]["following"] is True

        switch.uncheck()
        state = _state(page)
        assert state["toggles"]["showFlowline"] is False
        assert state["iceProjection"]["flowlines"]["visible"] is False

        # ...and follows it when the layer is switched from the main list.
        page.locator("#showFlowline").check()
        assert switch.is_checked() is True
        assert _state(page)["iceProjection"]["flowlines"]["visible"] is True

        # A choice made by hand outlasts the projection.
        _disable(page)
        assert _state(page)["toggles"]["showFlowline"] is True
        assert errors == []

    def test_closing_the_projection_takes_its_flowlines_with_it(self, projection_page):
        page, errors = projection_page
        _enable(page)
        _wait_for_flowlines(page)
        _disable(page)
        state = _state(page)
        assert state["toggles"]["showFlowline"] is False
        assert state["iceProjection"]["flowlines"]["visible"] is False
        assert page.locator("#projectionFlowline").is_checked() is False
        assert errors == []

    def test_a_scenario_change_keeps_a_flowline_choice(self, playwright_browser, explorer_url, synthetic_projection):
        # Serve the synthetic package for the low-emissions scenario too, so the switch is cheap.
        routes = _synthetic_routes(synthetic_projection) + _synthetic_routes(synthetic_projection, "ae10")
        context, page, errors = _open(playwright_browser, explorer_url, routes)
        try:
            _enable(page)
            page.locator("#projectionFlowline").uncheck()
            page.locator("#projectionScenario").select_option("ae10")
            page.wait_for_function(
                "() => { const p = JSON.parse(window.render_game_to_text()).iceProjection;"
                " return p.active && p.scenario === 'ae10'; }",
                timeout=READY_TIMEOUT_MS,
            )
            # Switching scenario re-activates the projection; it must not bring the flowlines back.
            assert page.locator("#projectionFlowline").is_checked() is False
            assert _state(page)["toggles"]["showFlowline"] is False
            assert errors == []
        finally:
            context.close()

    def test_playback_runs_to_the_last_year_and_stops(self, projection_page):
        page, _ = projection_page
        _enable(page)
        page.locator("#projectionPlay").click()
        assert _state(page)["iceProjection"]["playing"] is True
        # Fast-forward the runtime clock rather than waiting ~24 s of wall time.
        page.evaluate("window.advanceTime(30000)")
        projection = _state(page)["iceProjection"]
        assert projection["year"] == 2300
        assert projection["playing"] is False
        assert page.locator("#projectionPlay").inner_text() == "Play"

    def test_a_present_day_overlay_hands_the_scene_back(self, projection_page):
        page, _ = projection_page
        _enable(page)
        page.locator("#showVelocity").check()
        page.wait_for_function(
            "() => !JSON.parse(window.render_game_to_text()).iceProjection.active", timeout=10_000
        )
        state = _state(page)
        assert state["toggles"]["showIceProjection"] is False
        assert state["meshes"]["iceProjection"] is False
        assert state["meshes"]["ice"] is True
        assert state["legends"]["iceProjection"] is False

    def test_enabling_rebound_switches_the_projection_off(self, projection_page):
        page, _ = projection_page
        _enable(page)
        page.locator("#showIsostaticRebound").check()
        page.wait_for_function(
            "() => !JSON.parse(window.render_game_to_text()).iceProjection.active", timeout=10_000
        )
        assert _state(page)["toggles"]["showIceProjection"] is False

    def test_enabling_the_projection_over_a_rebounded_bed_restores_it(self, projection_page):
        page, _ = projection_page
        page.locator("#showIsostaticRebound").check()
        page.wait_for_function(
            """() => {
                const s = JSON.parse(window.render_game_to_text());
                return s.ready && s.meshes.isostaticRebound && s.isostaticRebound.solved;
            }""",
            timeout=REBOUND_TIMEOUT_MS,
        )
        projection = _enable(page)
        assert _state(page)["toggles"]["showIsostaticRebound"] is False
        # The rebound rewrote every bed vertex; the projection replaces only its domain, so
        # the rest of the bed must be back at the dataset's own heights.
        assert projection["bedMatchesDatasetOutsideDomain"] is True

    def test_switching_the_colour_mode_keeps_the_geometry(self, projection_page):
        page, _ = projection_page
        before = _enable(page)
        page.locator("#projectionColorMode").select_option("type")
        after = _state(page)
        assert after["iceProjection"]["colorMode"] == "type"
        assert after["iceProjection"]["iceCells"] == before["iceCells"]
        assert after["legends"]["iceProjection"] is False


class TestPublicEditionProjection:
    def test_the_public_page_plays_the_projection_with_its_flowlines(
        self, playwright_browser, explore_url, synthetic_projection
    ):
        context, page, errors = _open(playwright_browser, explore_url, _synthetic_routes(synthetic_projection))
        try:
            projection = _state(page)["iceProjection"]
            assert projection["available"] is True
            assert page.locator("#iceProjectionRow").is_visible()
            assert _state(page)["toggles"]["showFlowline"] is False
            projection = _enable(page)
            assert projection["colorMode"] == "change"  # the page has no colour-mode picker
            # Ice flow comes on with the projection.
            assert _wait_for_flowlines(page)["following"] is True
            end = _set_year(page, 2300)
            assert end["iceCells"] == synthetic_projection["cells"] - synthetic_projection["west_cells"]
            assert page.locator("#projectionLegendSection").is_visible()

            # The section's own switch mirrors the "Ice flow" layer.
            switch = page.locator("#projectionFlowline")
            assert switch.is_visible() and switch.is_checked()
            switch.uncheck()
            assert page.locator("#showFlowline").is_checked() is False
            assert _state(page)["iceProjection"]["flowlines"]["visible"] is False
            assert errors == []
        finally:
            context.close()
