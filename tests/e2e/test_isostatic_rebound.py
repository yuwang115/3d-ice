"""E2E tests: the isostatic-rebound layer loads or solves in the browser and reads back correctly.

Controls are located by `#id` rather than by label so the same assertions work on the
Chinese page. The default Earth response is the published one of Paxman et al. (2022),
checked against the figures the worked example pins (examples/isostatic-rebound.mjs); the
idealised ELRA and Airy responses are checked against values verified offline against
BedMachine v4 and BedMachine Greenland v6.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.e2e

SOLVE_TIMEOUT_MS = 90_000
READY_TIMEOUT_MS = 60_000

# Navigation waits on domcontentloaded rather than on the network going quiet. The explorer
# pulls Google Fonts and an analytics tag, and wherever those are blocked or slow the
# network never settles, so a network-idle wait times out on a page that is in fact
# perfectly usable. Readiness comes from the application's own signal instead.


@pytest.fixture
def page(playwright_browser, explorer_url):
    """A fresh page on the English explorer, using the session-scoped browser."""
    context = playwright_browser.new_context(viewport={"width": 1280, "height": 800})
    try:
        page = context.new_page()
        page.goto(explorer_url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => window.render_game_to_text && JSON.parse(window.render_game_to_text()).ready",
            timeout=READY_TIMEOUT_MS,
        )
        yield page
    finally:
        context.close()


def _state(page) -> dict:
    return json.loads(page.evaluate("window.render_game_to_text()"))


def _enable_rebound(page) -> dict:
    page.locator("#showIsostaticRebound").check()
    page.wait_for_function(
        """() => {
            const s = JSON.parse(window.render_game_to_text());
            return s.ready && s.meshes.isostaticRebound
                && s.isostaticRebound.solved && !s.isostaticRebound.pendingResolve;
        }""",
        timeout=SOLVE_TIMEOUT_MS,
    )
    return _state(page)


def _select_model(page, model: str) -> dict:
    page.locator("#reboundModel").select_option(model)
    page.wait_for_function(
        f"""() => {{
            const s = JSON.parse(window.render_game_to_text()).isostaticRebound;
            return s.solved && s.solvedModel === '{model}' && !s.pendingResolve;
        }}""",
        timeout=SOLVE_TIMEOUT_MS,
    )
    return _state(page)["isostaticRebound"]


def _select_dataset(page, dataset: str) -> None:
    """Switch terrain in the panel; the URL's `preset` names a view, not a dataset."""
    page.locator("#resolutionPreset").select_option(dataset)
    page.wait_for_function(
        f"() => {{ const s = JSON.parse(window.render_game_to_text()); return s.ready && s.dataset === '{dataset}'; }}",
        timeout=READY_TIMEOUT_MS,
    )


class TestIsostaticRebound:
    def test_the_layer_is_offered_and_starts_disabled(self, page):
        toggle = page.locator("#showIsostaticRebound")
        assert not toggle.is_disabled()
        assert not toggle.is_checked()
        assert page.locator("#isostaticReboundControls").is_hidden()

        rebound = _state(page)["isostaticRebound"]
        assert rebound["available"] is True
        assert rebound["enabled"] is False
        assert rebound["solved"] is False

    def test_enabling_loads_the_published_response_for_antarctica(self, page):
        rebound = _enable_rebound(page)["isostaticRebound"]

        assert rebound["enabled"] is True
        assert rebound["model"] == "paxman2022"
        assert rebound["solvedModel"] == "paxman2022"
        assert rebound["responseDoi"] == "10.18739/A22Z12R8C"
        # Defaults land straight on "ice gone, rebound complete".
        assert rebound["progressPercent"] == 100
        assert rebound["seaLevelMeters"] == 0

        # The pinned figures of the worked example: 1028.6 m of solid-surface uplift,
        # 940.6 m above the ice-free sea surface, 2.94 million km^2 newly emergent.
        assert abs(rebound["maxUpliftMeters"] - 1028.6) < 0.2
        assert abs(rebound["maxTopographyChangeMeters"] - 940.6) < 0.2
        assert abs(rebound["emergentAreaKm2"] - 2_941_863) < 5
        # Ice above flotation, close to the published 57.9 m for this product.
        assert 54 < rebound["seaLevelEquivalentMeters"] < 59

    def test_the_published_response_fixes_the_datum_and_cites_its_grids(self, page):
        rebound = _enable_rebound(page)["isostaticRebound"]

        assert rebound["seaLevelControlDisabled"] is True
        assert page.locator("#reboundSeaLevel").is_disabled()
        assert "Paxman" in page.locator("#reboundModelNote").inner_text()
        assert "ice-free sea surface" in page.locator("#reboundSeaLevelNote").inner_text()
        assert page.locator('a.meta-link[href="https://doi.org/10.18739/A22Z12R8C"]').count() >= 1

    def test_switching_to_regional_flexure_re_enables_the_datum_and_solves(self, page):
        published = _enable_rebound(page)["isostaticRebound"]

        flexural = _select_model(page, "flexural")

        assert flexural["seaLevelControlDisabled"] is False
        assert flexural["responseDoi"] is None
        # The idealised solve keeps its own documented figures (docs/example.md).
        assert abs(flexural["maxUpliftMeters"] - 1026.8) < 0.2
        # Without the eustatic rise or the post-LGM term it emerges more land.
        assert flexural["emergentAreaKm2"] > published["emergentAreaKm2"]

        restored = _select_model(page, "paxman2022")
        assert restored["emergentAreaKm2"] == published["emergentAreaKm2"]

    def test_enabling_reveals_the_controls_and_the_sea_surface(self, page):
        _enable_rebound(page)
        assert page.locator("#isostaticReboundControls").is_visible()
        # Uplift is unreadable without a waterline, so the sea plane is switched on.
        assert page.locator("#showSea").is_checked()
        assert page.locator("#reboundProgressNote").inner_text().strip() != ""
        assert page.locator("#reboundSeaLevelNote").inner_text().strip() != ""

    def test_enabling_clears_overlays_baked_onto_the_present_day_surface(self, page):
        page.locator("#showVelocity").check()
        page.wait_for_function(
            "() => JSON.parse(window.render_game_to_text()).meshes.velocity",
            timeout=SOLVE_TIMEOUT_MS,
        )
        _enable_rebound(page)

        state = _state(page)
        assert state["toggles"]["showVelocity"] is False
        assert state["meshes"]["velocity"] is False

    def test_the_local_model_gives_a_higher_peak_than_regional_flexure(self, page):
        _enable_rebound(page)
        flexural = _select_model(page, "flexural")
        local = _select_model(page, "local")

        # Airy isostasy has no lithospheric strength to spread the load, so it is the
        # upper bound on peak uplift and emerges more land.
        assert local["maxUpliftMeters"] > flexural["maxUpliftMeters"]
        assert local["emergentAreaKm2"] > flexural["emergentAreaKm2"]

    def test_raising_the_sea_level_datum_emerges_less_land(self, page):
        _enable_rebound(page)
        present = _select_model(page, "flexural")

        page.locator("#reboundSeaLevel").fill("57")
        page.locator("#reboundSeaLevel").dispatch_event("change")
        page.wait_for_function(
            """() => {
                const s = JSON.parse(window.render_game_to_text()).isostaticRebound;
                return s.solved && s.solvedSeaLevelMeters === 57 && !s.pendingResolve;
            }""",
            timeout=SOLVE_TIMEOUT_MS,
        )
        raised = _state(page)["isostaticRebound"]

        assert raised["emergentAreaKm2"] < present["emergentAreaKm2"]

    def test_the_progress_slider_reports_scenario_state(self, page):
        _enable_rebound(page)

        page.locator("#reboundProgress").fill("0")
        page.locator("#reboundProgress").dispatch_event("change")
        page.wait_for_function(
            "() => JSON.parse(window.render_game_to_text()).isostaticRebound.progressPercent === 0",
            timeout=10_000,
        )
        assert page.locator("#reboundProgressValue").inner_text().strip() == "0%"
        # The equilibrium solve is cached, so the statistics survive a scrub to zero.
        assert _state(page)["isostaticRebound"]["solved"] is True

        page.locator("#reboundProgress").fill("100")
        page.locator("#reboundProgress").dispatch_event("change")
        page.wait_for_function(
            "() => JSON.parse(window.render_game_to_text()).isostaticRebound.progressPercent === 100",
            timeout=10_000,
        )
        assert page.locator("#reboundProgressValue").inner_text().strip() == "100%"

    def test_the_emergent_land_highlight_can_be_turned_off(self, page):
        _enable_rebound(page)
        highlight = page.locator("#highlightEmergentLand")
        assert highlight.is_checked()

        highlight.uncheck()
        page.wait_for_function(
            "() => !JSON.parse(window.render_game_to_text()).isostaticRebound.highlightEmergentLand",
            timeout=10_000,
        )
        # Turning the tint off must not disturb the solved field.
        assert _state(page)["isostaticRebound"]["solved"] is True

    def test_disabling_restores_the_present_day_bed(self, page):
        _enable_rebound(page)
        page.locator("#showIsostaticRebound").uncheck()
        page.wait_for_function(
            "() => !JSON.parse(window.render_game_to_text()).meshes.isostaticRebound",
            timeout=10_000,
        )

        state = _state(page)
        assert state["toggles"]["showIsostaticRebound"] is False
        assert state["meshes"]["bed"] is True
        assert page.locator("#isostaticReboundControls").is_hidden()

    def test_selecting_a_bed_draped_overlay_clears_the_rebound_layer(self, page):
        _enable_rebound(page)
        page.locator("#showBasalFriction").check()
        page.wait_for_function(
            "() => !JSON.parse(window.render_game_to_text()).toggles.showIsostaticRebound",
            timeout=SOLVE_TIMEOUT_MS,
        )
        assert _state(page)["meshes"]["isostaticRebound"] is False

    def test_changing_the_model_mid_solve_still_lands_on_the_new_model(self, page):
        """A superseded in-flight solve must not leave the display on the old settings."""
        page.locator("#showIsostaticRebound").check()
        # Switch model immediately, while the first solve is still running.
        page.locator("#reboundModel").select_option("local")
        page.wait_for_function(
            """() => {
                const s = JSON.parse(window.render_game_to_text()).isostaticRebound;
                return s.solved && s.solvedModel === 'local' && !s.pendingResolve;
            }""",
            timeout=SOLVE_TIMEOUT_MS,
        )
        rebound = _state(page)["isostaticRebound"]
        assert rebound["model"] == "local"
        # Airy isostasy over BedMachine Antarctica v4 peaks above the flexural solution.
        assert rebound["maxUpliftMeters"] > 1100

    def test_the_layer_re_solves_for_greenland(self, page):
        _enable_rebound(page)
        page.locator("#regionPreset").select_option("greenland")
        page.wait_for_function(
            """() => {
                const s = JSON.parse(window.render_game_to_text());
                return s.ready && s.region === 'greenland'
                    && s.isostaticRebound.solved && !s.isostaticRebound.pendingResolve;
            }""",
            timeout=SOLVE_TIMEOUT_MS,
        )
        rebound = _state(page)["isostaticRebound"]

        # The published response for BedMachine Greenland v6: 829.0 m of solid-surface
        # uplift and 0.39 million km^2 newly emergent, over ice holding ~7.3 m of sea level.
        assert rebound["solvedModel"] == "paxman2022"
        assert abs(rebound["maxUpliftMeters"] - 829.0) < 0.2
        assert abs(rebound["emergentAreaKm2"] - 394_199) < 5
        assert 6.5 < rebound["seaLevelEquivalentMeters"] < 8.0

    def test_the_published_response_loads_for_bedmap3(self, page):
        _select_dataset(page, "bedmap3")
        rebound = _enable_rebound(page)["isostaticRebound"]

        assert _state(page)["dataset"] == "bedmap3"
        assert rebound["solvedModel"] == "paxman2022"
        # Bedmap3's own grid file: 1030.8 m of peak solid-surface uplift.
        assert abs(rebound["maxUpliftMeters"] - 1030.8) < 0.2

    def test_qrf_borrows_the_bedmachine_response_and_says_so(self, page):
        page.locator("#regionPreset").select_option("greenland")
        page.wait_for_function(
            "() => { const s = JSON.parse(window.render_game_to_text()); return s.ready && s.region === 'greenland'; }",
            timeout=READY_TIMEOUT_MS,
        )
        _select_dataset(page, "qrf")
        rebound = _enable_rebound(page)["isostaticRebound"]

        assert rebound["solvedModel"] == "paxman2022"
        assert abs(rebound["maxUpliftMeters"] - 829.0) < 0.2
        assert "BedMachine Greenland v6" in page.locator("#reboundModelNote").inner_text()

    def test_the_layer_works_on_the_chinese_explorer(self, playwright_browser, server):
        context = playwright_browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()
        try:
            page.goto(
                f"{server}/zh/tools/3D-interactive-cryosphere-explorer.html",
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            page.wait_for_function(
                "() => window.render_game_to_text"
                " && JSON.parse(window.render_game_to_text()).ready",
                timeout=READY_TIMEOUT_MS,
            )
            rebound = _enable_rebound(page)["isostaticRebound"]
            assert rebound["solved"] is True
            assert 900 < rebound["maxUpliftMeters"] < 1150

            # The notes must render translated copy, not raw key paths.
            note = page.locator("#reboundProgressNote").inner_text()
            assert "explorer.rebound" not in note
            assert any("一" <= ch <= "鿿" for ch in note), note
        finally:
            context.close()

    def test_solving_the_rebound_raises_no_page_errors(self, playwright_browser, explorer_url):
        context = playwright_browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        try:
            page.goto(explorer_url, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_function(
                "() => window.render_game_to_text"
                " && JSON.parse(window.render_game_to_text()).ready",
                timeout=READY_TIMEOUT_MS,
            )
            _enable_rebound(page)
            page.locator("#reboundProgress").fill("40")
            page.locator("#reboundProgress").dispatch_event("change")
            page.wait_for_timeout(500)
            assert errors == [], f"page errors: {errors}"
        finally:
            context.close()
