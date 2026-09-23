"""E2E tests: verify the 3D ICE explorer loads and initializes correctly."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


@pytest.fixture
def page(playwright_browser, explorer_url):
    """Create a fresh browser page navigated to the explorer."""
    context = playwright_browser.new_context(
        viewport={"width": 1280, "height": 800},
    )
    try:
        page = context.new_page()
        page.goto(explorer_url, wait_until="networkidle", timeout=30_000)
        yield page
    finally:
        context.close()


@pytest.fixture
def playwright_browser(request):
    """Launch a headless Chromium browser."""
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    try:
        browser = pw.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()
    finally:
        pw.stop()


class TestExplorerLoads:
    def test_page_loads_without_errors(self, playwright_browser, explorer_url):
        context = playwright_browser.new_context(viewport={"width": 1280, "height": 800})
        try:
            page = context.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda err: errors.append(str(err)))
            page.goto(explorer_url, wait_until="networkidle", timeout=30_000)
            assert len(errors) == 0, f"Page errors: {errors}"
        finally:
            context.close()

    def test_canvas_exists(self, page):
        canvas = page.locator("#viewer canvas")
        canvas.wait_for(state="visible", timeout=30_000)
        box = canvas.bounding_box()
        assert box is not None
        assert box["width"] > 0
        assert box["height"] > 0

    def test_flow_animation_can_be_toggled(self, page):
        """Users can pause the directional ice and ocean flow-light effect."""
        flow_animation = page.locator("#animateFlow")
        assert flow_animation.is_checked()

        initial_state = page.evaluate("JSON.parse(window.render_game_to_text())")
        assert initial_state["flowAnimation"]["enabled"]

        flow_animation.uncheck()
        page.wait_for_function(
            """() => !JSON.parse(window.render_game_to_text()).flowAnimation.enabled""",
            timeout=5_000,
        )

    def test_enabling_flow_animation_reveals_ice_flowlines_when_no_flow_layer_is_visible(self, page):
        """The animation control avoids an invisible-on-first-use state."""
        flow_animation = page.locator("#animateFlow")
        assert not page.locator("#showFlowline").is_checked()
        assert not page.locator("#showOceanCurrents").is_checked()

        flow_animation.uncheck()
        flow_animation.check()

        page.wait_for_function(
            """() => document.querySelector('#showFlowline').checked""",
            timeout=30_000,
        )

    def test_flow_animation_time_advances_while_enabled(self, page):
        """The visible flow-light phase advances rather than remaining a static texture."""
        initial_state = page.evaluate("JSON.parse(window.render_game_to_text())")
        initial_time = initial_state["flowAnimation"]["timeSeconds"]

        page.wait_for_timeout(300)
        updated_state = page.evaluate("JSON.parse(window.render_game_to_text())")
        assert updated_state["flowAnimation"]["timeSeconds"] > initial_time

    def test_reduced_motion_disables_flow_animation(self, playwright_browser, explorer_url):
        """The explorer starts static for people who request reduced motion."""
        context = playwright_browser.new_context(reduced_motion="reduce")
        try:
            page = context.new_page()
            page.goto(explorer_url, wait_until="networkidle", timeout=30_000)

            flow_animation = page.locator("#animateFlow")
            assert not flow_animation.is_checked()
            state = page.evaluate("JSON.parse(window.render_game_to_text())")
            assert state["flowAnimation"] == {
                "enabled": False,
                "reducedMotion": True,
                "timeSeconds": 0,
                "iceReady": False,
                "oceanReady": False,
                "iceParticles": False,
                "oceanParticles": False,
            }
        finally:
            context.close()

    def test_flowlines_rebuild_when_animation_is_reenabled_after_reduced_motion(self, page):
        """A static flowline layer upgrades to a shader flow-light layer on demand."""
        page.emulate_media(reduced_motion="reduce")
        page.wait_for_function(
            """() => JSON.parse(window.render_game_to_text()).flowAnimation.reducedMotion""",
            timeout=5_000,
        )
        page.locator("#showFlowline").check()
        page.wait_for_function(
            """() => {
                const state = JSON.parse(window.render_game_to_text());
                return state.ready && state.meshes.flowline && !state.flowAnimation.iceReady;
            }""",
            timeout=90_000,
        )

        page.emulate_media(reduced_motion="no-preference")
        page.locator("#animateFlow").check()
        page.wait_for_function(
            """() => {
                const state = JSON.parse(window.render_game_to_text());
                return state.flowAnimation.enabled &&
                  state.flowAnimation.iceReady &&
                  state.flowAnimation.iceParticles;
            }""",
            timeout=30_000,
        )

    def test_title_is_set(self, page):
        title = page.title()
        assert "3D" in title or "ICE" in title or "Cryosphere" in title

    def test_can_switch_to_the_bedmap3_terrain_package(self, page):
        page.select_option("#resolutionPreset", "bedmap3")
        page.wait_for_function(
            """() => {
                const state = JSON.parse(window.render_game_to_text());
                return state.ready && state.dataset === 'bedmap3';
            }""",
            timeout=30_000,
        )

        state = page.evaluate("JSON.parse(window.render_game_to_text())")
        assert state["grid"]["nx"] == 667
        assert state["grid"]["ny"] == 667
        assert state["grid"]["dx_m"] == 10_000
        assert state["grid"]["dy_m"] == -10_000
        for control in (
            "#showVelocity",
            "#showFlowline",
            "#showBasalFriction",
            "#showEffectivePressure",
            "#showSubglacialChannels",
            "#showOceanCurrents",
        ):
            assert not page.locator(control).is_disabled(), control
        assert "Bedmap3" in (page.text_content("#metaList") or "")

    def test_can_switch_to_the_bedmap3_hd_terrain_package(self, page):
        page.select_option("#resolutionPreset", "bedmap3-hd")
        page.wait_for_function(
            """() => {
                const state = JSON.parse(window.render_game_to_text());
                return state.ready && state.dataset === 'bedmap3-hd';
            }""",
            timeout=30_000,
        )

        state = page.evaluate("JSON.parse(window.render_game_to_text())")
        assert state["grid"]["nx"] == 1667
        assert state["grid"]["ny"] == 1667
        assert state["grid"]["dx_m"] == 4000
        assert state["grid"]["dy_m"] == -4000
        for control in (
            "#showVelocity",
            "#showFlowline",
            "#showBasalFriction",
            "#showEffectivePressure",
            "#showSubglacialChannels",
            "#showOceanCurrents",
        ):
            assert not page.locator(control).is_disabled(), control

    def test_bedmap3_balanced_loads_each_enabled_overlay(self, page):
        page.select_option("#resolutionPreset", "bedmap3")
        page.wait_for_function(
            """() => {
                const state = JSON.parse(window.render_game_to_text());
                return state.ready && state.dataset === 'bedmap3';
            }""",
            timeout=30_000,
        )

        for control, mesh in (
            ("#showVelocity", "velocity"),
            ("#showFlowline", "flowline"),
            ("#showBasalFriction", "basalFriction"),
            ("#showEffectivePressure", "hydrology"),
            ("#showSubglacialChannels", "hydrology"),
            ("#showOceanCurrents", "oceanCurrents"),
        ):
            page.locator(control).check()
            page.wait_for_function(
                f"""() => {{
                    const state = JSON.parse(window.render_game_to_text());
                    return state.ready && state.meshes.{mesh};
                }}""",
                timeout=90_000,
            )

        state = page.evaluate("JSON.parse(window.render_game_to_text())")
        assert state["flowAnimation"]["iceReady"]
        assert state["flowAnimation"]["oceanReady"]
        assert state["flowAnimation"]["iceParticles"]
        assert state["flowAnimation"]["oceanParticles"]

    @pytest.mark.parametrize(
        ("dataset", "nx", "ny", "spacing"),
        (("qrf", 511, 918, 3000), ("qrf-hd", 1533, 2752, 1000)),
    )
    def test_can_switch_to_the_qrf_greenland_terrain_package(self, page, dataset, nx, ny, spacing):
        page.select_option("#regionPreset", "greenland")
        page.select_option("#resolutionPreset", dataset)
        timeout = 90_000 if dataset == "qrf-hd" else 30_000
        page.wait_for_function(
            f"""() => {{
                const state = JSON.parse(window.render_game_to_text());
                return state.ready && state.region === 'greenland' && state.dataset === '{dataset}';
            }}""",
            timeout=timeout,
        )

        state = page.evaluate("JSON.parse(window.render_game_to_text())")
        assert state["grid"]["nx"] == nx
        assert state["grid"]["ny"] == ny
        assert state["grid"]["dx_m"] == spacing
        assert state["grid"]["dy_m"] == -spacing
        meta_text = page.text_content("#metaList") or ""
        assert "QRF 2025" in meta_text
        assert "QRF bed for" in meta_text

    def test_qrf_balanced_loads_the_reused_greenland_overlays(self, page):
        page.select_option("#regionPreset", "greenland")
        page.select_option("#resolutionPreset", "qrf")
        page.wait_for_function(
            """() => {
                const state = JSON.parse(window.render_game_to_text());
                return state.ready && state.region === 'greenland' && state.dataset === 'qrf';
            }""",
            timeout=30_000,
        )

        for control, mesh in (
            ("#showVelocity", "velocity"),
            ("#showFlowline", "flowline"),
            ("#showBasalFriction", "basalFriction"),
            ("#showOceanCurrents", "oceanCurrents"),
        ):
            assert not page.locator(control).is_disabled(), control
            page.locator(control).check()
            page.wait_for_function(
                f"""() => {{
                    const state = JSON.parse(window.render_game_to_text());
                    return state.ready && state.meshes.{mesh};
                }}""",
                timeout=90_000,
            )


class TestHomePage:
    def test_home_loads(self, playwright_browser, home_url):
        context = playwright_browser.new_context()
        try:
            page = context.new_page()
            response = page.goto(home_url, wait_until="domcontentloaded", timeout=15_000)
            assert response is not None
            assert response.status == 200
        finally:
            context.close()

    def test_mit_license_in_footer(self, playwright_browser, home_url):
        context = playwright_browser.new_context()
        try:
            page = context.new_page()
            page.goto(home_url, wait_until="domcontentloaded", timeout=15_000)
            footer_text = page.text_content("footer") or ""
            assert "MIT" in footer_text
        finally:
            context.close()
