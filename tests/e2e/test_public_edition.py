"""E2E tests: the public edition offers only its core layers and plays its guided tour.

The public pages run the same runtime as the research pages with <html data-edition="public">
(static/tools/js/editions.js). These tests check what that restriction guarantees in a real
browser: no research-only package is ever requested, no preset can switch a research layer
on, and the tour and the layer explainers work in both locales.

Most tour tests run with reduced motion, under which camera flights and slider animations
jump straight to their end, so they check state rather than timing.
"""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.e2e

READY_TIMEOUT_MS = 60_000
STOP_TIMEOUT_MS = 120_000

# Packages only the research edition may fetch: its extra layers and its other terrain grids.
RESEARCH_ONLY_PACKAGE = re.compile(
    r"basal_friction|rise_antarctica|hydrology|refined_basins|imbie|greenland_basins|bedmap3|"
    r"_741\.|_1km\.|qrf|cmocean"
)

# What each tour stop leaves switched on, in order (see static/tools/js/explore-content.js).
TOUR_STOPS = [
    ("ice-continent", "antarctica", {"showIce", "showBed"}),
    ("land-beneath", "antarctica", {"showIce", "showBed"}),
    ("rivers-of-ice", "antarctica", {"showIce", "showBed", "showFlowline"}),
    ("floating-ice", "antarctica", {"showIce", "showBed", "showSea"}),
    ("southern-ocean", "antarctica", {"showIce", "showBed", "showOceanCurrents"}),
    ("without-ice", "antarctica", {"showIce", "showBed", "showIsostaticRebound", "showSea"}),
    ("greenland", "greenland", {"showIce", "showBed", "showFlowline"}),
    ("your-turn", "greenland", {"showIce", "showBed"}),
]
PUBLIC_TOGGLES = {
    "showIce",
    "showBed",
    "showFlowline",
    "showOceanCurrents",
    "showSea",
    "showIsostaticRebound",
    "showGeographicNames",
    "showResearchStations",
}


def _state(page) -> dict:
    return json.loads(page.evaluate("window.render_game_to_text()"))


def _wait_ready(page) -> None:
    page.wait_for_function(
        """() => window.render_game_to_text && JSON.parse(window.render_game_to_text()).ready
            && document.documentElement.dataset.guide === 'ready'""",
        timeout=READY_TIMEOUT_MS,
    )


def _open(browser, url: str, *, reduced_motion: str = "reduce", requests: list[str] | None = None):
    context = browser.new_context(viewport={"width": 1280, "height": 800}, reduced_motion=reduced_motion)
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    if requests is not None:
        page.on("request", lambda request: requests.append(request.url))
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    _wait_ready(page)
    return context, page, errors


def _wait_for_stop(page, stop_id: str) -> dict:
    """Wait until a stop has been applied: its card is up and nothing is loading or flying."""
    page.wait_for_function(
        f"""() => {{
            const card = document.getElementById('tourCard');
            const s = JSON.parse(window.render_game_to_text());
            const statusText = card.querySelector('.tour-card__status')?.textContent || '';
            return card.dataset.stop === '{stop_id}' && !card.hidden && s.ready && !s.cameraFlight
                && statusText === '';
        }}""",
        timeout=STOP_TIMEOUT_MS,
    )
    return _state(page)


def _switched_on(state: dict) -> set[str]:
    return {name for name in PUBLIC_TOGGLES if state["toggles"].get(name)}


@pytest.fixture
def public_page(playwright_browser, explore_url):
    context, page, errors = _open(playwright_browser, explore_url)
    try:
        yield page, errors
    finally:
        context.close()


class TestPublicEditionCore:
    def test_loads_the_public_edition_without_errors(self, public_page):
        page, errors = public_page
        state = _state(page)
        assert errors == []
        assert state["edition"] == "public"
        assert (state["region"], state["dataset"]) == ("antarctica", "balanced")
        assert state["status"] == "Ready"

    def test_research_controls_are_not_on_the_page(self, public_page):
        page, _errors = public_page
        for control in (
            "resolutionPreset",
            "showVelocity",
            "showBasalFriction",
            "showBasalMelt",
            "showThermalDriving",
            "showEffectivePressure",
            "showSubglacialChannels",
            "showRefinedBasins",
            "reboundModel",
            "reboundSeaLevel",
            "wireframe",
            "metaList",
        ):
            assert page.locator(f"#{control}").count() == 0, control

    def test_controls_the_page_leaves_out_hold_the_editions_fixed_state(self, public_page):
        page, _errors = public_page
        toggles = _state(page)["toggles"]
        assert toggles["showIceBottom"] and toggles["highlightEmergentLand"]
        for band in ("Surface", "Upper", "Mid", "Lower"):
            assert toggles[f"showOceanLayer{band}"], band
        for research_layer in ("showVelocity", "showBasalFriction", "showEffectivePressure", "showRefinedBasins"):
            assert not toggles[research_layer], research_layer

    def test_a_preset_cannot_switch_research_layers_on(self, playwright_browser, explore_url):
        for preset in ("antarctica-subglacial-features", "antarctica-basin-boundary"):
            context, page, errors = _open(playwright_browser, f"{explore_url}?preset={preset}")
            try:
                state = _state(page)
                assert errors == []
                assert not state["toggles"]["showEffectivePressure"]
                assert not state["toggles"]["showSubglacialChannels"]
                assert not state["toggles"]["showRefinedBasins"]
                assert not state["meshes"]["hydrology"] and not state["meshes"]["basins"]
            finally:
                context.close()

    def test_never_requests_a_research_only_package(self, playwright_browser, explore_url):
        requests: list[str] = []
        context, page, errors = _open(playwright_browser, explore_url, requests=requests)
        try:
            page.locator("#showFlowline").check()
            page.wait_for_function(
                "() => JSON.parse(window.render_game_to_text()).meshes.flowline", timeout=READY_TIMEOUT_MS
            )
            page.locator("#polarSearchInput").fill("Thwaites")
            page.wait_for_function(
                "() => document.getElementById('polarSearchStatus').textContent.trim().length > 0",
                timeout=READY_TIMEOUT_MS,
            )
            assert errors == []
            data_requests = [url for url in requests if "/tools/data/" in url]
            assert any("bedmachine_antarctica_v4_480" in url for url in data_requests)
            offending = [url for url in data_requests if RESEARCH_ONLY_PACKAGE.search(url)]
            assert offending == []
        finally:
            context.close()

    def test_an_explainer_opens_and_closes(self, public_page):
        page, _errors = public_page
        button = page.locator('button[data-info="iceFlow"]')
        card = page.locator('[data-info-card="iceFlow"]')
        assert button.get_attribute("aria-expanded") == "false"
        assert button.get_attribute("aria-label") == "About Ice flow"
        button.click()
        assert button.get_attribute("aria-expanded") == "true"
        assert card.is_visible()
        assert "satellites" in card.inner_text()
        assert card.locator("a[href^='https://']").count() == 2
        card.locator(".info-card__close").click()
        assert card.is_hidden()
        assert page.evaluate("document.activeElement.dataset.info") == "iceFlow"


class TestGuidedTour:
    def test_every_stop_sets_its_region_and_layers(self, playwright_browser, explore_url):
        requests: list[str] = []
        context, page, errors = _open(playwright_browser, explore_url, requests=requests)
        try:
            self._walk_the_tour(page, errors, requests)
        finally:
            context.close()

    @staticmethod
    def _walk_the_tour(page, errors, requests) -> None:
        page.locator("#tourStartButton").click()
        assert page.locator("#tourCard").get_attribute("aria-label") == "Guided tour"
        for index, (stop_id, region, switched_on) in enumerate(TOUR_STOPS):
            if index > 0:
                page.locator(".tour-card__nav--primary").click()
            state = _wait_for_stop(page, stop_id)
            assert state["region"] == region, stop_id
            assert _switched_on(state) == switched_on, stop_id
            assert page.locator(".tour-card__counter").inner_text() == f"Stop {index + 1} of {len(TOUR_STOPS)}"
            if stop_id == "land-beneath":
                assert float(page.locator("#iceOpacity").input_value()) == pytest.approx(0.06)
            if stop_id == "rivers-of-ice":
                assert state["meshes"]["flowline"] and state["legends"]["velocity"]
            if stop_id == "southern-ocean":
                assert state["meshes"]["oceanCurrents"] and state["legends"]["oceanCurrents"]
            if stop_id == "without-ice":
                # The stop's slider animation runs after the card settles.
                _wait_rebound_complete(page)
                rebound = _state(page)["isostaticRebound"]
                assert rebound["solved"] and rebound["model"] == "paxman2022"
                assert page.locator("[data-rebound-legend]").is_visible()
        assert errors == []
        offending = [url for url in requests if "/tools/data/" in url and RESEARCH_ONLY_PACKAGE.search(url)]
        assert offending == [], "the whole tour stays on public-edition packages"

        page.locator(".tour-card__nav--primary").click()  # Finish
        assert page.locator("#tourCard").is_hidden()
        assert page.locator("#tourStartButton").inner_text() == "Start the tour"

    def test_keyboard_moves_between_stops_and_closes_the_tour(self, public_page):
        page, _errors = public_page
        page.locator("#tourStartButton").click()
        _wait_for_stop(page, "ice-continent")
        page.keyboard.press("ArrowRight")
        _wait_for_stop(page, "land-beneath")
        page.keyboard.press("ArrowLeft")
        _wait_for_stop(page, "ice-continent")
        page.keyboard.press("ArrowRight")
        _wait_for_stop(page, "land-beneath")
        # A held key's auto-repeat does not race through the stops.
        page.evaluate("window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', repeat: true }))")
        assert page.locator("#tourCard").get_attribute("data-stop") == "land-beneath"
        page.keyboard.press("Escape")
        assert page.locator("#tourCard").is_hidden()
        assert page.locator("#tourStartButton").inner_text() == "Resume the tour"
        assert page.evaluate("document.activeElement.id") == "tourStartButton"
        # Focus goes back to whichever button opened the tour.
        page.locator("#viewerTourButton").click()
        _wait_for_stop(page, "land-beneath")
        page.keyboard.press("Escape")
        assert page.evaluate("document.activeElement.id") == "viewerTourButton"

    def test_rapid_navigation_settles_on_the_last_stop_chosen(self, public_page):
        page, errors = public_page
        page.locator("#tourStartButton").click()
        _wait_for_stop(page, "ice-continent")
        dots = page.locator(".tour-card__stop")
        for stop in (5, 0, 6, 2):
            dots.nth(stop).click()
            page.wait_for_timeout(40)
        state = _wait_for_stop(page, "rivers-of-ice")
        assert state["region"] == "antarctica"
        assert _switched_on(state) == {"showIce", "showBed", "showFlowline"}
        assert not state["isostaticRebound"]["enabled"]
        assert errors == []

    def test_while_the_tour_plays_the_panel_shows_the_card_and_legends(self, public_page):
        page, _errors = public_page
        page.locator("#tourStartButton").click()
        _wait_for_stop(page, "ice-continent")
        assert page.locator("#tourCard").is_visible()
        assert page.locator("#viewControlsSection").is_hidden()
        assert page.locator("#bedLegendSection").is_visible()
        page.locator(".tour-card__header-buttons button").first.click()  # collapse
        assert page.locator(".tour-card__title").is_hidden()
        page.locator(".tour-card__header-buttons button").first.click()  # expand
        assert page.locator(".tour-card__title").is_visible()

    def test_the_camera_flies_between_stops(self, playwright_browser, explore_url):
        context, page, _errors = _open(playwright_browser, explore_url, reduced_motion="no-preference")
        try:
            page.locator("#tourStartButton").click()
            before = _wait_for_stop(page, "ice-continent")["camera"]
            # The Ross Ice Shelf stop needs no new layer data, so this times the flight alone.
            page.locator(".tour-card__stop").nth(3).click()
            page.wait_for_function(
                "() => JSON.parse(window.render_game_to_text()).cameraFlight", timeout=READY_TIMEOUT_MS
            )
            after = _wait_for_stop(page, "floating-ice")["camera"]
            # The 180th meridian runs along +z, so the view moved out over the Ross Sea side.
            assert after["target"]["z"] > before["target"]["z"] + 10, "the view moved over the Ross Ice Shelf"
        finally:
            context.close()

    def test_a_url_can_open_the_tour_at_a_stop(self, playwright_browser, explore_url):
        context, page, _errors = _open(playwright_browser, f"{explore_url}?tour=greenland")
        try:
            state = _wait_for_stop(page, "greenland")
            assert state["region"] == "greenland"
        finally:
            context.close()


SOLVE_TIMEOUT_MS = 90_000

# Times the melt by the rebound slider's own events, from the first frame of the run to the
# change that commits 100 %, on the frame clock the animation itself runs on. The run's
# first frames still read 0 % (the slider moves in 0.5 % steps); the reset to 0 % before
# the run is a committed change, which is how the probe tells the two apart.
# Sampling frames instead misses up to a frame at each end, and the software renderer on a
# CI runner can take a second over one, so the probe also reports the longest frame it saw.
# Each frame it notes whether the runtime showed today's ice (0 %) before the melt and
# part-melted ice on the way.
INSTALL_MELT_PROBE = """() => {
    const slider = document.getElementById("reboundProgress");
    const frameTime = () => document.timeline.currentTime;
    const probe = {
        sawToday: false, sawPartMelted: false, rose: false,
        start: null, seconds: null, lastFrame: null, longestFrameMs: 0,
    };
    slider.addEventListener("input", () => {
        if (Number(slider.value) > 0) probe.rose = true;
        else if (probe.start === null) probe.start = frameTime();
    });
    slider.addEventListener("change", () => {
        const value = Number(slider.value);
        if (value === 0) probe.start = null;
        if (probe.rose && probe.start !== null && probe.seconds === null && value === 100) {
            probe.seconds = (frameTime() - probe.start) / 1000;
        }
    });
    const sample = (now) => {
        if (probe.lastFrame !== null) probe.longestFrameMs = Math.max(probe.longestFrameMs, now - probe.lastFrame);
        probe.lastFrame = now;
        const state = JSON.parse(window.render_game_to_text()).isostaticRebound;
        if (state.solved && state.progressPercent === 0 && !probe.rose) probe.sawToday = true;
        if (state.solved && state.progressPercent > 0 && state.progressPercent < 100) probe.sawPartMelted = true;
        if (probe.seconds === null) requestAnimationFrame(sample);
    };
    requestAnimationFrame(sample);
    window.__meltProbe = probe;
}"""

# The lowest rebound progress seen over the next `ms` milliseconds.
LOWEST_PROGRESS = """async (ms) => {
    let lowest = 100;
    const end = performance.now() + ms;
    while (performance.now() < end) {
        lowest = Math.min(lowest, JSON.parse(window.render_game_to_text()).isostaticRebound.progressPercent);
        await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    return lowest;
}"""


def _wait_rebound_complete(page) -> None:
    page.wait_for_function(
        """() => { const r = JSON.parse(window.render_game_to_text()).isostaticRebound;
            return r.solved && r.progressPercent === 100; }""",
        timeout=SOLVE_TIMEOUT_MS,
    )


class TestReboundDemonstration:
    def test_the_first_switch_on_melts_the_ice_away_over_three_seconds(self, playwright_browser, explore_url):
        context, page, errors = _open(playwright_browser, explore_url, reduced_motion="no-preference")
        try:
            page.evaluate(INSTALL_MELT_PROBE)
            page.locator("#showIsostaticRebound").check()
            page.wait_for_function("() => window.__meltProbe.seconds !== null", timeout=SOLVE_TIMEOUT_MS)
            melt = page.evaluate("() => window.__meltProbe")
            assert errors == []
            assert melt["sawToday"], "the demonstration starts from today's ice"
            assert melt["sawPartMelted"], "the view passes through part-melted ice"
            # The run lasts 3 s by the frame clock and ends on the first frame after that.
            assert 2.95 <= melt["seconds"] <= 3.05 + melt["longestFrameMs"] / 1000, melt
        finally:
            context.close()

    def test_later_switches_leave_the_slider_where_it_is(self, playwright_browser, explore_url):
        context, page, _errors = _open(playwright_browser, explore_url, reduced_motion="no-preference")
        try:
            toggle = page.locator("#showIsostaticRebound")
            toggle.check()
            _wait_rebound_complete(page)
            toggle.uncheck()
            toggle.check()
            _wait_rebound_complete(page)
            assert page.evaluate(LOWEST_PROGRESS, 1500) == 100
        finally:
            context.close()

    def test_reduced_motion_shows_the_ice_free_state_at_once(self, playwright_browser, explore_url):
        context, page, _errors = _open(playwright_browser, explore_url, reduced_motion="reduce")
        try:
            page.locator("#showIsostaticRebound").check()
            _wait_rebound_complete(page)
            assert page.evaluate(LOWEST_PROGRESS, 1000) == 100
        finally:
            context.close()


class TestPublicEditionOnAPhone:
    def test_the_tour_card_floats_over_the_view_with_its_buttons_in_reach(self, playwright_browser, explore_url):
        context = playwright_browser.new_context(
            viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, reduced_motion="reduce"
        )
        try:
            page = context.new_page()
            page.goto(explore_url, wait_until="domcontentloaded", timeout=30_000)
            _wait_ready(page)
            assert page.evaluate("document.body.classList.contains('mobile-drawer')")
            page.locator("#viewerTourButton").tap()
            state = _wait_for_stop(page, "ice-continent")
            card = page.locator("#tourCard")
            assert page.evaluate("document.getElementById('tourCard').parentElement.id") == "viewerShell"
            next_box = card.locator(".tour-card__nav--primary").bounding_box()
            assert next_box is not None and next_box["y"] + next_box["height"] <= 844
            # The continent does not fit a phone at the usual field of view, so it widens.
            camera = state["camera"]
            assert camera["fov"] > 42
            # The view turns about the pole it shows, shifted up clear of the card.
            assert abs(camera["target"]["x"]) < 1 and abs(camera["target"]["z"]) < 1, camera["target"]
            assert camera["shift"][1] > 0.2, camera["shift"]
            card.locator(".tour-card__header-buttons button").last.tap()  # close
            page.wait_for_function(
                "() => JSON.parse(window.render_game_to_text()).camera.shift.every((value) => value === 0)",
                timeout=READY_TIMEOUT_MS,
            )
        finally:
            context.close()


class TestAssetBase:
    def test_a_cross_origin_asset_base_in_the_url_is_ignored(self, playwright_browser, explore_url):
        requests: list[str] = []
        context, page, errors = _open(
            playwright_browser, f"{explore_url}?assetBase=https://example.invalid/tools/", requests=requests
        )
        try:
            assert errors == []
            assert _state(page)["edition"] == "public"
            assert not [url for url in requests if urlparse(url).hostname == "example.invalid"]
        finally:
            context.close()


class TestChinesePublicEdition:
    def test_the_tour_and_explainers_are_in_chinese(self, playwright_browser, explore_zh_url):
        context, page, errors = _open(playwright_browser, f"{explore_zh_url}?tour=1")
        try:
            _wait_for_stop(page, "ice-continent")
            assert errors == []
            assert page.locator(".tour-card__title").inner_text() == "被冰封的大陆"
            assert page.locator(".tour-card__counter").inner_text() == "第 1 站，共 8 站"
            assert page.locator('button[data-info="rebound"]').get_attribute("aria-label") == "关于移除冰层"
            assert _state(page)["status"] == "就绪"
        finally:
            context.close()

    def test_the_language_switcher_maps_between_the_public_pages(self, playwright_browser, explore_zh_url):
        context, page, _errors = _open(playwright_browser, f"{explore_zh_url}?tour=1")
        try:
            english = page.locator('#panelLocaleSwitcher a[data-3d-ice-locale="en-US"]').get_attribute("href")
            assert english.endswith("/explore/?tour=1")
            research = page.locator(".about-edition__link").get_attribute("href")
            assert page.evaluate(f"new URL({json.dumps(research)}, location.href).pathname") == (
                "/zh/tools/3D-interactive-cryosphere-explorer.html"
            )
        finally:
            context.close()
