"""The home pages in a browser: both editions are one click away, and the page works as a page.

The static structure (links, sections, parity between locales) is checked without a browser
in tests/test_home_page.py; these tests cover what only a browser shows.
"""

from __future__ import annotations

from urllib.parse import urlparse

import pytest


pytestmark = pytest.mark.e2e

HOME_PATHS = {"en": "/", "zh": "/zh/"}
NAVIGATION_TIMEOUT_MS = 60_000
TOUR_TIMEOUT_MS = 120_000


def _open(browser, url: str, *, viewport=(1440, 900), reduced_motion: str = "reduce", live_showcase: bool = False):
    context = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]}, reduced_motion=reduced_motion)
    # Keep third-party services out of the tests.
    context.route("https://www.googletagmanager.com/**", lambda route: route.abort())
    context.route("https://js.usebasin.com/**", lambda route: route.abort())
    if not live_showcase:
        # The hero's live 3D panel boots the whole explorer; most tests only need the page around it.
        context.route(
            lambda target: "mode=showcase" in target,
            lambda route: route.fulfill(body="<!doctype html><title>showcase</title>", content_type="text/html"),
        )
    page = context.new_page()
    errors: list[str] = []
    failed: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on(
        "response",
        lambda response: failed.append(f"{response.status} {response.url}")
        if response.status >= 400 and urlparse(response.url).hostname in ("127.0.0.1", "localhost")
        else None,
    )
    page.goto(url, wait_until="load", timeout=NAVIGATION_TIMEOUT_MS)
    # Layout is measured as visitors see it once the typefaces arrive. The heading face is not
    # preloaded and can land after "load"; until then a wider fallback (DejaVu on Linux)
    # wraps the hero onto extra lines.
    page.evaluate("document.fonts.ready.then(() => true)")
    return context, page, errors, failed


@pytest.fixture(params=sorted(HOME_PATHS))
def home_locale(request) -> str:
    return request.param


class TestHomePageInABrowser:
    def test_loads_without_errors_or_missing_files(self, playwright_browser, server, home_locale):
        context, page, errors, failed = _open(playwright_browser, server + HOME_PATHS[home_locale], live_showcase=True)
        try:
            page.wait_for_timeout(1500)
            assert errors == []
            assert failed == []
            assert page.locator("#homeLocaleSwitcher a").count() >= 2, "the language switcher is mounted"
        finally:
            context.close()

    def test_both_editions_are_in_the_first_screen_on_a_laptop(self, playwright_browser, server, home_locale):
        # 1440 x 900 is a common laptop screen; the browser's own bars leave about 800 px of page.
        context, page, _errors, _failed = _open(playwright_browser, server + HOME_PATHS[home_locale], viewport=(1440, 800))
        try:
            loaded = page.evaluate("[...document.fonts].filter((face) => face.status === 'loaded').map((face) => face.family)")
            assert {"Space Grotesk", "Playfair Display"} <= set(loaded), f"the bundled typefaces did not load: {loaded}"
            buttons = page.locator(".explorer-ice-hero .explorer-actions a")
            assert buttons.count() == 2
            for index in range(2):
                box = buttons.nth(index).bounding_box()
                assert box is not None and box["y"] + box["height"] <= 800, f"hero button {index} is below the fold"
        finally:
            context.close()

    def test_the_hero_starts_the_guided_tour(self, playwright_browser, server):
        context, page, errors, _failed = _open(playwright_browser, server + "/")
        try:
            page.locator(".explorer-ice-hero .explorer-button--primary").click()
            page.wait_for_url("**/explore/?tour=1", timeout=NAVIGATION_TIMEOUT_MS)
            page.wait_for_function(
                "() => { const card = document.getElementById('tourCard'); return card && !card.hidden; }",
                timeout=TOUR_TIMEOUT_MS,
            )
            assert errors == []
        finally:
            context.close()

    def test_the_research_card_opens_the_research_edition(self, playwright_browser, server, home_locale):
        context, page, _errors, _failed = _open(playwright_browser, server + HOME_PATHS[home_locale])
        try:
            page.locator('[data-edition="research"] .explorer-button--primary').click()
            prefix = "/zh" if home_locale == "zh" else ""
            page.wait_for_url(f"**{prefix}/tools/3D-interactive-cryosphere-explorer.html", wait_until="commit", timeout=NAVIGATION_TIMEOUT_MS)
        finally:
            context.close()

    def test_the_comparison_table_opens_from_its_summary(self, playwright_browser, server, home_locale):
        context, page, _errors, _failed = _open(playwright_browser, server + HOME_PATHS[home_locale])
        try:
            table = page.locator(".explorer-edition-compare table")
            assert not table.is_visible()
            page.locator(".explorer-edition-compare summary").click()
            assert table.is_visible()
            assert page.locator(".explorer-edition-compare tbody tr").count() >= 8
        finally:
            context.close()

    def test_the_theme_toggle_switches_remembers_and_swaps_the_logos(self, playwright_browser, server):
        context, page, _errors, _failed = _open(playwright_browser, server + "/")
        try:
            page.evaluate("() => { localStorage.setItem('wc-color-theme', 'light'); }")
            page.reload(wait_until="load")
            logo = page.locator(".explorer-ice-logo")
            assert logo.get_attribute("src") == "/tools/3d-ice-logo-light.jpg"
            toggle = page.locator("#themeToggle")
            toggle.click()
            assert page.evaluate("document.documentElement.classList.contains('dark')")
            assert toggle.get_attribute("aria-pressed") == "true"
            assert logo.get_attribute("src") == "/tools/3d-ice-logo.jpg"
            assert page.locator(".explorer-footer-logo--aapp img").get_attribute("src") == "/logos/aapp-dark.png"
            page.reload(wait_until="load")
            assert page.evaluate("document.documentElement.classList.contains('dark')"), "the choice survives a reload"
        finally:
            context.close()

    def test_the_language_switcher_maps_between_the_home_pages(self, playwright_browser, server):
        context, page, _errors, _failed = _open(playwright_browser, server + "/")
        try:
            page.locator('#homeLocaleSwitcher a[hreflang="zh-CN"]').click()
            page.wait_for_url("**/zh/", timeout=NAVIGATION_TIMEOUT_MS)
            assert page.evaluate("document.documentElement.lang") == "zh-CN"
        finally:
            context.close()

    def test_a_phone_gets_one_column_and_no_sideways_scrolling(self, playwright_browser, server, home_locale):
        context, page, _errors, _failed = _open(playwright_browser, server + HOME_PATHS[home_locale], viewport=(390, 844))
        try:
            assert page.evaluate("document.documentElement.scrollWidth") <= 390
            public = page.locator('[data-edition="public"]').bounding_box()
            research = page.locator('[data-edition="research"]').bounding_box()
            assert research["y"] >= public["y"] + public["height"], "the edition cards stack"
        finally:
            context.close()

    def test_previews_stay_still_under_reduced_motion(self, playwright_browser, server):
        context, page, _errors, _failed = _open(playwright_browser, server + "/", reduced_motion="reduce")
        try:
            page.locator("#antarctica-features").scroll_into_view_if_needed()
            page.wait_for_timeout(800)
            states = page.evaluate(
                "() => [...document.querySelectorAll('.explorer-feature-video')].map((video) => ({"
                " paused: video.paused, active: video.closest('a').classList.contains('is-video-active') }))"
            )
            assert states and all(state["paused"] and not state["active"] for state in states)
        finally:
            context.close()

    def test_previews_only_try_to_play_while_on_screen(self, playwright_browser, server):
        context, page, _errors, _failed = _open(playwright_browser, server + "/", reduced_motion="no-preference")
        try:
            page.wait_for_timeout(800)
            # The first screen shows the hero, so no preview is asked to play yet.
            assert page.locator(".explorer-preview-link.is-video-active").count() == 0
            page.locator('[data-demo-slot="antarctica-velocity-flowlines"]').scroll_into_view_if_needed()
            page.wait_for_function(
                "() => document.querySelector('[data-demo-slot=\"antarctica-velocity-flowlines\"] a').classList.contains('is-video-active')"
                " || document.querySelector('[data-demo-slot=\"antarctica-velocity-flowlines\"] video').error !== null",
                timeout=10_000,
            )
        finally:
            context.close()


class TestHomePageEdgeCases:
    def test_the_open_comparison_table_scrolls_inside_its_box_on_a_phone(self, playwright_browser, server, home_locale):
        context, page, _errors, _failed = _open(playwright_browser, server + HOME_PATHS[home_locale], viewport=(390, 844))
        try:
            page.locator(".explorer-edition-compare summary").click()
            assert page.evaluate("document.documentElement.scrollWidth") <= 390, "the page widened"
            assert page.locator(".explorer-edition-table-wrap").evaluate("(el) => el.scrollWidth > el.clientWidth")
        finally:
            context.close()

    def test_wheel_scrolling_over_the_showcase_moves_the_page(self, playwright_browser, server):
        # The showcase explorer posts the wheel deltas it does not use. Twenty ticks of 100 px must
        # move the page 2000 px, although in-page links scroll smoothly.
        context, page, _errors, _failed = _open(playwright_browser, server + "/", reduced_motion="no-preference")
        try:
            showcase = next(frame for frame in page.frames if "mode=showcase" in frame.url)
            showcase.evaluate(
                "() => { for (let i = 0; i < 20; i += 1) window.parent.postMessage("
                "{ type: '3d-ice:showcase-scroll', deltaX: 0, deltaY: 100 }, window.location.origin); }"
            )
            page.wait_for_timeout(600)
            assert page.evaluate("window.scrollY") >= 1990
        finally:
            context.close()

    def test_a_keyboard_user_sees_which_preview_has_focus(self, playwright_browser, server):
        context, page, _errors, _failed = _open(playwright_browser, server + "/")
        try:
            page.locator("#antarctica-features-title").evaluate("(el) => { el.tabIndex = -1; el.focus(); }")
            page.keyboard.press("Tab")
            link = page.locator('[data-demo-slot="antarctica-velocity-flowlines"] .explorer-preview-link')
            assert link.evaluate("(el) => el === document.activeElement && el.matches(':focus-visible')")
            ring = link.evaluate("(el) => getComputedStyle(el, '::after')")
            assert link.evaluate("(el) => getComputedStyle(el, '::after').borderTopWidth") == "3px", ring
        finally:
            context.close()

    def test_a_missing_form_field_does_not_take_the_language_switcher_down(self, playwright_browser, server):
        context = playwright_browser.new_context()
        context.route("https://**", lambda route: route.abort())

        def without_page_url_field(route):
            response = route.fetch()
            body = response.text().replace('<input type="hidden" name="page_url" data-feedback-page-url />', "")
            route.fulfill(response=response, body=body)

        context.route(lambda target: urlparse(target).path == "/", without_page_url_field)
        try:
            page = context.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server + "/", wait_until="load", timeout=NAVIGATION_TIMEOUT_MS)
            assert page.locator("#faq-feedback-3d-ice-form [data-feedback-page-url]").count() == 0
            assert page.locator("#homeLocaleSwitcher a").count() == 2
            assert errors == []
        finally:
            context.close()

    def test_a_failed_send_keeps_the_message_and_frees_the_button(self, playwright_browser, server):
        context, page, _errors, _failed = _open(playwright_browser, server + "/")
        try:
            page.fill("#faq-feedback-3d-ice-message", "The tour is great")
            state = page.evaluate(
                """async () => {
                    const form = document.getElementById("faq-feedback-3d-ice-form");
                    const submit = form.querySelector(".tool-feedback-submit");
                    form.dispatchEvent(new Event("submit", { cancelable: true }));
                    document.dispatchEvent(new CustomEvent("basinjsFormSubmitted", { detail: { form } }));
                    const busy = { disabled: submit.disabled, label: submit.textContent };
                    // Basin clears and hides the form when sending fails, then reports it.
                    form.reset();
                    form.style.display = "none";
                    document.dispatchEvent(new CustomEvent("basinjsFormError", { detail: { form } }));
                    await new Promise((resolve) => setTimeout(resolve, 50));
                    return {
                        busy,
                        disabled: submit.disabled,
                        label: submit.textContent,
                        message: document.getElementById("faq-feedback-3d-ice-message").value,
                        formShown: form.style.display !== "none",
                        errorShown: getComputedStyle(document.getElementById("faq-feedback-3d-ice-error")).display !== "none",
                    };
                }"""
            )
            assert state["busy"] == {"disabled": True, "label": "Sending..."}
            assert (state["disabled"], state["label"]) == (False, "Send feedback")
            assert state["message"] == "The tour is great"
            assert state["formShown"] and state["errorShown"]
        finally:
            context.close()

    def test_a_sent_message_moves_focus_to_the_confirmation_and_back(self, playwright_browser, server):
        context, page, _errors, _failed = _open(playwright_browser, server + "/")
        try:
            page.evaluate(
                """() => {
                    const form = document.getElementById("faq-feedback-3d-ice-form");
                    document.dispatchEvent(new CustomEvent("basinjsFormSuccess", { detail: { form } }));
                }"""
            )
            page.wait_for_function("() => document.activeElement.id === 'faq-feedback-3d-ice-success'", timeout=5_000)
            page.locator(".tool-feedback-reset").click()
            assert page.evaluate("document.activeElement.id") == "faq-feedback-3d-ice-message"
        finally:
            context.close()
