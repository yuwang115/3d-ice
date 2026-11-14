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


def _open(browser, url: str, *, viewport=(1440, 900), reduced_motion: str = "reduce", live_showcase: bool = False, requests=None):
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
    if requests is not None:
        page.on("request", lambda request: requests.append(urlparse(request.url).path))
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


TOWER_MODULE = "/js/3d-ice-data-tower.js"
TOWER_READY = "() => document.querySelector('[data-tower-stage]')?.classList.contains('is-ready')"
FOCUSED_LAYER = """() => {
    const item = document.querySelector('[data-tower-region]:not([hidden]) .explorer-tower-layer.is-focused');
    return item ? { layer: item.dataset.towerLayer, href: item.querySelector('a').href } : null;
}"""


def _show_tower(page) -> None:
    page.evaluate("document.getElementById('data-tower').scrollIntoView({ behavior: 'instant', block: 'start' })")
    page.wait_for_function(TOWER_READY, timeout=30_000)
    # Let the layers fade in once their images have loaded.
    page.wait_for_timeout(1200)


def _canvas_pixels(page) -> str:
    return page.locator("[data-tower-canvas]").evaluate("(canvas) => canvas.toDataURL()")


def _canvas_points(page, columns: int = 9, rows: int = 12):
    """Points over the stack, top to bottom, where a layer may be under the pointer."""
    box = page.locator("[data-tower-canvas]").bounding_box()
    side = page.locator("[data-tower-stage]").evaluate("(stage) => stage.classList.contains('is-side-labels')")
    width = box["width"] * (0.6 if side else 1)
    for row in range(rows):
        for column in range(columns):
            yield box["x"] + width * (0.2 + 0.6 * column / (columns - 1)), box["y"] + box["height"] * (0.2 + 0.7 * row / (rows - 1))


def _hover_a_layer(page) -> tuple[float, float, dict]:
    for x, y in _canvas_points(page):
        page.mouse.move(x, y)
        focused = page.evaluate(FOCUSED_LAYER)
        if focused:
            return x, y, focused
    raise AssertionError("no point over the stack put a layer in focus")


class TestDataTower:
    def test_the_tower_loads_only_once_its_section_nears_the_screen(self, playwright_browser, server):
        # The tower sits right under the hero, within a screen of it on a laptop; a short window
        # keeps it more than a screen away at first.
        requested: list[str] = []
        context, page, errors, failed = _open(playwright_browser, server + "/", viewport=(1440, 420), requests=requested)
        try:
            page.wait_for_timeout(800)
            assert TOWER_MODULE not in requested, "the first screen is the hero; the tower can wait"
            _show_tower(page)
            assert TOWER_MODULE in requested
            images = [path for path in requested if path.startswith("/tools/media/3d-ice/tower/")]
            assert images and all(path.split("/")[-1].startswith("antarctica-") for path in images), images
            assert page.locator("[data-tower-controls]").is_visible()
            assert errors == [] and failed == []
        finally:
            context.close()

    def test_the_stack_turns_by_itself_but_not_under_reduced_motion(self, playwright_browser, server):
        for motion, turns in (("no-preference", True), ("reduce", False)):
            context, page, errors, _failed = _open(playwright_browser, server + "/", reduced_motion=motion)
            try:
                _show_tower(page)
                page.mouse.move(2, 2)
                before = _canvas_pixels(page)
                page.wait_for_timeout(900)
                assert (_canvas_pixels(page) != before) == turns, motion
                assert errors == []
            finally:
                context.close()

    def test_a_drag_turns_the_stack(self, playwright_browser, server):
        context, page, _errors, _failed = _open(playwright_browser, server + "/")
        try:
            _show_tower(page)
            box = page.locator("[data-tower-canvas]").bounding_box()
            before = _canvas_pixels(page)
            yaw_before = page.evaluate("document.getElementById('data-tower').dataTowerState().yaw")
            x, y = box["x"] + box["width"] * 0.3, box["y"] + box["height"] * 0.5
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + 160, y, steps=8)
            yaw_dragged = page.evaluate("document.getElementById('data-tower').dataTowerState().yaw")
            page.mouse.up()
            page.wait_for_timeout(300)
            assert _canvas_pixels(page) != before
            # A drag to the right brings the front of the stack to the right: the yaw falls.
            assert yaw_dragged < yaw_before - 0.5, (yaw_before, yaw_dragged)
            assert page.url.rstrip("/") == server.rstrip("/"), "a drag is not a click"
        finally:
            context.close()

    def test_a_layer_under_the_pointer_lights_its_label_and_a_click_opens_it(self, playwright_browser, server, home_locale):
        context, page, errors, _failed = _open(playwright_browser, server + HOME_PATHS[home_locale])
        try:
            _show_tower(page)
            x, y, focused = _hover_a_layer(page)
            assert page.locator("[data-tower-canvas]").evaluate("(canvas) => canvas.style.cursor") == "pointer"
            assert page.locator("[data-tower-stage]").evaluate("(stage) => stage.classList.contains('has-focus')")
            page.mouse.click(x, y)
            page.wait_for_url(focused["href"], wait_until="commit", timeout=NAVIGATION_TIMEOUT_MS)
            assert errors == []
        finally:
            context.close()

    def test_the_region_switch_stacks_greenland(self, playwright_browser, server):
        context, page, errors, _failed = _open(playwright_browser, server + "/")
        try:
            _show_tower(page)
            before = _canvas_pixels(page)
            page.locator('[data-tower-region-button="greenland"]').click()
            page.wait_for_timeout(1600)
            assert page.locator('[data-tower-region-button="greenland"]').get_attribute("aria-pressed") == "true"
            assert page.locator('[data-tower-region="greenland"]').is_visible()
            assert page.locator('[data-tower-region="antarctica"]').is_hidden()
            assert _canvas_pixels(page) != before
            _x, _y, focused = _hover_a_layer(page)
            assert "region=greenland" in focused["href"]
            assert errors == []
        finally:
            context.close()

    def test_a_keyboard_user_reaches_every_layer_and_sees_which_has_focus(self, playwright_browser, server):
        context, page, _errors, _failed = _open(playwright_browser, server + "/")
        try:
            _show_tower(page)
            page.locator('[data-tower-region-button="greenland"]').focus()
            page.keyboard.press("Tab")
            link = page.locator('[data-tower-region="antarctica"] [data-tower-layer="ice-surface"] a')
            assert link.evaluate("(el) => el === document.activeElement && el.matches(':focus-visible')")
            assert link.evaluate("(el) => getComputedStyle(el).outlineWidth") == "3px"
            assert page.evaluate(FOCUSED_LAYER)["layer"] == "ice-surface"
            count = page.locator('[data-tower-region="antarctica"] a').count()
            assert count == 13
        finally:
            context.close()

    def test_a_phone_lists_the_layers_under_the_stack(self, playwright_browser, server, home_locale):
        context, page, _errors, _failed = _open(playwright_browser, server + HOME_PATHS[home_locale], viewport=(390, 844))
        try:
            _show_tower(page)
            assert not page.locator("[data-tower-stage]").evaluate("(stage) => stage.classList.contains('is-side-labels')")
            canvas = page.locator("[data-tower-canvas]").bounding_box()
            first = page.locator('[data-tower-region="antarctica"] .explorer-tower-layer').first.bounding_box()
            assert first["y"] >= canvas["y"] + canvas["height"], "the labels follow the stack"
            assert page.evaluate("document.documentElement.scrollWidth") <= 390
        finally:
            context.close()


# Antarctica's layers that only the research edition has (tests/test_home_page.py lists them all).
RESEARCH_ONLY_ANTARCTICA = ["surface-change", "sea-ice", "basal-melt", "basal-friction", "subglacial", "basins"]
TOWER_STATE = "(tower) => tower.dataTowerState()"


class TestEditionsAndTheTower:
    def test_the_public_card_fades_the_layers_only_the_research_edition_has(self, playwright_browser, server, home_locale):
        context, page, errors, _failed = _open(playwright_browser, server + HOME_PATHS[home_locale])
        try:
            _show_tower(page)
            tower = page.locator("[data-tower]")
            page.locator('.explorer-edition-card[data-edition="public"] h3').hover()
            state = tower.evaluate(TOWER_STATE)
            assert state["edition"] == "public"
            assert state["muted"] == RESEARCH_ONLY_ANTARCTICA
            muted = page.locator('[data-tower-region="antarctica"] .explorer-tower-layer.is-muted')
            assert muted.evaluate_all("(items) => items.map((item) => item.dataset.towerLayer)") == RESEARCH_ONLY_ANTARCTICA
            page.locator('.explorer-edition-card[data-edition="research"] h3').hover()
            state = tower.evaluate(TOWER_STATE)
            assert (state["edition"], state["muted"]) == ("research", [])
            page.mouse.move(2, 2)
            assert tower.evaluate(TOWER_STATE)["edition"] is None
            assert muted.count() == 0
            assert errors == []
        finally:
            context.close()

    def test_a_keyboard_user_in_an_edition_card_sees_its_layers(self, playwright_browser, server):
        context, page, _errors, _failed = _open(playwright_browser, server + "/")
        try:
            _show_tower(page)
            tower = page.locator("[data-tower]")
            # The last layer's label, then Tab: the public card's first button.
            page.locator('[data-tower-region="antarctica"] [data-tower-layer="grounding-line"] a').focus()
            page.keyboard.press("Tab")
            assert page.evaluate("document.activeElement.closest('[data-edition]')?.dataset.edition") == "public"
            assert tower.evaluate(TOWER_STATE)["edition"] == "public"
            page.keyboard.press("Shift+Tab")
            assert tower.evaluate(TOWER_STATE)["edition"] is None
        finally:
            context.close()

    def test_a_click_in_an_edition_card_does_not_hold_the_emphasis(self, playwright_browser, server):
        context, page, _errors, _failed = _open(playwright_browser, server + "/")
        try:
            _show_tower(page)
            link = page.locator('.explorer-edition-card[data-edition="research"] a[href="#source-data"]')
            # A middle click leaves the page where it is and focus in the card.
            link.click(button="middle")
            page.mouse.move(2, 2)
            assert page.locator("[data-tower]").evaluate(TOWER_STATE)["edition"] is None
        finally:
            context.close()


DRAWER_OPEN = "() => document.getElementById('all-updates').open"


class TestUpdatesDrawer:
    @pytest.mark.parametrize("closing", ["escape", "close-button", "backdrop"])
    def test_the_drawer_opens_from_its_link_and_closes(self, playwright_browser, server, home_locale, closing):
        context, page, errors, _failed = _open(playwright_browser, server + HOME_PATHS[home_locale])
        try:
            trigger = page.locator('[data-drawer-open="all-updates"]')
            drawer = page.locator("#all-updates")
            assert not drawer.is_visible()
            trigger.click()
            assert page.evaluate(DRAWER_OPEN) and drawer.is_visible()
            assert page.evaluate("getComputedStyle(document.documentElement).overflow") == "hidden", "the page holds still"
            assert urlparse(page.url).fragment == "", "the address stays as it was"
            box = drawer.bounding_box()
            assert abs(box["x"] + box["width"] - 1440) <= 1 and box["height"] >= 899, "a panel along the right edge"
            assert drawer.locator(".explorer-drawer-item").count() >= 8
            if closing == "escape":
                page.keyboard.press("Escape")
            elif closing == "close-button":
                drawer.locator("[data-drawer-close]").click()
            else:
                page.mouse.click(box["x"] / 2, 450)
            page.wait_for_function("() => !document.getElementById('all-updates').open", timeout=5_000)
            assert not drawer.is_visible()
            # Focus comes back in the dialog's close event, a task after it closes.
            page.wait_for_function("() => document.activeElement.dataset.drawerOpen === 'all-updates'", timeout=5_000)
            assert page.evaluate("window.location.hash") == ""
            assert errors == []
        finally:
            context.close()

    def test_the_drawer_slides_out_rather_than_vanishing(self, playwright_browser, server):
        context, page, _errors, _failed = _open(playwright_browser, server + "/", reduced_motion="no-preference")
        try:
            page.locator('[data-drawer-open="all-updates"]').click()
            page.wait_for_timeout(400)
            page.keyboard.press("Escape")
            assert page.locator("#all-updates").evaluate("(drawer) => drawer.open && drawer.classList.contains('is-closing')")
            page.wait_for_function("() => !document.getElementById('all-updates').open", timeout=5_000)
        finally:
            context.close()

    def test_an_address_naming_the_drawer_opens_it_over_its_section(self, playwright_browser, server, home_locale):
        context, page, errors, _failed = _open(playwright_browser, server + HOME_PATHS[home_locale] + "#all-updates")
        try:
            page.wait_for_function(DRAWER_OPEN, timeout=5_000)
            section = page.locator("#latest-updates").bounding_box()
            # At the top of the window, less the sections' scroll margin (1.25rem).
            assert 0 <= section["y"] <= 24, "the page behind it shows the updates, not its foot"
            page.keyboard.press("Escape")
            page.wait_for_function("() => !document.getElementById('all-updates').open", timeout=5_000)
            assert not page.locator("#all-updates").is_visible(), "a closed drawer does not stay in the page"
            # The dialog's close event, which clears the address and returns focus, comes a task later.
            page.wait_for_function("() => document.activeElement.dataset.drawerOpen === 'all-updates'", timeout=5_000)
            assert page.evaluate("window.location.hash") == "", "a reload does not open it again"
            assert errors == []
        finally:
            context.close()

    def test_space_on_the_close_button_closes_the_drawer(self, playwright_browser, server):
        context, page, _errors, _failed = _open(playwright_browser, server + "/")
        try:
            page.locator('[data-drawer-open="all-updates"]').click()
            close = page.locator("#all-updates [data-drawer-close]")
            assert close.get_attribute("role") == "button"
            close.focus()
            page.keyboard.press("Space")
            page.wait_for_function("() => !document.getElementById('all-updates').open", timeout=5_000)
        finally:
            context.close()

    def test_without_the_script_the_link_still_reaches_every_update(self, playwright_browser, server):
        # Reduced motion: no smooth scroll to the drawer, which would keep its links moving.
        context = playwright_browser.new_context(viewport={"width": 1440, "height": 900}, java_script_enabled=False, reduced_motion="reduce")
        try:
            page = context.new_page()
            page.goto(server + "/", wait_until="load", timeout=NAVIGATION_TIMEOUT_MS)
            drawer = page.locator("#all-updates")
            assert not drawer.is_visible()
            page.locator('[data-drawer-open="all-updates"]').click()
            assert drawer.is_visible()
            assert drawer.evaluate("(el) => getComputedStyle(el).position") == "static", "shown in place, not as a modal"
            drawer.locator("[data-drawer-close]").click()
            assert not drawer.is_visible()
        finally:
            context.close()


def test_a_tap_picks_a_layer_and_a_second_tap_opens_it(playwright_browser, server):
    context = playwright_browser.new_context(viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True, reduced_motion="reduce")
    context.route("https://js.usebasin.com/**", lambda route: route.abort())
    context.route(
        lambda target: "mode=showcase" in target,
        lambda route: route.fulfill(body="<!doctype html><title>showcase</title>", content_type="text/html"),
    )
    try:
        page = context.new_page()
        page.goto(server + "/", wait_until="load", timeout=NAVIGATION_TIMEOUT_MS)
        _show_tower(page)
        chip = page.locator("[data-tower-open]")
        for x, y in _canvas_points(page, columns=5, rows=8):
            page.touchscreen.tap(x, y)
            if chip.is_visible():
                break
        else:
            raise AssertionError("no tap on the stack picked a layer")
        focused = page.evaluate(FOCUSED_LAYER)
        link = chip.locator("a")
        assert link.evaluate("(el) => el.href") == focused["href"]
        assert link.text_content().strip()
        assert page.url.rstrip("/") == server.rstrip("/"), "the first tap only picks the layer"
        page.touchscreen.tap(x, y)
        page.wait_for_url(focused["href"], wait_until="commit", timeout=NAVIGATION_TIMEOUT_MS)
    finally:
        context.close()
