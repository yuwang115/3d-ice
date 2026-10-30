"""Analytics load only on 3d-ice.com, never in local previews, test runs or copies of the site.

Every page loads Google Analytics and GoatCounter through static/js/3d-ice-analytics.js.
Served from anywhere but https://3d-ice.com, a page requests no tracker at all, so local
previews, these tests and the copy of the explorer that yuwang.blog serves are never counted.
On 3d-ice.com each page view loads each tracker once, and the explorer that the home page
shows in its showcase iframe adds nothing. There, GoatCounter also counts events: a click
into either edition, named for where it came from, and the guided tour's start and
completion (tests/e2e/test_public_edition.py checks that the tour reports them).

The production host is simulated by answering https://3d-ice.com/ requests from static/, and
every tracker request is aborted, so no test ever sends a hit. Where a test needs count.js
to run, it is answered with a copy of the pinned script.
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest


pytestmark = pytest.mark.e2e

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
LOADER = STATIC_DIR / "js" / "3d-ice-analytics.js"
LOADER_PATH = "/js/3d-ice-analytics.js"
PRODUCTION_ORIGIN = "https://3d-ice.com"
GA_TAG_URL = "https://www.googletagmanager.com/gtag/js?id=GT-WPQPVBWP"
GOATCOUNTER_SCRIPT_URL = "https://gc.zgo.at/count.v5.js"
# count.v5.js's published Subresource Integrity hash (https://www.goatcounter.com/help/countjs-versions).
GOATCOUNTER_INTEGRITY = "sha384-atnOLvQb9t+jTSipvd75X2yginT4PjVbqDdlJAmxMm+wYElFmeR6EmLP5bYeoRVQ"
GOATCOUNTER_CODE_DECLARATION = re.compile(r'(const GOATCOUNTER_CODE = ")([^"]*)(";)')
# A byte-for-byte copy of count.v5.js (ISC licence), so that it passes the integrity check.
GOATCOUNTER_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "goatcounter-count.v5.js"
GOATCOUNTER_READY = "() => typeof window.goatcounter?.count === 'function'"
TRACKER_URL = re.compile(r"^https?://([^/]+\.)?(googletagmanager\.com|google-analytics\.com|zgo\.at|goatcounter\.com)/")
NAVIGATION_TIMEOUT_MS = 60_000
# Time for a late tracker request to show up after the page's head has run.
SETTLE_MS = 1_000

# Every page that carries analytics, by its path on 3d-ice.com. GitHub Pages answers a
# missing path with 404.html, so the not-found page is reached through one.
PAGES = {
    "home-en": "/",
    "home-zh": "/zh/",
    "public-en": "/explore/",
    "public-zh": "/zh/explore/",
    "research-en": "/tools/3D-interactive-cryosphere-explorer.html",
    "research-zh": "/zh/tools/3D-interactive-cryosphere-explorer.html",
    "not-found": "/no/such/page",
}
# The test server has no not-found fallback, so there the page is opened by name.
LOCAL_PATHS = {**PAGES, "not-found": "/404.html"}
# Links into an edition: the page each is on, the link, and the event a click on it counts.
EDITION_ENTRIES = {
    "home-hero-tour": ("/", ".explorer-ice-hero .explorer-actions a[href='./explore/?tour=1']", "open-public-edition:home-hero"),
    "home-zh-editions": (
        "/zh/",
        "#editions a[href='./tools/3D-interactive-cryosphere-explorer.html']",
        "open-research-edition:home-editions",
    ),
    "home-preview": ("/", "#greenland-features a[href^='./explore/']", "open-public-edition:home-greenland-features"),
    "research-to-tour": ("/tools/3D-interactive-cryosphere-explorer.html", "a[href='../explore/?tour=1']", "open-public-edition:research-edition"),
    "public-zh-to-research": ("/zh/explore/", "a.about-edition__link", "open-research-edition:public-edition"),
}
CLICK_NEW_LINK = """href => {
    const link = Object.assign(document.createElement("a"), { href, textContent: "link" });
    document.body.append(link);
    link.click();
}"""


def _goatcounter_code() -> str:
    match = GOATCOUNTER_CODE_DECLARATION.search(LOADER.read_text(encoding="utf-8"))
    assert match, "static/js/3d-ice-analytics.js should declare GOATCOUNTER_CODE"
    return match.group(2)


def _record_trackers(context) -> list[tuple[str, bool]]:
    """Abort every tracker request, noting its URL and whether the top-level page made it."""
    requests: list[tuple[str, bool]] = []

    def abort(route):
        requests.append((route.request.url, route.request.frame.parent_frame is None))
        route.abort()

    context.route(TRACKER_URL, abort)
    return requests


def _site_file(path: str) -> Path | None:
    """The static/ file that answers a site path, or None where GitHub Pages would answer 404."""
    relative = path.lstrip("/")
    if not relative or relative.endswith("/"):
        relative += "index.html"
    candidate = STATIC_DIR / relative
    return candidate if candidate.is_file() else None


def _serve_site(context, *, loader_source: str | None = None) -> None:
    """Answer https://3d-ice.com/ with static/'s pages and analytics script; abort the rest.

    Styles, data and the explorer's code are not needed to see what a page's head loads.
    """

    def answer(route):
        request = route.request
        path = urlparse(request.url).path
        if path == LOADER_PATH:
            body = LOADER.read_text(encoding="utf-8") if loader_source is None else loader_source
            route.fulfill(body=body, content_type="text/javascript")
        elif request.resource_type == "document":
            page_file = _site_file(path)
            if page_file is None:
                route.fulfill(status=404, path=STATIC_DIR / "404.html")
            else:
                route.fulfill(path=page_file)
        else:
            route.abort()

    context.route(f"{PRODUCTION_ORIGIN}/**", answer)


def _serve_goatcounter(context) -> None:
    """Answer count.js with the pinned script, so that it runs and counts.

    Register it after _record_trackers: the route registered last answers first, and the
    counts the script sends are still recorded and aborted there.
    """
    context.route(
        GOATCOUNTER_SCRIPT_URL,
        lambda route: route.fulfill(
            path=GOATCOUNTER_FIXTURE, content_type="text/javascript", headers={"Access-Control-Allow-Origin": "*"}
        ),
    )


def _counted_events(trackers: list[tuple[str, bool]]) -> list[tuple[str, str]]:
    """The events sent to GoatCounter, as (name, title), in the order they were sent."""
    endpoint = f"https://{_goatcounter_code()}.goatcounter.com/count?"
    events = []
    for url, _top_level in trackers:
        query = parse_qs(urlparse(url).query, keep_blank_values=True)
        if url.startswith(endpoint) and query.get("e") == ["true"]:
            events.append((query["p"][0], query["t"][0]))
    return events


def _open_on_the_site(browser, path: str):
    """Open a page as on 3d-ice.com, with count.js running, and wait until it can count."""
    context = browser.new_context()
    trackers = _record_trackers(context)
    _serve_goatcounter(context)
    _serve_site(context)
    page = context.new_page()
    page.goto(PRODUCTION_ORIGIN + path, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
    page.wait_for_function(GOATCOUNTER_READY, timeout=NAVIGATION_TIMEOUT_MS)
    return context, page, trackers


def _wait_for_new_page(page, previous_url: str) -> None:
    """Wait until a click has led to another page and that page has counted its view."""
    page.wait_for_url(lambda url: url != previous_url, timeout=NAVIGATION_TIMEOUT_MS)
    page.wait_for_function(GOATCOUNTER_READY, timeout=NAVIGATION_TIMEOUT_MS)
    page.wait_for_timeout(SETTLE_MS)


class TestAwayFromTheSite:
    @pytest.mark.parametrize("page_id", sorted(PAGES))
    def test_a_page_loads_the_analytics_script_but_no_tracker(self, playwright_browser, server, page_id):
        context = playwright_browser.new_context()
        trackers = _record_trackers(context)
        context.route("**/tools/data/**", lambda route: route.abort())
        context.route(
            lambda url: "mode=showcase" in url,
            lambda route: route.fulfill(body="<!doctype html><title>showcase</title>", content_type="text/html"),
        )
        page = context.new_page()
        loader_statuses: list[int] = []
        page.on(
            "response",
            lambda response: loader_statuses.append(response.status)
            if urlparse(response.url).path == LOADER_PATH
            else None,
        )
        try:
            page.goto(server + LOCAL_PATHS[page_id], wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
            page.wait_for_timeout(SETTLE_MS)
            assert loader_statuses == [200], "the page should load the analytics script from its own site"
            assert trackers == []
            assert page.evaluate("typeof window.gtag") == "function", "page scripts can still queue their events"
        finally:
            context.close()

    def test_events_and_clicks_into_an_edition_send_nothing(self, playwright_browser, server):
        context = playwright_browser.new_context()
        trackers = _record_trackers(context)
        context.route("**/tools/data/**", lambda route: route.abort())
        context.route(
            lambda url: "mode=showcase" in url,
            lambda route: route.fulfill(body="<!doctype html><title>showcase</title>", content_type="text/html"),
        )
        page = context.new_page()
        try:
            page.goto(server + "/index.html", wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
            page.evaluate("window.__3dIceAnalytics.event('tour-start', 'Guided tour started')")
            page.locator(".explorer-ice-hero .explorer-actions a").first.evaluate("link => link.click()")
            page.wait_for_url(lambda url: "/explore/" in url, timeout=NAVIGATION_TIMEOUT_MS)
            page.wait_for_timeout(SETTLE_MS)
            assert trackers == []
        finally:
            context.close()


class TestOnTheSite:
    @pytest.mark.parametrize("page_id", sorted(PAGES))
    def test_a_page_view_loads_each_tracker_once(self, playwright_browser, page_id):
        context = playwright_browser.new_context()
        trackers = _record_trackers(context)
        _serve_site(context)
        page = context.new_page()
        try:
            page.goto(PRODUCTION_ORIGIN + PAGES[page_id], wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
            page.wait_for_timeout(SETTLE_MS)
            expected = [GA_TAG_URL] + ([GOATCOUNTER_SCRIPT_URL] if _goatcounter_code() else [])
            assert sorted(url for url, top_level in trackers if top_level) == sorted(expected)
            assert [url for url, top_level in trackers if not top_level] == []
        finally:
            context.close()

    def test_the_showcase_iframe_on_the_home_page_adds_no_page_view(self, playwright_browser):
        context = playwright_browser.new_context()
        trackers = _record_trackers(context)
        _serve_site(context)
        page = context.new_page()
        loader_in_iframe: list[bool] = []
        page.on(
            "request",
            lambda request: loader_in_iframe.append(request.frame.parent_frame is not None)
            if urlparse(request.url).path == LOADER_PATH
            else None,
        )
        try:
            page.goto(PRODUCTION_ORIGIN + "/", wait_until="load", timeout=NAVIGATION_TIMEOUT_MS)
            page.wait_for_timeout(SETTLE_MS)
            assert True in loader_in_iframe, "the showcase explorer should have run the analytics script"
            assert [url for url, top_level in trackers if not top_level] == []
            assert [url for url, top_level in trackers if top_level].count(GA_TAG_URL) == 1
        finally:
            context.close()

    def test_goatcounter_loads_its_pinned_script_once_a_site_code_is_set(self, playwright_browser):
        source, replaced = GOATCOUNTER_CODE_DECLARATION.subn(r"\g<1>test-site\g<3>", LOADER.read_text(encoding="utf-8"))
        assert replaced == 1, "static/js/3d-ice-analytics.js should declare GOATCOUNTER_CODE once"
        context = playwright_browser.new_context()
        trackers = _record_trackers(context)
        _serve_site(context, loader_source=source)
        page = context.new_page()
        try:
            page.goto(PRODUCTION_ORIGIN + "/explore/", wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
            page.wait_for_timeout(SETTLE_MS)
            assert sorted(url for url, top_level in trackers if top_level) == sorted([GA_TAG_URL, GOATCOUNTER_SCRIPT_URL])
            script = page.locator("script[data-goatcounter]")
            assert script.count() == 1
            assert script.get_attribute("data-goatcounter") == "https://test-site.goatcounter.com/count"
            assert script.get_attribute("integrity") == GOATCOUNTER_INTEGRITY
            assert script.get_attribute("crossorigin") == "anonymous"
        finally:
            context.close()


class TestEventsOnTheSite:
    def test_the_count_script_copy_is_the_pinned_script(self):
        digest = base64.b64encode(hashlib.sha384(GOATCOUNTER_FIXTURE.read_bytes()).digest()).decode()
        assert f"sha384-{digest}" == GOATCOUNTER_INTEGRITY

    @pytest.mark.parametrize("entry", sorted(EDITION_ENTRIES))
    def test_a_click_into_an_edition_counts_where_it_came_from(self, playwright_browser, entry):
        path, link, expected = EDITION_ENTRIES[entry]
        context, page, trackers = _open_on_the_site(playwright_browser, path)
        try:
            page.locator(link).first.evaluate("link => link.click()")
            _wait_for_new_page(page, PRODUCTION_ORIGIN + path)
            assert [name for name, _title in _counted_events(trackers)] == [expected]
        finally:
            context.close()

    @pytest.mark.parametrize("href", ["/explore/", "https://example.org/"], ids=["language-switch", "other-site"])
    def test_a_link_within_an_edition_or_off_the_site_is_no_entry(self, playwright_browser, href):
        context, page, trackers = _open_on_the_site(playwright_browser, "/zh/explore/")
        context.route(
            "https://example.org/**",
            lambda route: route.fulfill(body="<!doctype html><title>elsewhere</title>", content_type="text/html"),
        )
        try:
            page.evaluate(CLICK_NEW_LINK, href)
            page.wait_for_url(lambda url: not url.endswith("/zh/explore/"), timeout=NAVIGATION_TIMEOUT_MS)
            page.wait_for_timeout(SETTLE_MS)
            assert _counted_events(trackers) == []
        finally:
            context.close()

    def test_an_event_reported_before_goatcounter_loads_is_sent_once_it_has(self, playwright_browser):
        context = playwright_browser.new_context()
        trackers = _record_trackers(context)
        _serve_goatcounter(context)
        _serve_site(context)
        # The page reports its event at once, while count.js is still on its way.
        context.route(
            f"{PRODUCTION_ORIGIN}/events.html",
            lambda route: route.fulfill(
                body='<!doctype html><title>events</title><script src="/js/3d-ice-analytics.js"></script>'
                '<script>window.__3dIceAnalytics.event("tour-start", "Guided tour started")</script>',
                content_type="text/html",
            ),
        )
        page = context.new_page()
        try:
            page.goto(PRODUCTION_ORIGIN + "/events.html", wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
            page.wait_for_function(GOATCOUNTER_READY, timeout=NAVIGATION_TIMEOUT_MS)
            page.wait_for_timeout(SETTLE_MS)
            assert _counted_events(trackers) == [("tour-start", "Guided tour started")]
        finally:
            context.close()
