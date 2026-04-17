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
    page = context.new_page()
    page.goto(explorer_url, wait_until="networkidle", timeout=30_000)
    yield page
    context.close()


@pytest.fixture
def playwright_browser(request):
    """Launch a headless Chromium browser."""
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    yield browser
    browser.close()
    pw.stop()


class TestExplorerLoads:
    def test_page_loads_without_errors(self, playwright_browser, explorer_url):
        context = playwright_browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.goto(explorer_url, wait_until="networkidle", timeout=30_000)
        assert len(errors) == 0, f"Page errors: {errors}"
        context.close()

    def test_canvas_exists(self, page):
        canvas = page.query_selector("canvas")
        assert canvas is not None, "WebGL canvas not found"
        box = canvas.bounding_box()
        assert box is not None
        assert box["width"] > 0
        assert box["height"] > 0

    def test_title_is_set(self, page):
        title = page.title()
        assert "3D" in title or "ICE" in title or "Cryosphere" in title


class TestHomePage:
    def test_home_loads(self, playwright_browser, home_url):
        context = playwright_browser.new_context()
        page = context.new_page()
        response = page.goto(home_url, wait_until="domcontentloaded", timeout=15_000)
        assert response is not None
        assert response.status == 200
        context.close()

    def test_mit_license_in_footer(self, playwright_browser, home_url):
        context = playwright_browser.new_context()
        page = context.new_page()
        page.goto(home_url, wait_until="domcontentloaded", timeout=15_000)
        footer_text = page.text_content("footer") or ""
        assert "MIT" in footer_text
        context.close()
