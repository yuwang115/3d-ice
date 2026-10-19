"""E2E tests for behaviour the runtime gained in 0.3.1.

A phone opens on the whole ice sheet and loads the light Antarctic ocean package; a desktop
keeps the full one. The hydrology legends lay their tick labels out without overlaps once
their sections are shown. The Data Snapshot reports ice thickness and ice speed over
ice-covered cells, not over a grid that includes open ocean at 0.
"""

from __future__ import annotations

import re
import time

import pytest

pytestmark = pytest.mark.e2e

STATE = "JSON.parse(window.render_game_to_text())"
READY = "() => { try { return JSON.parse(window.render_game_to_text()).ready; } catch (e) { return false; } }"
CHECK = "(selector) => { const box = document.querySelector(selector); if (box && !box.checked) box.click(); }"


def _open(playwright_browser, url, viewport, **context_args):
    context = playwright_browser.new_context(viewport=viewport, **context_args)
    page = context.new_page()
    requests: list[str] = []
    page.on("request", lambda request: requests.append(request.url))
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_function(READY, timeout=120_000, polling=250)
    return context, page, requests


def _wait_for_request(page, requests, needle, timeout_s=60.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        matching = [url for url in requests if needle in url]
        if matching:
            return matching
        page.wait_for_timeout(250)
    raise AssertionError(f"no request for {needle!r}")


class TestNarrowViewers:
    def test_a_phone_opens_on_the_whole_ice_sheet_and_loads_the_light_ocean_package(
        self, playwright_browser, explore_url
    ):
        context, page, requests = _open(
            playwright_browser,
            explore_url,
            {"width": 390, "height": 844},
            is_mobile=True,
            has_touch=True,
            device_scale_factor=2,
        )
        try:
            # The framed opening view looks down on the pole from afar; the fixed pose that
            # suits a landscape viewer has a 45 deg field of view and crops to a close-up.
            page.wait_for_function(f"() => {STATE}.camera.fov > 50", timeout=60_000, polling=250)
            target = page.evaluate(f"() => {STATE}.camera.target")
            assert abs(target["x"]) < 5 and abs(target["z"]) < 5, target
            page.evaluate(CHECK, "#showOceanCurrents")
            ocean = _wait_for_request(page, requests, "ocean_currents")
            assert all("combined_light" in url for url in ocean), ocean
        finally:
            context.close()

    def test_a_desktop_keeps_the_full_ocean_package(self, playwright_browser, explore_url):
        context, page, requests = _open(playwright_browser, explore_url, {"width": 1440, "height": 900})
        try:
            assert page.evaluate(f"() => {STATE}.camera.fov") == 45
            page.evaluate(CHECK, "#showOceanCurrents")
            ocean = _wait_for_request(page, requests, "ocean_currents")
            assert not any("combined_light" in url for url in ocean), ocean
        finally:
            context.close()


class TestHydrologyLegends:
    def test_tick_labels_are_laid_out_without_overlaps(self, playwright_browser, explorer_url):
        context, page, _ = _open(playwright_browser, explorer_url, {"width": 1280, "height": 800})
        try:
            page.evaluate(CHECK, "#showEffectivePressure")
            page.evaluate(CHECK, "#showSubglacialChannels")
            for legend in ("effectivePressureLegendLabels", "channelLegendLabels"):
                spans = f"[...document.querySelectorAll('#{legend} span')]"
                # Laid out means pixel positions; percentages are the unlaid-out fallback.
                page.wait_for_function(
                    f"() => {spans}.length > 0 && {spans}.every((s) => s.style.left.endsWith('px'))",
                    timeout=30_000,
                    polling=250,
                )
                boxes = page.evaluate(
                    f"() => {spans}.map((s) => {{ const r = s.getBoundingClientRect(); return [r.left, r.right, r.top, r.bottom]; }})"
                )
                for index, first in enumerate(boxes):
                    for second in boxes[index + 1 :]:
                        horizontal = min(first[1], second[1]) - max(first[0], second[0])
                        vertical = min(first[3], second[3]) - max(first[2], second[2])
                        assert horizontal <= 0 or vertical <= 0, (legend, first, second)
        finally:
            context.close()


class TestDataSnapshot:
    def test_thickness_and_speed_are_summarised_over_the_ice(self, playwright_browser, explorer_url):
        context, page, _ = _open(playwright_browser, explorer_url, {"width": 1280, "height": 800})
        try:
            page.evaluate(CHECK, "#showVelocity")
            page.wait_for_function(
                "() => /Speed Quantiles: median [0-9]/.test(document.getElementById('metaList').textContent)",
                timeout=120_000,
                polling=500,
            )
            text = page.evaluate("() => document.getElementById('metaList').textContent")
            thickness = int(re.search(r"Mean Ice Thickness: ([0-9,]+) m", text).group(1).replace(",", ""))
            median = int(re.search(r"Speed Quantiles: median ([0-9,]+) m/yr", text).group(1).replace(",", ""))
            # Over every grid cell these read 590 m and 0 m/yr; over the ice, 1,940 m and 7 m/yr.
            assert 1_800 <= thickness <= 2_100, thickness
            assert median >= 3, median
        finally:
            context.close()
