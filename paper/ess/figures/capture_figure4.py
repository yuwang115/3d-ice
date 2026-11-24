"""Capture the panels of Figure 4 (Figure 3 until 2026-11-24): the ISMIP6 ensemble-mean geometry in 2300 for each
scenario, from two viewpoints, as the public edition draws it.

The public page drives its camera through the guide interface; this script exposes that
interface to itself by injecting one line into explore-guide.js as the page loads it (the
repository is not changed). Only the viewer element is captured.

    python capture_figure4.py <base url of static/> <output dir>
"""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE, OUT = sys.argv[1].rstrip("/"), Path(sys.argv[2])
ARGS = ["--host-resolver-rules=MAP www.googletagmanager.com ~NOTFOUND, MAP www.google-analytics.com ~NOTFOUND, MAP gc.zgo.at ~NOTFOUND"]
STATE = "() => { try { return JSON.parse(window.render_game_to_text()); } catch (e) { return null; } }"
SCENARIOS = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("ae10", "ae05", "ae14")
VIEWS = {
    # The public tour's high-emissions framing: West Antarctica, the Ross and the Weddell sectors.
    "west": {"lat": -82, "lon": -100, "fitKm": 3600, "azimuthDeg": -50, "elevationDeg": 52},
    # The Wilkes Subglacial Basin and its outlets, Cook and Ninnis, seen from the sea. The tour's
    # stops put the camera on the seaward side of longitude L with azimuth -(180 + L).
    "wilkes": {"lat": -72, "lon": 148, "fitKm": 2000, "azimuthDeg": 32, "elevationDeg": 58},
}
YEAR = 2300
HIDE_UI = """
  #status, .viewer-toolbar, .hud { visibility: hidden !important; }
"""


def wait(page, predicate, timeout=240, label="state"):
    end = time.time() + timeout
    while time.time() < end:
        state = page.evaluate(STATE)
        if state and predicate(state):
            return state
        time.sleep(0.5)
    raise TimeoutError(f"{label} not reached")


def set_check(page, selector, on):
    page.evaluate(
        """([sel, on]) => { const e = document.querySelector(sel); if (!e) throw new Error('no ' + sel);
             if (e.checked !== on) { e.checked = on; e.dispatchEvent(new Event('change', { bubbles: true })); } }""",
        [selector, on],
    )


def set_value(page, selector, value, event):
    page.evaluate(
        """([sel, value, event]) => { const e = document.querySelector(sel); if (!e) throw new Error('no ' + sel);
             e.value = value; e.dispatchEvent(new Event(event, { bubbles: true })); }""",
        [selector, str(value), event],
    )


def inject_guide_hook(route):
    response = route.fetch()
    body = response.text().replace(
        "export function mountExploreGuide(api) {",
        "export function mountExploreGuide(api) { window.__guideApi = api;",
        1,
    )
    if "window.__guideApi" not in body:
        raise RuntimeError("explore-guide.js no longer has the expected entry point")
    route.fulfill(response=response, body=body)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chromium", args=ARGS)
        page = browser.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1.5)
        errors = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.route("**/js/explore-guide.js*", inject_guide_hook)
        page.goto(f"{BASE}/explore/", wait_until="domcontentloaded")
        wait(page, lambda s: s.get("ready"), label="explorer ready")
        page.wait_for_function("() => Boolean(window.__guideApi)", timeout=120000)
        page.add_style_tag(content=HIDE_UI)
        set_check(page, "#showIceProjection", True)
        set_check(page, "#showSea", True)
        for scenario in SCENARIOS:
            set_value(page, "#projectionScenario", scenario, "change")
            wait(page, lambda s: s["iceProjection"]["loaded"] and s["iceProjection"]["scenario"] == scenario,
                 label=f"{scenario} loaded")
            set_value(page, "#projectionYear", YEAR, "input")
            wait(page, lambda s: s["iceProjection"]["active"] and s["iceProjection"]["year"] == YEAR,
                 label=f"{scenario} {YEAR}")
            for name, look in VIEWS.items():
                page.evaluate(
                    """(look) => { const api = window.__guideApi; const pose = api.getLookAtPose(look);
                         if (!pose) throw new Error('no pose'); return api.flyTo(pose, { durationMs: 0 }); }""",
                    look,
                )
                wait(page, lambda s: s.get("ready") and not s.get("cameraFlight"), label="camera")
                time.sleep(8)  # meshes, flowline recolouring and a few rendered frames
                path = OUT / f"figure4_{scenario}_{name}.png"
                page.locator("#viewer").screenshot(path=str(path))
                state = page.evaluate(STATE)["iceProjection"]
                print(path.name, state["scenario"], state["year"], state["iceCells"], state["seaLevelContributionMeters"])
        print("page errors:", errors)
        browser.close()


if __name__ == "__main__":
    main()
