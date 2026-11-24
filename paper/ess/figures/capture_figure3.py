"""Capture the panels of Figure 3 of the ESS manuscript: the research edition, with the panel's
controls in view, showing (a) the projection layer and (b) the observed elevation change layer.

The camera is placed with the explorer's own look-at framing (the one the public tour uses),
which this script exposes to itself by adding one line to explorer-app.js as the page loads it;
the repository is not changed.

    python capture_figure3.py <research page url> <output dir> [projection|surface ...]

Captured on 2026-11-24 from 8fa6068 (pull request #16, merged into main as 12c0a61), with the
repository's static/ folder served by a local file server, at 1100 x 688 CSS px and device
scale 3.
"""
import json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

URL, OUT = sys.argv[1], Path(sys.argv[2])
ONLY = set(sys.argv[3:])
VIEWPORT = {"width": 1100, "height": 688}
SCALE = 3
ARGS = ["--host-resolver-rules=MAP www.googletagmanager.com ~NOTFOUND, MAP www.google-analytics.com ~NOTFOUND, "
        "MAP gc.zgo.at ~NOTFOUND, MAP *.goatcounter.com ~NOTFOUND"]
STATE = "() => { try { return JSON.parse(window.render_game_to_text()); } catch (e) { return null; } }"
HOOK = "window.render_game_to_text = () => JSON.stringify(collectExplorerState());"
EXPOSE = "\nwindow.__cam = { lookAt: getLookAtPose, apply: applyCameraPose, get: getCurrentCameraPose };"
SET = """([sel, value, event]) => { const e = document.querySelector(sel); if (!e) throw new Error('no ' + sel);
         e.value = value; e.dispatchEvent(new Event(event, { bubbles: true })); }"""
# Collapse the groups a reader would close, then bring a control into view inside the panel.
PANEL = """([closed, selector, margin]) => {
  for (const g of closed) { const d = document.querySelector(`aside.panel details[data-group="${g}"]`); if (d) d.open = false; }
  const panel = document.querySelector('aside.panel'); const el = document.querySelector(selector);
  const top = el.getBoundingClientRect().top - panel.getBoundingClientRect().top + panel.scrollTop;
  panel.scrollTop = Math.max(0, top - margin); return [panel.scrollTop, panel.scrollHeight, panel.clientHeight]; }"""
# The onboarding hint over the viewer is hidden; nothing else on the page is changed.
HIDE_HINT = "#interactionHint { display: none !important; }"

PANELS = {
    "projection": {
        "preset": "antarctica-projection", "year": 2200,
        "look": {"lat": -82, "lon": -100, "fitKm": 3400, "azimuthDeg": -50, "elevationDeg": 50},
        "closed": ["scene", "flow"], "scrollTo": "#interactiveScenariosSection", "margin": 6,
    },
    "surface": {
        "preset": "antarctica-surface-change",
        "look": {"lat": -77.5, "lon": -105, "fitKm": 1900, "azimuthDeg": -75, "elevationDeg": 42},
        "closed": ["scene", "flow"], "scrollTo": "#surfaceChangeRow", "margin": 6,
    },
}


def inject(route):
    response = route.fetch()
    body = response.text()
    if HOOK not in body:
        raise RuntimeError("explorer-app.js no longer has the expected state hook")
    route.fulfill(response=response, body=body.replace(HOOK, HOOK + EXPOSE, 1))


def wait(page, predicate, timeout=240, label="state"):
    end = time.time() + timeout
    while time.time() < end:
        state = page.evaluate(STATE)
        if state and predicate(state):
            return state
        time.sleep(0.5)
    raise TimeoutError(f"{label} not reached")


def capture(browser, name, spec):
    page = browser.new_page(viewport=VIEWPORT, device_scale_factor=SCALE)
    errors = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    page.route("**/tools/js/explorer-app.js*", inject)
    page.goto(f"{URL}?preset={spec['preset']}", wait_until="domcontentloaded")
    wait(page, lambda s: s.get("ready"), label="ready")
    if name.startswith("projection"):
        page.evaluate(SET, ["#projectionScenario", spec.get("scenario", "ae05"), "change"])
        wait(page, lambda s: s["iceProjection"]["loaded"] and s["iceProjection"]["scenario"] == spec.get("scenario", "ae05"),
             label="scenario")
        page.evaluate(SET, ["#projectionColorMode", spec.get("colorMode", "change"), "change"])
        page.evaluate(SET, ["#projectionYear", str(spec.get("year", 2300)), "input"])
        wait(page, lambda s: s["iceProjection"]["year"] == spec.get("year", 2300), label="year")
    else:
        wait(page, lambda s: s["surfaceChange"]["shown"] and s["surfaceChange"]["epochCount"], label="surface change")
        page.evaluate(SET, ["#surfaceChangeBasin", spec.get("basin", "basin:9"), "change"])
        wait(page, lambda s: s["surfaceChange"]["selection"] == spec.get("basin", "basin:9") and s["surfaceChange"]["chartShown"],
             label="basin")
    pose = page.evaluate("(look) => { const pose = window.__cam.lookAt(look); return window.__cam.apply(pose); }", spec["look"])
    wait(page, lambda s: s.get("ready") and not s.get("cameraFlight"), label="camera")
    page.add_style_tag(content=HIDE_HINT)
    scroll = page.evaluate(PANEL, [spec["closed"], spec["scrollTo"], spec["margin"]])
    time.sleep(8)  # meshes, flowline recolouring, the chart and a few frames
    path = OUT / f"figure3_{name}.png"
    page.screenshot(path=str(path))
    state = page.evaluate(STATE)
    print(path.name, json.dumps({"pose": pose, "panelScroll": scroll}))
    print("  ", json.dumps(state["iceProjection" if name.startswith("projection") else "surfaceChange"])[:400])
    print("   readout:", page.evaluate("(sel) => document.querySelector(sel)?.textContent.trim()",
                                      "#projectionReadout" if name.startswith("projection") else "#surfaceChangeReadout"))
    print("   errors:", errors)
    page.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chromium", args=ARGS)
        for name, spec in PANELS.items():
            if not ONLY or name in ONLY:
                capture(browser, name, spec)
        browser.close()


if __name__ == "__main__":
    main()
