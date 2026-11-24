"""Capture Figure 1(a) of the ESS manuscript: the research edition, with ice-surface speed and
surface-band WAOM2 streamlines over BedMachine Antarctica v4, at 1600 x 900 and device scale
1.5, the panel shown from its top with the flow and ocean groups open. Adapted from the script
that captured Figure 1(a) of the JOSS draft (paper.md) on 2026-10-18; the figure in the ESS
manuscript was captured on 2026-11-11.

Usage: python capture_figure1a.py <research page url> <output png> [time-open]

The capture then replaces panel (a) of docs/images/explorer-overview.jpg, whose panels (b) and
(c) stay as they are (composite_panel_a.py, checked by check_composite.py).
"""
import json
import sys
import time

from playwright.sync_api import sync_playwright

URL, OUT = sys.argv[1], sys.argv[2]
TIME_OPEN = len(sys.argv) > 3 and sys.argv[3] == "time-open"
ARGS = ["--host-resolver-rules=MAP www.googletagmanager.com ~NOTFOUND, MAP www.google-analytics.com ~NOTFOUND, MAP gc.zgo.at ~NOTFOUND"]
SET_CHECK = """([sel, on]) => { const e = document.querySelector(sel);
  if (!e) throw new Error('no control ' + sel);
  if (e.checked !== on) { e.checked = on; e.dispatchEvent(new Event('change', { bubbles: true })); } }"""
STATE = "() => { try { return JSON.parse(window.render_game_to_text()); } catch (e) { return null; } }"
PACKAGES = ("antarctic_ice_velocity_phase_v01_480.bin", "combined_cavity80km_remote_open_ocean.bin")
OPEN_GROUPS = {"scene": True, "flow": True, "ocean": True, "time": TIME_OPEN}


def wait(page, predicate, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        state = page.evaluate(STATE)
        if state and predicate(state):
            return state
        time.sleep(0.4)
    raise TimeoutError("state not reached")


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chromium", args=ARGS)
    page = browser.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1.5)
    errors = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    page.goto(URL, wait_until="domcontentloaded")
    # A 28 px page margin instead of 16, so the figure's "a" label sits on the panel's
    # corner and clears the wordmark. Nothing inside the panel or the viewer changes.
    page.add_style_tag(content=".app { padding: 28px !important; }")
    wait(page, lambda s: s.get("ready"))
    page.evaluate(SET_CHECK, ["#showVelocity", True])
    page.evaluate(SET_CHECK, ["#showOceanCurrents", True])
    time.sleep(1)
    for sel, on in (("#showOceanLayerSurface", True), ("#showOceanLayerUpper", False),
                    ("#showOceanLayerMid", False), ("#showOceanLayerLower", False)):
        page.evaluate(SET_CHECK, [sel, on])
    page.evaluate(SET_CHECK, ["#animateFlow", True])
    end = time.time() + 180
    while time.time() < end:
        names = page.evaluate("() => performance.getEntriesByType('resource').map(e => e.name).join(' ')")
        if all(name in names for name in PACKAGES):
            break
        time.sleep(0.5)
    else:
        raise TimeoutError("packages not fetched")
    wait(page, lambda s: s.get("ready") and not s.get("cameraFlight"))
    time.sleep(6)  # meshes, legends and the flow animation
    page.evaluate("""(groups) => { for (const [name, open] of Object.entries(groups)) {
        const d = document.querySelector(`aside.panel details[data-group="${name}"]`); if (d) d.open = open; }
        document.querySelector('aside.panel').scrollTop = 0; }""", OPEN_GROUPS)
    time.sleep(1.5)
    page.screenshot(path=OUT)
    geometry = page.evaluate("""() => {
      const box = (sel) => { const e = document.querySelector(sel); if (!e) return null;
        const r = e.getBoundingClientRect(); return [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)]; };
      return { panel: box('aside.panel'), masthead: box('.rp-masthead'), wordmark: box('.rp-masthead__wordmark, .rp-wordmark, .rp-masthead h1'),
               viewer: box('#viewer'), scrollHeight: document.querySelector('aside.panel').scrollHeight,
               open: [...document.querySelectorAll('aside.panel details[open]')].map((d) => d.dataset.group || d.id) };
    }""")
    s = page.evaluate(STATE)
    print("toggles on:", sorted(k for k, v in s["toggles"].items() if v))
    print("geometry (CSS px):", json.dumps(geometry))
    print("page errors:", errors)
    browser.close()
