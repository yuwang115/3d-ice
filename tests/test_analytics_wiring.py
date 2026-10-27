"""Pages reach analytics only through static/js/3d-ice-analytics.js.

That script decides where trackers may load: on 3d-ice.com, in a top-level window. A page or
script that named a tracker itself would bypass that check, and count local previews, test
runs and copies of the site again. tests/e2e/test_analytics.py checks the behaviour in a
browser.
"""

from __future__ import annotations

from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
LOADER = STATIC_DIR / "js" / "3d-ice-analytics.js"
TRACKER_HOSTS = ("googletagmanager.com", "google-analytics.com", "gc.zgo.at", "goatcounter.com")


def test_only_the_analytics_script_names_a_tracker():
    sources = [path for pattern in ("*.html", "*.js") for path in STATIC_DIR.rglob(pattern) if path != LOADER]
    assert sources, "static/ should hold the site's pages and scripts"
    offenders = sorted(
        f"{path.relative_to(STATIC_DIR)}: {host}"
        for path in sources
        for host in TRACKER_HOSTS
        if host in path.read_text(encoding="utf-8", errors="replace")
    )
    assert offenders == [], "load trackers through static/js/3d-ice-analytics.js"


def test_the_analytics_script_names_the_trackers_it_loads():
    source = LOADER.read_text(encoding="utf-8")
    assert "https://www.googletagmanager.com/gtag/js?id=" in source
    assert "https://gc.zgo.at/count.v5.js" in source
