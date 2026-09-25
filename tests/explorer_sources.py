"""Source text of the explorer pages together with the shared runtime they load.

Both locale pages are thin HTML shells around one module (static/tools/js/explorer-app.js)
and one stylesheet (static/tools/css/explorer.css). Wiring checks that look for runtime
code therefore read each page plus whatever local module scripts and stylesheets that page
actually links, so a page that stopped loading the shared runtime would fail them.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"
EXPLORER_PAGES = (
    STATIC_DIR / "tools" / "3D-interactive-cryosphere-explorer.html",
    STATIC_DIR / "zh" / "tools" / "3D-interactive-cryosphere-explorer.html",
)

_LINKED_ASSET = re.compile(
    r'<script type="module" src="(?P<script>[^"]+)"></script>'
    r'|<link rel="stylesheet" href="(?P<style>[^"]+)" />'
)


def linked_local_assets(page: Path) -> list[Path]:
    """Module scripts and stylesheets the page loads from this repository, in page order."""
    assets = []
    for match in _LINKED_ASSET.finditer(page.read_text(encoding="utf-8")):
        ref = match.group("script") or match.group("style")
        if ref.startswith(("http:", "https:", "//")):
            continue
        base = STATIC_DIR if ref.startswith("/") else page.parent
        assets.append((base / ref.lstrip("/")).resolve())
    return assets


def explorer_source(page: Path) -> str:
    """The page's HTML followed by every local module and stylesheet it links."""
    parts = [page.read_text(encoding="utf-8")]
    parts.extend(asset.read_text(encoding="utf-8") for asset in linked_local_assets(page))
    return "\n".join(parts)
