"""What a host that serves the compatibility bundle relies on.

yuwang.blog mounts the bundle at its site root, so both editions run there at the same
paths as on 3d-ice.com. It also builds its own copy of the home page at /tools/3d-ice/
from the bundle's home/ sources: it takes the page's header and main content, resolves
their relative links against the page's own address, and places them inside its own
navbar and footer. These tests pin that contract, so a change here that would break
yuwang.blog fails in this repository first.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlsplit

import pytest

from tests.test_home_page import HOME_PAGES, LOCALES, ROOT_DIR, STATIC_DIR, Node, _TreeBuilder, parse


BUILDER = ROOT_DIR / "scripts" / "build_compat_bundle.mjs"
PAGE_URLS = {"en": "https://3d-ice.com/", "zh": "https://3d-ice.com/zh/"}
RESEARCH_PAGES = {
    "en": STATIC_DIR / "tools" / "3D-interactive-cryosphere-explorer.html",
    "zh": STATIC_DIR / "zh" / "tools" / "3D-interactive-cryosphere-explorer.html",
}
PUBLIC_PAGES = {
    "en": STATIC_DIR / "explore" / "index.html",
    "zh": STATIC_DIR / "zh" / "explore" / "index.html",
}
# What the home pages load; yuwang.blog loads the same files from the bundle. The typefaces
# come too: yuwang.blog declares the same variable fonts at single weights, which draws
# other weights than 3d-ice.com does.
HOME_ASSETS = {"/css/3d-ice-type.css", "/css/3d-ice-home.css", "/js/3d-ice-locale.js", "/js/3d-ice-home.js"}
# Site furniture every host has its own copy of.
HOST_FURNITURE = re.compile(r"^/(favicon[^/]*|apple-touch-icon[^/]*)$")
TYPE_CSS = STATIC_DIR / "css" / "3d-ice-type.css"
URL_ATTRS = ("href", "src", "poster", "data-light-src", "data-dark-src", "data-src")
# The explorer pages resolve data-src images against their asset base, not the page.
EXPLORER_ASSET_BASE = "/tools/"


def served_paths() -> list[str]:
    block = re.search(r"const SERVED_PATHS = Object\.freeze\(\[(.*?)\]\);", BUILDER.read_text(), re.S)
    assert block, "scripts/build_compat_bundle.mjs should declare SERVED_PATHS"
    return re.findall(r'"([^"]+)"', block.group(1))


def in_bundle(path: str) -> bool:
    """Whether a site path is served from the bundle (a directory path means its index.html)."""
    relative = path.lstrip("/")
    if not relative or relative.endswith("/"):
        relative += "index.html"
    for served in served_paths():
        if relative == served or relative.startswith(f"{served}/"):
            return (STATIC_DIR / relative).is_file()
    return False


def parse_file(path) -> Node:
    builder = _TreeBuilder()
    builder.feed(path.read_text(encoding="utf-8"))
    return builder.root


def local_urls(nodes: list[Node], page_url: str, asset_base: str | None = None) -> set[str]:
    """Site paths the nodes reference, resolved against the page's address (data-src
    against the asset base, when the page has one)."""
    paths = set()
    for node in nodes:
        for element in [node, *node.iter()]:
            for attr in URL_ATTRS:
                value = element.attrs.get(attr, "").strip()
                if not value or value.startswith(("#", "mailto:", "data:", "javascript:")):
                    continue
                base = urljoin(page_url, asset_base) if asset_base and attr == "data-src" else page_url
                parts = urlsplit(urljoin(base, value))
                if parts.netloc == urlsplit(page_url).netloc:
                    paths.add(parts.path)
            for candidate in element.attrs.get("srcset", "").split(","):
                if candidate.strip():
                    parts = urlsplit(urljoin(page_url, candidate.split()[0]))
                    if parts.netloc == urlsplit(page_url).netloc:
                        paths.add(parts.path)
    return paths


def embedded_parts(doc: Node) -> tuple[Node, Node]:
    shells = doc.find_all("div", cls="explorer-page-shell--ice")
    assert len(shells) == 1, "one .explorer-page-shell--ice wraps the page"
    headers = [child for child in shells[0].children if isinstance(child, Node) and child.tag == "header"]
    mains = [child for child in shells[0].children if isinstance(child, Node) and child.tag == "main"]
    assert len(headers) == 1 and "explorer-page-header" in headers[0].classes
    assert len(mains) == 1 and mains[0].attrs.get("id") == "main"
    return headers[0], mains[0]


@pytest.mark.parametrize("locale", LOCALES)
class TestHomePageEmbedContract:
    def test_the_shell_holds_exactly_one_header_and_one_main(self, locale):
        header, main = embedded_parts(parse(locale))
        assert header.find("h1", cls="explorer-page-title")
        assert main.find_all("section")

    def test_every_local_link_in_the_embedded_parts_is_served_by_the_bundle(self, locale):
        header, main = embedded_parts(parse(locale))
        missing = sorted(
            path
            for path in local_urls([header, main], PAGE_URLS[locale])
            if not in_bundle(path) and not HOST_FURNITURE.match(path)
        )
        assert missing == [], f"yuwang.blog's copy would link to paths the bundle does not carry: {missing}"

    def test_the_page_loads_only_assets_a_host_can_supply(self, locale):
        doc = parse(locale)
        loaded = {
            urlsplit(node.attrs["src"]).path for node in doc.find_all("script") if node.attrs.get("src", "").startswith("/")
        } | {
            urlsplit(node.attrs["href"]).path
            for node in doc.find_all("link")
            if node.attrs.get("rel") == "stylesheet" and node.attrs.get("href", "").startswith("/")
        }
        assert loaded == HOME_ASSETS, "a new home-page asset must be added to the bundle and to yuwang.blog's layout"
        assert all(in_bundle(path) for path in HOME_ASSETS)

    def test_scripts_outside_the_head_are_external(self, locale):
        body = parse(locale).find("body")
        for script in body.find_all("script"):
            assert script.attrs.get("src", "").startswith("https://"), "yuwang.blog copies only external body scripts"


@pytest.mark.parametrize("locale", LOCALES)
class TestBundledEditionPages:
    def test_the_research_page_links_to_the_tour_on_the_same_site(self, locale):
        doc = parse_file(RESEARCH_PAGES[locale])
        tour = [node.attrs["href"] for node in doc.find_all("a") if "tour=1" in node.attrs.get("href", "")]
        assert tour == ["../explore/?tour=1"], "the tour link must stay on whichever site serves the page"

    @pytest.mark.parametrize("pages", [RESEARCH_PAGES, PUBLIC_PAGES], ids=["research", "public"])
    def test_every_local_reference_is_served_by_the_bundle(self, locale, pages):
        page = pages[locale]
        page_url = "https://3d-ice.com/" + page.relative_to(STATIC_DIR).as_posix()
        missing = sorted(
            path
            for path in local_urls([parse_file(page)], page_url, EXPLORER_ASSET_BASE)
            if not in_bundle(path) and not HOST_FURNITURE.match(path)
        )
        assert missing == [], f"{PurePosixPath(page.relative_to(STATIC_DIR))} references paths outside the bundle: {missing}"


def test_the_bundle_carries_every_font_file_the_type_stylesheet_uses():
    fonts = set(re.findall(r"url\(['\"]?(/fonts/[^'\")]+)['\"]?\)", TYPE_CSS.read_text()))
    assert fonts, "css/3d-ice-type.css should declare the self-hosted typefaces"
    assert sorted(path for path in fonts if not in_bundle(path)) == []


def test_the_bundle_never_serves_a_root_home_page():
    assert not {"index.html", "zh/index.html", "zh"} & set(served_paths())
    assert set(HOME_PAGES) == set(PAGE_URLS)
