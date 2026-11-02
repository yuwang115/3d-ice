"""The panel's masthead in both editions, and the public edition's panel.

Every explorer page opens its panel with the same masthead: the site's icon and the
wordmark, the ⓘ menu, then a row with the edition switch and the language switch. The
switch is a pair of links, so the analytics counts a move between editions as it counts
any link into one (js/3d-ice-analytics.js).

The public panel takes the research panel's look, with four groups that stay open and each
legend under the layer that draws it. Its copy is the copy the author reviewed, moved.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from urllib.parse import urljoin, urlsplit

import pytest

from tests.test_home_page import STATIC_DIR, Node, _TreeBuilder

PAGES = {
    ("research", "en"): STATIC_DIR / "tools" / "3D-interactive-cryosphere-explorer.html",
    ("research", "zh"): STATIC_DIR / "zh" / "tools" / "3D-interactive-cryosphere-explorer.html",
    ("public", "en"): STATIC_DIR / "explore" / "index.html",
    ("public", "zh"): STATIC_DIR / "zh" / "explore" / "index.html",
}
EDITION_NAMES = {"en": ("Public", "Research"), "zh": ("科普版", "专业版")}
ICON = "/tools/3d-ice-icon.png"

# The public groups in order, by heading, and the legends each holds after its layer.
PUBLIC_GROUPS = ("iceHeading", "motionHeading", "futureHeading", "meltHeading")
PUBLIC_LEGENDS = {
    "bedLegendSection": ("showBed", None),
    "velocityLegendSection": ("showFlowline", "showOceanCurrents"),
    "oceanLegendSection": ("showOceanCurrents", "showSea"),
    "projectionLegendSection": ("showIceProjection", None),
}


def parse_page(path) -> Node:
    builder = _TreeBuilder()
    builder.feed(path.read_text(encoding="utf-8"))
    return builder.root


def page_url(path) -> str:
    return "https://3d-ice.com/" + path.relative_to(STATIC_DIR).as_posix()


def site_path(path, href: str) -> str:
    """The site path a link on the page leads to, an index page by its folder."""
    target = urlsplit(urljoin(page_url(path), href)).path
    return target[: -len("index.html")] if target.endswith("/index.html") else target


def ancestors(node: Node) -> list[Node]:
    chain = []
    while node.parent is not None:
        node = node.parent
        chain.append(node)
    return chain


def order_of(root: Node) -> dict[str, int]:
    return {node.attrs["id"]: index for index, node in enumerate(root.iter()) if node.attrs.get("id")}


@pytest.fixture(params=sorted(PAGES), ids=lambda key: f"{key[0]}-{key[1]}")
def page(request):
    edition, locale = request.param
    path = PAGES[request.param]
    return edition, locale, path, parse_page(path)


def test_the_wordmark_has_the_site_icon_in_front(page) -> None:
    _edition, _locale, path, root = page
    wordmark = root.find("h1", cls="rp-wordmark")
    icon = wordmark.children[0]
    assert isinstance(icon, Node) and icon.tag == "img" and "rp-wordmark__icon" in icon.classes
    # The name says "3D ICE", so the icon is decoration.
    assert icon.attrs["alt"] == ""
    assert site_path(path, icon.attrs["src"]) == ICON
    assert (STATIC_DIR / PurePosixPath(ICON).relative_to("/")).is_file()
    assert wordmark.find("span", cls="rp-wordmark__name").text().startswith("3D ICE")


def test_the_masthead_rows_hold_the_wordmark_and_menu_then_the_two_switches(page) -> None:
    _edition, _locale, _path, root = page
    masthead = root.find("header", cls="rp-masthead")
    first, second = masthead.find_all("div", cls="rp-masthead__row")
    assert first.find("h1", cls="rp-wordmark") and first.find("button", id="panelInfoButton")
    assert second.find("nav", cls="rp-editions") and second.find("div", id="panelLocaleSwitcher")
    assert masthead.find("div", id="panelInfoMenu").attrs.get("hidden") == ""
    assert masthead.find_all(cls="rp-edition-tag") == []


def test_the_edition_switch_marks_this_edition_and_links_to_the_other_in_this_language(page) -> None:
    edition, locale, path, root = page
    switch = root.find("nav", cls="rp-editions")
    options = switch.find_all(cls="rp-editions__option")
    assert [option.text() for option in options] == list(EDITION_NAMES[locale])
    current = [option for option in options if option.attrs.get("aria-current") == "page"]
    links = [option for option in options if option.tag == "a"]
    assert len(current) == 1 and current[0].tag == "span"
    assert current[0].text() == EDITION_NAMES[locale][0 if edition == "public" else 1]
    assert len(links) == 1 and "data-edition-link" in links[0].attrs


def test_the_edition_link_names_the_other_edition_s_page(page) -> None:
    edition, locale, path, root = page
    link = root.find("a", cls="rp-editions__option")
    other = PAGES[("research" if edition == "public" else "public", locale)]
    expected = "/" + other.relative_to(STATIC_DIR).as_posix()
    assert site_path(path, link.attrs["href"]) == expected.removesuffix("index.html")
    # No query in the page itself: the runtime adds the region once it knows it.
    assert "?" not in link.attrs["href"]


# ------------------------------------------------------------------ the public panel


@pytest.fixture(params=["en", "zh"])
def public(request):
    path = PAGES[("public", request.param)]
    return request.param, path, parse_page(path)


def test_the_public_panel_has_no_logo_banner_and_keeps_its_intro_in_the_menu(public) -> None:
    _locale, path, root = public
    html = path.read_text(encoding="utf-8")
    assert "panel-brand" not in html and "about-edition" not in html
    menu = root.find("div", id="panelInfoMenu")
    assert menu.find("p", id="panelSubtitle")
    assert menu.find("p", cls="rp-info-menu__about").text()
    # The heading still names the page in full for screen readers and search engines.
    assert len(root.find_all("h1")) == 1
    assert ":" in root.find("h1").text() or "：" in root.find("h1").text()


def test_the_public_groups_stay_open_in_order(public) -> None:
    _locale, _path, root = public
    panel = root.find("aside", cls="panel")
    groups = panel.find_all("section", cls="rp-group")
    assert [group.find("h2", cls="rp-group__heading").attrs["id"] for group in groups] == list(PUBLIC_GROUPS)
    assert panel.find_all("details") == []


def test_each_public_legend_sits_in_its_layer_s_group_after_the_layer(public) -> None:
    _locale, _path, root = public
    order = order_of(root)
    by_id = {node.attrs["id"]: node for node in root.iter() if node.attrs.get("id")}
    for legend_id, (layer_id, next_layer_id) in PUBLIC_LEGENDS.items():
        legend = by_id[legend_id]
        assert "rp-legend" in legend.classes
        group = next(node for node in ancestors(legend) if "rp-group" in node.classes)
        assert group in ancestors(by_id[layer_id]), f"{legend_id} is not in {layer_id}'s group"
        assert order[legend_id] > order[layer_id], f"{legend_id} comes before {layer_id}"
        if next_layer_id:
            assert order[legend_id] < order[next_layer_id], f"{legend_id} comes after {next_layer_id}"


def test_the_public_viewer_holds_reset_view_and_the_panel_no_buttons(public) -> None:
    _locale, _path, root = public
    viewer = root.find("section", id="viewerShell")
    toolbar = viewer.find("div", cls="viewer-toolbar")
    assert [node.attrs["id"] for node in toolbar.find_all("button")] == [
        "viewerTourButton",
        "interactionToggle",
        "resetView",
        "viewerFullscreenToggle",
    ]
    assert root.find_all(id="fullscreenToggle") == []


def test_the_region_select_hides_behind_its_buttons(public) -> None:
    _locale, _path, root = public
    where = root.find("section", id="viewControlsSection")
    buttons = where.find("div", cls="rp-segmented").find_all("button")
    assert [button.attrs["data-value"] for button in buttons] == ["antarctica", "greenland"]
    select = where.find("select", id="regionPreset")
    assert "hidden" in select.attrs
    assert [option.text() for option in select.find_all("option")] == [button.text() for button in buttons]


def test_both_public_locales_carry_the_same_ids_in_the_same_order() -> None:
    english, chinese = (list(order_of(parse_page(PAGES[("public", locale)]))) for locale in ("en", "zh"))
    assert english == chinese
