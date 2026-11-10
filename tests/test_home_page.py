"""The home pages route visitors to the two editions and describe each of them.

The page runs in five parts: the hero, Explore (the data tower with the two edition cards
under it), the latest updates (three on the page, every one in a drawer), the source data
grouped by what they show, and feedback. Both locale pages share one structure, one
stylesheet and one script, so most checks run on both and a parity test keeps them in step.
Page links are relative to the page (so /zh/ reaches the Chinese editions), and shared assets
are absolute from the site root.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "static"
HOME_PAGES = {
    "en": STATIC_DIR / "index.html",
    "zh": STATIC_DIR / "zh" / "index.html",
}
LOCALES = tuple(HOME_PAGES)
TOUR = "./explore/?tour=1"
PUBLIC = "./explore/"
RESEARCH = "./tools/3D-interactive-cryosphere-explorer.html"
RUNTIME = STATIC_DIR / "tools" / "js" / "explorer-app.js"
# The data tower's layers, top to bottom, and those only the research edition offers.
TOWER_LAYERS = {
    "antarctica": [
        "ice-surface", "surface-change", "velocity", "projection", "sea-ice", "ocean", "basal-melt",
        "basal-friction", "subglacial", "bed", "rebound", "basins", "grounding-line",
    ],
    "greenland": [
        "ice-surface", "velocity", "sea-ice", "ocean", "basal-friction", "bed", "rebound", "basins", "grounding-line",
    ],
}
RESEARCH_ONLY_LAYERS = {"surface-change", "sea-ice", "basal-melt", "basal-friction", "subglacial", "basins"}
# Layers whose view goes by another name than <region>-<layer>.
TOWER_PRESETS = {
    "velocity": "{region}-velocity-flowlines",
    "ocean": "{region}-ocean-circulations",
    "subglacial": "{region}-subglacial-features",
    "bed": "{region}-bed-topography",
    "rebound": "{region}-isostatic-rebound",
    "basins": "{region}-basin-boundary",
}
# The page's sections, in order; Key Features is gone and Editions and Datasets are one.
SECTIONS = ["explore", "latest-updates", "source-data", "feedback"]
# The source data, one group per kind of dataset.
SOURCE_GROUPS = 8
# The updates on the page itself; the drawer holds every one.
UPDATES_ON_PAGE = 3
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class Node:
    def __init__(self, tag: str, attrs: dict[str, str], parent: "Node | None") -> None:
        self.tag = tag
        self.attrs = attrs
        self.parent = parent
        self.children: list[Node | str] = []

    @property
    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    def iter(self):
        for child in self.children:
            if isinstance(child, Node):
                yield child
                yield from child.iter()

    def find_all(self, tag: str | None = None, cls: str | None = None, attrs_present: str | None = None, **attrs: str) -> list["Node"]:
        found = []
        for node in self.iter():
            if tag and node.tag != tag:
                continue
            if cls and cls not in node.classes:
                continue
            if attrs_present and attrs_present not in node.attrs:
                continue
            if any(node.attrs.get(key.replace("_", "-")) != value for key, value in attrs.items()):
                continue
            found.append(node)
        return found

    def find(self, tag: str | None = None, cls: str | None = None, attrs_present: str | None = None, **attrs: str) -> "Node":
        matches = self.find_all(tag, cls, attrs_present, **attrs)
        assert matches, f"no <{tag or '*'} class={cls} {attrs_present or ''} {attrs}> under <{self.tag}>"
        return matches[0]

    def text(self) -> str:
        parts = []
        for child in self.children:
            parts.append(child if isinstance(child, str) else child.text())
        return re.sub(r"\s+", " ", " ".join(parts)).strip()


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("#document", {}, None)
        self.current = self.root

    def handle_starttag(self, tag, attrs):
        node = Node(tag, {key: value or "" for key, value in attrs}, self.current)
        self.current.children.append(node)
        if tag not in VOID_TAGS:
            self.current = node

    def handle_startendtag(self, tag, attrs):
        self.current.children.append(Node(tag, {key: value or "" for key, value in attrs}, self.current))

    def handle_endtag(self, tag):
        node = self.current
        while node is not self.root and node.tag != tag:
            node = node.parent
        if node is not self.root:
            self.current = node.parent

    def handle_data(self, data):
        self.current.children.append(data)


def tower_preset(region: str, layer: str) -> str:
    """The explorer view a data tower layer opens (static/tools/js/view-presets.js)."""
    return TOWER_PRESETS.get(layer, "{region}-" + layer).format(region=region)


def parse(locale: str) -> Node:
    builder = _TreeBuilder()
    builder.feed(HOME_PAGES[locale].read_text(encoding="utf-8"))
    return builder.root


def section_ids(doc: Node) -> list[str]:
    return [node.attrs["id"] for node in doc.find_all("section") if node.attrs.get("id")]


def runtime_source_urls() -> list[str]:
    source = RUNTIME.read_text(encoding="utf-8")
    urls = re.findall(r'text:\s*"[^"]+",\s*\n\s*url:\s*"([^"]+)"', source)
    assert len(urls) >= 10, "the runtime's source registry was not found"
    return sorted(set(urls))


def place_catalogue_source_urls() -> list[str]:
    """The sources the station and place-name catalogues credit, shown on each place's card."""
    urls = set()
    for catalogue in sorted((STATIC_DIR / "tools" / "data").glob("*_research_stations.json")) + sorted(
        (STATIC_DIR / "tools" / "data").glob("*_geographic_names.json")
    ):
        urls.update(item["url"] for item in json.loads(catalogue.read_text(encoding="utf-8"))["sources"])
    assert len(urls) >= 5, "the place catalogues' sources were not found"
    return sorted(urls)


@pytest.fixture(params=LOCALES)
def locale(request) -> str:
    return request.param


@pytest.mark.integration
class TestEntries:
    def test_the_hero_leads_with_the_guided_tour_and_offers_the_research_edition(self, locale):
        hero = parse(locale).find(cls="explorer-ice-hero")
        actions = hero.find("div", cls="explorer-actions")
        buttons = actions.find_all("a", cls="explorer-button")
        assert [button.attrs["href"] for button in buttons] == [TOUR, RESEARCH]
        assert "explorer-button--primary" in buttons[0].classes
        assert "explorer-button--primary" not in buttons[1].classes
        assert all(button.text() for button in buttons)

    def test_the_page_runs_explore_updates_data_and_feedback_after_the_hero(self, locale):
        doc = parse(locale)
        assert section_ids(doc) == SECTIONS
        ids = {node.attrs["id"] for node in doc.iter() if node.attrs.get("id")}
        for gone in ("key-features", "editions-title", "data-tower-title", "antarctica-features", "greenland-features"):
            assert gone not in ids, gone

    def test_the_explore_section_holds_the_tower_then_the_two_editions(self, locale):
        explore = parse(locale).find("section", id="explore")
        assert explore.attrs.get("aria-labelledby") == "explore-title"
        assert explore.find("h2", id="explore-title").text()
        parts = [node.attrs["id"] for node in explore.iter() if node.attrs.get("id") in ("data-tower", "editions")]
        assert parts == ["data-tower", "editions"]
        # Each names itself to the analytics, as the sections they used to be did.
        for part in parts:
            assert explore.find(id=part).attrs.get("data-entry-source") == part

    def test_the_tower_says_what_its_edition_tags_mean(self, locale):
        key = parse(locale).find(id="data-tower").find("p", cls="explorer-tower-key")
        # Outside the controls, which wait for the script: the key reads without it too.
        assert key.parent.attrs.get("data-tower-controls") is None
        tags = key.find_all("span", cls="explorer-tower-tag")
        assert [("explorer-tower-tag--public" in tag.classes, "explorer-tower-tag--research" in tag.classes) for tag in tags] == [
            (True, False),
            (False, True),
        ]
        assert key.text()

    def test_each_edition_has_an_introduction_and_a_way_in(self, locale):
        editions = parse(locale).find(id="editions")
        cards = editions.find_all("article", cls="explorer-edition-card")
        assert [card.attrs.get("data-edition") for card in cards] == ["public", "research"]
        expected_primary = {"public": TOUR, "research": RESEARCH}
        for card in cards:
            edition = card.attrs["data-edition"]
            title = card.find("h3")
            assert title.attrs.get("id") == f"{edition}-edition-title"
            assert card.attrs.get("aria-labelledby") == title.attrs["id"]
            audience = card.find("p", cls="explorer-edition-audience")
            assert audience.text()
            # The card wears the tag the tower's labels wear for the same edition.
            assert f"explorer-tower-tag--{edition}" in audience.find("span", cls="explorer-tower-tag").classes
            assert card.find("p", cls="explorer-edition-lead").text()
            assert 3 <= len(card.find("ul", cls="explorer-edition-points").find_all("li")) <= 4
            assert card.find("p", cls="explorer-edition-meta").text()
            primary = card.find("a", cls="explorer-button--primary")
            assert primary.attrs["href"] == expected_primary[edition]
        public_links = [a.attrs["href"] for a in cards[0].find_all("a")]
        assert PUBLIC in public_links, "the public card also opens the edition without the tour"

    def test_the_editions_can_be_compared_in_a_table(self, locale):
        compare = parse(locale).find("section", id="explore").find("details", cls="explorer-edition-compare")
        assert "open" not in compare.attrs, "the comparison starts folded"
        assert compare.find("summary").text()
        # On a phone the table scrolls sideways inside a region a keyboard can reach and name.
        wrap = compare.find("div", cls="explorer-edition-table-wrap")
        assert (wrap.attrs.get("role"), wrap.attrs.get("tabindex")) == ("region", "0")
        table = compare.find("table")
        caption = table.find("caption")
        assert caption.text() and wrap.attrs.get("aria-labelledby") == caption.attrs.get("id")
        header = table.find("thead").find_all("th")
        assert len(header) == 3
        rows = table.find("tbody").find_all("tr")
        assert len(rows) >= 8
        for row in rows:
            assert len(row.find_all("th")) == 1 and len(row.find_all("td")) == 2, row.text()
            for cell in row.find_all("td"):
                assert cell.text(), f"an empty cell in {row.text()!r}"

    def test_the_data_tower_holds_both_regions_from_the_ice_surface_down(self, locale):
        tower = parse(locale).find(id="data-tower")
        assert "data-tower" in tower.attrs
        lists = tower.find_all("ol", cls="explorer-tower-layers")
        assert [node.attrs.get("data-tower-region") for node in lists] == list(TOWER_LAYERS)
        for node in lists:
            region = node.attrs["data-tower-region"]
            layers = [item.attrs["data-tower-layer"] for item in node.find_all("li", cls="explorer-tower-layer")]
            assert layers == TOWER_LAYERS[region], region
            assert node.attrs.get("aria-label"), f"the {region} list names itself"
            assert node.attrs.get("data-shadow-src") == f"/tools/media/3d-ice/tower/{region}-shadow.webp"

    def test_every_tower_layer_opens_its_view_in_the_edition_that_has_it(self, locale):
        tower = parse(locale).find(id="data-tower")
        for item in tower.find_all("li", cls="explorer-tower-layer"):
            region = item.parent.attrs["data-tower-region"]
            layer = item.attrs["data-tower-layer"]
            assert item.attrs.get("data-src") == f"/tools/media/3d-ice/tower/{region}-{layer}.webp"
            links = item.find_all("a")
            assert len(links) == 1, f"{region} {layer}: one link per layer"
            link = links[0]
            opens = "research" if layer in RESEARCH_ONLY_LAYERS else "public"
            assert link.attrs.get("data-opens") == opens, f"{region} {layer}"
            target = PUBLIC if opens == "public" else RESEARCH
            assert link.attrs["href"] == f"{target}?region={region}&preset={tower_preset(region, layer)}", f"{region} {layer}"
            assert link.find(attrs_present="data-tower-name").text(), f"{region} {layer} has a name"
            assert link.find("span", cls="explorer-tower-layer-source").text(), f"{region} {layer} names its source"
            tag = link.find("span", cls="explorer-tower-tag")
            assert f"explorer-tower-tag--{opens}" in tag.classes and tag.text(), f"{region} {layer} says where it opens"

    def test_the_tower_controls_wait_for_the_script(self, locale):
        tower = parse(locale).find(id="data-tower")
        controls = tower.find("div", cls="explorer-tower-controls")
        # Without the script the region switch would do nothing, so it starts hidden.
        assert "hidden" in controls.attrs
        buttons = controls.find_all("button")
        assert [button.attrs.get("data-tower-region-button") for button in buttons] == list(TOWER_LAYERS)
        assert [button.attrs.get("aria-pressed") for button in buttons] == ["true", "false"]
        assert controls.find("div", cls="explorer-tower-switch").attrs.get("role") == "group"
        canvas = tower.find("canvas")
        assert canvas.attrs.get("aria-hidden") == "true", "the lists of links are what assistive technology reads"
        chip = tower.find(attrs_present="data-tower-open")
        assert "hidden" in chip.attrs and "{layer}" in chip.attrs.get("data-template", "")

    def test_the_page_shows_the_newest_updates_and_a_way_to_all_of_them(self, locale):
        doc = parse(locale)
        updates = doc.find("section", id="latest-updates")
        cards = updates.find_all("article", cls="explorer-update-card")
        assert len(cards) == UPDATES_ON_PAGE
        drawer = doc.find("dialog", id="all-updates")
        everything = [node.attrs["datetime"] for node in drawer.find_all("time", cls="explorer-update-date")]
        shown = [card.find("time", cls="explorer-update-date").attrs["datetime"] for card in cards]
        assert shown == everything[:UPDATES_ON_PAGE], "the page shows the drawer's newest updates"
        for card in cards:
            title = card.find("h3")
            assert card.attrs.get("aria-labelledby") == title.attrs.get("id") and title.text()
            assert card.find("span", cls="explorer-update-badge").text()
            assert len(card.find_all("a", cls="explorer-update-link")) == 1, "each update links to where it can be seen"
        trigger = updates.find("a", attrs_present="data-drawer-open")
        assert (trigger.attrs["href"], trigger.attrs["data-drawer-open"], trigger.attrs.get("aria-haspopup")) == (
            "#all-updates",
            "all-updates",
            "dialog",
        )

    def test_the_drawer_lists_every_update_newest_first(self, locale):
        drawer = parse(locale).find("dialog", id="all-updates")
        assert "open" not in drawer.attrs, "the drawer starts closed"
        assert drawer.attrs.get("aria-labelledby") == "all-updates-title"
        assert drawer.find("h2", id="all-updates-title").text()
        assert drawer.attrs.get("data-entry-source") == "all-updates"
        close = drawer.find(attrs_present="data-drawer-close")
        # Without the script the close link leaves the drawer, which then hides again.
        assert close.attrs.get("href") == "#latest-updates" and close.attrs.get("aria-label")
        items = drawer.find("ol", cls="explorer-drawer-list").find_all("li", cls="explorer-drawer-item")
        dates = [item.find("time", cls="explorer-update-date").attrs["datetime"] for item in items]
        assert len(items) >= 8
        assert dates == sorted(dates, reverse=True)
        for item in items:
            assert item.find("h3").text() and item.find("span", cls="explorer-update-badge").text()
            assert len([child for child in item.find_all("p") if "explorer-update-meta" not in child.classes]) == 1
        # The releases and features the page has announced before, still on record.
        for date in ("2026-10-06", "2026-09-30", "2026-09-22", "2026-08-13"):
            assert date in dates, date
        changelog = drawer.find("p", cls="explorer-drawer-foot").find("a")
        assert changelog.attrs["href"].endswith("/CHANGELOG.md")

    def test_update_badges_name_their_edition_in_its_colour(self, locale):
        doc = parse(locale)
        for badge in doc.find_all("span", cls="explorer-update-badge"):
            variants = [name for name in badge.classes if name.startswith("explorer-update-badge--")]
            assert variants in (["explorer-update-badge--both"], ["explorer-update-badge--public"], ["explorer-update-badge--research"]), badge.text()


@pytest.mark.integration
class TestContent:
    def test_source_data_lists_every_source_the_explorer_cites(self, locale):
        sources = parse(locale).find("section", id="source-data")
        listed = {a.attrs["href"] for a in sources.find_all("a")}
        missing = [url for url in runtime_source_urls() + place_catalogue_source_urls() if url not in listed]
        assert not missing, f"{locale} source data misses {missing}"

    def test_source_data_names_the_greenland_basins(self, locale):
        # The runtime names this product only by its file name, Greenland_Basins_PS_v1.4.2.
        sources = parse(locale).find("section", id="source-data")
        assert "https://doi.org/10.7280/D1WT11" in {a.attrs["href"] for a in sources.find_all("a")}

    def test_source_data_is_grouped_by_what_it_shows(self, locale):
        sources = parse(locale).find("section", id="source-data")
        groups = sources.find("div", cls="explorer-source-columns").find_all("div", cls="explorer-source-group")
        assert len(groups) == SOURCE_GROUPS
        entries = 0
        for group in groups:
            assert group.find("h3").text()
            items = group.find("ul", cls="explorer-source-list").find_all("li")
            assert items, group.find("h3").text()
            entries += len(items)
        # One line per dataset, with its paper or data alongside: no longer a list of 42 titles.
        assert entries <= 24

    def test_source_links_open_in_a_new_tab_safely(self, locale):
        sources = parse(locale).find("section", id="source-data")
        for link in sources.find("div", cls="explorer-source-columns").find_all("a"):
            assert link.attrs.get("target") == "_blank"
            assert "noopener" in link.attrs.get("rel", "")

    def test_the_hero_holds_the_one_page_title(self, locale):
        doc = parse(locale)
        titles = doc.find_all("h1")
        assert len(titles) == 1 and "explorer-page-title" in titles[0].classes
        hero = doc.find(cls="explorer-ice-hero")
        assert hero.find("h1") is titles[0]
        assert titles[0].find("img", cls="explorer-ice-logo").attrs.get("alt")
        assert titles[0].find("span", cls="explorer-hero-title").text()

    def test_in_page_links_point_at_sections_on_the_page(self, locale):
        doc = parse(locale)
        ids = {node.attrs["id"] for node in doc.iter() if node.attrs.get("id")}
        anchors = [a.attrs["href"][1:] for a in doc.find_all("a") if a.attrs.get("href", "").startswith("#")]
        assert anchors, "the page has in-page navigation"
        assert [anchor for anchor in anchors if anchor not in ids] == []

    def test_the_page_names_its_language(self, locale):
        html = parse(locale).find("html")
        expected = {"en": ("en", "en-US"), "zh": ("zh-CN", "zh-CN")}[locale]
        assert (html.attrs.get("lang"), html.attrs.get("data-locale")) == expected

    def test_structured_data_parses_and_mentions_the_tour(self, locale):
        blocks = [json.loads(node.text()) for node in parse(locale).find_all("script", type="application/ld+json")]
        kinds = [block["@type"] for block in blocks]
        assert kinds == ["WebApplication", "WebSite", "FAQPage"]
        features = " ".join(blocks[0]["featureList"])
        assert ("guided tour" if locale == "en" else "导览") in features


@pytest.mark.integration
class TestParity:
    def test_both_locales_share_one_structure(self):
        en, zh = parse("en"), parse("zh")
        assert section_ids(en) == section_ids(zh)

        def shape(doc: Node) -> list[tuple]:
            return [
                (node.tag, node.attrs.get("id"), node.attrs.get("data-edition"), node.attrs.get("data-tower-layer"),
                 node.attrs.get("data-tower-region"), node.attrs.get("data-src"), node.attrs.get("data-opens"),
                 node.attrs.get("href") if node.tag == "a" else None)
                for node in doc.find("body").iter()
                if node.attrs.get("id") or node.attrs.get("data-edition") or node.attrs.get("data-tower-layer")
                or node.attrs.get("data-tower-region")
                or (node.tag == "a" and not node.attrs.get("href", "").startswith("mailto:"))
            ]

        assert shape(en) == shape(zh)

    def test_both_locales_compare_the_same_features(self):
        def rows(locale):
            return len(parse(locale).find("details", cls="explorer-edition-compare").find("tbody").find_all("tr"))

        assert rows("en") == rows("zh")


@pytest.mark.integration
class TestAssets:
    def test_pages_load_the_rebuilt_stylesheets_and_scripts(self, locale):
        doc = parse(locale)
        styles = [node.attrs["href"] for node in doc.find_all("link", rel="stylesheet")]
        assert styles == ["/css/3d-ice-type.css", "/css/3d-ice-home.css"]
        scripts = [node.attrs.get("src") for node in doc.find_all("script") if node.attrs.get("src")]
        assert scripts.count("/js/3d-ice-locale.js") == 1
        home_script = doc.find("script", src="/js/3d-ice-home.js")
        assert "defer" in home_script.attrs

    def test_pages_drop_the_framework_leftovers(self, locale):
        page = HOME_PAGES[locale].read_text(encoding="utf-8")
        for leftover in ("wc.min.css", "sky.min.css", "Inter.var", "buttons.github.io", "Hugo Blox", "hugoblox", "task-list"):
            assert leftover not in page, leftover

    def test_every_local_file_a_page_references_exists(self, locale):
        doc = parse(locale)
        page_dir = HOME_PAGES[locale].parent
        refs = []
        for node in doc.iter():
            for attr in ("href", "src", "poster", "data-light-src", "data-dark-src", "data-src", "data-shadow-src"):
                value = node.attrs.get(attr, "")
                if value and not value.startswith(("http:", "https:", "mailto:", "#", "//")):
                    refs.append(value)
        assert refs
        for ref in refs:
            path = ref.split("?")[0].split("#")[0]
            target = STATIC_DIR / path.lstrip("/") if path.startswith("/") else page_dir / path
            if path.endswith("/"):
                target = target / "index.html"
            assert target.resolve().is_file(), f"{locale} references missing {ref}"

    def test_the_feedback_form_carries_its_messages_in_the_page_language(self, locale):
        form = parse(locale).find("form", id="faq-feedback-3d-ice-form")
        for key in ("data-label-submit", "data-label-sending", "data-status-sending", "data-status-sent", "data-status-failed"):
            assert form.attrs.get(key), key
        chinese = re.compile(r"[一-鿿]")
        assert bool(chinese.search(form.attrs["data-status-sent"])) == (locale == "zh")


@pytest.mark.parametrize(("locale", "path"), [("en", "/"), ("zh", "/zh/")])
def test_the_feedback_form_names_the_page_it_was_sent_from(locale, path):
    field = parse(locale).find("input", name="source_page")
    assert field.attrs["value"] == path
