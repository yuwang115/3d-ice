"""The home pages route visitors to the two editions and describe each of them.

Both locale pages share one structure, one stylesheet and one script, so most checks run
on both and a parity test keeps them in step. Page links are relative to the page (so
/zh/ reaches the Chinese editions), and shared assets are absolute from the site root.
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

    def find_all(self, tag: str | None = None, cls: str | None = None, **attrs: str) -> list["Node"]:
        found = []
        for node in self.iter():
            if tag and node.tag != tag:
                continue
            if cls and cls not in node.classes:
                continue
            if any(node.attrs.get(key.replace("_", "-")) != value for key, value in attrs.items()):
                continue
            found.append(node)
        return found

    def find(self, tag: str | None = None, cls: str | None = None, **attrs: str) -> "Node":
        matches = self.find_all(tag, cls, **attrs)
        assert matches, f"no <{tag or '*'} class={cls} {attrs}> under <{self.tag}>"
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

    def test_the_editions_section_follows_the_hero(self, locale):
        ids = section_ids(parse(locale))
        assert ids[0] == "editions"
        assert ids.index("editions") < ids.index("antarctica-features")

    def test_each_edition_has_an_introduction_and_a_way_in(self, locale):
        editions = parse(locale).find("section", id="editions")
        assert editions.attrs.get("aria-labelledby") == "editions-title"
        assert editions.find("h2", id="editions-title").text()
        cards = editions.find_all("article", cls="explorer-edition-card")
        assert [card.attrs.get("data-edition") for card in cards] == ["public", "research"]
        expected_primary = {"public": TOUR, "research": RESEARCH}
        for card in cards:
            edition = card.attrs["data-edition"]
            title = card.find("h3")
            assert title.attrs.get("id") == f"{edition}-edition-title"
            assert card.attrs.get("aria-labelledby") == title.attrs["id"]
            assert card.find("p", cls="explorer-edition-audience").text()
            assert card.find("p", cls="explorer-edition-lead").text()
            assert len(card.find("ul", cls="explorer-edition-points").find_all("li")) >= 4
            assert card.find("p", cls="explorer-edition-meta").text()
            primary = card.find("a", cls="explorer-button--primary")
            assert primary.attrs["href"] == expected_primary[edition]
        public_links = [a.attrs["href"] for a in cards[0].find_all("a")]
        assert PUBLIC in public_links, "the public card also opens the edition without the tour"

    def test_the_editions_can_be_compared_in_a_table(self, locale):
        editions = parse(locale).find("section", id="editions")
        compare = editions.find("details", cls="explorer-edition-compare")
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

    def test_every_preview_names_the_editions_that_offer_it_and_opens_one(self, locale):
        doc = parse(locale)
        cards = doc.find_all("article", cls="explorer-video-card")
        assert len(cards) == 6
        for card in cards:
            slot = card.attrs["data-demo-slot"]
            editions = card.attrs.get("data-editions")
            assert editions in {"public research", "research"}, slot
            tags = card.find_all("span", cls="explorer-edition-tag")
            assert len(tags) == len(editions.split()), slot
            link = card.find("a", cls="explorer-preview-link")
            # The link's name starts with its visible badge text, then says where it goes.
            assert "aria-label" not in link.attrs, slot
            assert link.find("span", cls="visually-hidden").text(), slot
            region = slot.split("-")[0]
            target = PUBLIC if editions.startswith("public") else RESEARCH
            assert link.attrs["href"] == f"{target}?region={region}&preset={slot}", slot
            if editions == "public research":
                research_links = [a for a in card.find_all("a") if a.attrs["href"].startswith(RESEARCH)]
                assert research_links, f"{slot} also opens in the research edition"

    def test_research_only_previews_are_the_research_layers(self, locale):
        cards = parse(locale).find_all("article", cls="explorer-video-card")
        research_only = {card.attrs["data-demo-slot"] for card in cards if card.attrs["data-editions"] == "research"}
        assert research_only == {"antarctica-basin-boundary", "antarctica-subglacial-features"}

    def test_the_newest_update_introduces_the_public_edition(self, locale):
        updates = parse(locale).find("section", id="latest-updates")
        first = updates.find("article", cls="explorer-update-card")
        assert first.attrs.get("aria-labelledby") == "public-edition-update-title"
        assert first.find("time", cls="explorer-update-date").attrs.get("datetime") == "2026-09-30"
        assert TOUR in [a.attrs["href"] for a in first.find_all("a")]


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

    def test_source_links_open_in_a_new_tab_safely(self, locale):
        sources = parse(locale).find("section", id="source-data")
        for link in sources.find("div", cls="explorer-source-columns").find_all("a"):
            assert link.attrs.get("target") == "_blank"
            assert "noopener" in link.attrs.get("rel", "")

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
                (node.tag, node.attrs.get("id"), node.attrs.get("data-edition"), node.attrs.get("data-editions"),
                 node.attrs.get("data-demo-slot"), node.attrs.get("href") if node.tag == "a" else None)
                for node in doc.find("body").iter()
                if node.attrs.get("id") or node.attrs.get("data-edition") or node.attrs.get("data-demo-slot")
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
            for attr in ("href", "src", "poster", "data-light-src", "data-dark-src"):
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
