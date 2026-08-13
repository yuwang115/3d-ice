"""Regression tests for the localized latest-updates block on the home page."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parent.parent
EXPLORER_PATH = "./tools/3D-interactive-cryosphere-explorer.html"
HOME_CASES = (
    (
        ROOT_DIR / "static" / "index.html",
        "August 13, 2026",
        ("Places & Geographic Features", "Search", "refined basins"),
    ),
    (
        ROOT_DIR / "static" / "zh" / "index.html",
        "2026年8月13日",
        ("地点与地理特征", "搜索", "细化流域"),
    ),
)


class _HomeStructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.id_counts: dict[str, int] = {}
        self.section_stack: list[str | None] = []
        self.latest_attrs: dict[str, str] = {}
        self.latest_tags: list[tuple[str, dict[str, str]]] = []
        self.latest_text: list[str] = []
        self.stylesheets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        element_id = attr_map.get("id")
        if element_id:
            self.ids.append(element_id)
            self.id_counts[element_id] = self.id_counts.get(element_id, 0) + 1

        if tag == "link" and attr_map.get("rel") == "stylesheet":
            self.stylesheets.append(attr_map.get("href", ""))

        if tag == "section":
            self.section_stack.append(element_id)

        if self.section_stack and self.section_stack[-1] == "latest-updates":
            if tag == "section":
                self.latest_attrs = attr_map
            self.latest_tags.append((tag, attr_map))

    def handle_endtag(self, tag: str) -> None:
        if tag == "section" and self.section_stack:
            self.section_stack.pop()

    def handle_data(self, data: str) -> None:
        if self.section_stack and self.section_stack[-1] == "latest-updates":
            value = data.strip()
            if value:
                self.latest_text.append(value)


@pytest.mark.integration
@pytest.mark.parametrize(("page_path", "visible_date", "expected_terms"), HOME_CASES)
def test_latest_updates_is_localized_and_between_region_and_features(
    page_path: Path,
    visible_date: str,
    expected_terms: tuple[str, ...],
) -> None:
    parser = _HomeStructureParser()
    parser.feed(page_path.read_text(encoding="utf-8"))

    assert parser.id_counts.get("latest-updates") == 1
    assert "/css/3d-ice-updates.css" in parser.stylesheets
    assert parser.ids.index("greenland-features") < parser.ids.index("latest-updates")
    assert parser.ids.index("latest-updates") < parser.ids.index("key-features")
    assert parser.latest_attrs.get("aria-labelledby") == "latest-updates-title"
    assert any(
        tag == "h2" and attrs.get("id") == "latest-updates-title"
        for tag, attrs in parser.latest_tags
    )
    assert any(
        tag == "article"
        and attrs.get("aria-labelledby") == "polar-place-search-update-title"
        for tag, attrs in parser.latest_tags
    )
    assert any(
        tag == "h3" and attrs.get("id") == "polar-place-search-update-title"
        for tag, attrs in parser.latest_tags
    )
    assert any(
        tag == "time"
        and attrs.get("datetime") == "2026-08-13"
        and attrs.get("class") == "explorer-update-date"
        for tag, attrs in parser.latest_tags
    )
    assert any(
        tag == "a" and attrs.get("href") == EXPLORER_PATH
        for tag, attrs in parser.latest_tags
    )

    section_text = " ".join(parser.latest_text)
    assert visible_date in section_text
    for term in expected_terms:
        assert term in section_text
