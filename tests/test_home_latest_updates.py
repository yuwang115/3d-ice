"""Regression tests for the localized latest-updates block on the home page.

The page shows the three newest updates; a drawer (a <dialog>) holds every one of them, newest
first. These pin the copy each locale carries for updates the page has announced before, so a
rewrite of the block cannot drop one or leave it in the other language.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_home_page import Node, _TreeBuilder


ROOT_DIR = Path(__file__).resolve().parent.parent
HOME_PAGES = {
    "en": ROOT_DIR / "static" / "index.html",
    "zh": ROOT_DIR / "static" / "zh" / "index.html",
}
# (datetime, the date as the locale prints it, terms its entry in the drawer carries)
ANNOUNCED = {
    "en": (
        ("2026-11-13", "November 13, 2026", ("Mankoff et al. (2020)", "outlet catchments", "Mouginot & Rignot")),
        ("2026-11-03", "November 3, 2026", ("grounding line", "2300 projection")),
        ("2026-11-01", "November 1, 2026", ("NOAA/NSIDC", "1981–2010 median ice edge")),
        ("2026-10-06", "October 6, 2026", ("ISMIP6", "Seroussi et al., 2024", "ice-shelf collapse")),
        ("2026-09-30", "September 30, 2026", ("public edition", "guided tour")),
        ("2026-09-22", "September 22, 2026", ("rebound", "Paxman et al. (2022)", "Airy")),
        ("2026-08-13", "August 13, 2026", ("Research stations", "refined drainage basins")),
    ),
    "zh": (
        ("2026-11-13", "2026年11月13日", ("Mankoff 等（2020）", "出口集水区", "Mouginot 与 Rignot")),
        ("2026-11-03", "2026年11月3日", ("接地线", "2300 年预估")),
        ("2026-11-01", "2026年11月1日", ("NOAA/NSIDC", "1981–2010 年的中位冰缘线")),
        ("2026-10-06", "2026年10月6日", ("ISMIP6", "Seroussi 等，2024", "冰架崩解")),
        ("2026-09-30", "2026年9月30日", ("科普版", "导览")),
        ("2026-09-22", "2026年9月22日", ("地壳回弹", "Paxman 等（2022）", "Airy")),
        ("2026-08-13", "2026年8月13日", ("科考站", "细化流域")),
    ),
}


def parse(locale: str) -> Node:
    builder = _TreeBuilder()
    builder.feed(HOME_PAGES[locale].read_text(encoding="utf-8"))
    return builder.root


def ids_in_order(doc: Node) -> list[str]:
    return [node.attrs["id"] for node in doc.iter() if node.attrs.get("id")]


@pytest.fixture(params=sorted(HOME_PAGES))
def locale(request) -> str:
    return request.param


@pytest.mark.integration
def test_the_updates_sit_between_explore_and_the_source_data(locale):
    ids = ids_in_order(parse(locale))
    assert ids.count("latest-updates") == 1
    assert ids.index("explore") < ids.index("latest-updates") < ids.index("source-data")
    assert "key-features" not in ids


@pytest.mark.integration
def test_every_announced_update_is_in_the_drawer_in_the_page_language(locale):
    drawer = parse(locale).find("dialog", id="all-updates")
    entries = {}
    for item in drawer.find_all("li", cls="explorer-drawer-item"):
        time = item.find("time", cls="explorer-update-date")
        entries.setdefault(time.attrs["datetime"], []).append((time.text(), item.text()))
    for datetime, printed, terms in ANNOUNCED[locale]:
        assert datetime in entries, f"{locale}: no update dated {datetime}"
        assert all(shown == printed for shown, _ in entries[datetime]), f"{locale}: {datetime} printed as {entries[datetime][0][0]!r}"
        text = " ".join(body for _, body in entries[datetime])
        for term in terms:
            assert term in text, f"{locale} {datetime}: {term!r}"


@pytest.mark.integration
def test_the_page_and_the_drawer_agree_on_the_newest_updates(locale):
    # By date: a card on the page may sum up two entries of the drawer that share one.
    doc = parse(locale)
    on_page = [
        card.find("time").attrs["datetime"]
        for card in doc.find("section", id="latest-updates").find_all("article", cls="explorer-update-card")
    ]
    in_drawer = [
        item.find("time").attrs["datetime"]
        for item in doc.find("dialog", id="all-updates").find_all("li", cls="explorer-drawer-item")
    ]
    assert on_page == in_drawer[: len(on_page)]
    assert on_page == sorted(on_page, reverse=True)


@pytest.mark.integration
def test_both_locales_list_the_same_updates():
    def dates(locale):
        return [node.attrs["datetime"] for node in parse(locale).find("dialog", id="all-updates").find_all("time")]

    assert dates("en") == dates("zh")
