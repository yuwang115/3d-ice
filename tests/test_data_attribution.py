"""Each data set is credited by its own title and linked to its own record.

The 2026-10-18 audit found the friction data sets (CC BY 4.0) never linked, home-page
titles that match no record, the NSIDC-0709 refined basins called "IMBIE", the Greenland
basins uncredited in the explorer, links whose labels name a different work than the one
they open, and the Greenland velocity named as ITS_LIVE "version 2" although the package is
the V02.1 summary mosaic (2014-2024).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
RUNTIME = STATIC / "tools" / "js" / "explorer-app.js"
PUBLIC_COPY = STATIC / "tools" / "js" / "explore-content.js"
HOME_PAGES = (STATIC / "index.html", STATIC / "zh" / "index.html")
README = ROOT / "README.md"

AISEFI = ("Antarctic Ice Sheet Ensemble Friction Inversion (AISEFI): Basal shear stress with uncertainties", "https://doi.org/10.5281/zenodo.18508904")
GRISEFI = ("Greenland Ice Sheet Ensemble Friction Inversion (GrISEFI): Basal shear stress with uncertainties", "https://doi.org/10.5281/zenodo.18508850")
RISE_DATA = ("Multi-model estimate of Antarctic ice-shelf basal melting and ocean drivers: data", "https://doi.org/10.26179/q0rh-gt74")
RISE_ARTICLE = "https://doi.org/10.5194/tc-19-6507-2025"
GLADS_DATA = ("Data for: Antarctic wide subglacial hydrology modeling", "https://doi.org/10.5281/zenodo.12738170")
GLADS_ARTICLE = "https://doi.org/10.1029/2024GL111386"
GREENLAND_BASINS = ("Mouginot & Rignot (2019)", "https://doi.org/10.7280/D1WT11")
PAXMAN_ARTICLE = "https://doi.org/10.1038/s41598-022-15440-y"
PAXMAN_GRIDS = "https://doi.org/10.18739/A22Z12R8C"
ITS_LIVE = "https://its-live.jpl.nasa.gov/"

_REGISTRY_LINK = re.compile(r'\{\s*text:\s*"(?P<text>[^"]+)",\s*url:\s*"(?P<url>[^"]+)",?\s*\}')
_ANCHOR = re.compile(r'<a href="(?P<url>[^"]+)"[^>]*>(?P<text>.*?)</a>', re.S)


def runtime_links() -> list[tuple[str, str]]:
    return [(m["text"], m["url"]) for m in _REGISTRY_LINK.finditer(RUNTIME.read_text(encoding="utf-8"))]


def public_links() -> list[tuple[str, str]]:
    source = PUBLIC_COPY.read_text(encoding="utf-8")
    table = dict(re.findall(r'^\s*(\w+):\s*"(https?://[^"]+)",', source, re.M))
    links = []
    for m in re.finditer(r'\{\s*text:\s*"(?P<text>[^"]+)",\s*url:\s*(?:SOURCES\.(?P<key>\w+)|"(?P<url>[^"]+)")', source):
        links.append((m["text"], table[m["key"]] if m["key"] else m["url"]))
    return links


def page_links(page: Path) -> list[tuple[str, str]]:
    return [(re.sub(r"\s+", " ", m["text"]).strip(), m["url"]) for m in _ANCHOR.finditer(page.read_text(encoding="utf-8"))]


def links_to(links: list[tuple[str, str]], url: str) -> list[str]:
    return [text for text, link in links if link == url]


@pytest.mark.parametrize("title, url", [AISEFI, GRISEFI])
def test_the_friction_data_sets_are_linked_by_their_titles_in_the_explorer(title, url):
    assert any(title in text for text in links_to(runtime_links(), url)), f"no runtime link to {url} titled {title!r}"


@pytest.mark.parametrize("page", HOME_PAGES, ids=lambda page: page.parent.name or "en")
@pytest.mark.parametrize("title, url", [AISEFI, GRISEFI])
def test_the_home_pages_list_the_friction_data_sets_by_their_titles(page, title, url):
    texts = links_to(page_links(page), url)
    assert any(title.lower() in text.lower() for text in texts), f"{page} lists {url} as {texts}"


def test_no_list_uses_invented_friction_titles():
    for page in HOME_PAGES:
        source = page.read_text(encoding="utf-8")
        assert "Basal Friction from Elmer/Ice Inversions" not in source
        assert "Basal Friction Ensemble Inversion Reference" not in source


def test_the_refined_basins_are_not_called_imbie():
    texts = [text for text, _ in runtime_links()]
    assert not any("IMBIE refined" in text for text in texts)
    assert any("MEaSUREs Antarctic Boundaries v2, refined basins" in text for text in texts)
    for doc in (README, ROOT / "docs" / "architecture.md"):
        assert "IMBIE-refined" not in doc.read_text(encoding="utf-8")
        assert "IMBIE refined" not in doc.read_text(encoding="utf-8")


def test_the_explorer_credits_the_greenland_basins():
    label, url = GREENLAND_BASINS
    assert any(label in text for text in links_to(runtime_links(), url))
    assert "Greenland Basins PS v1.4.2 catchment boundaries" not in RUNTIME.read_text(encoding="utf-8")


def test_the_data_snapshot_lists_basin_boundaries_for_both_ice_sheets():
    source = RUNTIME.read_text(encoding="utf-8")
    assert 'dataset.regionKey === "antarctica" && dataset.sources.basins' not in source
    assert "if (capabilities.refinedBasins && dataset.sources.basins) {" in source
    # Both ice sheets offer the basin layer, and each registry entry credits its basins.
    registry = source[source.index("const ALL_REGIONS") : source.index("// Phones, small windows")]
    for region in ("antarctica", "greenland"):
        block = registry[registry.index(f"  {region}: {{") :]
        capabilities = block[block.index("capabilities: {") : block.index("}", block.index("capabilities: {"))]
        assert "refinedBasins: true" in capabilities, region
        assert "      basins: {" in block[: block.index("datasets: {")], region


def test_rise_is_credited_with_its_data_citation_and_article():
    links = runtime_links()
    title, url = RISE_DATA
    assert any(title in text for text in links_to(links, url))
    assert links_to(links, RISE_ARTICLE), "the RISE article (The Cryosphere, 2025) is linked"
    assert "https://data.aad.gov.au/metadata/RISE" not in [link for _, link in links]
    assert "15,269 ice-shelf cells" not in RUNTIME.read_text(encoding="utf-8")


def test_the_rise_preview_index_names_the_published_article():
    index = (STATIC / "tools" / "data" / "rise_antarctica_preview_index.json").read_text(encoding="utf-8")
    assert "Realistic ice-shelf/ocean state estimates" not in index
    assert "egusphere" not in index.lower()
    assert RISE_ARTICLE in index


def test_glads_labels_match_their_links():
    links = runtime_links()
    title, url = GLADS_DATA
    assert any(title in text for text in links_to(links, url))
    assert links_to(links, GLADS_ARTICLE)
    assert "https://zenodo.org/records/12738170" not in [link for _, link in links]


@pytest.mark.parametrize("source", ["runtime", "public", "home-en", "home-zh"])
def test_paxman_links_open_the_work_their_labels_name(source):
    links = {
        "runtime": runtime_links,
        "public": public_links,
        "home-en": lambda: page_links(HOME_PAGES[0]),
        "home-zh": lambda: page_links(HOME_PAGES[1]),
    }[source]()
    grids = [text for text, url in links if url == PAXMAN_GRIDS]
    articles = [text for text, url in links if url == PAXMAN_ARTICLE]
    assert grids, "the grid files are linked"
    for text in grids:
        assert re.search(r"grid|网格", text, re.I), f"a grid-file link reads {text!r}"
    for text in articles:
        assert not re.search(r"grid files|网格文件", text, re.I), f"an article link reads {text!r}"


def test_its_live_is_named_as_the_v02_1_mosaic_everywhere():
    # The archived NSIDC-0776 Version 2 may still be credited; the mosaic shown may not be
    # called version 2.
    places = [RUNTIME, PUBLIC_COPY, README, *HOME_PAGES, ROOT / "docs" / "architecture.md"]
    mislabels = re.compile(r"velocity mosaic, version 2|速度镶嵌图，第 2 版|ITS_LIVE v2\b|ITS_LIVE v2 velocity|MEaSUREs ITS_LIVE v2\]")
    for place in places:
        assert not mislabels.search(place.read_text(encoding="utf-8")), place
    for place in (RUNTIME, PUBLIC_COPY, README, *HOME_PAGES):
        source = place.read_text(encoding="utf-8")
        assert "V02.1" in source and "2014–2024" in source, place
    assert links_to(runtime_links(), ITS_LIVE), "the explorer links ITS_LIVE, whose archive serves the mosaic"
