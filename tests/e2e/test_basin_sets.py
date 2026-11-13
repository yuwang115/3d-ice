"""E2E coverage for the basin layer's choice of set in Greenland.

Greenland's basins come as Mouginot & Rignot (2019)'s seven labelled regions, the default,
or Mankoff et al. (2020)'s outlet catchments, drawn as the divides between the large ones
(js/basin-divides.js). The chooser sits under the basin toggle while the layer is on.
"""

from __future__ import annotations

import pytest

from tests.e2e.panel import reveal

pytestmark = pytest.mark.e2e

MANKOFF_DATA = "https://doi.org/10.22008/FK2/XKQVL7"
MANKOFF_ARTICLE = "https://doi.org/10.5194/essd-12-2811-2020"
MOUGINOT = "https://doi.org/10.7280/D1WT11"

_BUILT_FOR = """(setKey) => {
  const state = JSON.parse(window.render_game_to_text());
  return state.ready && state.basins.set === setKey && state.basins.builtForSet && state.meshes.basins;
}"""


@pytest.fixture
def context(playwright_browser):
    context = playwright_browser.new_context(viewport={"width": 1280, "height": 800})
    yield context
    context.close()


def _open(context, url: str):
    page = context.new_page()
    page.goto(url, wait_until="networkidle", timeout=60_000)
    return page


def _state(page) -> dict:
    return page.evaluate("JSON.parse(window.render_game_to_text())")


def _source_links(page) -> list[str]:
    return page.evaluate("[...document.querySelectorAll('#metaList a')].map((a) => a.href)")


def test_the_outlet_catchments_view_draws_mankoffs_divides(context, explorer_url):
    page = _open(context, f"{explorer_url}?region=greenland&preset=greenland-outlet-catchments")
    page.wait_for_function(_BUILT_FOR, arg="mankoff2020", timeout=90_000)
    basins = _state(page)["basins"]
    assert basins["kind"] == "divides"
    assert basins["sets"] == ["mouginot2019", "mankoff2020"]
    assert basins["chooserVisible"]
    assert basins["labels"] == 0, "the catchments have no names to label"
    assert basins["basinCount"] == 23_906 and basins["largeBasinCount"] == 3_126
    assert basins["surfaceTriangles"] > 10_000
    assert "3,126 of 23,906" in basins["note"]
    links = _source_links(page)
    assert MANKOFF_DATA in links and MANKOFF_ARTICLE in links
    assert MOUGINOT not in links


def test_the_chooser_switches_sets_and_their_credits(context, explorer_url):
    page = _open(context, f"{explorer_url}?region=greenland&preset=greenland-basin-boundary")
    page.wait_for_function(_BUILT_FOR, arg="mouginot2019", timeout=90_000)
    basins = _state(page)["basins"]
    assert basins["kind"] == "basins" and basins["labels"] > 0
    assert MOUGINOT in _source_links(page)

    chooser = reveal(page, "#basinSet")
    chooser.select_option("mankoff2020")
    page.wait_for_function(_BUILT_FOR, arg="mankoff2020", timeout=90_000)
    assert _state(page)["basins"]["labels"] == 0
    assert MANKOFF_DATA in _source_links(page)

    chooser.select_option("mouginot2019")
    page.wait_for_function(_BUILT_FOR, arg="mouginot2019", timeout=90_000)
    assert _state(page)["basins"]["labels"] > 0
    assert MOUGINOT in _source_links(page)


def test_the_chooser_hides_with_the_layer_and_in_antarctica(context, explorer_url):
    page = _open(context, f"{explorer_url}?region=greenland&preset=greenland-outlet-catchments")
    page.wait_for_function(_BUILT_FOR, arg="mankoff2020", timeout=90_000)
    reveal(page, "#showRefinedBasins").uncheck()
    assert not _state(page)["basins"]["chooserVisible"]
    page.locator("#showRefinedBasins").check()
    page.wait_for_function(_BUILT_FOR, arg="mankoff2020", timeout=90_000)
    assert _state(page)["basins"]["chooserVisible"], "the set chosen is kept while the layer is off"

    antarctica = _open(context, f"{explorer_url}?region=antarctica&preset=antarctica-basin-boundary")
    antarctica.wait_for_function(
        "() => { const s = JSON.parse(window.render_game_to_text()); return s.ready && s.meshes.basins; }",
        timeout=90_000,
    )
    basins = _state(antarctica)["basins"]
    assert basins["sets"] == [] and basins["set"] is None
    assert not basins["chooserVisible"]


def test_a_searched_region_brings_back_the_set_it_belongs_to(context, explorer_url):
    page = _open(context, f"{explorer_url}?region=greenland&preset=greenland-outlet-catchments")
    page.wait_for_function(_BUILT_FOR, arg="mankoff2020", timeout=90_000)
    search = page.get_by_role("combobox", name="Search places and features")
    search.fill("Central East")
    search.press("ArrowDown")
    search.press("Enter")
    page.wait_for_function(
        """() => {
          const state = JSON.parse(window.render_game_to_text());
          return state.selectedFeature?.id === 'greenland-refined-basin-CE' &&
            state.basins.set === 'mouginot2019' && state.basins.builtForSet && state.basins.labels > 0;
        }""",
        timeout=90_000,
    )


MANKOFF_PACKAGE = "**/greenland_mankoff_ice_basins_k1.json*"
BOUNDARY_VIEW = "{}?region=greenland&preset=greenland-basin-boundary"


def test_a_set_chosen_while_another_loads_is_the_one_drawn(context, explorer_url):
    page = context.new_page()
    held = []
    page.route(MANKOFF_PACKAGE, lambda route: held.append(route))
    page.goto(BOUNDARY_VIEW.format(explorer_url), wait_until="networkidle", timeout=60_000)
    page.wait_for_function(_BUILT_FOR, arg="mouginot2019", timeout=90_000)
    chooser = reveal(page, "#basinSet")
    with page.expect_request(MANKOFF_PACKAGE):
        chooser.select_option("mankoff2020")
    chooser.select_option("mouginot2019")
    page.wait_for_function(_BUILT_FOR, arg="mouginot2019", timeout=90_000)
    assert held, "the Mankoff package was requested and held"
    with page.expect_response(MANKOFF_PACKAGE):
        held[0].continue_()
    page.wait_for_timeout(1_000)
    basins = _state(page)["basins"]
    assert basins["set"] == "mouginot2019" and basins["kind"] == "basins" and basins["labels"] > 0
    assert MOUGINOT in _source_links(page) and MANKOFF_DATA not in _source_links(page)


def test_a_set_that_fails_to_load_gives_way_to_the_default(context, explorer_url):
    page = context.new_page()
    page.route(MANKOFF_PACKAGE, lambda route: route.fulfill(status=503, body="unavailable"))
    page.goto(BOUNDARY_VIEW.format(explorer_url), wait_until="networkidle", timeout=60_000)
    page.wait_for_function(_BUILT_FOR, arg="mouginot2019", timeout=90_000)
    reveal(page, "#basinSet").select_option("mankoff2020")
    page.wait_for_function(
        """() => {
          const state = JSON.parse(window.render_game_to_text());
          const option = document.querySelector('#basinSet option[value="mankoff2020"]');
          return state.basins.set === 'mouginot2019' && state.basins.builtForSet && option.disabled;
        }""",
        timeout=90_000,
    )
    state = _state(page)
    assert state["toggles"]["showRefinedBasins"] and state["meshes"]["basins"]
