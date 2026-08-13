"""E2E coverage for polar feature layers, search, and camera focus."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect


pytestmark = pytest.mark.e2e


@pytest.fixture
def page(playwright_browser, explorer_url):
    context = playwright_browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    page.goto(explorer_url, wait_until="networkidle", timeout=30_000)
    yield page
    context.close()


def test_feature_layers_can_be_toggled_independently(page):
    page.get_by_label("Show research stations").check()
    page.wait_for_function(
        "() => JSON.parse(window.render_game_to_text()).featureLayers.researchStations.visibleCount > 0",
        timeout=30_000,
    )
    state = page.evaluate("JSON.parse(window.render_game_to_text())")
    assert state["featureLayers"]["researchStations"]["enabled"]
    assert state["featureLayers"]["researchStations"]["totalCount"] == 82
    assert not state["featureLayers"]["geographicNames"]["enabled"]

    page.get_by_label("Show geographic names").check()
    page.wait_for_function(
        "() => JSON.parse(window.render_game_to_text()).featureLayers.geographicNames.visibleCount > 0",
        timeout=30_000,
    )
    state = page.evaluate("JSON.parse(window.render_game_to_text())")
    assert state["featureLayers"]["geographicNames"]["totalCount"] == 160


def test_search_selects_feature_enables_layer_and_moves_camera(page):
    before = page.evaluate("JSON.parse(window.render_game_to_text()).camera.target")
    search = page.get_by_role("combobox", name="Search places and features")
    search.fill("Amundsen-Scott")
    page.get_by_role("option", name=re.compile("Amundsen", re.IGNORECASE)).click()
    page.wait_for_function(
        """() => {
          const state = JSON.parse(window.render_game_to_text());
          return state.selectedFeature?.id?.includes('amundsen') && state.featureLayers.researchStations.enabled;
        }""",
        timeout=30_000,
    )
    after = page.evaluate("JSON.parse(window.render_game_to_text()).camera.target")
    assert after != before


def test_search_selects_refined_basin_enables_layer_and_moves_camera(page):
    before = page.evaluate("JSON.parse(window.render_game_to_text()).camera.target")
    search = page.get_by_role("combobox", name="Search places and features")
    search.fill("Abbot refined basin")
    page.get_by_role("option", name=re.compile("Abbot", re.IGNORECASE)).click()
    page.wait_for_function(
        """() => {
          const state = JSON.parse(window.render_game_to_text());
          return state.selectedFeature?.layer === 'refined_basins' &&
            state.selectedFeature?.name === 'Abbot' &&
            state.toggles.showRefinedBasins &&
            state.meshes.basins;
        }""",
        timeout=30_000,
    )
    after = page.evaluate("JSON.parse(window.render_game_to_text()).camera.target")
    assert after != before


def test_refined_basin_search_restores_a_visible_surface(page):
    page.get_by_label("Show bed topography").uncheck()
    page.get_by_label("Show ice surface", exact=True).uncheck()
    search = page.get_by_role("combobox", name="Search places and features")
    search.fill("Abbot refined basin")
    page.get_by_role("option", name=re.compile("Abbot", re.IGNORECASE)).click()
    page.wait_for_function(
        """() => {
          const state = JSON.parse(window.render_game_to_text());
          return state.selectedFeature?.id === 'antarctica-refined-basin-1' &&
            state.toggles.showIce && state.toggles.showRefinedBasins && state.meshes.basins;
        }""",
        timeout=30_000,
    )


def test_keyboard_search_can_switch_region_and_focus(page):
    search = page.get_by_role("combobox", name="Search places and features")
    search.fill("Summit Station")
    search.press("ArrowDown")
    search.press("Enter")
    page.wait_for_function(
        """() => {
          const state = JSON.parse(window.render_game_to_text());
          return state.ready && state.region === 'greenland' && state.selectedFeature?.id?.includes('summit');
        }""",
        timeout=90_000,
    )


def test_keyboard_search_can_switch_region_and_focus_refined_basin(page):
    search = page.get_by_role("combobox", name="Search places and features")
    search.fill("Central East")
    search.press("ArrowDown")
    search.press("Enter")
    page.wait_for_function(
        """() => {
          const state = JSON.parse(window.render_game_to_text());
          return state.ready && state.region === 'greenland' &&
            state.selectedFeature?.id === 'greenland-refined-basin-CE' &&
            state.selectedFeature?.layer === 'refined_basins' &&
            state.toggles.showRefinedBasins && state.meshes.basins;
        }""",
        timeout=90_000,
    )


def test_search_reports_no_results_accessibly(page):
    search = page.get_by_role("combobox", name="Search places and features")
    search.fill("not-a-real-polar-feature-xyz")
    expect(page.get_by_role("status")).to_contain_text("No matches", timeout=30_000)


def test_chinese_explorer_searches_localized_feature_names(playwright_browser, server):
    context = playwright_browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    page.goto(f"{server}/zh/tools/3D-interactive-cryosphere-explorer.html", wait_until="networkidle", timeout=30_000)
    search = page.get_by_role("combobox", name="搜索地点和地理特征")
    search.fill("横贯南极山脉")
    page.get_by_role("option", name=re.compile("横贯南极山脉")).click()
    page.wait_for_function(
        """() => {
          const state = JSON.parse(window.render_game_to_text());
          return state.selectedFeature?.id?.includes('14887') && state.featureLayers.geographicNames.enabled;
        }""",
        timeout=30_000,
    )
    context.close()
