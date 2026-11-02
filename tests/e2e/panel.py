"""Helpers for driving the research page's grouped panel from Playwright.

The panel's groups are <details> elements, and Playwright cannot click a control inside a
closed one, so a test opens the group first. The region and dataset <select>s stay the
runtime's source of truth but are hidden behind segmented controls; choose_dataset() sets
the select directly, as a preset does, while the panel's own controls are exercised in
test_research_panel.py.
"""

from __future__ import annotations

_OPEN_GROUP_OF = """(selector) => {
    const element = document.querySelector(selector);
    const group = element && element.closest("details.rp-group");
    if (group) group.open = true;
    return Boolean(element);
}"""


def open_group(page, group: str) -> None:
    """Open a panel group by its data-group name: scene, flow, bed, ocean, time, places,
    display or data."""
    page.locator(f'details.rp-group[data-group="{group}"]').evaluate("(details) => { details.open = true; }")


def reveal(page, selector: str):
    """The locator for a control, with the group that holds it opened first."""
    assert page.evaluate(_OPEN_GROUP_OF, selector), f"no element matches {selector}"
    return page.locator(selector)


def choose_region(page, region: str) -> None:
    """Switch region with the panel's segmented control, as a reader would."""
    page.locator(f'[data-segmented-for="regionPreset"] [data-value="{region}"]').click()


def choose_dataset(page, dataset: str) -> None:
    """Load a terrain dataset by its registry key ("bedmap3", "qrf-hd", ...), once the
    picker is free again: it is disabled while a terrain loads."""
    page.wait_for_function("() => !document.getElementById('resolutionPreset').disabled", timeout=90_000)
    page.evaluate(
        """(key) => {
            const select = document.getElementById("resolutionPreset");
            if (![...select.options].some((option) => option.value === key)) {
                throw new Error(`no dataset ${key} in this region`);
            }
            select.value = key;
            select.dispatchEvent(new Event("change", { bubbles: true }));
        }""",
        dataset,
    )
