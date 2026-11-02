/**
 * The panel's masthead and region buttons, in both editions.
 *
 * - The ⓘ button opens a menu with the intro, and on the public page what the explorer is
 *   built from. A click outside it or Escape closes it.
 * - Segmented buttons over the region <select>. The select stays the runtime's source of
 *   truth: a press sets it and dispatches its change event, and sync() mirrors back
 *   whatever the runtime wrote.
 * - The edition switch is a pair of links, the page's own edition marked. The link to the
 *   other edition carries the region on screen (?region=, which the runtime reads), so the
 *   reader lands on the same ice sheet, in the same language.
 */

/** The region a page opens on when its URL names none. */
export const DEFAULT_REGION = "antarctica";

/**
 * A link to another edition, at the region on screen: the link's own path and no query
 * for the default region, else ?region=.
 * @param {string} href
 * @param {string | null | undefined} region
 */
export function editionHref(href, region) {
  const path = String(href ?? "").split(/[?#]/, 1)[0];
  return region && region !== DEFAULT_REGION ? `${path}?region=${encodeURIComponent(region)}` : path;
}

/** The button a click landed on, or null. */
export function closestButton(event) {
  return event.target instanceof Element ? event.target.closest("button[data-value]") : null;
}

/** Choose an option of a select the way a reader would, so its change handlers run. */
export function chooseOption(select, value) {
  if (!select || select.value === value) return;
  select.value = value;
  select.dispatchEvent(new Event("change", { bubbles: true }));
}

function bindInfoMenu(panel) {
  const button = panel.querySelector("#panelInfoButton");
  const menu = panel.querySelector("#panelInfoMenu");
  if (!button || !menu) return;
  const setOpen = (open) => {
    menu.hidden = !open;
    button.setAttribute("aria-expanded", String(open));
  };
  button.addEventListener("click", () => setOpen(menu.hidden));
  document.addEventListener("click", (event) => {
    if (menu.hidden || !(event.target instanceof Node)) return;
    if (!menu.contains(event.target) && !button.contains(event.target)) setOpen(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || menu.hidden) return;
    setOpen(false);
    button.focus();
  });
}

function bindRegionSegments(panel, regionSelect) {
  const segments = panel.querySelector('[data-segmented-for="regionPreset"]');
  if (!segments || !regionSelect) return () => {};
  segments.addEventListener("click", (event) => {
    const button = closestButton(event);
    if (button && !button.disabled) chooseOption(regionSelect, button.dataset.value);
  });
  return () => {
    for (const button of segments.querySelectorAll("button[data-value]")) {
      button.setAttribute("aria-pressed", String(button.dataset.value === regionSelect.value));
      button.disabled = regionSelect.disabled;
    }
  };
}

function bindEditionLink(panel, regionSelect) {
  const link = panel.querySelector("a[data-edition-link]");
  if (!link || !regionSelect) return () => {};
  const base = link.getAttribute("href") || "";
  return () => link.setAttribute("href", editionHref(base, regionSelect.value));
}

/**
 * @param {{ panel: HTMLElement | null, regionSelect: HTMLSelectElement | null }} options
 * @returns {{ sync(): void }}
 */
export function bindMasthead({ panel, regionSelect }) {
  if (!panel) return { sync() {} };
  bindInfoMenu(panel);
  const syncRegion = bindRegionSegments(panel, regionSelect);
  const syncEditionLink = bindEditionLink(panel, regionSelect);
  const sync = () => {
    syncRegion();
    syncEditionLink();
  };
  sync();
  return { sync };
}
