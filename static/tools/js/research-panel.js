/**
 * The research edition's panel, on top of the controls the runtime reads and writes.
 *
 * - Groups: collapsible <details class="rp-group" data-group>. Each opens as the reader
 *   last left it (localStorage), opens by itself when a layer in it is switched on, and
 *   counts the layers on in a pill beside its title. The panel watches its layer toggles
 *   (input[data-layer]) itself, including script writes to `checked`.
 * - Where: segmented buttons over the region <select>, and a terrain <select> and grid
 *   buttons over the dataset <select>. The selects stay the runtime's source of truth: a
 *   choice here sets one and dispatches its change event, and syncWhere() mirrors back
 *   whatever the runtime wrote.
 * - The reason a disabled layer gives, on its row.
 * - The ⓘ menu in the masthead.
 *
 * The public page has no groups, so there the panel is inert. The panel takes every
 * string it shows from its caller, so all localisation stays in the runtime's t().
 */

import {
  findDatasetForChoice,
  getTerrainChoice,
  groupsToOpen,
  listTerrainChoices,
  offersChoice,
  readGroupState,
  recordGroupState,
} from "./panel-state.js";

export const GROUP_STATE_STORAGE_KEY = "3d-ice-research-panel-groups";

const INERT_PANEL = Object.freeze({
  syncWhere() {},
  setReason() {},
});

function readStorage(storage) {
  try {
    return storage?.getItem(GROUP_STATE_STORAGE_KEY) ?? null;
  } catch (_error) {
    // Blocked storage (a sandboxed frame, strict privacy settings) only costs the memory.
    return null;
  }
}

function writeStorage(storage, value) {
  try {
    storage?.setItem(GROUP_STATE_STORAGE_KEY, value);
  } catch (_error) {
    // As above: the panel still works, it just forgets.
  }
}

function closestButton(event) {
  return event.target instanceof Element ? event.target.closest("button[data-value]") : null;
}

function chooseOption(select, value) {
  if (!select || select.value === value) return;
  select.value = value;
  select.dispatchEvent(new Event("change", { bubbles: true }));
}

function createSegment(value, label) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "rp-segmented__option";
  button.dataset.value = value;
  button.textContent = label;
  return button;
}

function layerInputs(root) {
  return Array.from(root.querySelectorAll("input[data-layer]"));
}

// The browser's own `checked` property of an <input>.
const CHECKED = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "checked");

/**
 * Call `onWrite` whenever script sets `input.checked`. The runtime and the place search
 * switch layers on and off that way, in many places and without an event; a click still
 * toggles the box natively and fires `change`.
 */
function watchChecked(input, onWrite) {
  Object.defineProperty(input, "checked", {
    configurable: true,
    enumerable: CHECKED.enumerable,
    get() {
      return CHECKED.get.call(this);
    },
    set(value) {
      CHECKED.set.call(this, value);
      onWrite();
    },
  });
}

function bindGroups({ panel, groups, storage, formatCount, onGroupToggle }) {
  const keys = groups.map((group) => group.dataset.group);
  const byKey = new Map(groups.map((group) => [group.dataset.group, group]));
  const groupOfLayer = new Map();
  for (const group of groups) {
    for (const input of layerInputs(group)) groupOfLayer.set(input.id, group.dataset.group);
  }
  // Groups opened or closed by the panel rather than the reader: not a choice to remember.
  // The toggle event that follows arrives as a task of its own, after this one.
  const unprompted = new WeakSet();
  const setOpen = (group, open) => {
    if (!group || group.open === open) return;
    unprompted.add(group);
    group.open = open;
  };

  const defaults = groups.filter((group) => group.open).map((group) => group.dataset.group);
  const state = readGroupState(readStorage(storage), keys, defaults);
  for (const group of groups) {
    setOpen(group, state[group.dataset.group]);
    group.addEventListener("toggle", () => {
      if (unprompted.has(group)) {
        unprompted.delete(group);
      } else {
        writeStorage(storage, recordGroupState(readStorage(storage), group.dataset.group, group.open));
      }
      onGroupToggle(group);
    });
  }

  let layersOn = new Set();
  const syncGroups = () => {
    const now = new Set(layerInputs(panel).filter((input) => input.checked).map((input) => input.id));
    for (const key of groupsToOpen(layersOn, now, (id) => groupOfLayer.get(id) || null)) setOpen(byKey.get(key), true);
    layersOn = now;
    for (const group of groups) {
      const pill = group.querySelector(".rp-group__count");
      if (!pill) continue;
      const count = layerInputs(group).filter((input) => input.checked).length;
      pill.textContent = count ? formatCount(count) : "";
      pill.hidden = count === 0;
    }
  };

  // The panel watches its own layer toggles, so the reader's clicks, presets, the place
  // search and the runtime's own switches all keep the groups in step. One sync follows a
  // batch of writes, such as a time feature switching several layers off at once.
  let syncQueued = false;
  const queueSync = () => {
    if (syncQueued) return;
    syncQueued = true;
    queueMicrotask(() => {
      syncQueued = false;
      syncGroups();
    });
  };
  for (const input of layerInputs(panel)) {
    input.addEventListener("change", queueSync);
    watchChecked(input, queueSync);
  }
  // Layers already on, such as a checkbox the browser restored, open their groups now.
  syncGroups();
}

function bindWhere({ panel, regionSelect, datasetSelect }) {
  const regionSegments = panel.querySelector('[data-segmented-for="regionPreset"]');
  const terrainSelect = panel.querySelector("#terrainPreset");
  const gridSegments = panel.querySelector("#gridPreset");
  const terrainHint = panel.querySelector("#terrainHint");
  let datasets = null;

  const shownChoice = () => getTerrainChoice(datasets, datasetSelect?.value) || { terrain: "", grid: "" };
  const chooseDataset = (choice) => {
    const key = datasets ? findDatasetForChoice(datasets, choice) : null;
    if (key) chooseOption(datasetSelect, key);
  };

  regionSegments?.addEventListener("click", (event) => {
    const button = closestButton(event);
    if (button && !button.disabled) chooseOption(regionSelect, button.dataset.value);
  });
  terrainSelect?.addEventListener("change", () => chooseDataset({ ...shownChoice(), terrain: terrainSelect.value }));
  gridSegments?.addEventListener("click", (event) => {
    const button = closestButton(event);
    if (button && !button.disabled) chooseDataset({ ...shownChoice(), grid: button.dataset.value });
  });

  const renderChoices = () => {
    const { terrains, grids } = listTerrainChoices(datasets);
    terrainSelect?.replaceChildren(...terrains.map(({ key, label }) => new Option(label, key)));
    gridSegments?.replaceChildren(...grids.map(({ key, label }) => createSegment(key, label)));
  };

  const syncRegion = () => {
    if (!regionSegments || !regionSelect) return;
    for (const button of regionSegments.querySelectorAll("button[data-value]")) {
      button.setAttribute("aria-pressed", String(button.dataset.value === regionSelect.value));
      button.disabled = regionSelect.disabled;
    }
  };

  const syncDataset = () => {
    if (!datasets || !datasetSelect) return;
    const choice = shownChoice();
    if (terrainSelect) {
      terrainSelect.value = choice.terrain;
      terrainSelect.disabled = datasetSelect.disabled || terrainSelect.options.length < 2;
    }
    for (const button of gridSegments?.querySelectorAll("button[data-value]") || []) {
      const option = { terrain: choice.terrain, grid: button.dataset.value };
      button.setAttribute("aria-pressed", String(option.grid === choice.grid));
      button.disabled = datasetSelect.disabled || !offersChoice(datasets, option);
      button.title = datasets[findDatasetForChoice(datasets, option)]?.summary || "";
    }
    if (terrainHint) {
      // Only a heavy grid earns a line of its own: its size, and that it wants a desktop.
      const shown = datasets[datasetSelect.value];
      terrainHint.textContent = shown?.desktopRecommended ? shown.summary || "" : "";
      terrainHint.hidden = !terrainHint.textContent;
    }
  };

  return function syncWhere({ datasets: nextDatasets } = {}) {
    if (nextDatasets && nextDatasets !== datasets) {
      datasets = nextDatasets;
      renderChoices();
    }
    syncRegion();
    syncDataset();
  };
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

/**
 * The reason a disabled layer gives, appended to its row; an empty reason removes it.
 * Inside the row's label it becomes part of the layer's accessible name as well.
 * @param {Element | null} control
 * @param {string} text
 */
function setRowReason(control, text) {
  const row = control instanceof Element ? control.closest(".rp-row, .rp-chip") : null;
  if (!row) return;
  let reason = row.querySelector(":scope > .rp-reason");
  if (!reason && !text) return;
  if (!reason) {
    reason = document.createElement("span");
    reason.className = "rp-reason";
    row.append(reason);
  }
  reason.textContent = text || "";
  reason.hidden = !text;
  row.classList.toggle("is-unavailable", Boolean(text));
}

/**
 * @param {{
 *   panel: HTMLElement | null,
 *   regionSelect: HTMLSelectElement | null,
 *   datasetSelect: HTMLSelectElement | null,
 *   storage?: Storage | null,
 *   formatCount?: (count: number) => string,
 *   onGroupToggle?: (group: HTMLDetailsElement) => void,
 * }} options
 * @returns {{ syncWhere(options?: { datasets?: object }): void, setReason(control: Element | null, text: string): void }}
 */
export function createResearchPanel({
  panel,
  regionSelect,
  datasetSelect,
  storage = null,
  formatCount = String,
  onGroupToggle = () => {},
}) {
  const groups = panel ? Array.from(panel.querySelectorAll("details.rp-group[data-group]")) : [];
  if (!groups.length) return INERT_PANEL;
  bindGroups({ panel, groups, storage, formatCount, onGroupToggle });
  const syncWhere = bindWhere({ panel, regionSelect, datasetSelect });
  bindInfoMenu(panel);
  return { syncWhere, setReason: setRowReason };
}
