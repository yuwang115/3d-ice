/**
 * The research edition's panel, on top of the controls the runtime reads and writes.
 *
 * - Groups: collapsible <details class="rp-group" data-group>. Each opens as the reader
 *   last left it (localStorage), opens by itself when a layer in it is switched on, and
 *   counts the layers on in a pill beside its title. The panel watches its layer toggles
 *   (input[data-layer]) itself, including script writes to `checked`.
 * - Where: a terrain <select> and grid buttons over the dataset <select>. The select stays
 *   the runtime's source of truth: a choice here sets it and dispatches its change event,
 *   and syncWhere() mirrors back whatever the runtime wrote. The region buttons, like the
 *   masthead, are js/panel-masthead.js's, on both editions.
 * - One-of choices ([data-choice]): a surface's colourings as pills over their layer
 *   checkboxes, with a first pill (Elevation, Off) that switches them all off.
 * - Notes on a layer's row: the reason a disabled layer gives, and the time feature that
 *   switched it off; a row in a one-of choice says it on the choice's note line.
 * - Change over time, one feature at a time (js/research-time.js).
 *
 * The public page has no groups, so there the panel is inert. The panel takes every
 * string it shows from its caller, so all localisation stays in the runtime's t().
 */

import { chooseOption, closestButton } from "./panel-masthead.js";
import { bindTimeFeatures } from "./research-time.js";
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
  bindTimeFeatures() {},
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

function bindGroups({ panel, groups, storage, formatCount, onGroupToggle, onSynced }) {
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
    // The pill that is not a layer (Elevation, Off, Present day) is pressed while none is on.
    for (const none of panel.querySelectorAll("[data-choice-none]")) {
      none.setAttribute("aria-pressed", String(!none.closest("[data-choice]")?.querySelector("input:checked")));
    }
    panel.querySelector("[data-time-none]")?.setAttribute("aria-pressed", String(!panel.querySelector("input[data-time]:checked")));
    onSynced();
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

function bindWhere({ panel, datasetSelect }) {
  const terrainSelect = panel.querySelector("#terrainPreset");
  const gridSegments = panel.querySelector("#gridPreset");
  const terrainHint = panel.querySelector("#terrainHint");
  let datasets = null;

  const shownChoice = () => getTerrainChoice(datasets, datasetSelect?.value) || { terrain: "", grid: "" };
  const chooseDataset = (choice) => {
    const key = datasets ? findDatasetForChoice(datasets, choice) : null;
    if (key) chooseOption(datasetSelect, key);
  };

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
    syncDataset();
  };
}

function optionLabel(input) {
  return input.closest(".rp-choice__opt")?.textContent.trim() || "";
}

/**
 * A one-of choice's note line: one reason when every option is off for the same one (in
 * Greenland, both ice-shelf colourings are "Antarctica only"), else a line per option
 * that is off, with its label.
 */
function renderChoiceNote(field, formatOptionNote) {
  const note = field?.querySelector(".rp-choice__note");
  if (!note) return;
  const options = Array.from(field.querySelectorAll(".rp-choice__opt input[data-layer]"));
  const reasons = options.map((input) => input.dataset.reason || "");
  const lines =
    reasons.every(Boolean) && new Set(reasons).size === 1
      ? [reasons[0]]
      : options.flatMap((input) => {
          const text = input.dataset.reason || input.dataset.takeover || "";
          return text ? [formatOptionNote(optionLabel(input), text)] : [];
        });
  note.textContent = lines.join("\n");
  note.hidden = !lines.length;
}

/**
 * What a layer's row says besides its label: the reason a disabled layer gives (`reason`),
 * or the time feature that switched it off (`takeover`); an empty text removes it. Inside
 * the row's label it is part of the layer's accessible name as well. An option of a one-of
 * choice says it on the choice's note line, so each pill stays one line.
 */
function setRowNote(control, kind, text, formatOptionNote) {
  if (!(control instanceof Element)) return;
  if (control.closest(".rp-choice__opt")) {
    if (text) control.dataset[kind] = text;
    else delete control.dataset[kind];
    renderChoiceNote(control.closest("[data-choice]"), formatOptionNote);
    return;
  }
  const row = control.closest(".rp-row, .rp-chip");
  if (!row) return;
  const className = kind === "reason" ? "rp-reason" : "rp-takeover";
  let note = row.querySelector(`:scope > .${className}`);
  if (!note && !text) return;
  if (!note) {
    note = document.createElement("span");
    note.className = className;
    row.append(note);
  }
  note.textContent = text || "";
  note.hidden = !text;
  if (kind === "reason") row.classList.toggle("is-unavailable", Boolean(text));
}

/** The first pill of a one-of choice switches every colouring of its surface off. */
function bindChoices(panel) {
  for (const field of panel.querySelectorAll("[data-choice]")) {
    field.querySelector("[data-choice-none]")?.addEventListener("click", () => {
      for (const input of field.querySelectorAll(".rp-choice__opt input:checked")) {
        input.checked = false;
        input.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });
  }
}

/**
 * @param {{
 *   panel: HTMLElement | null,
 *   datasetSelect: HTMLSelectElement | null,
 *   storage?: Storage | null,
 *   formatCount?: (count: number) => string,
 *   formatOptionNote?: (layer: string, text: string) => string,
 *   onGroupToggle?: (group: HTMLDetailsElement) => void,
 * }} options
 * @returns {{
 *   syncWhere(options?: { datasets?: object }): void,
 *   setReason(control: Element | null, text: string): void,
 *   bindTimeFeatures(options: { features: readonly import("./research-time.js").TimeFeature[], layer: (id: string) => HTMLInputElement | null | undefined }): void,
 * }}
 */
export function createResearchPanel({
  panel,
  datasetSelect,
  storage = null,
  formatCount = String,
  formatOptionNote = (layer, text) => `${layer}: ${text}`,
  onGroupToggle = () => {},
}) {
  const groups = panel ? Array.from(panel.querySelectorAll("details.rp-group[data-group]")) : [];
  if (!groups.length) return INERT_PANEL;
  let time = null;
  bindGroups({ panel, groups, storage, formatCount, onGroupToggle, onSynced: () => time?.sync() });
  const syncWhere = bindWhere({ panel, datasetSelect });
  bindChoices(panel);
  return {
    syncWhere,
    setReason: (control, text) => setRowNote(control, "reason", text, formatOptionNote),
    bindTimeFeatures: ({ features, layer }) => {
      time = bindTimeFeatures({
        panel,
        features,
        layer,
        setTakeover: (control, text) => setRowNote(control, "takeover", text, formatOptionNote),
      });
    },
  };
}
