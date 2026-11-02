/**
 * Change over time on the research page: one feature at a time, from the present day.
 *
 * Each feature switches off the layers drawn against today's ice or bed that it replaces,
 * and switches on the ones it needs (the sea plane, the flowlines). Choosing one remembers
 * what it changed; leaving it, for the present day or straight for another feature, puts
 * those back, unless the next feature changes them the same way. Switching one of the
 * replaced layers on instead ends the feature, and only that layer changes; a layer it
 * raised that the reader then sets by hand is the reader's. While a feature shows, the
 * rows it switched off say so.
 *
 * The feature toggles stay the runtime's. The panel stops the reader's own click on one
 * and makes the change itself, in order, each step as the toggle's own change event: the
 * feature that was on goes first, so the runtime lowers whatever it raised (the sea plane,
 * the flowlines), then the layers come back, then the next feature starts.
 */

import { planTimeChange } from "./panel-state.js";

/**
 * @typedef {{
 *   key: string,
 *   toggle: HTMLInputElement,
 *   switchesOff: readonly string[],
 *   switchesOn: readonly string[],
 *   takeoverText: string,
 * }} TimeFeature
 */

/**
 * @param {{
 *   panel: HTMLElement,
 *   features: readonly TimeFeature[],
 *   layer: (id: string) => HTMLInputElement | null | undefined,
 *   setTakeover: (control: Element, text: string) => void,
 * }} options
 * @returns {{ sync(): void }}
 */
export function bindTimeFeatures({ panel, features, layer, setTakeover }) {
  const byKey = new Map(features.map((feature) => [feature.key, feature]));
  const byToggle = new Map(features.map((feature) => [feature.toggle, feature]));
  const switchesOff = Object.fromEntries(features.map((feature) => [feature.key, feature.switchesOff]));
  const switchesOn = Object.fromEntries(features.map((feature) => [feature.key, feature.switchesOn]));
  const replaced = new Set(features.flatMap((feature) => feature.switchesOff));
  let active = features.find((feature) => feature.toggle.checked)?.key ?? null;
  let saved = [];
  let raised = [];
  // While the panel makes a change itself, the events it dispatches are its own.
  let changing = false;

  const isOn = (id) => Boolean(layer(id)?.checked);
  const setToggle = (control, checked) => {
    if (!control || control.checked === checked) return;
    control.checked = checked;
    control.dispatchEvent(new Event("change", { bubbles: true }));
  };
  const inChange = (steps) => {
    changing = true;
    try {
      steps();
    } finally {
      changing = false;
    }
  };

  const showTakeovers = () => {
    const text = active ? byKey.get(active).takeoverText : "";
    for (const id of replaced) {
      const control = layer(id);
      if (control) setTakeover(control, saved.includes(id) ? text : "");
    }
  };

  const choose = (next) => {
    if (changing || next === active) return;
    inChange(() => {
      const previous = active;
      if (previous) setToggle(byKey.get(previous).toggle, false);
      const plan = planTimeChange({ active: previous, saved, raised, next, switchesOff, switchesOn, isOn });
      for (const id of plan.restore) {
        const control = layer(id);
        if (control && !control.disabled) setToggle(control, true);
      }
      for (const id of plan.lower) setToggle(layer(id), false);
      active = next;
      saved = plan.save;
      raised = plan.raise;
      if (next) setToggle(byKey.get(next).toggle, true);
    });
    showTakeovers();
  };

  // A layer the feature replaced, switched on by the reader: the feature ends, nothing
  // comes back, and the layer's own change then runs on the scene the feature handed back.
  const endFor = () => {
    const feature = byKey.get(active);
    active = null;
    saved = [];
    raised = [];
    inChange(() => setToggle(feature.toggle, false));
    showTakeovers();
  };

  // Capture, so the panel sees the reader's change before the runtime's own listeners.
  panel.addEventListener(
    "change",
    (event) => {
      const target = event.target;
      if (!event.isTrusted || changing || !(target instanceof HTMLInputElement)) return;
      const feature = byToggle.get(target);
      if (feature) {
        event.stopImmediatePropagation();
        const wanted = target.checked;
        target.checked = !wanted;
        choose(wanted ? feature.key : null);
        return;
      }
      // A layer the reader sets by hand is theirs: the feature neither lowers nor restores it.
      if (raised.includes(target.id)) raised = raised.filter((id) => id !== target.id);
      if (saved.includes(target.id)) saved = saved.filter((id) => id !== target.id);
      if (target.checked && active && switchesOff[active].includes(target.id)) endFor();
    },
    true
  );
  panel.querySelector("[data-time-none]")?.addEventListener("click", () => choose(null));

  return {
    /** Follow a feature the runtime switched on or off itself: a preset, a terrain without it. */
    sync() {
      if (changing) return;
      if (active && !byKey.get(active).toggle.checked) {
        active = null;
        saved = [];
        raised = [];
        showTakeovers();
      }
      if (!active) active = features.find((feature) => feature.toggle.checked)?.key ?? null;
    },
  };
}
