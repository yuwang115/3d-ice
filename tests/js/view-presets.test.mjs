/**
 * Every named view the explorer opens on keeps the panel's one-of choices: no showcase
 * hero, `?preset=` view or guided-tour stop switches on two colourings of one surface, or
 * two time features, at once.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { TOUR_CHAPTERS } from "../../static/tools/js/explore-content.js";
import { resolveChapterControls } from "../../static/tools/js/explore-tour.js";
import { findChoiceConflicts } from "../../static/tools/js/panel-state.js";
import {
  EXPLORER_PRESETS,
  SHOWCASE_PRESETS,
  getExplorerPreset,
  getShowcasePreset,
} from "../../static/tools/js/view-presets.js";

const VIEWS = Object.entries({ ...SHOWCASE_PRESETS, ...EXPLORER_PRESETS });

test("there are views to check, each with its toggles", () => {
  assert.ok(VIEWS.length >= 8);
  for (const [name, preset] of VIEWS) assert.equal(typeof preset.toggles, "object", name);
});

test("no showcase or preset view switches on two layers of one one-of choice", () => {
  for (const [name, preset] of VIEWS) {
    assert.deepEqual(findChoiceConflicts(preset.toggles), [], name);
  }
});

test("no guided-tour stop switches on two time features at once", () => {
  assert.ok(TOUR_CHAPTERS.length > 0);
  for (const chapter of TOUR_CHAPTERS) {
    assert.deepEqual(findChoiceConflicts(resolveChapterControls(chapter.view.controls)), [], chapter.id);
  }
});

test("the check fails a view that would break a choice", () => {
  const [name, preset] = VIEWS.find(([, view]) => view.toggles.showEffectivePressure);
  const broken = { ...preset.toggles, showBasalFriction: true };
  assert.deepEqual(findChoiceConflicts(broken), [{ choice: "bed", toggles: ["showBasalFriction", "showEffectivePressure"] }], name);
});

test("an unknown name opens no preset view, and the home hero in a showcase", () => {
  assert.equal(getExplorerPreset("no-such-view"), null);
  assert.equal(getExplorerPreset("antarctica-ocean-circulations"), EXPLORER_PRESETS["antarctica-ocean-circulations"]);
  assert.equal(getShowcasePreset("no-such-view"), SHOWCASE_PRESETS["home-hero"]);
  assert.equal(getShowcasePreset("tools-hero"), SHOWCASE_PRESETS["tools-hero"]);
});

test("views are frozen, so the runtime cannot change one under another page", () => {
  for (const [name, preset] of VIEWS) {
    assert.ok(Object.isFrozen(preset) && Object.isFrozen(preset.toggles), name);
  }
});
