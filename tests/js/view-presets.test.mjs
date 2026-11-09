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

test("every layer of the home page's data tower opens a view of its own region", async () => {
  const { readFile } = await import("node:fs/promises");
  const { getEditionProfile } = await import("../../static/tools/js/editions.js");
  // The layer toggles a public-edition view may not switch on: their capability is off there.
  const capabilityOf = {
    showBasalFriction: "basalFriction",
    showBasalMelt: "rise",
    showThermalDriving: "rise",
    showEffectivePressure: "hydrology",
    showSubglacialChannels: "hydrology",
    showRefinedBasins: "refinedBasins",
    showSurfaceChange: "surfaceChange",
    showSeaIce: "seaIce",
  };
  const publicOff = new Set(getEditionProfile("public").disabledCapabilities);
  for (const page of ["../../static/index.html", "../../static/zh/index.html"]) {
    const html = await readFile(new URL(page, import.meta.url), "utf8");
    const links = [...html.matchAll(/href="\.\/(explore\/|tools\/3D-interactive-cryosphere-explorer\.html)\?region=([a-z]+)&amp;preset=([a-z0-9-]+)" data-opens="([a-z]+)"/g)];
    assert.equal(links.length, 22, page);
    for (const [, target, region, presetKey, opens] of links) {
      const preset = getExplorerPreset(presetKey);
      assert.ok(preset, `${page}: no view ${presetKey}`);
      assert.equal(preset.regionKey, region, presetKey);
      assert.equal(opens, target === "explore/" ? "public" : "research", presetKey);
      if (opens !== "public") continue;
      const needs = Object.entries(preset.toggles)
        .filter(([id, on]) => on && publicOff.has(capabilityOf[id]))
        .map(([id]) => id);
      assert.deepEqual(needs, [], `${presetKey} opens in the public edition, which lacks ${needs}`);
    }
  }
});
