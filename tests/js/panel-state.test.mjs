/**
 * The research panel's decisions that need no DOM: which terrain and grid a dataset is,
 * why a layer is off on the terrain shown, and which groups start open.
 *
 * The registries here have the shape of the runtime's REGIONS, cut down to the fields
 * these functions read.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  describeUnavailability,
  findDatasetForChoice,
  getTerrainChoice,
  groupsToOpen,
  listTerrainChoices,
  offersChoice,
  readGroupState,
  recordGroupState,
} from "../../static/tools/js/panel-state.js";

const ANTARCTIC_DATASETS = Object.freeze({
  balanced: { id: "balanced", terrain: "bedmachine", terrainLabel: "BedMachine v4", grid: "10km", gridLabel: "10 km" },
  hd: { id: "hd", terrain: "bedmachine", terrainLabel: "BedMachine v4", grid: "4km", gridLabel: "4 km" },
  bedmap3: { id: "bedmap3", terrain: "bedmap3", terrainLabel: "Bedmap3", grid: "10km", gridLabel: "10 km" },
  "bedmap3-hd": { id: "bedmap3-hd", terrain: "bedmap3", terrainLabel: "Bedmap3", grid: "4km", gridLabel: "4 km" },
});

const REGIONS = Object.freeze({
  antarctica: { key: "antarctica", datasets: ANTARCTIC_DATASETS },
  greenland: {
    key: "greenland",
    datasets: {
      "3km": { id: "3km", terrain: "bedmachine", terrainLabel: "BedMachine v6", grid: "3km", gridLabel: "3 km" },
      "1km": { id: "1km", terrain: "bedmachine", terrainLabel: "BedMachine v6", grid: "1km", gridLabel: "1 km" },
    },
  },
});

/** A layer offered on the listed datasets, as `regionKey/datasetKey`. */
function offeredOn(...places) {
  const offered = new Set(places);
  return (regionKey, datasetKey) => offered.has(`${regionKey}/${datasetKey}`);
}

test("a region's datasets split into terrains and grids, in registry order", () => {
  assert.deepEqual(listTerrainChoices(ANTARCTIC_DATASETS), {
    terrains: [
      { key: "bedmachine", label: "BedMachine v4" },
      { key: "bedmap3", label: "Bedmap3" },
    ],
    grids: [
      { key: "10km", label: "10 km" },
      { key: "4km", label: "4 km" },
    ],
  });
});

test("a dataset without a terrain or grid of its own is a terrain and grid of its own", () => {
  assert.deepEqual(listTerrainChoices({ only: { id: "only", label: "Only terrain" } }), {
    terrains: [{ key: "only", label: "Only terrain" }],
    grids: [{ key: "only", label: "only" }],
  });
  assert.deepEqual(listTerrainChoices(undefined), { terrains: [], grids: [] });
});

test("each dataset names one terrain and one grid", () => {
  assert.deepEqual(getTerrainChoice(ANTARCTIC_DATASETS, "bedmap3-hd"), { terrain: "bedmap3", grid: "4km" });
  assert.deepEqual(getTerrainChoice(ANTARCTIC_DATASETS, "balanced"), { terrain: "bedmachine", grid: "10km" });
  assert.equal(getTerrainChoice(ANTARCTIC_DATASETS, "missing"), null);
});

test("a terrain and a grid pick out their dataset", () => {
  assert.equal(findDatasetForChoice(ANTARCTIC_DATASETS, { terrain: "bedmap3", grid: "4km" }), "bedmap3-hd");
  assert.equal(findDatasetForChoice(ANTARCTIC_DATASETS, { terrain: "bedmachine", grid: "10km" }), "balanced");
});

test("a pair the region lacks keeps the terrain, then falls back to the first dataset", () => {
  const datasets = { a: ANTARCTIC_DATASETS.balanced, b: ANTARCTIC_DATASETS.bedmap3 };
  assert.equal(findDatasetForChoice(datasets, { terrain: "bedmap3", grid: "4km" }), "bedmap3");
  assert.equal(findDatasetForChoice(datasets, { terrain: "qrf", grid: "1km" }), "balanced");
  assert.equal(findDatasetForChoice({}, { terrain: "bedmap3", grid: "4km" }), null);
});

test("a grid is on offer only where the terrain has a dataset on it", () => {
  const datasets = { a: ANTARCTIC_DATASETS.balanced, b: ANTARCTIC_DATASETS.hd, c: ANTARCTIC_DATASETS.bedmap3 };
  assert.equal(offersChoice(datasets, { terrain: "bedmachine", grid: "4km" }), true);
  assert.equal(offersChoice(datasets, { terrain: "bedmap3", grid: "4km" }), false);
});

test("a layer the terrain shown offers needs no reason", () => {
  const offers = offeredOn("antarctica/balanced");
  assert.equal(describeUnavailability(REGIONS, "antarctica", "balanced", offers), null);
});

test("a layer no terrain of this region offers names the region that has it", () => {
  const hydrology = offeredOn("antarctica/balanced", "antarctica/hd", "antarctica/bedmap3", "antarctica/bedmap3-hd");
  assert.deepEqual(describeUnavailability(REGIONS, "greenland", "3km", hydrology), { kind: "region", region: "antarctica" });
});

test("a layer one terrain offers on every grid names that terrain", () => {
  const melt = offeredOn("antarctica/balanced", "antarctica/hd");
  assert.deepEqual(describeUnavailability(REGIONS, "antarctica", "bedmap3", melt), {
    kind: "terrain",
    terrain: "BedMachine v4",
  });
});

test("a layer only one dataset offers names its terrain and grid", () => {
  const projection = offeredOn("antarctica/balanced");
  assert.deepEqual(describeUnavailability(REGIONS, "antarctica", "hd", projection), {
    kind: "dataset",
    terrain: "BedMachine v4",
    grid: "10 km",
  });
});

test("a layer every terrain offers on one grid names that grid", () => {
  const coarseOnly = offeredOn("antarctica/balanced", "antarctica/bedmap3");
  assert.deepEqual(describeUnavailability(REGIONS, "antarctica", "bedmap3-hd", coarseOnly), { kind: "grid", grid: "10 km" });
});

test("a layer with no single place to point to, or none at all, has no specific reason", () => {
  const scattered = offeredOn("antarctica/balanced", "antarctica/bedmap3-hd");
  assert.deepEqual(describeUnavailability(REGIONS, "antarctica", "hd", scattered), { kind: "none" });
  assert.deepEqual(describeUnavailability(REGIONS, "antarctica", "hd", () => false), { kind: "none" });
  const threeRegions = { ...REGIONS, svalbard: { key: "svalbard", datasets: { s: { id: "s" } } } };
  const elsewhere = offeredOn("greenland/3km", "svalbard/s");
  assert.deepEqual(describeUnavailability(threeRegions, "antarctica", "balanced", elsewhere), { kind: "none" });
});

test("groups start as stored, and as the page marks them where nothing is stored", () => {
  const keys = ["scene", "flow", "bed", "time"];
  const defaults = ["scene", "flow", "time"];
  assert.deepEqual(readGroupState(null, keys, defaults), { scene: true, flow: true, bed: false, time: true });
  assert.deepEqual(readGroupState('{"flow":false,"bed":true}', keys, defaults), {
    scene: true,
    flow: false,
    bed: true,
    time: true,
  });
});

test("a stored state that is not a map of group names to booleans is ignored", () => {
  const keys = ["scene", "bed"];
  const defaults = ["scene"];
  const expected = { scene: true, bed: false };
  for (const stored of ["not json", "[true,false]", '"open"', "42", '{"scene":"no","bed":1}']) {
    assert.deepEqual(readGroupState(stored, keys, defaults), expected, stored);
  }
  assert.deepEqual(readGroupState('{"gone":true,"bed":true}', keys, defaults), { scene: true, bed: true });
});

test("only the group the reader opened or closed is recorded, over what was stored", () => {
  assert.deepEqual(JSON.parse(recordGroupState(null, "bed", true)), { bed: true });
  assert.deepEqual(JSON.parse(recordGroupState('{"flow":false}', "bed", true)), { flow: false, bed: true });
  assert.deepEqual(JSON.parse(recordGroupState('{"bed":true}', "bed", false)), { bed: false });
  // Whatever else was stored is dropped rather than carried forward.
  assert.deepEqual(JSON.parse(recordGroupState('{"flow":"no"}', "bed", true)), { bed: true });
  assert.deepEqual(JSON.parse(recordGroupState("[1]", "bed", false)), { bed: false });
});

test("a group opens when a layer in it has just been switched on", () => {
  const groupOf = (id) => ({ showVelocity: "flow", showBasalFriction: "bed", showSea: "scene" })[id] || null;
  const before = new Set(["showVelocity"]);
  const after = new Set(["showVelocity", "showBasalFriction", "showUnknown"]);
  assert.deepEqual([...groupsToOpen(before, after, groupOf)], ["bed"]);
  assert.deepEqual([...groupsToOpen(after, after, groupOf)], []);
  assert.deepEqual([...groupsToOpen(new Set(), new Set(["showSea", "showVelocity"]), groupOf)], ["scene", "flow"]);
});
