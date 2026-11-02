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
  ONE_OF_CHOICES,
  TIME_FEATURE_TOGGLES,
  describeUnavailability,
  findChoiceConflicts,
  findDatasetForChoice,
  getTerrainChoice,
  groupsToOpen,
  listTerrainChoices,
  offersChoice,
  planTimeChange,
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

// ------------------------------------------------ one-of choices and time features

test("the bed and the ice-shelf base each take one colouring, and the four time features one at a time", () => {
  assert.deepEqual(ONE_OF_CHOICES.bed, ["showBasalFriction", "showEffectivePressure"]);
  assert.deepEqual(ONE_OF_CHOICES.shelf, ["showBasalMelt", "showThermalDriving"]);
  assert.deepEqual(TIME_FEATURE_TOGGLES, ["showSurfaceChange", "showSeaIce", "showIceProjection", "showIsostaticRebound"]);
  assert.ok(Object.isFrozen(ONE_OF_CHOICES) && Object.isFrozen(ONE_OF_CHOICES.bed) && Object.isFrozen(TIME_FEATURE_TOGGLES));
});

test("a set of toggles that switches on two colourings of one surface, or two time features, is a conflict", () => {
  assert.deepEqual(findChoiceConflicts({ showBasalFriction: true, showVelocity: true }), []);
  assert.deepEqual(findChoiceConflicts({ showBasalFriction: true, showEffectivePressure: true }), [
    { choice: "bed", toggles: ["showBasalFriction", "showEffectivePressure"] },
  ]);
  assert.deepEqual(findChoiceConflicts({ showBasalMelt: true, showThermalDriving: true, showSeaIce: true, showSurfaceChange: true }), [
    { choice: "shelf", toggles: ["showBasalMelt", "showThermalDriving"] },
    { choice: "time", toggles: ["showSurfaceChange", "showSeaIce"] },
  ]);
  // Only a true value switches a layer on.
  assert.deepEqual(findChoiceConflicts({ showBasalFriction: "yes", showEffectivePressure: 1 }), []);
});

const SWITCHES_OFF = Object.freeze({
  projection: ["showVelocity", "showBasalFriction", "showOceanCurrents"],
  rebound: ["showVelocity", "showFlowline"],
  surfaceChange: ["showVelocity"],
  seaIce: [],
});

const SWITCHES_ON = Object.freeze({
  projection: ["showFlowline", "showSea"],
  rebound: ["showSea"],
  seaIce: ["showSea"],
});

function onlyOn(...ids) {
  const on = new Set(ids);
  return (id) => on.has(id);
}

function plan(change) {
  return planTimeChange({ saved: [], raised: [], switchesOff: SWITCHES_OFF, switchesOn: SWITCHES_ON, ...change });
}

test("entering a time feature from the present day remembers what it will switch off and on", () => {
  assert.deepEqual(plan({ active: null, next: "projection", isOn: onlyOn("showVelocity", "showOceanCurrents", "showSea") }), {
    restore: [],
    lower: [],
    save: ["showVelocity", "showOceanCurrents"],
    // The sea plane is on already, so only the flowlines are the projection's to put away.
    raise: ["showFlowline"],
  });
});

test("returning to the present day puts back what the feature switched off, and lowers what it raised", () => {
  assert.deepEqual(
    plan({
      active: "rebound",
      saved: ["showVelocity", "showFlowline"],
      raised: ["showSea"],
      next: null,
      isOn: onlyOn("showFlowline", "showSea"),
    }),
    { restore: ["showVelocity"], lower: ["showSea"], save: [], raise: [] }
  );
});

test("going straight to another time feature keeps what that one switches off or on too", () => {
  assert.deepEqual(
    plan({
      active: "projection",
      saved: ["showVelocity", "showBasalFriction"],
      raised: ["showSea"],
      next: "rebound",
      isOn: onlyOn("showFlowline", "showSea"),
    }),
    // Velocity stays off and the sea plane on, both now the rebound's; friction comes back;
    // the flowlines are on, and the rebound switches them off.
    { restore: ["showBasalFriction"], lower: [], save: ["showVelocity", "showFlowline"], raise: ["showSea"] }
  );
});

test("a layer that comes back is the reader's again, even if the next feature also shows it", () => {
  // The reader had the flowlines on; the rebound switched them off, and the projection,
  // which shows them too, gets them back. They are not the projection's to put away.
  assert.deepEqual(
    plan({ active: "rebound", saved: ["showVelocity", "showFlowline"], raised: [], next: "projection", isOn: onlyOn() }),
    { restore: ["showFlowline"], lower: [], save: ["showVelocity"], raise: ["showSea"] }
  );
});

test("a raised layer the reader has since switched off needs no lowering", () => {
  assert.deepEqual(plan({ active: "seaIce", raised: ["showSea"], next: null, isOn: onlyOn() }), {
    restore: [],
    lower: [],
    save: [],
    raise: [],
  });
});

test("choosing the feature already on changes nothing", () => {
  assert.deepEqual(plan({ active: "seaIce", saved: ["showVelocity"], raised: ["showSea"], next: "seaIce", isOn: onlyOn() }), {
    restore: [],
    lower: [],
    save: ["showVelocity"],
    raise: ["showSea"],
  });
  assert.deepEqual(
    planTimeChange({ active: null, saved: [], next: "unknown", switchesOff: SWITCHES_OFF, isOn: onlyOn("showVelocity") }),
    { restore: [], lower: [], save: [], raise: [] }
  );
});
