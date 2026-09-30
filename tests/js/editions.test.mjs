/**
 * Edition profiles decide which datasets, layers and behaviours a 3D ICE page offers.
 *
 * The research page must see the registry exactly as the runtime defines it, and the
 * public page must never be able to reach a research-only package: restricting a layer
 * drops the URLs of its packages, not just its toggle.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  EDITION_KEYS,
  applyEditionToRegions,
  getEditionProfile,
  getStandInSpec,
  resolveEditionKey,
} from "../../static/tools/js/editions.js";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const RUNTIME_SOURCE = readFileSync(resolve(repoRoot, "static/tools/js/explorer-app.js"), "utf8");
const RESEARCH_PAGE = readFileSync(resolve(repoRoot, "static/tools/3D-interactive-cryosphere-explorer.html"), "utf8");

/** The shape of the runtime's REGIONS registry, cut down to what the profiles touch. */
function makeRegions() {
  return {
    antarctica: {
      key: "antarctica",
      refinedBasinsUrl: "data/imbie.json",
      defaultDatasetKey: "balanced",
      capabilities: {
        velocity: true,
        basalFriction: true,
        flowline: true,
        rise: true,
        oceanCurrents: true,
        refinedBasins: true,
        hydrology: true,
        isostaticRebound: true,
      },
      sources: { geometry: { text: "BedMachine" } },
      datasets: {
        balanced: {
          id: "balanced",
          metaUrl: "data/bm_480.meta.json",
          binUrl: "data/bm_480.bin",
          reboundMetaUrl: "data/rebound_480.meta.json",
          reboundBinUrl: "data/rebound_480.bin",
          velocityMetaUrl: "data/vel_480.meta.json",
          velocityBinUrl: "data/vel_480.bin",
          basalFrictionMetaUrl: "data/taub_480.meta.json",
          basalFrictionBinUrl: "data/taub_480.bin",
          riseMetaUrl: "data/rise_480.meta.json",
          riseBinUrl: "data/rise_480.bin",
          oceanCurrentsMetaUrl: "data/waom.meta.json",
          oceanCurrentsBinUrl: "data/waom.bin",
          hydrologyMetaUrl: "data/hydro_480.meta.json",
          hydrologyBinUrl: "data/hydro_480.bin",
        },
        hd: { id: "hd", metaUrl: "data/bm_741.meta.json", binUrl: "data/bm_741.bin" },
        bedmap3: {
          id: "bedmap3",
          metaUrl: "data/bedmap3.meta.json",
          binUrl: "data/bedmap3.bin",
          capabilities: { velocity: true, basalFriction: true, rise: false, hydrology: true },
        },
      },
    },
    greenland: {
      key: "greenland",
      refinedBasinsUrl: "data/greenland_basins.json",
      defaultDatasetKey: "3km",
      capabilities: { velocity: true, basalFriction: true, oceanCurrents: true, refinedBasins: true, isostaticRebound: true },
      datasets: {
        "3km": {
          id: "3km",
          metaUrl: "data/gl_3km.meta.json",
          binUrl: "data/gl_3km.bin",
          velocityMetaUrl: "data/gl_vel.meta.json",
          velocityBinUrl: "data/gl_vel.bin",
          basalFrictionMetaUrl: "data/gl_taub.meta.json",
          basalFrictionBinUrl: "data/gl_taub.bin",
        },
        "1km": { id: "1km", metaUrl: "data/gl_1km.meta.json", binUrl: "data/gl_1km.bin" },
      },
    },
  };
}

test("an unknown or missing edition name falls back to the research edition", () => {
  assert.equal(resolveEditionKey("public"), EDITION_KEYS.public);
  assert.equal(resolveEditionKey("research"), EDITION_KEYS.research);
  for (const value of [undefined, null, "", "PUBLIC ", "pro", "kiosk"]) {
    assert.equal(resolveEditionKey(value), EDITION_KEYS.research, `edition ${String(value)}`);
  }
  assert.equal(getEditionProfile("nonsense").key, EDITION_KEYS.research);
});

test("the research edition leaves the dataset registry untouched", () => {
  const regions = makeRegions();
  const restricted = applyEditionToRegions(regions, getEditionProfile("research"));
  assert.deepEqual(restricted, makeRegions());
  assert.notEqual(restricted, regions, "a new object is returned");
  assert.notEqual(restricted.antarctica.datasets, regions.antarctica.datasets);
});

test("the public edition offers one terrain dataset per region", () => {
  const restricted = applyEditionToRegions(makeRegions(), getEditionProfile("public"));
  assert.deepEqual(Object.keys(restricted.antarctica.datasets), ["balanced"]);
  assert.deepEqual(Object.keys(restricted.greenland.datasets), ["3km"]);
  assert.equal(restricted.antarctica.defaultDatasetKey, "balanced");
  assert.equal(restricted.greenland.defaultDatasetKey, "3km");
});

test("the public edition switches off research layers at the region and the dataset level", () => {
  const restricted = applyEditionToRegions(makeRegions(), getEditionProfile("public"));
  for (const region of Object.values(restricted)) {
    for (const capability of ["basalFriction", "rise", "hydrology", "refinedBasins"]) {
      assert.equal(region.capabilities[capability], false, `${region.key}.${capability}`);
      for (const dataset of Object.values(region.datasets)) {
        if (dataset.capabilities) {
          assert.equal(dataset.capabilities[capability], false, `${region.key}/${dataset.id}.${capability}`);
        }
      }
    }
    assert.equal(region.capabilities.velocity, true, "flowlines need the velocity field");
    assert.equal(region.capabilities.oceanCurrents, true);
    assert.equal(region.capabilities.isostaticRebound, true);
  }
});

test("the public edition drops every package URL owned by a research layer", () => {
  const restricted = applyEditionToRegions(makeRegions(), getEditionProfile("public"));
  const balanced = restricted.antarctica.datasets.balanced;
  for (const key of [
    "basalFrictionMetaUrl",
    "basalFrictionBinUrl",
    "riseMetaUrl",
    "riseBinUrl",
    "hydrologyMetaUrl",
    "hydrologyBinUrl",
  ]) {
    assert.equal(key in balanced, false, `antarctica/balanced keeps ${key}`);
  }
  assert.equal("refinedBasinsUrl" in restricted.antarctica, false);
  assert.equal("refinedBasinsUrl" in restricted.greenland, false);
  assert.equal("basalFrictionMetaUrl" in restricted.greenland.datasets["3km"], false);

  for (const key of [
    "metaUrl",
    "binUrl",
    "velocityMetaUrl",
    "velocityBinUrl",
    "oceanCurrentsMetaUrl",
    "oceanCurrentsBinUrl",
    "reboundMetaUrl",
    "reboundBinUrl",
  ]) {
    assert.equal(balanced[key], makeRegions().antarctica.datasets.balanced[key], `antarctica/balanced loses ${key}`);
  }
});

test("restricting an edition never mutates the registry it is given", () => {
  const regions = makeRegions();
  applyEditionToRegions(regions, getEditionProfile("public"));
  assert.deepEqual(regions, makeRegions());
});

test("a region whose default dataset is excluded falls back to the first one offered", () => {
  const regions = makeRegions();
  regions.greenland.defaultDatasetKey = "1km";
  const restricted = applyEditionToRegions(regions, getEditionProfile("public"));
  assert.equal(restricted.greenland.defaultDatasetKey, "3km");
});

test("only the public edition stands in for controls its page leaves out", () => {
  assert.equal(getStandInSpec(getEditionProfile("research"), "showBasalFriction"), null);

  const publicProfile = getEditionProfile("public");
  assert.deepEqual(getStandInSpec(publicProfile, "showBasalFriction"), {});
  assert.deepEqual(getStandInSpec(publicProfile, "showIceBottom"), { type: "checkbox", checked: true });
  for (const layer of ["Surface", "Upper", "Mid", "Lower"]) {
    assert.equal(
      getStandInSpec(publicProfile, `showOceanLayer${layer}`).checked,
      true,
      `the ${layer} ocean band rides on the single ocean toggle`
    );
  }
  assert.equal(getStandInSpec(publicProfile, "reboundSeaLevel").value, 0);
  assert.equal(getStandInSpec(publicProfile, "resolutionPreset").tag, "select");
  assert.equal(getStandInSpec(publicProfile, "oceanLegendCanvas").tag, "canvas", "the legend needs a 2D context");
});

test("the public edition pins the Earth response to the published model the research page offers", () => {
  const pinned = getStandInSpec(getEditionProfile("public"), "reboundModel").value;
  assert.match(RUNTIME_SOURCE, new RegExp(`published: "${pinned}"`));
  assert.match(RESEARCH_PAGE, new RegExp(`<option value="${pinned}" selected>`));
});

test("every dataset the public edition offers exists in the runtime registry", () => {
  const publicProfile = getEditionProfile("public");
  for (const [regionKey, datasetKeys] of Object.entries(publicProfile.datasets)) {
    assert.match(RUNTIME_SOURCE, new RegExp(`\\n  ${regionKey}: \\{`), `region ${regionKey}`);
    for (const datasetKey of datasetKeys) {
      assert.match(RUNTIME_SOURCE, new RegExp(`id: "${datasetKey}",`), `dataset ${regionKey}/${datasetKey}`);
    }
  }
});

test("background warm-up fetches only layers the edition offers, the public one no more than flowlines need", () => {
  // The research order is the runtime's historical warm-up sequence.
  assert.deepEqual(getEditionProfile("research").backgroundWarmup, ["velocity", "basalFriction", "hydrology", "oceanCurrents"]);
  // The ocean package is tens of megabytes: the public edition fetches it only on request.
  assert.deepEqual(getEditionProfile("public").backgroundWarmup, ["velocity"]);
  for (const key of Object.values(EDITION_KEYS)) {
    const profile = getEditionProfile(key);
    for (const layer of profile.backgroundWarmup) {
      assert.ok(!profile.disabledCapabilities.includes(layer), `${key} warms up disabled layer ${layer}`);
    }
  }
});

test("profiles are frozen so a page cannot reconfigure its edition at run time", () => {
  for (const key of Object.values(EDITION_KEYS)) {
    const profile = getEditionProfile(key);
    assert.ok(Object.isFrozen(profile), `${key} profile`);
    assert.ok(Object.isFrozen(profile.fixedControls), `${key} fixed controls`);
    // Test modules run in strict mode, where writing to a frozen object throws.
    assert.throws(() => {
      profile.guide = !profile.guide;
    }, TypeError);
  }
});
