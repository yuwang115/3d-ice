/**
 * The guided tour places its camera by latitude and longitude, so the browser needs the
 * same polar stereographic projections the preparation pipeline used. The place catalogues
 * store both coordinates for every feature, which makes them the reference: the projection
 * must reproduce each stored x/y, as scripts/prepare_polar_features.py computed it.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { POLAR_PROJECTIONS, projectLatLon } from "../../static/tools/js/polar-projection.js";

const here = dirname(fileURLToPath(import.meta.url));
const dataDir = resolve(here, "..", "..", "static", "tools", "data");

const CATALOGUES = [
  ["antarctica", "antarctica_research_stations.json"],
  ["antarctica", "antarctica_geographic_names.json"],
  ["greenland", "greenland_research_stations.json"],
  ["greenland", "greenland_geographic_names.json"],
];

// The catalogues round x/y to the millimetre.
const TOLERANCE_M = 0.002;

for (const [region, file] of CATALOGUES) {
  test(`${file}: every feature projects onto its stored position`, () => {
    const catalogue = JSON.parse(readFileSync(resolve(dataDir, file), "utf8"));
    assert.equal(catalogue.projection, POLAR_PROJECTIONS[region].epsg);
    assert.ok(catalogue.items.length >= 5, "catalogue has features");
    for (const item of catalogue.items) {
      const { x, y } = projectLatLon(item.lat, item.lon, region);
      assert.ok(Math.abs(x - item.x_m) <= TOLERANCE_M, `${item.id}: x ${x} vs ${item.x_m}`);
      assert.ok(Math.abs(y - item.y_m) <= TOLERANCE_M, `${item.id}: y ${y} vs ${item.y_m}`);
    }
  });
}

test("the poles project onto the origin of their grid", () => {
  const south = projectLatLon(-90, 0, "antarctica");
  const north = projectLatLon(90, 0, "greenland");
  for (const point of [south, north]) {
    assert.ok(Math.abs(point.x) < 1e-6 && Math.abs(point.y) < 1e-6, JSON.stringify(point));
  }
});

test("grid north follows the central meridian of each projection", () => {
  // EPSG:3031 puts 0 deg E along +y; EPSG:3413 puts 45 deg W along -y.
  const antarctic = projectLatLon(-70, 0, "antarctica");
  assert.ok(antarctic.y > 0 && Math.abs(antarctic.x) < 1e-6);
  const greenland = projectLatLon(70, -45, "greenland");
  assert.ok(greenland.y < 0 && Math.abs(greenland.x) < 1e-6);
});

test("coordinates outside a region's hemisphere or range are rejected", () => {
  assert.throws(() => projectLatLon(60, 0, "antarctica"), RangeError);
  assert.throws(() => projectLatLon(-60, 0, "greenland"), RangeError);
  assert.throws(() => projectLatLon(-91, 0, "antarctica"), RangeError);
  assert.throws(() => projectLatLon(-70, 181, "antarctica"), RangeError);
  assert.throws(() => projectLatLon(Number.NaN, 0, "antarctica"), RangeError);
  assert.throws(() => projectLatLon(-70, 0, "mars"), RangeError);
});

test("unprojectXY inverts projectLatLon on both grids", async () => {
  const { projectLatLon, unprojectXY } = await import("../../static/tools/js/polar-projection.js");
  const cases = [
    ["antarctica", -74.9, -102.5],
    ["antarctica", -60, 140],
    ["antarctica", -89.5, 10],
    ["antarctica", -65, 179.9],
    ["greenland", 72.5, -38.5],
    ["greenland", 61, -45],
  ];
  for (const [region, lat, lon] of cases) {
    const { x, y } = projectLatLon(lat, lon, region);
    const back = unprojectXY(x, y, region);
    assert.ok(Math.abs(back.lat - lat) < 1e-9, `${region} ${lat}: ${back.lat}`);
    assert.ok(Math.abs(((back.lon - lon + 540) % 360) - 180) < 1e-9, `${region} ${lon}: ${back.lon}`);
  }
  assert.throws(() => unprojectXY(0, 0, "mars"), RangeError);
  assert.throws(() => unprojectXY(Number.NaN, 0, "antarctica"), RangeError);
});
