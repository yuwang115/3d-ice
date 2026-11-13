/**
 * The divide packages of js/basin-divides.js: the committed Mankoff et al. (2020) package
 * validates and decodes onto Greenland, and a malformed package is refused before any of it
 * reaches the scene.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  BASIN_DIVIDE_SCHEMA,
  decodeBasinDivideArc,
  decodeBasinDivides,
  isBasinDivideDataset,
  validateBasinDivideDataset,
} from "../../static/tools/js/basin-divides.js";
import { fetchRefinedBasinJson, validateRefinedBasinDataset } from "../../static/tools/js/polar-refined-basins.js";

const PACKAGE = JSON.parse(
  readFileSync(new URL("../../static/tools/data/greenland_mankoff_ice_basins_k1.json", import.meta.url), "utf8")
);

function minimal(overrides = {}) {
  return {
    schema: BASIN_DIVIDE_SCHEMA,
    schema_version: 1,
    lattice_m: 100,
    min_area_km2: 10,
    basin_count: 3,
    large_basin_count: 2,
    large_area_fraction: 0.9,
    divide_count: 2,
    point_count: 5,
    divides: [
      [10, -20, 1, 0, 0, 2],
      [11, -18, 3, 0],
    ],
    ...overrides,
  };
}

test("the committed package validates and decodes onto Greenland", () => {
  assert.equal(validateBasinDivideDataset(PACKAGE), PACKAGE);
  const divides = decodeBasinDivides(PACKAGE);
  assert.equal(divides.length, PACKAGE.divide_count);
  assert.equal(
    divides.reduce((sum, line) => sum + line.length, 0),
    PACKAGE.point_count
  );
  for (const line of divides) {
    for (const [x, y] of line) {
      assert.ok(x > -700_000 && x < 950_000, `x ${x}`);
      assert.ok(y > -3_450_000 && y < -550_000, `y ${y}`);
    }
  }
});

test("an arc decodes from lattice deltas to metres", () => {
  assert.deepEqual(decodeBasinDivideArc([10, -20, 1, 0, 0, 2], 100), [
    [1000, -2000],
    [1100, -2000],
    [1100, -1800],
  ]);
  assert.deepEqual(decodeBasinDivides(minimal())[1], [
    [1100, -1800],
    [1400, -1800],
  ]);
});

test("only a divide package is taken for one", () => {
  assert.ok(isBasinDivideDataset(minimal()));
  assert.ok(!isBasinDivideDataset({ basins: [], basin_count: 0 }));
  assert.ok(!isBasinDivideDataset(null));
});

test("a malformed divide package is refused", () => {
  const cases = [
    [{ schema: "something-else" }, /Invalid basin divide dataset/],
    [{ schema_version: 2 }, /schema version/],
    [{ lattice_m: 0 }, /lattice/],
    [{ lattice_m: 99.5 }, /lattice/],
    [{ basin_count: -1 }, /basin_count/],
    [{ large_basin_count: 4 }, /more large basins/],
    [{ large_area_fraction: 1.2 }, /large_area_fraction/],
    [{ min_area_km2: "10" }, /min_area_km2/],
    [{ min_area_km2: Infinity }, /min_area_km2/],
    [{ large_area_fraction: Number.NaN }, /large_area_fraction/],
    [{ divide_count: 3 }, /divide_count/],
    [{ point_count: 6 }, /point_count/],
    [{ divides: [[10, -20, 1]], divide_count: 1, point_count: 1 }, /invalid arc/],
    [{ divides: [[10, -20]], divide_count: 1, point_count: 1 }, /invalid arc/],
    [{ divides: [[10, -20, 0.5, 0]], divide_count: 1, point_count: 2 }, /integers/],
    [{ divides: [[100_000, 0, 1, 0]], divide_count: 1, point_count: 2 }, /out of range/],
    [{ divides: "nope" }, /must be a list/],
  ];
  for (const [overrides, pattern] of cases) {
    assert.throws(() => validateBasinDivideDataset(minimal(overrides)), pattern, JSON.stringify(overrides));
  }
});

test("the basin fetcher validates with the validator it is given", async () => {
  const fetchImpl = async () => new Response(JSON.stringify(minimal()), { status: 200 });
  const validate = (payload) =>
    isBasinDivideDataset(payload) ? validateBasinDivideDataset(payload) : validateRefinedBasinDataset(payload);
  const loaded = await fetchRefinedBasinJson("/divides.json", "Basin load failed", { fetchImpl, validate });
  assert.equal(loaded.divide_count, 2);
  // Without it, a divide package is not a set of named basins.
  await assert.rejects(fetchRefinedBasinJson("/divides.json", "Basin load failed", { fetchImpl }), /Invalid refined basin/);
});
