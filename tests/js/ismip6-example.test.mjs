/**
 * docs/example.md and the ESS manuscript quote these figures as the expected output of the
 * ISMIP6 worked example, so they are pinned here: a change to the projection module, the
 * decoder or a bundled package that moves them must update the documentation too.
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  BASINS_FILE,
  ISLANDS_ENTRY,
  SCENARIOS,
  assignCellsToBasins,
  runIsmip6ProjectionExample,
} from "../../examples/ismip6-projection.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const DATA_DIR = join(ROOT, "static", "tools", "data");
const EXAMPLE = join(ROOT, "examples", "ismip6-projection.mjs");

const near = (actual, expected, tolerance, label) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${label}: ${actual} is not within ${tolerance} of ${expected}`);

const memo = new Map();
const run = (scenario) => {
  if (!memo.has(scenario)) memo.set(scenario, runIsmip6ProjectionExample({ scenario }));
  return memo.get(scenario);
};
const at = (result, year) => result.series.find((row) => row.year === year);
const basinEnd = (result, name) => {
  const row = result.basins.find((basin) => basin.name === name);
  assert.ok(row, `no basin ${name}`);
  return row.sleMeters.at(-1);
};

test("the high-emissions example reproduces the documented ensemble-mean figures", () => {
  const result = run("ae05");

  assert.equal(result.package, "ismip6_2300_mean8_ae05_480");
  assert.equal(result.models.length, 8);
  assert.deepEqual(result.series.map((row) => row.year), [2015, 2100, 2150, 2200, 2250, 2300]);

  const start = at(result, 2015);
  const end = at(result, 2300);
  near(start.sleMeters, 0, 1e-12, "no change at the first keyframe");
  near(start.groundedAreaKm2 / 1e6, 12.281, 0.0005, "grounded area in 2015 (million km²)");
  near(end.groundedAreaKm2 / 1e6, 11.192, 0.0005, "grounded area in 2300 (million km²)");
  near(start.floatingAreaKm2 / 1e6, 1.561, 0.0005, "floating area in 2015 (million km²)");
  near(end.floatingAreaKm2 / 1e6, 2.166, 0.0005, "floating area in 2300 (million km²)");
  near(at(result, 2100).sleMeters, 0.018, 0.0005, "implied sea-level rise by 2100 (m)");
  near(at(result, 2200).sleMeters, 0.689, 0.0005, "implied sea-level rise by 2200 (m)");
  near(end.sleMeters, 1.974, 0.0005, "implied sea-level rise by 2300 (m)");
  near(end.publishedMeanSleMeters, 1.463, 0.0005, "mean of the eight published values in 2300 (m)");
  near(end.publishedMinSleMeters, -0.084, 0.0005, "lowest published value in 2300 (m)");
  near(end.publishedMaxSleMeters, 3.99, 0.0005, "highest published value in 2300 (m)");
});

test("the basin figures quoted in the documentation and the manuscript are pinned", () => {
  const result = run("ae05");
  assert.equal(result.basins[0].name, "Thwaites", "basins are ranked by their contribution at the last year");
  const quoted = {
    Thwaites: 0.333, "Pine Island": 0.186, MacAyeal: 0.148, Whillans: 0.146, Kamb: 0.13,
    Bindschadler: 0.12, Cook: 0.099, Institute: 0.098, Evans: 0.073, Moller: 0.048,
    Bailey: 0.041, Rutford: 0.036, Ninnis: 0.026, Totten: 0.021,
  };
  for (const [name, value] of Object.entries(quoted)) near(basinEnd(result, name), value, 0.0005, `${name} by 2300 (m)`);
  const whillans = result.basins.find((basin) => basin.name === "Whillans");
  near(whillans.groundedAreaChangeKm2.at(-1) / 1e3, -91.7, 0.05, "Whillans grounded-area change (10³ km²)");

  near(basinEnd(run("ae10"), "Thwaites"), 0.119, 0.0005, "Thwaites under SSP1-2.6 (m)");
  near(basinEnd(run("ae14"), "Pine Island"), 0.346, 0.0005, "Pine Island with shelf collapse (m)");
});

test("ice rises and islands are reported beside the basins, not ranked with them", () => {
  const result = run("ae05");
  assert.ok(!result.basins.some((basin) => basin.name === ISLANDS_ENTRY));
  assert.equal(result.islands.name, ISLANDS_ENTRY);
  near(result.islands.sleMeters.at(-1), 0.073, 0.0005, "ice rises and islands by 2300 (m)");
  near(result.islands.groundedAreaChangeKm2.at(-1) / 1e3, -99.5, 0.05, "their grounded-area change (10³ km²)");
  near(result.outside.initialGroundedAreaKm2 / 1e3, 25.0, 0.5, "grounded ice outside every outline in 2015 (10³ km²)");
});

test("the basins, the islands and the cells outside every outline add up to the ice sheet", () => {
  for (const scenario of Object.keys(SCENARIOS)) {
    const result = run(scenario);
    const regions = [...result.basins, result.islands, result.outside];
    const total = (pick) => regions.reduce((sum, row) => sum + pick(row), 0);
    near(total((row) => row.initialGroundedAreaKm2), at(result, 2015).groundedAreaKm2, 1e-6, `${scenario} 2015 area`);
    result.series.forEach((row, i) => {
      near(total((r) => r.sleMeters[i]), row.sleMeters, 1e-9, `${scenario} ${row.year} sea level`);
      near(total((r) => r.groundedAreaChangeKm2[i]), row.groundedAreaKm2 - at(result, 2015).groundedAreaKm2, 1e-6,
        `${scenario} ${row.year} grounded area`);
    });
  }
});

test("the low-emissions and shelf-collapse examples bracket the high-emissions one", () => {
  const low = run("ae10");
  const collapse = run("ae14");

  near(at(low, 2100).sleMeters, 0.011, 0.0005, "SSP1-2.6 by 2100 (m)");
  near(at(low, 2300).sleMeters, 0.151, 0.0005, "SSP1-2.6 by 2300 (m)");
  near(at(low, 2300).publishedMeanSleMeters, 0.052, 0.0005, "SSP1-2.6 published mean (m)");
  near(at(collapse, 2100).sleMeters, 0.059, 0.0005, "shelf collapse by 2100 (m)");
  near(at(collapse, 2300).sleMeters, 3.19, 0.0005, "shelf collapse by 2300 (m)");
  near(at(collapse, 2300).publishedMeanSleMeters, 2.429, 0.0005, "shelf collapse published mean (m)");
  near(at(collapse, 2300).groundedAreaKm2 / 1e6, 10.892, 0.0005, "shelf collapse grounded area in 2300 (million km²)");
  near(at(collapse, 2300).floatingAreaKm2 / 1e6, 1.951, 0.0005, "shelf collapse floating area in 2300 (million km²)");
  near(at(low, 2300).groundedAreaKm2 / 1e6, 12.158, 0.0005, "SSP1-2.6 grounded area in 2300 (million km²)");
});

test("nominal grid cells give the figures the package's own validation block records", () => {
  const result = runIsmip6ProjectionExample({ scenario: "ae05", years: [2300], cellArea: "grid" });
  const recorded = result.meta.validation.sea_level_of_packed_geometry_m.end_packed;
  near(at(result, 2300).sleMeters, recorded, 0.0005, "grid-area SLE against the packer's validation");
});

test("a package without a published series still reports its own geometry", () => {
  const dir = mkdtempSync(join(tmpdir(), "ismip6-example-"));
  try {
    const name = SCENARIOS.ae10;
    const meta = JSON.parse(readFileSync(join(DATA_DIR, `${name}.meta.json`), "utf8"));
    delete meta.series;
    writeFileSync(join(dir, `${name}.meta.json`), JSON.stringify(meta));
    symlinkSync(join(DATA_DIR, `${name}.bin`), join(dir, `${name}.bin`));
    symlinkSync(join(DATA_DIR, BASINS_FILE), join(dir, BASINS_FILE));
    const result = runIsmip6ProjectionExample({ scenario: "ae10", years: [2300], dataDir: dir });
    near(at(result, 2300).sleMeters, 0.151, 0.0005, "implied rise without the series (m)");
    assert.ok(Number.isNaN(at(result, 2300).publishedMeanSleMeters));
    assert.ok(Number.isNaN(at(result, 2300).publishedMaxSleMeters));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("cells are assigned to the union of a basin's rings, and to the first of overlapping outlines", () => {
  const projection = {
    nx: 4,
    grid: { x0_m: 0, y0_m: 0, dx_m: 10, dy_m: -10 },
    cellIndex: Int32Array.from([0, 1, 2, 3, 5]),
    domainCount: 5,
  };
  const square = (x0, x1, y0, y1) => [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]];
  const basins = [
    // Two rings: one around (0, 0), one around (20, 0).
    { name: "Split", segments_xy_m: [square(-5, 5, 5, -5), square(15, 25, 5, -5)] },
    // Overlaps the second ring of Split at (20, 0), which Split keeps, and holds (30, 0).
    { name: "Later", segments_xy_m: [square(15, 35, 5, -5)] },
  ];
  // Cells: 0 (0, 0), 1 (10, 0), 2 (20, 0), 3 (30, 0), 5 (10, -10).
  assert.deepEqual([...assignCellsToBasins(projection, basins)], [0, -1, 0, 1, -1]);
});

test("the command line writes the documented JSON and rejects a malformed year list", () => {
  const ok = spawnSync(process.execPath, [EXAMPLE, "--scenario", "ae10", "--years", "2100,2300", "--json"], {
    encoding: "utf8",
  });
  assert.equal(ok.status, 0, ok.stderr);
  const parsed = JSON.parse(ok.stdout);
  assert.deepEqual(parsed.series.map((row) => row.year), [2100, 2300]);
  assert.ok(parsed.basins.length > 100 && parsed.islands && parsed.outside && parsed.publishedSeriesSource);
  assert.equal(parsed.meta, undefined, "the package metadata is not repeated in the output");

  const bad = spawnSync(process.execPath, [EXAMPLE, "--years", "2100,"], { encoding: "utf8" });
  assert.equal(bad.status, 1);
  assert.match(bad.stderr, /between 2015 and 2300/);
});

test("the worked example rejects an unknown scenario, year or cell-area convention", () => {
  assert.deepEqual(Object.keys(SCENARIOS), ["ae10", "ae05", "ae14"]);
  assert.throws(() => runIsmip6ProjectionExample({ scenario: "ae99" }), /Unknown scenario/);
  assert.throws(() => runIsmip6ProjectionExample({ years: [2400] }), /between 2015 and 2300/);
  assert.throws(() => runIsmip6ProjectionExample({ years: [] }), /at least one year/);
  assert.throws(() => runIsmip6ProjectionExample({ cellArea: "sphere" }), /Unknown cell-area/);
});
