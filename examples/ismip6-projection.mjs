/**
 * Worked example: how much ice does the ISMIP6 Antarctic ensemble lose by 2300, and where?
 *
 * It runs the explorer's own decoder (static/tools/js/ice-projection.js) under Node on the
 * three projection packages that ship with the repository. Each package carries the
 * equal-weight mean, over eight ISMIP6-2300 models, of the change in ice thickness since
 * 2015, applied to BedMachine Antarctica v4. The example rebuilds the geometry by flotation,
 * as the explorer does, and reports the grounded area and the sea-level rise implied by the
 * loss of ice above flotation: for the whole ice sheet, for each drainage basin of MEaSUREs
 * Antarctic Boundaries v2, for that product's ice rises and islands, and for the cells that
 * fall inside no outline, so that the parts add up to the whole. It also carries the mean and
 * range of the same eight models' published values, so the effect of mapping the
 * ensemble-mean change onto one geometry can be read off directly. See docs/example.md.
 *
 *   node examples/ismip6-projection.mjs [--scenario ae10|ae05|ae14] [--years 2100,2200,2300]
 *                                       [--top 10] [--cell-area ground|grid] [--json]
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

import { ANTARCTIC_STANDARD_PARALLEL_DEGREES } from "../static/tools/js/gia-rebound.js";
import {
  blendThicknessInto,
  decodeIceProjection,
  domainCellAreasKm2,
  flotationGeometryInto,
  sampleSeries,
} from "../static/tools/js/ice-projection.js";
import { validateRefinedBasinDataset } from "../static/tools/js/polar-refined-basins.js";

const DATA_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..", "static", "tools", "data");
export const BASINS_FILE = "imbie_refined_basins_v2.json";

export const SCENARIOS = Object.freeze({
  ae10: "ismip6_2300_mean8_ae10_480",
  ae05: "ismip6_2300_mean8_ae05_480",
  ae14: "ismip6_2300_mean8_ae14_480",
});

export const DEFAULT_YEARS = Object.freeze([2015, 2100, 2150, 2200, 2250, 2300]);
const CELL_AREAS = Object.freeze(["ground", "grid"]);

/**
 * MEaSUREs Antarctic Boundaries v2 records its ice rises and islands as one entry of 656
 * rings. It is not a drainage basin, so it is reported beside the basins, not ranked with them.
 */
export const ISLANDS_ENTRY = "Islands";

/** ISMIP6's conversion, used when a package does not record its own: 362.5 Gt per mm. */
const DEFAULT_GT_PER_MM_SLE = 362.5;
const DEFAULT_RHO_ICE = 917;

function readPackage(name, dataDir) {
  const meta = JSON.parse(readFileSync(resolve(dataDir, `${name}.meta.json`), "utf8"));
  const bytes = readFileSync(resolve(dataDir, `${name}.bin`));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return { meta, buffer };
}

function pointInRing(x, y, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function ringBounds(basin) {
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const ring of basin.segments_xy_m) {
    for (const [x, y] of ring) {
      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
  }
  return { xMin, xMax, yMin, yMax };
}

/**
 * The basin (index into `basins`) whose outline holds each domain cell's centre, or -1.
 * Outlines are closed rings in EPSG:3031 metres; a basin of several rings is their union,
 * and where outlines overlap the first basin that holds a cell wins.
 */
export function assignCellsToBasins(projection, basins) {
  const { nx, cellIndex, domainCount, grid } = projection;
  const bounds = basins.map(ringBounds);
  const basinOfCell = new Int16Array(domainCount).fill(-1);
  for (let v = 0; v < domainCount; v += 1) {
    const col = cellIndex[v] % nx;
    const row = (cellIndex[v] - col) / nx;
    const x = grid.x0_m + col * grid.dx_m;
    const y = grid.y0_m + row * grid.dy_m;
    for (let k = 0; k < basins.length; k += 1) {
      const b = bounds[k];
      if (x < b.xMin || x > b.xMax || y < b.yMin || y > b.yMax) continue;
      if (basins[k].segments_xy_m.some((ring) => pointInRing(x, y, ring))) {
        basinOfCell[v] = k;
        break;
      }
    }
  }
  return basinOfCell;
}

/** Ice above flotation (km³) of every cell, and the grounded and floating areas (km²). */
function iceAboveFlotation(projection, thickness, areasKm2, scratch) {
  const { bed, densityRatio, domainCount } = projection;
  flotationGeometryInto(bed, thickness, densityRatio, scratch.surface, scratch.base, scratch.grounded);
  let groundedAreaKm2 = 0;
  let floatingAreaKm2 = 0;
  for (let i = 0; i < domainCount; i += 1) {
    const h = thickness[i];
    scratch.vafKm3[i] = 0;
    if (!(h > 0)) continue;
    if (!scratch.grounded[i]) {
      floatingAreaKm2 += areasKm2[i];
      continue;
    }
    groundedAreaKm2 += areasKm2[i];
    const submergedBed = Math.max(0, -bed[i]);
    const thicknessAboveFlotation = Math.max(0, h - submergedBed / densityRatio);
    scratch.vafKm3[i] = (thicknessAboveFlotation / 1000) * areasKm2[i];
  }
  return { groundedAreaKm2, floatingAreaKm2 };
}

function checkYears(years, first, last) {
  if (!Array.isArray(years) || years.length === 0) throw new Error("Give at least one year.");
  for (const year of years) {
    if (!Number.isFinite(year) || year < first || year > last) {
      throw new Error(`Years must lie between ${first} and ${last}; got ${year}.`);
    }
  }
  return [...new Set(years)].sort((a, b) => a - b);
}

function cellAreasKm2(projection, cellArea) {
  if (cellArea === "ground") return domainCellAreasKm2(projection, ANTARCTIC_STANDARD_PARALLEL_DEGREES);
  const nominal = (Math.abs(projection.grid.dx_m) * Math.abs(projection.grid.dy_m)) / 1e6;
  return new Float64Array(projection.domainCount).fill(nominal);
}

/**
 * Totals at each year, and the grounded area and ice above flotation of each region: the
 * basins in order, then one region for the cells outside every outline (`basinOfCell` -1).
 */
function summarise(projection, areasKm2, basinOfCell, basinCount, years) {
  const n = projection.domainCount;
  const outside = basinCount;
  const thickness = new Float32Array(n);
  const scratch = {
    surface: new Float32Array(n),
    base: new Float32Array(n),
    grounded: new Uint8Array(n),
    vafKm3: new Float64Array(n),
  };
  return years.map((year) => {
    blendThicknessInto(projection, year, thickness);
    const totals = iceAboveFlotation(projection, thickness, areasKm2, scratch);
    const regionVafKm3 = new Float64Array(basinCount + 1);
    const regionGroundedKm2 = new Float64Array(basinCount + 1);
    let vafKm3 = 0;
    for (let i = 0; i < n; i += 1) {
      const region = basinOfCell[i] < 0 ? outside : basinOfCell[i];
      vafKm3 += scratch.vafKm3[i];
      regionVafKm3[region] += scratch.vafKm3[i];
      if (scratch.grounded[i]) regionGroundedKm2[region] += areasKm2[i];
    }
    return { year, ...totals, vafKm3, regionVafKm3, regionGroundedKm2 };
  });
}

function physicalConstants(meta) {
  const rhoIce = Number(meta.physical_constants?.rho_ice_kg_m3);
  const gtPerMm = Number(meta.physical_constants?.gt_per_mm_sle);
  const vafToSleMeters = (lossKm3) =>
    (lossKm3 * ((rhoIce > 0 ? rhoIce : DEFAULT_RHO_ICE) / 1000)) / (gtPerMm > 0 ? gtPerMm : DEFAULT_GT_PER_MM_SLE) / 1000;
  return { vafToSleMeters };
}

function seriesRows(states, reference, published, vafToSleMeters) {
  const sample = (values, year) => (published && values ? sampleSeries(published.years, values, year) : Number.NaN);
  return states.map((state) => ({
    year: state.year,
    groundedAreaKm2: state.groundedAreaKm2,
    floatingAreaKm2: state.floatingAreaKm2,
    sleMeters: vafToSleMeters(reference.vafKm3 - state.vafKm3),
    publishedMeanSleMeters: sample(published?.seaLevel, state.year),
    publishedMinSleMeters: sample(published?.seaLevelMin, state.year),
    publishedMaxSleMeters: sample(published?.seaLevelMax, state.year),
  }));
}

function regionRow(name, region, states, reference, vafToSleMeters) {
  return {
    name,
    initialGroundedAreaKm2: reference.regionGroundedKm2[region],
    groundedAreaChangeKm2: states.map((s) => s.regionGroundedKm2[region] - reference.regionGroundedKm2[region]),
    sleMeters: states.map((s) => vafToSleMeters(reference.regionVafKm3[region] - s.regionVafKm3[region])),
  };
}

/** Drainage basins ranked by their contribution in the last year, the islands entry and the rest. */
function regionRows(basins, states, reference, vafToSleMeters) {
  const rows = basins.map((basin, k) => regionRow(basin.name, k, states, reference, vafToSleMeters));
  const islands = rows.find((row) => row.name === ISLANDS_ENTRY) ?? null;
  const ranked = rows.filter((row) => row !== islands).sort((a, b) => b.sleMeters.at(-1) - a.sleMeters.at(-1));
  const outside = regionRow("Outside every outline", basins.length, states, reference, vafToSleMeters);
  return { basins: ranked, islands, outside };
}

export function runIsmip6ProjectionExample({
  scenario = "ae05",
  years = DEFAULT_YEARS,
  cellArea = "ground",
  dataDir = DATA_DIR,
} = {}) {
  const name = SCENARIOS[scenario];
  if (!name) throw new Error(`Unknown scenario "${scenario}"; use ${Object.keys(SCENARIOS).join(", ")}.`);
  if (!CELL_AREAS.includes(cellArea)) throw new Error(`Unknown cell-area convention "${cellArea}"; use ground or grid.`);

  const { meta, buffer } = readPackage(name, dataDir);
  const projection = decodeIceProjection(meta, buffer);
  const requested = checkYears(years, projection.firstYear, projection.lastYear);
  const basinsPayload = JSON.parse(readFileSync(resolve(dataDir, BASINS_FILE), "utf8"));
  validateRefinedBasinDataset(basinsPayload);
  const basins = basinsPayload.basins;
  const basinOfCell = assignCellsToBasins(projection, basins);

  // The first keyframe is the reference; it is summarised once, with the requested years.
  const computed = [...new Set([projection.firstYear, ...requested])];
  const all = summarise(projection, cellAreasKm2(projection, cellArea), basinOfCell, basins.length, computed);
  const reference = all[0];
  const states = all.filter((state) => requested.includes(state.year));
  const { vafToSleMeters } = physicalConstants(meta);

  return {
    scenario,
    package: name,
    meta,
    title: meta.title,
    models: meta.models,
    cellArea,
    series: seriesRows(states, reference, projection.series, vafToSleMeters),
    ...regionRows(basins, states, reference, vafToSleMeters),
  };
}

const fixed = (value, digits) => (Number.isFinite(value) ? value.toFixed(digits) : "n/a");

function formatRegion(row) {
  return `${row.name.padEnd(26)}${row.sleMeters.map((v) => fixed(v, 3).padStart(11)).join("")}` +
    `   ${Math.round(row.groundedAreaChangeKm2.at(-1)).toLocaleString("en-US")}`;
}

function formatReport(result, top) {
  const lines = [
    result.title,
    `Models: ${result.models.join(", ")}`,
    `Cell areas: ${result.cellArea === "ground" ? "ground area (polar stereographic scale factor applied)" : "nominal grid cells"}`,
    "",
    "Year   Grounded (10^6 km²)  Floating (10^6 km²)  Implied SLE (m)  Published mean [min, max] (m)",
  ];
  for (const row of result.series) {
    lines.push(
      `${row.year}   ${fixed(row.groundedAreaKm2 / 1e6, 3).padStart(10)}           ${fixed(row.floatingAreaKm2 / 1e6, 3).padStart(10)}` +
        `           ${fixed(row.sleMeters, 3).padStart(7)}        ${fixed(row.publishedMeanSleMeters, 3)} ` +
        `[${fixed(row.publishedMinSleMeters, 3)}, ${fixed(row.publishedMaxSleMeters, 3)}]`
    );
  }
  const years = result.series.map((row) => row.year);
  lines.push("", `Drainage basins with the largest contributions by ${years.at(-1)} (MEaSUREs Antarctic Boundaries v2)`);
  lines.push(`${"Basin".padEnd(26)}${years.map((y) => `SLE ${y}`.padStart(11)).join("")}   Grounded-area change (km²)`);
  for (const basin of result.basins.slice(0, top)) lines.push(formatRegion(basin));
  lines.push(`... ${Math.max(0, result.basins.length - top)} more basins`);
  if (result.islands) lines.push(formatRegion({ ...result.islands, name: "Ice rises and islands" }));
  lines.push(formatRegion(result.outside));
  return lines.join("\n");
}

function toJson(result) {
  const { meta, ...rest } = result;
  return JSON.stringify({ ...rest, publishedSeriesSource: meta.series ?? null }, null, 2);
}

function main() {
  const { values } = parseArgs({
    options: {
      scenario: { type: "string", default: "ae05" },
      years: { type: "string" },
      top: { type: "string", default: "10" },
      "cell-area": { type: "string", default: "ground" },
      json: { type: "boolean", default: false },
    },
  });
  const years = values.years ? values.years.split(",").map(Number) : DEFAULT_YEARS;
  const result = runIsmip6ProjectionExample({ scenario: values.scenario, years, cellArea: values["cell-area"] });
  console.log(values.json ? toJson(result) : formatReport(result, Math.max(1, Number(values.top) || 10)));
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    main();
  } catch (error) {
    console.error(`ismip6-projection example: ${error.message}`);
    process.exitCode = 1;
  }
}
