/**
 * Worked example: how much of the bed that lies below sea level today would emerge once
 * the ice is gone and isostatic rebound has run to completion?
 *
 * It runs the explorer's own decoder (static/tools/js/data-contract.js) and solver
 * (static/tools/js/gia-rebound.js) under Node on a terrain package that ships with the
 * repository, so it needs no download and reproduces the figures the explorer's metadata
 * panel shows for the isostatic-rebound layer. See docs/example.md.
 *
 *   node examples/isostatic-rebound.mjs [--region antarctica|greenland] [--dataset <package>]
 *                                       [--model flexural|local] [--sea-level <metres>]
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

import { decodeFieldToFloat32, parseField } from "../static/tools/js/data-contract.js";
import {
  ANTARCTIC_STANDARD_PARALLEL_DEGREES,
  GREENLAND_STANDARD_PARALLEL_DEGREES,
  REBOUND_MODELS,
  solveIsostaticRebound,
} from "../static/tools/js/gia-rebound.js";

const DATA_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..", "static", "tools", "data");

export const REGIONS = Object.freeze({
  antarctica: { dataset: "bedmachine_antarctica_v4_480", standardParallelDegrees: ANTARCTIC_STANDARD_PARALLEL_DEGREES },
  greenland: { dataset: "bedmachine_greenland_v6_3km", standardParallelDegrees: GREENLAND_STANDARD_PARALLEL_DEGREES },
});

/** Decode a terrain package exactly as the explorer does before it solves for rebound. */
export function loadTerrainPackage(dataset, dataDir = DATA_DIR) {
  const meta = JSON.parse(readFileSync(resolve(dataDir, `${dataset}.meta.json`), "utf8"));
  const bytes = readFileSync(resolve(dataDir, `${dataset}.bin`));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const { nx, ny } = meta.grid;
  return {
    meta,
    nx,
    ny,
    cellCount: nx * ny,
    grid: meta.grid,
    bedHeights: decodeFieldToFloat32(meta, buffer, "bed"),
    surfaceHeights: decodeFieldToFloat32(meta, buffer, "surface"),
    thickness: decodeFieldToFloat32(meta, buffer, "thickness"),
    mask: parseField(meta, buffer, "mask"),
  };
}

export function runIsostaticReboundExample({
  region = "antarctica",
  dataset = REGIONS[region]?.dataset,
  model = "flexural",
  seaLevelMeters = 0,
} = {}) {
  if (!REGIONS[region]) throw new Error(`Unknown region "${region}"; use ${Object.keys(REGIONS).join(" or ")}.`);
  if (!REBOUND_MODELS.includes(model)) throw new Error(`Unknown model "${model}"; use ${REBOUND_MODELS.join(" or ")}.`);
  if (!Number.isFinite(seaLevelMeters)) throw new Error("The sea-level datum must be a number of metres.");

  const { meta, ...terrain } = loadTerrainPackage(dataset);
  const { stats } = solveIsostaticRebound({
    ...terrain,
    model,
    seaLevelMeters,
    standardParallelDegrees: REGIONS[region].standardParallelDegrees,
  });
  return { dataset, title: meta.title, grid: meta.grid, stats };
}

function formatReport({ dataset, title, grid, stats }) {
  const km2 = (value) => `${Math.round(value).toLocaleString("en-US")} km²`;
  const earth =
    stats.model === "flexural"
      ? `flexural (D = ${stats.flexuralRigidityNm.toExponential(0).replace("e+", "e")} N m, length scale ${stats.flexuralLengthScaleKm.toFixed(0)} km)`
      : "local Airy isostasy (upper bound on peak uplift)";
  const rows = [
    ["Package", `${dataset} (${title}, ${grid.nx} × ${grid.ny} cells at ${Math.abs(grid.dx_m) / 1000} km)`],
    ["Earth response", earth],
    ["Sea-level datum", `${stats.seaLevelMeters} m`],
    ["Maximum equilibrium uplift", `${stats.maxUpliftMeters.toFixed(1)} m`],
    ["Mean uplift under grounded ice", `${stats.meanGroundedUpliftMeters.toFixed(1)} m`],
    ["Bed above sea level today", km2(stats.landAreaNowKm2)],
    ["Bed above the datum after rebound", km2(stats.landAreaAfterKm2)],
    ["Newly emergent land", km2(stats.emergentAreaKm2)],
    ["Still below the datum under the present ice", km2(stats.marineUnderIceAfterAreaKm2)],
    ["Closed basins below the datum", km2(stats.closedBasinAreaKm2)],
    ["Sea-level equivalent", `${stats.sleMeters.toFixed(2)} m`],
    ["Solve", `${stats.iterations} Picard iterations, residual ${stats.residualMeters.toFixed(3)} m, ${stats.solveCellKm.toFixed(1)} km solve grid`],
  ];
  const width = Math.max(...rows.map(([label]) => label.length));
  return rows.map(([label, value]) => `${label.padEnd(width)}  ${value}`).join("\n");
}

function main() {
  const { values } = parseArgs({
    options: {
      region: { type: "string", default: "antarctica" },
      dataset: { type: "string" },
      model: { type: "string", default: "flexural" },
      "sea-level": { type: "string", default: "0" },
    },
  });
  const report = runIsostaticReboundExample({
    region: values.region,
    dataset: values.dataset ?? REGIONS[values.region]?.dataset,
    model: values.model,
    seaLevelMeters: Number(values["sea-level"]),
  });
  console.log(formatReport(report));
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    main();
  } catch (error) {
    console.error(`isostatic-rebound example: ${error.message}`);
    process.exitCode = 1;
  }
}
