/**
 * Worked example: how much of the bed that lies below sea level today would emerge once
 * the ice is gone and isostatic rebound has run to completion?
 *
 * It runs the explorer's own decoder (static/tools/js/data-contract.js) and rebound module
 * (static/tools/js/gia-rebound.js) under Node on packages that ship with the repository, so
 * it needs no download and reproduces the figures the explorer's metadata panel shows for
 * the isostatic-rebound layer. See docs/example.md.
 *
 * The default Earth response is the published one of Paxman, Austermann & Hollyday (2022),
 * loaded from the response package that matches the terrain; `--model flexural|local`
 * solves the idealised responses instead, which alone accept a sea-level datum.
 *
 *   node examples/isostatic-rebound.mjs [--region antarctica|greenland] [--dataset <package>]
 *                                       [--model paxman2022|flexural|local] [--sea-level <metres>]
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

import { decodeFieldToFloat32, parseField } from "../static/tools/js/data-contract.js";
import {
  ANTARCTIC_STANDARD_PARALLEL_DEGREES,
  GREENLAND_STANDARD_PARALLEL_DEGREES,
  REBOUND_MODEL_PUBLISHED,
  REBOUND_MODELS,
  solveIsostaticRebound,
  summarisePublishedResponse,
} from "../static/tools/js/gia-rebound.js";

const DATA_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..", "static", "tools", "data");

export const REGIONS = Object.freeze({
  antarctica: { dataset: "bedmachine_antarctica_v4_480", standardParallelDegrees: ANTARCTIC_STANDARD_PARALLEL_DEGREES },
  greenland: { dataset: "bedmachine_greenland_v6_3km", standardParallelDegrees: GREENLAND_STANDARD_PARALLEL_DEGREES },
});

/**
 * Terrain package -> published response package, as the explorer pairs them. The QRF
 * packages share BedMachine Greenland v6's grids and borrow its response, which Paxman et
 * al. computed for the BedMachine ice load.
 */
export const RESPONSE_PACKAGES = Object.freeze({
  bedmachine_antarctica_v4_480: "antarctica_isostatic_response_480",
  bedmachine_antarctica_v4_741: "antarctica_isostatic_response_741",
  bedmap3_antarctica_10km: "bedmap3_antarctica_isostatic_response_10km",
  bedmap3_antarctica_4km: "bedmap3_antarctica_isostatic_response_4km",
  bedmachine_greenland_v6_3km: "greenland_isostatic_response_3km",
  bedmachine_greenland_v6_1km: "greenland_isostatic_response_1km",
  greenland_qrf_2025_3km: "greenland_isostatic_response_3km",
  greenland_qrf_2025_1km: "greenland_isostatic_response_1km",
});

function readPackage(name, dataDir) {
  const meta = JSON.parse(readFileSync(resolve(dataDir, `${name}.meta.json`), "utf8"));
  const bytes = readFileSync(resolve(dataDir, `${name}.bin`));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return { meta, buffer };
}

/** Decode a terrain package exactly as the explorer does before it solves for rebound. */
export function loadTerrainPackage(dataset, dataDir = DATA_DIR) {
  const { meta, buffer } = readPackage(dataset, dataDir);
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

const GRID_KEYS = ["nx", "ny", "x0_m", "y0_m", "dx_m", "dy_m"];

/** Decode the published response package that pairs with a terrain package. */
export function loadResponsePackage(dataset, terrainGrid, dataDir = DATA_DIR) {
  const name = RESPONSE_PACKAGES[dataset];
  if (!name) throw new Error(`No published isostatic response is packaged for "${dataset}".`);
  const { meta, buffer } = readPackage(name, dataDir);
  if (!GRID_KEYS.every((key) => Number(meta.grid[key]) === Number(terrainGrid[key]))) {
    throw new Error(`${name} is not on the ${dataset} grid.`);
  }
  return {
    name,
    meta,
    topographyChange: decodeFieldToFloat32(meta, buffer, "topography_change"),
    solidSurfaceChange: decodeFieldToFloat32(meta, buffer, "solid_surface_change"),
    standardDeviation: decodeFieldToFloat32(meta, buffer, "standard_deviation"),
  };
}

export function runIsostaticReboundExample({
  region = "antarctica",
  dataset = REGIONS[region]?.dataset,
  model = REBOUND_MODEL_PUBLISHED,
  seaLevelMeters = 0,
} = {}) {
  if (!REGIONS[region]) throw new Error(`Unknown region "${region}"; use ${Object.keys(REGIONS).join(" or ")}.`);
  if (!REBOUND_MODELS.includes(model)) throw new Error(`Unknown model "${model}"; use ${REBOUND_MODELS.join(", ")}.`);
  if (!Number.isFinite(seaLevelMeters)) throw new Error("The sea-level datum must be a number of metres.");
  if (model === REBOUND_MODEL_PUBLISHED && seaLevelMeters !== 0) {
    throw new Error("The published response fixes its own sea surface; a sea-level datum applies only to flexural or local.");
  }

  const { meta, ...terrain } = loadTerrainPackage(dataset);
  const standardParallelDegrees = REGIONS[region].standardParallelDegrees;
  if (model === REBOUND_MODEL_PUBLISHED) {
    const response = loadResponsePackage(dataset, meta.grid);
    const { stats } = summarisePublishedResponse({
      ...terrain,
      topographyChange: response.topographyChange,
      solidSurfaceChange: response.solidSurfaceChange,
      standardDeviation: response.standardDeviation,
      standardParallelDegrees,
      response: response.meta,
    });
    return { dataset, title: meta.title, grid: meta.grid, response: response.name, stats };
  }
  const { stats } = solveIsostaticRebound({ ...terrain, model, seaLevelMeters, standardParallelDegrees });
  return { dataset, title: meta.title, grid: meta.grid, response: null, stats };
}

const km2 = (value) => `${Math.round(value).toLocaleString("en-US")} km²`;

function publishedRows(stats, response) {
  const components = stats.components;
  return [
    ["Earth response", `published: Paxman et al. (2022), grids v3 (${response})`],
    ["Sea surface", `ice-free: ${stats.eustaticSeaLevelRiseMeters} m eustatic plus the residual post-LGM geoid`],
    ["Maximum solid-surface uplift (R)", `${stats.maxUpliftMeters.toFixed(1)} m`],
    ["Largest rise above the ice-free sea (T)", `${stats.maxTopographyChangeMeters.toFixed(1)} m`],
    ["Mean uplift under grounded ice (R)", `${stats.meanGroundedUpliftMeters.toFixed(1)} m`],
    ["Components under grounded ice (mean)",
      `ice unloading ${components.iceUnloading.mean.toFixed(1)} m, post-LGM ${components.postLgm.mean.toFixed(1)} m, water loading ${components.waterLoading.mean.toFixed(1)} m`],
    ["Model spread (1 sigma) under grounded ice",
      `${stats.meanGroundedSigmaMeters.toFixed(1)} m mean, ${stats.maxGroundedSigmaMeters.toFixed(1)} m max`],
  ];
}

function idealisedRows(stats) {
  const earth =
    stats.model === "flexural"
      ? `flexural (D = ${stats.flexuralRigidityNm.toExponential(0).replace("e+", "e")} N m, length scale ${stats.flexuralLengthScaleKm.toFixed(0)} km)`
      : "local Airy isostasy (upper bound on peak uplift)";
  return [
    ["Earth response", earth],
    ["Sea-level datum", `${stats.seaLevelMeters} m`],
    ["Maximum equilibrium uplift", `${stats.maxUpliftMeters.toFixed(1)} m`],
    ["Mean uplift under grounded ice", `${stats.meanGroundedUpliftMeters.toFixed(1)} m`],
  ];
}

function formatReport({ dataset, title, grid, response, stats }) {
  const published = stats.model === REBOUND_MODEL_PUBLISHED;
  const datum = published ? "the ice-free sea surface" : "the datum";
  const rows = [
    ["Package", `${dataset} (${title}, ${grid.nx} × ${grid.ny} cells at ${Math.abs(grid.dx_m) / 1000} km)`],
    ...(published ? publishedRows(stats, response) : idealisedRows(stats)),
    ["Bed above sea level today", km2(stats.landAreaNowKm2)],
    [`Bed above ${datum} after rebound`, km2(stats.landAreaAfterKm2)],
    ["Newly emergent land", km2(stats.emergentAreaKm2)],
    [`Still below ${datum} under the present ice`, km2(stats.marineUnderIceAfterAreaKm2)],
    ...(published ? [] : [["Closed basins below the datum", km2(stats.closedBasinAreaKm2)]]),
    ["Sea-level equivalent", `${stats.sleMeters.toFixed(2)} m`],
    ...(published
      ? []
      : [["Solve", `${stats.iterations} Picard iterations, residual ${stats.residualMeters.toFixed(3)} m, ${stats.solveCellKm.toFixed(1)} km solve grid`]]),
  ];
  const width = Math.max(...rows.map(([label]) => label.length));
  return rows.map(([label, value]) => `${label.padEnd(width)}  ${value}`).join("\n");
}

function main() {
  const { values } = parseArgs({
    options: {
      region: { type: "string", default: "antarctica" },
      dataset: { type: "string" },
      model: { type: "string", default: REBOUND_MODEL_PUBLISHED },
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
