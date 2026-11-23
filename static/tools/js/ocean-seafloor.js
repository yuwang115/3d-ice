/**
 * Observed seafloor water for the 3D ICE explorer: the Conservative Temperature and Absolute
 * Salinity of the water over the seafloor around Antarctica, from the OCEAN ICE Southern Ocean
 * Climatology (Zhou, Dutrieux & Giulivi, 2024), which maps ship CTD, Argo and instrumented-seal
 * profiles along the bathymetry.
 *
 * One package (docs/data-contract.md, "Observed seafloor water") holds, on the 10 km BedMachine
 * grid, the mean of each climatology column's lowest 150 dbar for every profile (1972-2023) and
 * for the profiles since 2000, the climatology's own seafloor pressure, and the share of those
 * levels that had a profile inside the mapping ellipse. Every Antarctic terrain samples it
 * bilinearly at its own open-ocean cells; ice-shelf cavities, where no profile was taken, are
 * left out.
 *
 * The module has no DOM or Three.js dependency and runs unchanged under Node's test runner.
 */

import { decodeFieldToFloat32, parseField } from "./data-contract.js";

export const SEAFLOOR_KIND = "observed_ocean_seafloor";
export const SEAFLOOR_PERIODS = Object.freeze(["full", "post2000"]);
export const SEAFLOOR_VARIABLES = Object.freeze(["temperature", "salinity"]);
/** Cells whose seafloor levels had a profile inside the mapping ellipse less often are faded. */
export const SEAFLOOR_FADE_BELOW = 0.5;
const SUPPORT_FILL = 255;

/**
 * Colour ramps as sRGB 0-1 stops on [0, 1]; the shader, the legend and the readout read the
 * same values. Temperature diverges at 0 degC, white between the cold shelf waters near the
 * surface freezing point (blue) and Circumpolar Deep Water (red); salinity rises from fresh
 * (dark blue) to the dense, salty shelf waters of the Ross and Weddell seas (yellow).
 */
export const SEAFLOOR_STYLE = Object.freeze({
  temperature: Object.freeze({
    range: Object.freeze([-2, 2]),
    ticks: Object.freeze([-2, -1, 0, 1, 2]),
    unit: "°C",
    digits: 2,
    stops: Object.freeze([
      Object.freeze([0, Object.freeze([0.106, 0.2, 0.471])]),
      Object.freeze([0.25, Object.freeze([0.255, 0.533, 0.796])]),
      Object.freeze([0.45, Object.freeze([0.78, 0.875, 0.937])]),
      Object.freeze([0.5, Object.freeze([0.965, 0.965, 0.945])]),
      Object.freeze([0.55, Object.freeze([0.992, 0.859, 0.733])]),
      Object.freeze([0.75, Object.freeze([0.902, 0.396, 0.212])]),
      Object.freeze([1, Object.freeze([0.565, 0.071, 0.098])]),
    ]),
  }),
  salinity: Object.freeze({
    range: Object.freeze([34.2, 35]),
    ticks: Object.freeze([34.2, 34.4, 34.6, 34.8, 35]),
    unit: "g/kg",
    digits: 3,
    stops: Object.freeze([
      Object.freeze([0, Object.freeze([0.165, 0.094, 0.424])]),
      Object.freeze([0.25, Object.freeze([0.149, 0.302, 0.569])]),
      Object.freeze([0.5, Object.freeze([0.071, 0.522, 0.553])]),
      Object.freeze([0.75, Object.freeze([0.341, 0.718, 0.392])]),
      Object.freeze([1, Object.freeze([0.937, 0.91, 0.471])]),
    ]),
  }),
});

function gridOf(meta) {
  const grid = meta?.grid || {};
  const out = {
    nx: Number(grid.nx),
    ny: Number(grid.ny),
    x0: Number(grid.x0_m),
    y0: Number(grid.y0_m),
    dx: Number(grid.dx_m),
    dy: Number(grid.dy_m),
  };
  if (!Number.isInteger(out.nx) || !Number.isInteger(out.ny) || out.nx < 2 || out.ny < 2) {
    throw new Error("Seafloor package has no valid grid.");
  }
  if (![out.x0, out.y0, out.dx, out.dy].every(Number.isFinite) || out.dx === 0 || out.dy === 0) {
    throw new Error("Seafloor package grid has no valid origin and step.");
  }
  return out;
}

function decodeDomainField(meta, buffer, name, count) {
  const values = decodeFieldToFloat32(meta, buffer, name);
  if (values.length !== count) throw new Error(`Seafloor field ${name} does not hold one value per domain cell.`);
  return values;
}

/**
 * Decode the package. Throws if the payload does not follow the declared layout, so a
 * truncated download fails loudly instead of drawing garbage.
 */
export function decodeSeafloorRecord(meta, buffer) {
  if (meta?.geometry_type !== "sparse_grid" || meta?.field_kind !== SEAFLOOR_KIND) {
    throw new Error("Package is not an observed seafloor-water record.");
  }
  const grid = gridOf(meta);
  const mask = parseField(meta, buffer, "domain_mask");
  if (mask.length !== grid.nx * grid.ny) throw new Error("Seafloor domain mask does not cover the grid.");
  const cellIndex = new Int32Array(mask.length).fill(-1);
  let count = 0;
  for (let cell = 0; cell < mask.length; cell += 1) {
    if (mask[cell]) {
      cellIndex[cell] = count;
      count += 1;
    }
  }
  if (count !== Number(meta.domain?.cell_count)) throw new Error("Seafloor domain size does not match its metadata.");

  const periods = {};
  for (const period of Array.isArray(meta.periods) ? meta.periods : []) {
    if (!SEAFLOOR_PERIODS.includes(period?.key)) continue;
    periods[period.key] = {
      key: period.key,
      label: String(period.label || period.key),
      temperature: decodeDomainField(meta, buffer, period.fields.ct, count),
      salinity: decodeDomainField(meta, buffer, period.fields.sa, count),
    };
  }
  for (const key of SEAFLOOR_PERIODS) {
    if (!periods[key]) throw new Error(`Seafloor package has no ${key} period.`);
  }
  const supportCodes = parseField(meta, buffer, meta.support?.field || "support");
  if (supportCodes.length !== count) throw new Error("Seafloor support does not hold one value per domain cell.");
  const support = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    support[i] = supportCodes[i] === SUPPORT_FILL ? Number.NaN : supportCodes[i] / 100;
  }
  return {
    grid,
    cellIndex,
    domainCount: count,
    periods,
    support,
    bottomPressure: decodeDomainField(meta, buffer, "bottom_pressure", count),
    fadeBelow: Number(meta.support?.fade_below_percent ?? SEAFLOOR_FADE_BELOW * 100) / 100,
  };
}

function fieldOf(record, period, variable) {
  const values = record.periods[period]?.[variable];
  if (!values) throw new Error(`Unknown seafloor view ${period}/${variable}.`);
  return values;
}

/**
 * Bilinear sample at projected (x, y), weights renormalised over the corners in the domain with
 * a value. Returns null where no corner has one.
 */
export function sampleSeafloor(record, x, y, { period = "full", variable = "temperature" } = {}) {
  const values = fieldOf(record, period, variable);
  const { grid, cellIndex } = record;
  const u = (x - grid.x0) / grid.dx;
  const v = (y - grid.y0) / grid.dy;
  const c0 = Math.floor(u);
  const r0 = Math.floor(v);
  const fu = u - c0;
  const fv = v - r0;
  let weightSum = 0;
  let value = 0;
  let support = 0;
  let pressure = 0;
  for (let k = 0; k < 4; k += 1) {
    const row = r0 + (k >> 1);
    const col = c0 + (k & 1);
    if (row < 0 || col < 0 || row >= grid.ny || col >= grid.nx) continue;
    const index = cellIndex[row * grid.nx + col];
    if (index < 0 || !Number.isFinite(values[index])) continue;
    const weight = (k & 1 ? fu : 1 - fu) * (k >> 1 ? fv : 1 - fv);
    if (weight <= 0) continue;
    weightSum += weight;
    value += weight * values[index];
    support += weight * (Number.isFinite(record.support[index]) ? record.support[index] : 0);
    pressure += weight * record.bottomPressure[index];
  }
  if (!(weightSum > 1e-9)) return null;
  return { value: value / weightSum, support: support / weightSum, bottomPressure: pressure / weightSum };
}

/**
 * The seafloor field at every wanted node of a terrain grid ({nx, ny, x0_m, y0_m, dx_m, dy_m}):
 * value and support per node (NaN elsewhere) and a mask of the nodes that have one.
 */
export function sampleSeafloorOnGrid(record, terrainGrid, wanted, view = {}) {
  const nx = Number(terrainGrid.nx);
  const ny = Number(terrainGrid.ny);
  if (wanted.length !== nx * ny) throw new Error("Seafloor sampling mask does not cover the terrain grid.");
  const values = new Float32Array(nx * ny).fill(Number.NaN);
  const support = new Float32Array(nx * ny).fill(Number.NaN);
  const valid = new Uint8Array(nx * ny);
  for (let row = 0; row < ny; row += 1) {
    const y = terrainGrid.y0_m + row * terrainGrid.dy_m;
    for (let col = 0; col < nx; col += 1) {
      const cell = row * nx + col;
      if (!wanted[cell]) continue;
      const sample = sampleSeafloor(record, terrainGrid.x0_m + col * terrainGrid.dx_m, y, view);
      if (!sample) continue;
      values[cell] = sample.value;
      support[cell] = sample.support;
      valid[cell] = 1;
    }
  }
  return { values, support, valid };
}

/** Position of a value on its variable's ramp, clamped to [0, 1]. */
export function seafloorRampPosition(variable, value) {
  const style = SEAFLOOR_STYLE[variable];
  if (!style) throw new Error(`Unknown seafloor variable ${variable}.`);
  const [low, high] = style.range;
  return Math.min(1, Math.max(0, (value - low) / (high - low)));
}

/** sRGB 0-1 colour of a ramp position. */
export function seafloorRampColor(variable, t) {
  const stops = SEAFLOOR_STYLE[variable]?.stops;
  if (!stops) throw new Error(`Unknown seafloor variable ${variable}.`);
  const position = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  for (let i = 1; i < stops.length; i += 1) {
    const [end, endColor] = stops[i];
    if (position <= end || i === stops.length - 1) {
      const [start, startColor] = stops[i - 1];
      const f = end > start ? (position - start) / (end - start) : 0;
      return startColor.map((channel, c) => channel + (endColor[c] - channel) * f);
    }
  }
  return [...stops[stops.length - 1][1]];
}

/** sRGB 0-1 colour of a value. */
export function seafloorColor(variable, value) {
  return seafloorRampColor(variable, seafloorRampPosition(variable, value));
}

/** CSS linear-gradient of a variable's ramp, left to right. */
export function seafloorLegendGradient(variable) {
  const stops = SEAFLOOR_STYLE[variable]?.stops;
  if (!stops) throw new Error(`Unknown seafloor variable ${variable}.`);
  const parts = stops.map(([position, color]) => {
    const rgb = color.map((channel) => Math.round(channel * 255)).join(", ");
    return `rgb(${rgb}) ${Math.round(position * 1000) / 10}%`;
  });
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}

/** Legend ticks of a variable, each placed at its value on the ramp (percent from the left). */
export function seafloorLegendTicks(variable) {
  const style = SEAFLOOR_STYLE[variable];
  if (!style) throw new Error(`Unknown seafloor variable ${variable}.`);
  return style.ticks.map((value) => ({ value, percent: seafloorRampPosition(variable, value) * 100 }));
}

/** GLSL source of a variable's ramp, `vec3 <name>(float t)`, from the same stops. */
export function seafloorRampGlsl(variable, name) {
  const stops = SEAFLOOR_STYLE[variable]?.stops;
  if (!stops) throw new Error(`Unknown seafloor variable ${variable}.`);
  const vec = (color) => `vec3(${color.map((channel) => channel.toFixed(4)).join(", ")})`;
  const lines = [`vec3 ${name}(float t) {`, "  float p = clamp(t, 0.0, 1.0);"];
  for (let i = 1; i < stops.length; i += 1) {
    const [start, startColor] = stops[i - 1];
    const [end, endColor] = stops[i];
    const span = Math.max(end - start, 1e-6).toFixed(4);
    const mixLine = `mix(${vec(startColor)}, ${vec(endColor)}, (p - ${start.toFixed(4)}) / ${span})`;
    lines.push(i === stops.length - 1 ? `  return ${mixLine};` : `  if (p <= ${end.toFixed(4)}) return ${mixLine};`);
  }
  lines.push("}");
  return lines.join("\n");
}
