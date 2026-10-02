/**
 * Ice-sheet projection time series for the 3D ICE explorer.
 *
 * A projection package (see docs/data-contract.md, "Ice-sheet projections") stores the
 * modelled ice thickness at keyframe years for every grid cell the ice ever covers, plus
 * the model's static bed there. Everything the viewer draws is rebuilt from those two
 * fields: thickness is interpolated linearly between keyframes, and the surface and base
 * follow from flotation,
 *
 *     base    = max(bed, -H * rho_ice / rho_seawater)
 *     surface = base + H,
 *
 * so grounded ice rests on the bed and floating ice sits in hydrostatic equilibrium about
 * sea level (0 m). The grounded/floating class therefore needs no storage of its own; the
 * packer records how often it matches the model's own mask.
 *
 * The domain cells are the vertices of the projection meshes, so every per-cell array here
 * is indexed by vertex. The module has no DOM or Three.js dependency and runs unchanged
 * under Node's test runner.
 */

import { getFieldDefinition, parseField, resolveInt16Quantization } from "./data-contract.js";

export const ICE_PROJECTION_GEOMETRY_TYPE = "sparse_grid_time_series";
export const DEFAULT_DENSITY_RATIO = 917 / 1028;

/** Thickness-change colour scale: linear below the soft width, logarithmic beyond it. */
export const THICKNESS_CHANGE_SOFT_METERS = 10;
export const THICKNESS_CHANGE_MAX_METERS = 1000;

// Diverging ramp: thinning red, thickening blue, unchanged ice near white. The stops are
// ColorBrewer RdBu with the neutral stop tinted towards the explorer's ice colour.
const THICKNESS_CHANGE_STOPS = [
  [0.0, [0.404, 0.0, 0.122]],
  [0.18, [0.698, 0.094, 0.169]],
  [0.33, [0.839, 0.376, 0.302]],
  [0.44, [0.957, 0.647, 0.51]],
  [0.5, [0.95, 0.97, 1.0]],
  [0.56, [0.573, 0.773, 0.871]],
  [0.67, [0.263, 0.576, 0.765]],
  [0.82, [0.129, 0.4, 0.675]],
  [1.0, [0.02, 0.188, 0.38]],
];

function increasing(values) {
  for (let i = 1; i < values.length; i += 1) {
    if (!(values[i] > values[i - 1])) return false;
  }
  return values.length > 0;
}

function readSeries(meta) {
  const series = meta.series;
  if (!series || !Array.isArray(series.years) || !Array.isArray(series.sea_level_contribution_m)) {
    return null;
  }
  const toArray = (values) => Float64Array.from(values || [], (value) => (value === null ? Number.NaN : value));
  return {
    years: toArray(series.years),
    seaLevel: toArray(series.sea_level_contribution_m),
    // Ensemble packages carry the spread of their models; single-model ones the model's areas.
    seaLevelMin: toArray(series.sea_level_contribution_min_m),
    seaLevelMax: toArray(series.sea_level_contribution_max_m),
    groundedArea: toArray(series.grounded_area_km2),
    floatingArea: toArray(series.floating_area_km2),
  };
}

function decodeSpeedChange(meta, buffer, frameCount, domainCount) {
  const field = (meta.fields || []).find((entry) => entry?.name === "speed_change");
  if (!field) return null;
  const frames = parseField(meta, buffer, "speed_change");
  if (frames.length !== frameCount * domainCount) {
    throw new Error("Projection speed-change frames do not match the keyframes and domain.");
  }
  const quant = resolveInt16Quantization(meta, field);
  return { frames, scale: quant.scale, offset: quant.offset, fill: quant.fillValue };
}

/**
 * Decode a projection package. Throws if the payload does not follow the layout the
 * metadata declares, so a truncated download fails loudly instead of drawing garbage.
 */
export function decodeIceProjection(meta, buffer) {
  if (meta?.geometry_type !== ICE_PROJECTION_GEOMETRY_TYPE) {
    throw new Error("Package is not an ice-sheet projection.");
  }
  const nx = Number(meta.grid?.nx);
  const ny = Number(meta.grid?.ny);
  const mask = parseField(meta, buffer, "domain_mask");
  if (!Number.isInteger(nx) || !Number.isInteger(ny) || mask.length !== nx * ny) {
    throw new Error("Projection domain mask does not cover the grid.");
  }

  const cellIndex = [];
  const vertexOfCell = new Int32Array(nx * ny).fill(-1);
  for (let cell = 0; cell < mask.length; cell += 1) {
    if (mask[cell]) {
      vertexOfCell[cell] = cellIndex.length;
      cellIndex.push(cell);
    }
  }
  const domainCount = cellIndex.length;
  if (domainCount !== Number(meta.domain?.cell_count)) {
    throw new Error("Projection domain size does not match its metadata.");
  }

  const frameYears = Float64Array.from(meta.frames?.years || []);
  if (!increasing(frameYears)) {
    throw new Error("Projection keyframe years must be strictly increasing.");
  }
  const thicknessField = getFieldDefinition(meta, "thickness");
  const frames = parseField(meta, buffer, "thickness");
  if (frames.length !== frameYears.length * domainCount) {
    throw new Error("Projection thickness frames do not match the keyframes and domain.");
  }
  const bedField = getFieldDefinition(meta, "bed");
  const bedCodes = parseField(meta, buffer, "bed");
  if (bedCodes.length !== domainCount) {
    throw new Error("Projection bed does not match the domain.");
  }

  const bedQuant = resolveInt16Quantization(meta, bedField);
  const bed = new Float32Array(domainCount);
  for (let i = 0; i < domainCount; i += 1) {
    bed[i] = bedCodes[i] === bedQuant.fillValue ? Number.NaN : bedCodes[i] * bedQuant.scale + bedQuant.offset;
  }
  const thicknessQuant = resolveInt16Quantization(meta, thicknessField);
  const ratio = Number(meta.physical_constants?.density_ratio);
  const speedChange = decodeSpeedChange(meta, buffer, frameYears.length, domainCount);

  return {
    grid: meta.grid,
    nx,
    ny,
    domainCount,
    cellIndex: Int32Array.from(cellIndex),
    vertexOfCell,
    bed,
    frames,
    frameCount: frameYears.length,
    frameYears,
    thicknessScale: thicknessQuant.scale,
    thicknessOffset: thicknessQuant.offset,
    thicknessFill: thicknessQuant.fillValue,
    firstYear: frameYears[0],
    lastYear: frameYears[frameYears.length - 1],
    densityRatio: Number.isFinite(ratio) && ratio > 0 && ratio < 1 ? ratio : DEFAULT_DENSITY_RATIO,
    series: readSeries(meta),
    speedChange,
  };
}

/** The keyframes around `year` and the blend weight of the later one; `year` is clamped. */
export function locateYear(frameYears, year) {
  const last = frameYears.length - 1;
  const clamped = Math.min(frameYears[last], Math.max(frameYears[0], Number(year)));
  let upper = 0;
  while (upper < last && frameYears[upper] < clamped) upper += 1;
  if (frameYears[upper] === clamped) {
    return { lower: upper, upper, weight: 0, year: clamped };
  }
  const lower = upper - 1;
  const weight = (clamped - frameYears[lower]) / (frameYears[upper] - frameYears[lower]);
  return { lower, upper, weight, year: clamped };
}

/** Thickness of every domain cell at `year`, written into `out`; returns the bracket. */
export function blendThicknessInto(projection, year, out) {
  const bracket = locateYear(projection.frameYears, year);
  const { domainCount, frames, thicknessScale: scale, thicknessOffset: offset, thicknessFill: fill } = projection;
  const lowerStart = bracket.lower * domainCount;
  const upperStart = bracket.upper * domainCount;
  const w = bracket.weight;
  for (let i = 0; i < domainCount; i += 1) {
    const a = frames[lowerStart + i];
    const b = frames[upperStart + i];
    const ha = a === fill ? 0 : a * scale + offset;
    const hb = b === fill ? 0 : b * scale + offset;
    out[i] = w === 0 ? ha : (1 - w) * ha + w * hb;
  }
  return bracket;
}

/**
 * Change in ice speed (m/yr) of every domain cell at `year`, written into `out`; zero
 * everywhere for a package without a `speed_change` field. Returns the bracket.
 */
export function blendSpeedChangeInto(projection, year, out) {
  const bracket = locateYear(projection.frameYears, year);
  const speed = projection.speedChange;
  if (!speed) {
    out.fill(0);
    return bracket;
  }
  const { domainCount } = projection;
  const { frames, scale, offset, fill } = speed;
  const lowerStart = bracket.lower * domainCount;
  const upperStart = bracket.upper * domainCount;
  const w = bracket.weight;
  for (let i = 0; i < domainCount; i += 1) {
    const a = frames[lowerStart + i];
    const b = frames[upperStart + i];
    const sa = a === fill ? 0 : a * scale + offset;
    const sb = b === fill ? 0 : b * scale + offset;
    out[i] = w === 0 ? sa : (1 - w) * sa + w * sb;
  }
  return bracket;
}

/**
 * The four domain cells around each point at fractional grid position (cols[i], rows[i]),
 * with their bilinear weights, so a per-cell field can be read along a line. A corner off
 * the grid or outside the domain is recorded as -1.
 */
export function buildDomainSampler(projection, cols, rows) {
  const { nx, ny, vertexOfCell } = projection;
  const count = Math.min(cols.length, rows.length);
  const vertices = new Int32Array(4 * count).fill(-1);
  const weights = new Float32Array(4 * count);
  for (let i = 0; i < count; i += 1) {
    const col = Number(cols[i]);
    const row = Number(rows[i]);
    if (!Number.isFinite(col) || !Number.isFinite(row)) continue;
    const c0 = Math.floor(col);
    const r0 = Math.floor(row);
    const tx = col - c0;
    const ty = row - r0;
    const corners = [
      [c0, r0, (1 - tx) * (1 - ty)],
      [c0 + 1, r0, tx * (1 - ty)],
      [c0, r0 + 1, (1 - tx) * ty],
      [c0 + 1, r0 + 1, tx * ty],
    ];
    corners.forEach(([c, r, w], k) => {
      weights[4 * i + k] = w;
      if (c >= 0 && c < nx && r >= 0 && r < ny) vertices[4 * i + k] = vertexOfCell[r * nx + c];
    });
  }
  return { count, vertices, weights };
}

/**
 * Bilinear sample of per-vertex `values` at every sampler point, into `out`. Only corners
 * that carry ice (`ice[vertex]`) contribute, and the sample is their weighted mean. A point
 * counts as covered (`coveredOut[i] = 1`) when those corners hold at least `minCover` of its
 * bilinear weight; by default all of it, so every corner with weight must carry ice. An
 * uncovered point samples NaN.
 */
export function sampleDomainInto(sampler, values, ice, out, coveredOut, minCover = 1) {
  const { count, vertices, weights } = sampler;
  const threshold = minCover - 1e-6;
  for (let i = 0; i < count; i += 1) {
    let sum = 0;
    let weight = 0;
    for (let k = 0; k < 4; k += 1) {
      const w = weights[4 * i + k];
      if (w === 0) continue;
      const v = vertices[4 * i + k];
      if (v < 0 || !ice[v]) continue;
      sum += w * values[v];
      weight += w;
    }
    const covered = weight > 0 && weight >= threshold ? 1 : 0;
    out[i] = covered ? sum / weight : Number.NaN;
    if (coveredOut) coveredOut[i] = covered;
  }
}

/**
 * Phase offsets that keep a travelling pulse continuous when its rate changes. A pulse at
 * distance d is drawn with phase fract(d * scale - time * rate - offset); switching from
 * `oldRates` to `newRates` at `timeTimesBaseRate` (time times the material's base rate)
 * leaves every phase where it was. Offsets are kept in [0, 1) to preserve precision.
 */
export function rephaseInto(offsets, oldRates, newRates, timeTimesBaseRate) {
  for (let i = 0; i < offsets.length; i += 1) {
    const next = offsets[i] + timeTimesBaseRate * (oldRates[i] - newRates[i]);
    offsets[i] = next - Math.floor(next);
  }
}

/**
 * Surface, base and grounded flag from flotation. Ice-free cells collapse onto the bed on
 * land and onto sea level over the ocean, which is where their vertices sit when drawn.
 */
export function flotationGeometryInto(bed, thickness, densityRatio, surfaceOut, baseOut, groundedOut) {
  for (let i = 0; i < thickness.length; i += 1) {
    const h = thickness[i];
    const b = bed[i];
    if (!(h > 0)) {
      const rest = b > 0 ? b : 0;
      surfaceOut[i] = rest;
      baseOut[i] = rest;
      if (groundedOut) groundedOut[i] = 0;
      continue;
    }
    const floatingBase = -densityRatio * h;
    const grounded = floatingBase <= b;
    const base = grounded ? b : floatingBase;
    baseOut[i] = base;
    surfaceOut[i] = base + h;
    if (groundedOut) groundedOut[i] = grounded ? 1 : 0;
  }
}

/** Linear interpolation of an annual series; NaN when the series is empty. */
export function sampleSeries(years, values, year) {
  const n = Math.min(years.length, values.length);
  if (n === 0) return Number.NaN;
  if (year <= years[0]) return values[0];
  if (year >= years[n - 1]) return values[n - 1];
  let upper = 1;
  while (upper < n - 1 && years[upper] < year) upper += 1;
  const t = (year - years[upper - 1]) / (years[upper] - years[upper - 1]);
  return values[upper - 1] + t * (values[upper] - values[upper - 1]);
}

/**
 * Triangles (as vertex indices) of every grid quad whose corners are domain cells, split
 * the same way as the explorer's terrain meshes so the two line up exactly.
 */
export function buildDomainTriangles(projection) {
  const { nx, ny, vertexOfCell } = projection;
  const triangles = [];
  for (let row = 0; row < ny - 1; row += 1) {
    for (let col = 0; col < nx - 1; col += 1) {
      const c0 = row * nx + col;
      const v0 = vertexOfCell[c0];
      const v1 = vertexOfCell[c0 + 1];
      const v2 = vertexOfCell[c0 + nx];
      const v3 = vertexOfCell[c0 + nx + 1];
      if (v0 >= 0 && v2 >= 0 && v1 >= 0) triangles.push(v0, v2, v1);
      if (v1 >= 0 && v2 >= 0 && v3 >= 0) triangles.push(v1, v2, v3);
    }
  }
  return Uint32Array.from(triangles);
}

/** Copy into `out` the triangles whose three corners all carry ice; returns the index count. */
export function collectIceTriangles(triangles, iceByVertex, out) {
  let count = 0;
  for (let i = 0; i < triangles.length; i += 3) {
    const a = triangles[i];
    const b = triangles[i + 1];
    const c = triangles[i + 2];
    if (iceByVertex[a] && iceByVertex[b] && iceByVertex[c]) {
      out[count] = a;
      out[count + 1] = b;
      out[count + 2] = c;
      count += 3;
    }
  }
  return count;
}

/**
 * Smooth vertex normals of a height field sampled on the domain cells, by central
 * differences (one-sided at the domain edge). Far cheaper than accumulating face normals,
 * which is what lets the projection re-light every animation frame. Neighbours outside
 * `mask` (when given) are ignored, so ice cells are not shaded by the collapsed vertices
 * of their ice-free neighbours.
 */
export function heightfieldNormalsInto(projection, heights, dxUnits, dzUnits, out, mask) {
  const { nx, ny, cellIndex, vertexOfCell, domainCount } = projection;
  const usable = (vertex) => vertex >= 0 && (!mask || mask[vertex]);
  for (let v = 0; v < domainCount; v += 1) {
    const cell = cellIndex[v];
    const col = cell % nx;
    const row = (cell - col) / nx;
    const h = heights[v];

    const left = col > 0 ? vertexOfCell[cell - 1] : -1;
    const right = col < nx - 1 ? vertexOfCell[cell + 1] : -1;
    const up = row > 0 ? vertexOfCell[cell - nx] : -1;
    const down = row < ny - 1 ? vertexOfCell[cell + nx] : -1;

    let dhdx = 0;
    if (usable(left) && usable(right)) dhdx = (heights[right] - heights[left]) / (2 * dxUnits);
    else if (usable(right)) dhdx = (heights[right] - h) / dxUnits;
    else if (usable(left)) dhdx = (h - heights[left]) / dxUnits;

    let dhdz = 0;
    if (usable(up) && usable(down)) dhdz = (heights[down] - heights[up]) / (2 * dzUnits);
    else if (usable(down)) dhdz = (heights[down] - h) / dzUnits;
    else if (usable(up)) dhdz = (h - heights[up]) / dzUnits;

    // `0 - x` rather than `-x`, so a flat cell gets +0 and not -0.
    const nxv = 0 - dhdx;
    const nzv = 0 - dhdz;
    const length = Math.hypot(nxv, 1, nzv);
    out[3 * v] = nxv / length;
    out[3 * v + 1] = 1 / length;
    out[3 * v + 2] = nzv / length;
  }
}

/**
 * Position of a thickness change on the diverging colour scale: 0 is the thinning end,
 * 0.5 no change, 1 the thickening end. asinh keeps metre-scale change visible without
 * saturating the kilometre-scale collapse of the marine basins.
 */
export function thicknessChangeScaleT(
  deltaMeters,
  softMeters = THICKNESS_CHANGE_SOFT_METERS,
  maxMeters = THICKNESS_CHANGE_MAX_METERS,
) {
  if (!Number.isFinite(deltaMeters) || deltaMeters === 0) return 0.5;
  const magnitude = Math.min(1, Math.asinh(Math.abs(deltaMeters) / softMeters) / Math.asinh(maxMeters / softMeters));
  return 0.5 + 0.5 * Math.sign(deltaMeters) * magnitude;
}

function sampleStops(stops, t) {
  if (t <= stops[0][0]) return stops[0][1];
  for (let i = 1; i < stops.length; i += 1) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1];
      const [t1, c1] = stops[i];
      const f = (t - t0) / (t1 - t0);
      return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f];
    }
  }
  return stops[stops.length - 1][1];
}

/** RGB (0-1) at position `t` (0-1) of the thickness-change ramp; for legends and lookup tables. */
export function thicknessChangeRampColor(t) {
  return sampleStops(THICKNESS_CHANGE_STOPS, Math.min(1, Math.max(0, Number(t) || 0)));
}

/** RGB (0-1) of a thickness change in metres. */
export function thicknessChangeColor(deltaMeters) {
  return thicknessChangeRampColor(thicknessChangeScaleT(deltaMeters));
}
