/**
 * Grid utilities for the 3D ICE isostatic-rebound layer: polar-stereographic area
 * weighting, load-conserving coarsening, C1 upsampling and ocean connectivity.
 *
 * These are separated from the physics in gia-rebound.js because they are generic
 * raster operations with their own correctness criteria, and each one is verifiable
 * in isolation: block averaging must conserve the integral, upsampling must be
 * C1-continuous and range-preserving, and the flood fill must respect 4-connectivity.
 */

import { nextPowerOfTwo } from "./fft2d.js";

// The Kelvin kernel kei(r/L_r) falls below 1e-3 of its peak only beyond ~7.6 flexural
// length scales, so 8 is the smallest margin that guarantees ~0.1% truncation. Rounding
// the transform up to a power of two usually pads considerably further than this.
const FLEXURE_MARGIN_LENGTH_SCALES = 8;
const PREFERRED_FFT_AXIS = 512;
const SOLVER_TARGET_CELL_M = 16000;
const SOLVER_MAX_CELL_M = 28000;

/**
 * Great-circle-correct cell area (m^2) for a polar-stereographic grid.
 *
 * The projection is conformal, so true area is the projected cell area divided by the
 * squared point scale factor k = (1 + sin|phi_c|) / (1 + sin|phi|). Latitude is recovered
 * from the polar radius on a sphere of radius `earthRadiusMeters`; against the WGS84
 * ellipsoid that leaves a sub-percent residual, far inside the several-percent spread the
 * mantle-density choice already imposes on these statistics. Ignoring the correction
 * entirely would instead bias cell areas by roughly -3 % at the pole to +8 % at 60 deg.
 */
export function polarStereographicCellAreaM2({
  xMeters,
  yMeters,
  dxMeters,
  dyMeters,
  standardParallelDegrees,
  earthRadiusMeters = 6371000,
}) {
  const projectedArea = Math.abs(dxMeters * dyMeters);
  const standardSine = Math.sin(Math.abs(standardParallelDegrees) * (Math.PI / 180));
  const radius = Math.hypot(xMeters, yMeters);
  const absoluteLatitude =
    Math.PI / 2 - 2 * Math.atan(radius / (earthRadiusMeters * (1 + standardSine)));
  const scaleFactor = (1 + standardSine) / (1 + Math.sin(absoluteLatitude));
  if (!Number.isFinite(scaleFactor) || scaleFactor <= 0) return projectedArea;
  return projectedArea / (scaleFactor * scaleFactor);
}

/**
 * Smallest coarsening factor whose zero-padded transform fits the preferred FFT axis,
 * bounded so the solve grid never exceeds SOLVER_MAX_CELL_M.
 *
 * Takes the flexural length scale directly rather than the rigidity so this module
 * stays free of any dependency on the physics module.
 */
export function chooseCoarseningFactor(nx, ny, dxMeters, dyMeters, lengthScale) {
  const finestSpacing = Math.min(Math.abs(dxMeters), Math.abs(dyMeters));
  const minFactor = Math.max(1, Math.round(SOLVER_TARGET_CELL_M / finestSpacing));
  const maxFactor = Math.max(minFactor, Math.floor(SOLVER_MAX_CELL_M / finestSpacing));

  let fallback = minFactor;
  for (let factor = minFactor; factor <= maxFactor; factor += 1) {
    const { sizeX, sizeY } = paddedTransformSize(nx, ny, dxMeters, dyMeters, factor, lengthScale);
    if (sizeX <= PREFERRED_FFT_AXIS && sizeY <= PREFERRED_FFT_AXIS) return factor;
    fallback = factor;
  }
  return fallback;
}

export function paddedTransformSize(nx, ny, dxMeters, dyMeters, factor, lengthScale) {
  const coarseX = Math.ceil(nx / factor);
  const coarseY = Math.ceil(ny / factor);
  const cellX = Math.abs(dxMeters) * factor;
  const cellY = Math.abs(dyMeters) * factor;
  const marginX = Math.ceil((FLEXURE_MARGIN_LENGTH_SCALES * lengthScale) / cellX);
  const marginY = Math.ceil((FLEXURE_MARGIN_LENGTH_SCALES * lengthScale) / cellY);
  return {
    coarseX,
    coarseY,
    cellX,
    cellY,
    sizeX: nextPowerOfTwo(coarseX + 2 * marginX),
    sizeY: nextPowerOfTwo(coarseY + 2 * marginY),
  };
}

/** Area-weighted block mean, skipping non-finite cells so gaps do not bias the load. */
export function blockMean(source, nx, ny, factor) {
  const coarseX = Math.ceil(nx / factor);
  const coarseY = Math.ceil(ny / factor);
  const totals = new Float64Array(coarseX * coarseY);
  const counts = new Float64Array(coarseX * coarseY);

  for (let row = 0; row < ny; row += 1) {
    const coarseRow = Math.min(coarseY - 1, (row / factor) | 0);
    for (let column = 0; column < nx; column += 1) {
      const value = source[row * nx + column];
      if (!Number.isFinite(value)) continue;
      const target = coarseRow * coarseX + Math.min(coarseX - 1, (column / factor) | 0);
      totals[target] += value;
      counts[target] += 1;
    }
  }
  const coverage = new Uint8Array(coarseX * coarseY);
  for (let index = 0; index < totals.length; index += 1) {
    coverage[index] = counts[index] > 0 ? 1 : 0;
    totals[index] = counts[index] > 0 ? totals[index] / counts[index] : 0;
  }
  return { data: totals, coverage, coarseX, coarseY };
}

function catmullRom(p0, p1, p2, p3, t) {
  const a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
  const b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3;
  const c = -0.5 * p0 + 0.5 * p2;
  return ((a * t + b) * t + c) * t + p1;
}

/**
 * Bicubic (Catmull-Rom) upsample of a cell-centred coarse field onto the native grid,
 * with half-cell registration and clamped edges.
 *
 * Bilinear interpolation would be adequate for the uplift values themselves, but the
 * rebounded bed is rendered as a lit 3-D surface: a piecewise-constant interpolation
 * gradient turns into visible facet seams on every coarse-cell boundary once vertex
 * normals are computed. A C1-continuous kernel removes that artefact.
 */
export function bicubicUpsampleInto(source, coarseX, coarseY, nx, ny, factor, out) {
  const sample = (column, row) =>
    source[
      Math.min(coarseY - 1, Math.max(0, row)) * coarseX + Math.min(coarseX - 1, Math.max(0, column))
    ];

  for (let row = 0; row < ny; row += 1) {
    const sampleY = (row + 0.5) / factor - 0.5;
    const baseRow = Math.floor(sampleY);
    const weightY = sampleY - baseRow;
    const outOffset = row * nx;

    for (let column = 0; column < nx; column += 1) {
      const sampleX = (column + 0.5) / factor - 0.5;
      const baseColumn = Math.floor(sampleX);
      const weightX = sampleX - baseColumn;

      let value = 0;
      const columns = [
        catmullRom(
          sample(baseColumn - 1, baseRow - 1),
          sample(baseColumn, baseRow - 1),
          sample(baseColumn + 1, baseRow - 1),
          sample(baseColumn + 2, baseRow - 1),
          weightX
        ),
        catmullRom(
          sample(baseColumn - 1, baseRow),
          sample(baseColumn, baseRow),
          sample(baseColumn + 1, baseRow),
          sample(baseColumn + 2, baseRow),
          weightX
        ),
        catmullRom(
          sample(baseColumn - 1, baseRow + 1),
          sample(baseColumn, baseRow + 1),
          sample(baseColumn + 1, baseRow + 1),
          sample(baseColumn + 2, baseRow + 1),
          weightX
        ),
        catmullRom(
          sample(baseColumn - 1, baseRow + 2),
          sample(baseColumn, baseRow + 2),
          sample(baseColumn + 1, baseRow + 2),
          sample(baseColumn + 2, baseRow + 2),
          weightX
        ),
      ];
      value = catmullRom(columns[0], columns[1], columns[2], columns[3], weightY);
      out[outOffset + column] = value;
    }
  }
  return out;
}

/**
 * Four-connected flood fill marking which sub-datum cells drain to the domain boundary.
 *
 * Without this test every depression below the sea-level datum is loaded as if it were
 * open ocean, including basins that rebound into closed, land-locked hollows. Those
 * carry no marine water column, and wrongly flooding one takes back rho_w/rho_m ~ 31 %
 * of its uplift, which is exactly the margin that decides whether it emerges.
 */
export function floodConnectedOcean({
  coarseX,
  coarseY,
  bed,
  uplift,
  seaLevelMeters,
  valid = null,
  out,
  queue,
}) {
  const connected = out || new Uint8Array(coarseX * coarseY);
  connected.fill(0);
  const stack = queue || new Int32Array(coarseX * coarseY);
  let top = 0;

  // Cells with no bed data conduct rather than dam. Bedmap3 leaves 97 % of the domain
  // edge as no-data, so treating those as land at 0 m would seed the fill from a handful
  // of cells and, for a future package with none, would wall the ocean off entirely and
  // silently classify every basin as closed.
  const isSubmerged = (index) =>
    (valid !== null && !valid[index]) || bed[index] + uplift[index] < seaLevelMeters;
  const push = (index) => {
    if (connected[index] || !isSubmerged(index)) return;
    connected[index] = 1;
    stack[top] = index;
    top += 1;
  };

  for (let column = 0; column < coarseX; column += 1) {
    push(column);
    push((coarseY - 1) * coarseX + column);
  }
  for (let row = 0; row < coarseY; row += 1) {
    push(row * coarseX);
    push(row * coarseX + coarseX - 1);
  }

  while (top > 0) {
    top -= 1;
    const index = stack[top];
    const row = (index / coarseX) | 0;
    const column = index - row * coarseX;
    if (column > 0) push(index - 1);
    if (column < coarseX - 1) push(index + 1);
    if (row > 0) push(index - coarseX);
    if (row < coarseY - 1) push(index + coarseX);
  }
  return connected;
}

/**
 * Sub-cell bathymetric hypsometry, so a coarse-grid solve can carry an exact water load.
 *
 * The water load a coarse cell carries is mean_fine(max(0, h - b)), but the obvious
 * coarse-grid shortcut max(0, h - mean_fine(b)) is biased low by Jensen's inequality
 * wherever the sub-cell bathymetry straddles the waterline - which is exactly the
 * marginal terrain that decides whether a basin emerges. Measured against a
 * native-resolution reference the shortcut costs ~3 m RMS and ~17 m peak of uplift, and
 * biases the emergent-area headline by about half a percent.
 *
 * The deflection varies over the flexural length scale, so h = datum - uplift is
 * effectively constant inside one coarse cell and the exact mean can be precomputed as a
 * function of that single scalar. With each cell's fine bed values sorted ascending and
 * prefix-summed, mean(max(0, h - b)) = (k*h - sum of the k values below h) / n, so a
 * binary search per coarse cell replaces a full fine-grid sweep every iteration.
 */
export function buildSubCellHypsometry(bedHeights, nx, ny, factor) {
  const coarseX = Math.ceil(nx / factor);
  const coarseY = Math.ceil(ny / factor);
  const coarseCount = coarseX * coarseY;

  const counts = new Int32Array(coarseCount);
  for (let row = 0; row < ny; row += 1) {
    const coarseRow = Math.min(coarseY - 1, (row / factor) | 0) * coarseX;
    for (let column = 0; column < nx; column += 1) {
      if (!Number.isFinite(bedHeights[row * nx + column])) continue;
      counts[coarseRow + Math.min(coarseX - 1, (column / factor) | 0)] += 1;
    }
  }

  const offsets = new Int32Array(coarseCount + 1);
  for (let index = 0; index < coarseCount; index += 1) offsets[index + 1] = offsets[index] + counts[index];
  const total = offsets[coarseCount];

  const sorted = new Float32Array(total);
  const cursor = offsets.slice(0, coarseCount);
  for (let row = 0; row < ny; row += 1) {
    const coarseRow = Math.min(coarseY - 1, (row / factor) | 0) * coarseX;
    for (let column = 0; column < nx; column += 1) {
      const bed = bedHeights[row * nx + column];
      if (!Number.isFinite(bed)) continue;
      const cell = coarseRow + Math.min(coarseX - 1, (column / factor) | 0);
      sorted[cursor[cell]] = bed;
      cursor[cell] += 1;
    }
  }

  // Insertion sort per group: groups hold factor^2 entries (25 at the default 4 km -> 20 km),
  // where insertion sort beats a generic sort and needs no comparator allocation.
  const prefix = new Float32Array(total);
  for (let cell = 0; cell < coarseCount; cell += 1) {
    const start = offsets[cell];
    const end = offsets[cell + 1];
    for (let i = start + 1; i < end; i += 1) {
      const value = sorted[i];
      let j = i - 1;
      while (j >= start && sorted[j] > value) {
        sorted[j + 1] = sorted[j];
        j -= 1;
      }
      sorted[j + 1] = value;
    }
    let running = 0;
    for (let i = start; i < end; i += 1) {
      running += sorted[i];
      prefix[i] = running;
    }
  }

  return { offsets, sorted, prefix, coarseX, coarseY, coarseCount };
}

/**
 * Mean submerged depth over one coarse cell for a waterline at `waterLevel` (metres in
 * bed coordinates, i.e. sea-level datum minus the cell's uplift).
 */
export function meanSubmergedDepth(hypsometry, cellIndex, waterLevel) {
  const { offsets, sorted, prefix } = hypsometry;
  const start = offsets[cellIndex];
  const end = offsets[cellIndex + 1];
  const count = end - start;
  if (count <= 0) return 0;
  if (waterLevel <= sorted[start]) return 0;

  // Upper bound: index of the first value >= waterLevel.
  let low = start;
  let high = end;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (sorted[middle] < waterLevel) low = middle + 1;
    else high = middle;
  }
  const submerged = low - start;
  if (submerged <= 0) return 0;
  const belowSum = prefix[low - 1];
  return (submerged * waterLevel - belowSum) / count;
}
