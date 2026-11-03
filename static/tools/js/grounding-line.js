/**
 * The grounding line of the 3D ICE explorer: the edge of grounded ice wherever it meets
 * floating ice or the ocean, traced from the grounded and floating ice of the terrain on
 * screen, or of a projection frame, so that it follows the ice the scene draws cell for cell.
 *
 * Grounded ice against ice-free land is not part of it. Where grounded ice meets floating
 * ice the line runs half-way between the two cell centres, as the subglacial channels end
 * (see overlay-cells.js). Where grounded ice meets the ocean directly (a tidewater front, or
 * a shelf gone in a projection) the grounding line is the ice front, and it runs through the
 * last grounded cell centres, which is where the scene draws the edge of the ice.
 *
 * The line is traced by marching squares over the grid of cell centres, with the saddle
 * squares split along the same diagonal as the terrain meshes' triangles, chained into
 * polylines and smoothed by corner cutting. Positions are in grid units (column, row).
 *
 * No DOM or Three.js dependency: this module runs unchanged under Node's test runner.
 */

import { isGroundedIceCell, MASK_FLOATING_ICE, MASK_OCEAN } from "./overlay-cells.js";

export const CELL_OTHER = 0;
export const CELL_GROUNDED = 1;
export const CELL_FLOATING = 2;
export const CELL_OCEAN = 3;

/** Corner-cutting passes applied to the traced line unless a caller asks otherwise. */
export const DEFAULT_SMOOTHING_PASSES = 2;

// The sharpest join the ribbon mitres; beyond it the ribbon narrows rather than spiking.
const MIN_MITRE_COSINE = 0.5;

/**
 * Classes of a terrain's cells: grounded ice (including the ice over Lake Vostok, which is
 * part of the ice sheet's interior), floating ice, the ocean, and everything else. Ice
 * classes need a drawn ice surface (`iceValid`), as the scene's ice meshes do.
 */
export function classifyTerrainCells(mask, iceValid, out = new Uint8Array(mask.length)) {
  for (let i = 0; i < mask.length; i += 1) {
    const value = mask[i];
    if (value === MASK_OCEAN) out[i] = CELL_OCEAN;
    else if (!iceValid[i]) out[i] = CELL_OTHER;
    else if (isGroundedIceCell(value)) out[i] = CELL_GROUNDED;
    else if (value === MASK_FLOATING_ICE) out[i] = CELL_FLOATING;
    else out[i] = CELL_OTHER;
  }
  return out;
}

/**
 * Classes of a projection frame's cells, from what the projection draws: inside its domain
 * each cell with ice is grounded or floating by flotation, and a cell without ice is ocean
 * where it rests at or below sea level. Outside the domain the projection never has ice, so
 * the terrain's ocean and floating ice are open water there and the rest is land.
 */
export function classifyProjectionCells(
  projection,
  ice,
  grounded,
  restingBed,
  terrainMask,
  out = new Uint8Array(projection.nx * projection.ny)
) {
  for (let cell = 0; cell < out.length; cell += 1) {
    const value = terrainMask[cell];
    out[cell] = value === MASK_OCEAN || value === MASK_FLOATING_ICE ? CELL_OCEAN : CELL_OTHER;
  }
  const { cellIndex, domainCount } = projection;
  for (let v = 0; v < domainCount; v += 1) {
    let cls;
    if (ice[v]) cls = grounded[v] ? CELL_GROUNDED : CELL_FLOATING;
    else cls = restingBed[v] > 0 ? CELL_OTHER : CELL_OCEAN;
    out[cellIndex[v]] = cls;
  }
  return out;
}

/** Spread per-domain-cell `values` over the grid: `fill` where the cell is outside the domain or `keep` is 0. */
export function scatterDomainInto(projection, values, keep, fill, out) {
  out.fill(fill);
  const { cellIndex, domainCount } = projection;
  for (let v = 0; v < domainCount; v += 1) {
    if (keep[v]) out[cellIndex[v]] = values[v];
  }
  return out;
}

// Segments of each marching-squares case, as pairs of square sides: 0 top (between the
// corners at row r), 1 right, 2 bottom, 3 left. Case bits: 1 top-left corner, 2 top-right,
// 4 bottom-right, 8 bottom-left. The terrain meshes split every square along the diagonal
// from its top-right to its bottom-left corner, so in the two saddle cases (5 and 10) the
// grounded corners on that diagonal are joined and the other two are cut off.
const CASE_SEGMENTS = [
  [],
  [3, 0],
  [0, 1],
  [3, 1],
  [1, 2],
  [3, 0, 1, 2],
  [0, 2],
  [3, 2],
  [2, 3],
  [0, 2],
  [3, 0, 1, 2],
  [1, 2],
  [3, 1],
  [0, 1],
  [3, 0],
  [],
];

// Edge keys: 2·cell for the edge from `cell` to its right-hand neighbour, 2·cell + 1 for the
// edge from `cell` to the cell below it.
function sideKey(nx, topLeft, side) {
  switch (side) {
    case 0:
      return 2 * topLeft;
    case 1:
      return 2 * (topLeft + 1) + 1;
    case 2:
      return 2 * (topLeft + nx);
    default:
      return 2 * topLeft + 1;
  }
}

function edgeCells(nx, key) {
  const a = key >> 1;
  return [a, key & 1 ? a + nx : a + 1];
}

/** Whether a crossing edge borders floating ice or the ocean, so that the line is drawn there. */
function isSeawardEdge(nx, classes, key) {
  const [a, b] = edgeCells(nx, key);
  const other = classes[a] === CELL_GROUNDED ? classes[b] : classes[a];
  return other === CELL_FLOATING || other === CELL_OCEAN;
}

/** Where the line crosses an edge, as [col, row]. */
function crossingPoint(nx, classes, key) {
  const [a, b] = edgeCells(nx, key);
  const groundedA = classes[a] === CELL_GROUNDED;
  const from = groundedA ? a : b;
  const to = groundedA ? b : a;
  const t = classes[to] === CELL_OCEAN ? 0 : 0.5;
  const fromCol = from % nx;
  const fromRow = (from - fromCol) / nx;
  const toCol = to % nx;
  const toRow = (to - toCol) / nx;
  return [fromCol + t * (toCol - fromCol), fromRow + t * (toRow - fromRow)];
}

function collectSegments(nx, ny, classes) {
  const ends = [];
  for (let row = 0; row < ny - 1; row += 1) {
    for (let col = 0; col < nx - 1; col += 1) {
      const i0 = row * nx + col;
      const code =
        (classes[i0] === CELL_GROUNDED ? 1 : 0) |
        (classes[i0 + 1] === CELL_GROUNDED ? 2 : 0) |
        (classes[i0 + nx + 1] === CELL_GROUNDED ? 4 : 0) |
        (classes[i0 + nx] === CELL_GROUNDED ? 8 : 0);
      if (code === 0 || code === 15) continue;
      const sides = CASE_SEGMENTS[code];
      for (let s = 0; s < sides.length; s += 2) {
        const keyA = sideKey(nx, i0, sides[s]);
        const keyB = sideKey(nx, i0, sides[s + 1]);
        if (isSeawardEdge(nx, classes, keyA) && isSeawardEdge(nx, classes, keyB)) ends.push(keyA, keyB);
      }
    }
  }
  return ends;
}

function pushDistinct(points, point) {
  const n = points.length;
  if (n >= 2 && Math.abs(points[n - 2] - point[0]) < 1e-9 && Math.abs(points[n - 1] - point[1]) < 1e-9) return;
  points.push(point[0], point[1]);
}

function chainSegments(nx, classes, ends) {
  const segmentCount = ends.length / 2;
  const byEdge = new Map();
  for (let s = 0; s < segmentCount; s += 1) {
    for (const key of [ends[2 * s], ends[2 * s + 1]]) {
      const list = byEdge.get(key);
      if (list) list.push(s);
      else byEdge.set(key, [s]);
    }
  }
  const used = new Uint8Array(segmentCount);
  const lines = [];
  const walk = (startKey, startSegment) => {
    const points = [];
    pushDistinct(points, crossingPoint(nx, classes, startKey));
    let key = startKey;
    let segment = startSegment;
    let closed = false;
    while (segment >= 0) {
      used[segment] = 1;
      key = ends[2 * segment] === key ? ends[2 * segment + 1] : ends[2 * segment];
      if (key === startKey) {
        closed = true;
        break;
      }
      pushDistinct(points, crossingPoint(nx, classes, key));
      segment = -1;
      for (const next of byEdge.get(key)) {
        if (!used[next]) {
          segment = next;
          break;
        }
      }
    }
    if (closed && points.length >= 4) {
      const n = points.length;
      if (Math.abs(points[0] - points[n - 2]) < 1e-9 && Math.abs(points[1] - points[n - 1]) < 1e-9) {
        points.length = n - 2;
      }
    }
    if (points.length >= 4) lines.push({ points, closed: closed && points.length >= 6 });
  };
  // Open lines first, from their ends, so that none is entered half-way along.
  for (const [key, list] of byEdge) {
    if (list.length === 1 && !used[list[0]]) walk(key, list[0]);
  }
  for (let s = 0; s < segmentCount; s += 1) {
    if (!used[s]) walk(ends[2 * s], s);
  }
  return lines;
}

/**
 * One corner-cutting pass (Chaikin): every segment is replaced by the points a quarter and
 * three quarters along it. An open line keeps its two ends.
 */
export function cutCorners(points, closed) {
  const n = points.length / 2;
  if (n < 3 && !closed) return points.slice();
  const out = [];
  if (!closed) out.push(points[0], points[1]);
  const last = closed ? n : n - 1;
  for (let i = 0; i < last; i += 1) {
    const j = (i + 1) % n;
    const ax = points[2 * i];
    const ay = points[2 * i + 1];
    const bx = points[2 * j];
    const by = points[2 * j + 1];
    out.push(0.75 * ax + 0.25 * bx, 0.75 * ay + 0.25 * by, 0.25 * ax + 0.75 * bx, 0.25 * ay + 0.75 * by);
  }
  if (!closed) out.push(points[2 * n - 2], points[2 * n - 1]);
  return out;
}

/**
 * The grounding line of a grid of cell classes, as polylines of [col, row] pairs. Each has
 * `points` (a flat array) and `closed`; a closed line does not repeat its first point.
 */
export function traceGroundingLine(nx, ny, classes, { smoothingPasses = DEFAULT_SMOOTHING_PASSES } = {}) {
  if (classes.length !== nx * ny) throw new Error("Cell classes do not cover the grid.");
  const lines = chainSegments(nx, classes, collectSegments(nx, ny, classes));
  for (const line of lines) {
    for (let pass = 0; pass < smoothingPasses; pass += 1) line.points = cutCorners(line.points, line.closed);
  }
  return lines;
}

/**
 * Height of a terrain mesh at a point in grid units, interpolated over the mesh's own
 * triangle there so that a line drawn with it lies on the mesh. Where that triangle is not
 * drawn the valid corners of the square are blended instead; NaN when none is valid.
 */
export function meshHeightAt(nx, ny, heights, valid, col, row) {
  const c0 = Math.min(Math.max(Math.floor(col), 0), nx - 2);
  const r0 = Math.min(Math.max(Math.floor(row), 0), ny - 2);
  const fx = Math.min(Math.max(col - c0, 0), 1);
  const fy = Math.min(Math.max(row - r0, 0), 1);
  const i0 = r0 * nx + c0;
  const i1 = i0 + 1;
  const i2 = i0 + nx;
  const i3 = i2 + 1;
  if (fx + fy <= 1) {
    if (valid[i0] && valid[i1] && valid[i2]) {
      return heights[i0] + fx * (heights[i1] - heights[i0]) + fy * (heights[i2] - heights[i0]);
    }
  } else if (valid[i1] && valid[i2] && valid[i3]) {
    return heights[i3] + (1 - fx) * (heights[i2] - heights[i3]) + (1 - fy) * (heights[i1] - heights[i3]);
  }
  const corners = [i0, i1, i2, i3];
  const weights = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
  let sum = 0;
  let total = 0;
  let nearest = -1;
  let nearestWeight = -1;
  for (let k = 0; k < 4; k += 1) {
    if (!valid[corners[k]]) continue;
    sum += weights[k] * heights[corners[k]];
    total += weights[k];
    if (weights[k] > nearestWeight) {
      nearestWeight = weights[k];
      nearest = corners[k];
    }
  }
  if (nearest < 0) return Number.NaN;
  return total > 1e-9 ? sum / total : heights[nearest];
}

// Runs of a line with a surface beneath them, as scene xyz with the grid position of each point.
function ribbonRuns(line, grid) {
  const { nx, ny, heights, valid, colUnits, rowUnits, verticalMetersPerUnit, offsetMeters } = grid;
  const halfX = (nx - 1) / 2;
  const halfY = (ny - 1) / 2;
  const runs = [];
  let run = { xyz: [], grid: [] };
  const n = line.points.length / 2;
  for (let i = 0; i < n; i += 1) {
    const col = line.points[2 * i];
    const row = line.points[2 * i + 1];
    const h = meshHeightAt(nx, ny, heights, valid, col, row);
    if (!Number.isFinite(h)) {
      if (run.xyz.length >= 6) runs.push({ ...run, closed: false });
      run = { xyz: [], grid: [] };
      continue;
    }
    run.xyz.push((col - halfX) * colUnits, (h + offsetMeters) / verticalMetersPerUnit, (row - halfY) * rowUnits);
    run.grid.push(col, row);
  }
  if (run.xyz.length >= 6) runs.push({ ...run, closed: line.closed && run.xyz.length === 3 * n });
  return runs;
}

function unitXZ(xyz, from, to) {
  const dx = xyz[3 * to] - xyz[3 * from];
  const dz = xyz[3 * to + 2] - xyz[3 * from + 2];
  const length = Math.hypot(dx, dz);
  return length > 1e-12 ? [dx / length, dz / length] : null;
}

/**
 * A ribbon along the grounding line, `halfWidthUnits` either side of it, draped at
 * `offsetMeters` above the mesh height field `heights` (cells where `valid` is 0 carry no
 * height; the line breaks over them). Each edge takes the height of the mesh beneath it, so
 * a slope across the line does not bury the uphill edge in the surface. Scene x and z follow the terrain meshes: the grid's
 * midpoint at the origin, `colUnits` and `rowUnits` scene units per column and row, and
 * height divided by `verticalMetersPerUnit`. Returns `positions` (xyz per vertex) and
 * `indices` (triangles), two vertices per line point.
 */
export function buildGroundingRibbon(lines, grid) {
  const { nx, ny, heights, valid, colUnits, rowUnits, verticalMetersPerUnit, offsetMeters } = grid;
  const halfWidth = grid.halfWidthUnits;
  // The scene height of a ribbon edge at a grid position, or the line's own where no surface is.
  const edgeY = (col, row, fallback) => {
    const h = meshHeightAt(nx, ny, heights, valid, col, row);
    return Number.isFinite(h) ? (h + offsetMeters) / verticalMetersPerUnit : fallback;
  };
  const positions = [];
  const indices = [];
  for (const line of lines) {
    for (const { xyz, grid: at, closed } of ribbonRuns(line, grid)) {
      const n = xyz.length / 3;
      const base = positions.length / 3;
      for (let i = 0; i < n; i += 1) {
        const prev = i > 0 ? i - 1 : closed ? n - 1 : -1;
        const next = i < n - 1 ? i + 1 : closed ? 0 : -1;
        const inbound = prev >= 0 ? unitXZ(xyz, prev, i) : null;
        const outbound = next >= 0 ? unitXZ(xyz, i, next) : null;
        let tangent = inbound || outbound || [1, 0];
        let scale = 1;
        if (inbound && outbound) {
          const tx = inbound[0] + outbound[0];
          const tz = inbound[1] + outbound[1];
          const length = Math.hypot(tx, tz);
          if (length > 1e-9) {
            tangent = [tx / length, tz / length];
            const cosine = tangent[0] * inbound[0] + tangent[1] * inbound[1];
            scale = 1 / Math.max(cosine, MIN_MITRE_COSINE);
          }
        }
        const ox = -tangent[1] * halfWidth * scale;
        const oz = tangent[0] * halfWidth * scale;
        const x = xyz[3 * i];
        const y = xyz[3 * i + 1];
        const z = xyz[3 * i + 2];
        const col = at[2 * i];
        const row = at[2 * i + 1];
        const dCol = ox / colUnits;
        const dRow = oz / rowUnits;
        positions.push(
          x + ox,
          edgeY(col + dCol, row + dRow, y),
          z + oz,
          x - ox,
          edgeY(col - dCol, row - dRow, y),
          z - oz
        );
      }
      const segments = closed ? n : n - 1;
      for (let i = 0; i < segments; i += 1) {
        const a = base + 2 * i;
        const b = base + 2 * ((i + 1) % n);
        indices.push(a, a + 1, b, a + 1, b + 1, b);
      }
    }
  }
  return { positions: Float32Array.from(positions), indices: Uint32Array.from(indices) };
}

/** Length of the traced line in kilometres on a grid of `dxMeters` by `dyMeters` cells. */
export function groundingLineLengthKm(lines, dxMeters, dyMeters) {
  let total = 0;
  for (const { points, closed } of lines) {
    const n = points.length / 2;
    const segments = closed ? n : n - 1;
    for (let i = 0; i < segments; i += 1) {
      const j = (i + 1) % n;
      total += Math.hypot((points[2 * j] - points[2 * i]) * dxMeters, (points[2 * j + 1] - points[2 * i + 1]) * dyMeters);
    }
  }
  return total / 1000;
}
