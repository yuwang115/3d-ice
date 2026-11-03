/**
 * The grounding line: the edge of grounded ice against floating ice or the ocean, traced
 * from the cell classes of the terrain on screen or of a projection frame.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { decodeFieldToFloat32, parseField } from "../../static/tools/js/data-contract.js";
import {
  buildGroundingRibbon,
  CELL_FLOATING,
  CELL_GROUNDED,
  CELL_OCEAN,
  CELL_OTHER,
  classifyProjectionCells,
  classifyTerrainCells,
  cutCorners,
  groundingLineLengthKm,
  meshHeightAt,
  scatterDomainInto,
  traceGroundingLine,
} from "../../static/tools/js/grounding-line.js";

const here = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(here, "..", "..", "static", "tools", "data");

const G = CELL_GROUNDED;
const F = CELL_FLOATING;
const O = CELL_OCEAN;
const L = CELL_OTHER;

/** A grid of `fill` with a block of `block` cells over rows r0..r1 and columns c0..c1. */
function blockGrid(nx, ny, fill, block, [r0, r1], [c0, c1]) {
  const classes = new Uint8Array(nx * ny).fill(fill);
  for (let row = r0; row <= r1; row += 1) {
    for (let col = c0; col <= c1; col += 1) classes[row * nx + col] = block;
  }
  return classes;
}

function pointsOf(line) {
  const out = [];
  for (let i = 0; i < line.points.length; i += 2) out.push([line.points[i], line.points[i + 1]]);
  return out;
}

const RAW = { smoothingPasses: 0 };

// ---------------------------------------------------------------- classes

test("terrain cells: grounded ice, Lake Vostok's ice, floating ice, ocean and the rest", () => {
  const mask = Uint8Array.from([0, 1, 2, 3, 4, 2, 3]);
  const iceValid = Uint8Array.from([0, 0, 1, 1, 1, 0, 0]);
  assert.deepEqual([...classifyTerrainCells(mask, iceValid)], [O, L, G, F, G, L, L]);
});

test("projection cells follow the frame's ice inside the domain and the terrain outside it", () => {
  // A 3 × 2 grid whose domain is the first four cells.
  const projection = { nx: 3, ny: 2, cellIndex: Int32Array.from([0, 1, 2, 3]), domainCount: 4 };
  const ice = Uint8Array.from([1, 1, 0, 0]);
  const grounded = Uint8Array.from([1, 0, 0, 0]);
  const restingBed = Float32Array.from([-500, -800, -300, 40]);
  const terrainMask = Uint8Array.from([2, 3, 3, 1, 3, 1]);
  assert.deepEqual(
    [...classifyProjectionCells(projection, ice, grounded, restingBed, terrainMask)],
    [G, F, O, L, O, L],
    "an ice-free cell below sea level is ocean; outside the domain floating ice counts as open water"
  );
});

test("scatterDomainInto spreads kept domain values over the grid", () => {
  const projection = { cellIndex: Int32Array.from([1, 3]), domainCount: 2 };
  const out = scatterDomainInto(projection, Float32Array.from([5, 7]), Uint8Array.from([1, 0]), Number.NaN, new Float32Array(4));
  assert.deepEqual([...out].map((v) => (Number.isNaN(v) ? null : v)), [null, 5, null, null]);
});

// ---------------------------------------------------------------- tracing

test("grounded ice in a shelf is ringed half-way to the floating cells", () => {
  const classes = blockGrid(7, 7, F, G, [2, 4], [2, 4]);
  const lines = traceGroundingLine(7, 7, classes, RAW);
  assert.equal(lines.length, 1);
  assert.equal(lines[0].closed, true);
  for (const [col, row] of pointsOf(lines[0])) {
    assert.equal(Math.max(Math.abs(col - 3), Math.abs(row - 3)), 1.5, `(${col}, ${row}) lies half a cell out`);
  }
});

test("against the ocean the line runs through the last grounded cells, the ice front", () => {
  const classes = blockGrid(7, 7, O, G, [2, 4], [2, 4]);
  const [line, ...rest] = traceGroundingLine(7, 7, classes, RAW);
  assert.equal(rest.length, 0);
  assert.equal(line.closed, true);
  for (const [col, row] of pointsOf(line)) {
    assert.equal(Math.max(Math.abs(col - 3), Math.abs(row - 3)), 1, `(${col}, ${row}) lies on the edge of the ice`);
  }
});

test("grounded ice against ice-free land has no grounding line", () => {
  assert.deepEqual(traceGroundingLine(7, 7, blockGrid(7, 7, L, G, [2, 4], [2, 4])), []);
  assert.deepEqual(traceGroundingLine(5, 5, new Uint8Array(25).fill(G)), [], "no edge, no line");
});

test("where a shelf meets land the line ends instead of following the land margin", () => {
  // Grounded ice on the left half, a shelf on the right half; land along the top row.
  const nx = 8;
  const ny = 6;
  const classes = new Uint8Array(nx * ny);
  for (let row = 0; row < ny; row += 1) {
    for (let col = 0; col < nx; col += 1) classes[row * nx + col] = row === 0 ? L : col < 4 ? G : F;
  }
  const lines = traceGroundingLine(nx, ny, classes, RAW);
  assert.equal(lines.length, 1);
  assert.equal(lines[0].closed, false);
  for (const [col] of pointsOf(lines[0])) assert.equal(col, 3.5);
  const rows = pointsOf(lines[0]).map(([, row]) => row);
  assert.ok(Math.min(...rows) >= 1, "nothing is drawn against the land");
});

test("saddles follow the terrain meshes' diagonal, top-right to bottom-left", () => {
  // Diagonal neighbours on that diagonal are one grounded patch...
  const joined = new Uint8Array(16).fill(F);
  joined[1 * 4 + 2] = G;
  joined[2 * 4 + 1] = G;
  assert.equal(traceGroundingLine(4, 4, joined, RAW).length, 1);
  // ...and on the other diagonal two.
  const apart = new Uint8Array(16).fill(F);
  apart[1 * 4 + 1] = G;
  apart[2 * 4 + 2] = G;
  assert.equal(traceGroundingLine(4, 4, apart, RAW).length, 2);
});

test("an ice rise in a shelf beside an ice sheet gives two separate lines", () => {
  const nx = 12;
  const ny = 6;
  const classes = new Uint8Array(nx * ny).fill(F);
  for (let row = 0; row < ny; row += 1) {
    for (let col = 0; col < 3; col += 1) classes[row * nx + col] = G;
  }
  classes[3 * nx + 8] = G;
  const lines = traceGroundingLine(nx, ny, classes, RAW);
  assert.equal(lines.length, 2);
  assert.deepEqual(
    lines.map((line) => line.closed).sort(),
    [false, true],
    "the ice sheet's line reaches the grid's edge; the ice rise's closes"
  );
});

test("an open line is walked from one end, not entered half-way along", () => {
  // A grounded block on the grid's bottom edge: its line arches from the edge and back to it.
  const classes = blockGrid(7, 5, F, G, [3, 4], [2, 4]);
  const lines = traceGroundingLine(7, 5, classes, RAW);
  assert.equal(lines.length, 1);
  assert.equal(lines[0].closed, false);
  const rows = pointsOf(lines[0]).map(([, row]) => row);
  assert.deepEqual([rows[0], rows[rows.length - 1]], [4, 4], "both ends on the grid's edge");
});

test("a lone grounded cell in the ocean has no line, as the ice meshes draw nothing there", () => {
  // Against the ocean every crossing falls on the cell itself, so the ring has one point.
  assert.deepEqual(traceGroundingLine(3, 3, blockGrid(3, 3, O, G, [1, 1], [1, 1])), []);
});

test("the traced line rejects classes that do not cover the grid", () => {
  assert.throws(() => traceGroundingLine(3, 3, new Uint8Array(8)), /cover the grid/);
});

// ---------------------------------------------------------------- smoothing

test("corner cutting doubles the points, keeps an open line's ends and closes a closed one", () => {
  const open = [0, 0, 1, 0, 1, 1];
  const cut = cutCorners(open, false);
  assert.equal(cut.length / 2, 6);
  assert.deepEqual(cut.slice(0, 2), [0, 0]);
  assert.deepEqual(cut.slice(-2), [1, 1]);
  const square = [0, 0, 1, 0, 1, 1, 0, 1];
  assert.equal(cutCorners(square, true).length / 2, 8);
  assert.deepEqual(cutCorners([0, 0, 2, 0], false), [0, 0, 2, 0], "a single segment stays as it is");
});

test("smoothing keeps the ring inside the band between the grounded and floating cells", () => {
  const classes = blockGrid(9, 9, F, G, [2, 6], [2, 6]);
  const [line] = traceGroundingLine(9, 9, classes);
  assert.equal(line.closed, true);
  for (const [col, row] of pointsOf(line)) {
    const distance = Math.max(Math.abs(col - 4), Math.abs(row - 4));
    assert.ok(distance > 2 && distance <= 2.5, `(${col}, ${row}) stays between the grounded and floating cells`);
  }
});

// ---------------------------------------------------------------- heights

test("heights follow the mesh's own triangles and skip cells with no surface", () => {
  // A plane h = 2·col + 3·row on a 3 × 3 grid: exact on every triangle.
  const heights = Float32Array.from([0, 2, 4, 3, 5, 7, 6, 8, 10]);
  const valid = new Uint8Array(9).fill(1);
  assert.equal(meshHeightAt(3, 3, heights, valid, 0.25, 0.25), 1.25);
  assert.equal(meshHeightAt(3, 3, heights, valid, 1.75, 1.5), 8);
  // Without the top-left cell the point is blended from the square's valid corners.
  const holed = Uint8Array.from([0, 1, 1, 1, 1, 1, 1, 1, 1]);
  const blended = meshHeightAt(3, 3, heights, holed, 0.5, 0.5);
  assert.ok(Math.abs(blended - (2 + 3 + 5) / 3) < 1e-6);
  assert.equal(meshHeightAt(3, 3, heights, holed, 0, 0), 2, "on an invalid corner the nearest valid one is used");
  assert.ok(Number.isNaN(meshHeightAt(3, 3, heights, new Uint8Array(9), 1, 1)));
});

test("heights are interpolated over the diagonal the meshes split their squares along", () => {
  // Only the bottom-right corner is raised. The meshes' diagonal runs from the top-right to
  // the bottom-left corner, so the top-left triangle stays flat and the other is a ramp.
  const heights = Float32Array.from([0, 0, 0, 1]);
  const valid = new Uint8Array(4).fill(1);
  assert.equal(meshHeightAt(2, 2, heights, valid, 0.25, 0.25), 0);
  assert.equal(meshHeightAt(2, 2, heights, valid, 0.75, 0.75), 0.5);
  assert.equal(meshHeightAt(2, 2, heights, valid, 0.8, 0.2), 0, "on the diagonal itself");
});

// ---------------------------------------------------------------- ribbon

const FLAT_GRID = {
  nx: 5,
  ny: 5,
  heights: new Float32Array(25).fill(100),
  valid: new Uint8Array(25).fill(1),
  colUnits: 2,
  rowUnits: 2,
  verticalMetersPerUnit: 10,
  offsetMeters: 20,
  halfWidthUnits: 0.5,
};

test("the ribbon is two vertices a half-width either side of each point, above the surface", () => {
  const line = { points: [0, 2, 4, 2], closed: false };
  const { positions, indices } = buildGroundingRibbon([line], FLAT_GRID);
  assert.equal(positions.length, 4 * 3);
  assert.deepEqual([...indices], [0, 1, 2, 1, 3, 2]);
  // Scene x from -4 to +4 along z = 0, height (100 + 20) / 10.
  assert.deepEqual([...positions.slice(0, 6)], [-4, 12, 0.5, -4, 12, -0.5]);
  assert.deepEqual([...positions.slice(6)], [4, 12, 0.5, 4, 12, -0.5]);
});

test("each edge of the ribbon rides on the surface beneath it, so a slope does not bury one", () => {
  // The surface rises 100 m a row across a line along row 2; the edges sit a quarter of a row
  // either side of it (half-width 0.5 scene units at 2 units a row).
  const heights = new Float32Array(25);
  for (let row = 0; row < 5; row += 1) for (let col = 0; col < 5; col += 1) heights[row * 5 + col] = 100 * row;
  const { positions } = buildGroundingRibbon([{ points: [0, 2, 4, 2], closed: false }], { ...FLAT_GRID, heights });
  const edgeHeights = [positions[1], positions[4]].sort((a, b) => a - b);
  assert.deepEqual(edgeHeights, [(175 + 20) / 10, (225 + 20) / 10]);
});

test("a sharp turn is mitred to at most twice the half-width", () => {
  const spike = { points: [0, 0, 2, 0, 0, 0.1], closed: false };
  const { positions } = buildGroundingRibbon([spike], FLAT_GRID);
  // The turn is the second point, whose vertex pair starts at the third vertex: scene (0, -4).
  const [x, , z] = positions.slice(6, 9);
  assert.ok(Math.abs(Math.hypot(x - 0, z - -4) - 2 * FLAT_GRID.halfWidthUnits) < 1e-6);
});

test("a closed ribbon joins its last point to its first, mitred at the corners", () => {
  const square = { points: [1, 1, 3, 1, 3, 3, 1, 3], closed: true };
  const { positions, indices } = buildGroundingRibbon([square], FLAT_GRID);
  assert.equal(indices.length, 4 * 6);
  // A right-angle corner is mitred to half-width · √2 from the point.
  const [x, , z] = positions;
  assert.ok(Math.abs(Math.hypot(x - -2, z - -2) - 0.5 * Math.SQRT2) < 1e-6);
});

test("the ribbon breaks where the surface it rides on is missing", () => {
  // Columns 2 and 3 have no surface, so the square from column 2 to 3 has no corner to
  // take a height from, while column 3 still takes one from column 4.
  const valid = new Uint8Array(25).fill(1);
  for (let row = 0; row < 5; row += 1) {
    valid[row * 5 + 2] = 0;
    valid[row * 5 + 3] = 0;
  }
  const line = { points: [0, 2, 1, 2, 2, 2, 3, 2, 4, 2], closed: false };
  const { indices } = buildGroundingRibbon([line], { ...FLAT_GRID, valid });
  // Points at columns 0-1 and 3-4 form two runs of one segment each.
  assert.equal(indices.length, 2 * 6);
});

test("the line's length counts every segment, closing segment included", () => {
  const square = { points: [0, 0, 1, 0, 1, 1, 0, 1], closed: true };
  assert.equal(groundingLineLengthKm([square], 10000, -10000), 40);
  assert.equal(groundingLineLengthKm([{ points: [0, 0, 3, 4], closed: false }], 1000, 1000), 5);
});

// ---------------------------------------------------------------- the shipped terrains

function loadTerrainClasses(name) {
  const meta = JSON.parse(readFileSync(resolve(DATA_DIR, `${name}.meta.json`), "utf8"));
  const bytes = readFileSync(resolve(DATA_DIR, `${name}.bin`));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const mask = parseField(meta, buffer, "mask");
  const surface = decodeFieldToFloat32(meta, buffer, "surface");
  const thickness = decodeFieldToFloat32(meta, buffer, "thickness");
  const iceValid = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i += 1) {
    iceValid[i] = Number((mask[i] >= 2 && mask[i] <= 4) && Number.isFinite(surface[i]) && thickness[i] > 0);
  }
  return { meta, nx: meta.grid.nx, ny: meta.grid.ny, classes: classifyTerrainCells(mask, iceValid) };
}

/** Whether (col, row) lies on an edge between a grounded cell and floating ice or ocean, where it is traced. */
function onSeawardEdge({ nx, classes }, col, row) {
  const c = Math.floor(col);
  const r = Math.floor(row);
  const fc = col - c;
  const fr = row - r;
  const at = (cc, rr) => classes[rr * nx + cc];
  if (fc === 0 && fr === 0) return at(c, r) === G;
  if (fr === 0 && fc === 0.5) {
    const pair = [at(c, r), at(c + 1, r)];
    return pair.includes(G) && (pair.includes(F) || pair.includes(O));
  }
  if (fc === 0 && fr === 0.5) {
    const pair = [at(c, r), at(c, r + 1)];
    return pair.includes(G) && (pair.includes(F) || pair.includes(O));
  }
  return false;
}

for (const name of ["bedmachine_antarctica_v4_480", "bedmap3_antarctica_10km", "bedmachine_greenland_v6_3km"]) {
  test(`${name}: every traced point lies where grounded ice meets floating ice or the ocean`, () => {
    const terrain = loadTerrainClasses(name);
    const lines = traceGroundingLine(terrain.nx, terrain.ny, terrain.classes, RAW);
    assert.ok(lines.length > 0);
    let points = 0;
    for (const line of lines) {
      for (const [col, row] of pointsOf(line)) {
        assert.ok(onSeawardEdge(terrain, col, row), `(${col}, ${row}) is not on a seaward edge of grounded ice`);
        points += 1;
      }
    }
    assert.ok(points > 1000);
  });
}

test("BedMachine v4: the grounding line runs past the Ross, Ronne and Amery shelves", () => {
  const terrain = loadTerrainClasses("bedmachine_antarctica_v4_480");
  const { grid } = terrain.meta;
  const lines = traceGroundingLine(terrain.nx, terrain.ny, terrain.classes);
  const near = (xKm, yKm, radiusKm) =>
    lines.some(({ points }) => {
      for (let i = 0; i < points.length; i += 2) {
        const x = (grid.x0_m + points[i] * grid.dx_m) / 1000;
        const y = (grid.y0_m + points[i + 1] * grid.dy_m) / 1000;
        if (Math.hypot(x - xKm, y - yKm) < radiusKm) return true;
      }
      return false;
    });
  // EPSG:3031 positions of points on each shelf's grounding zone, in km.
  assert.ok(near(-350, -650, 120), "Siple Coast, Ross Ice Shelf");
  assert.ok(near(-1000, 250, 150), "Foundation Ice Stream, Ronne Ice Shelf");
  assert.ok(near(1950, 650, 200), "Lambert Glacier, Amery Ice Shelf");
  const lengthKm = groundingLineLengthKm(lines, grid.dx_m, grid.dy_m);
  assert.ok(lengthKm > 30000 && lengthKm < 120000, `a 10 km grounding line of ${Math.round(lengthKm)} km`);
});
