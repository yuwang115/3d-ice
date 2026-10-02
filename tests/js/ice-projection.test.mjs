import assert from "node:assert/strict";
import test from "node:test";

import {
  blendSpeedChangeInto,
  blendThicknessInto,
  buildDomainSampler,
  buildDomainTriangles,
  collectIceTriangles,
  decodeIceProjection,
  DEFAULT_DENSITY_RATIO,
  flotationGeometryInto,
  heightfieldNormalsInto,
  ICE_PROJECTION_GEOMETRY_TYPE,
  locateYear,
  rephaseInto,
  sampleDomainInto,
  sampleSeries,
  thicknessChangeColor,
  thicknessChangeRampColor,
  thicknessChangeScaleT,
} from "../../static/tools/js/ice-projection.js";

const RATIO = 917 / 1028;

/**
 * A 4 x 3 grid whose middle 2 x 2 block plus one extra cell is the domain:
 *
 *   row 0:  .  .  .  .
 *   row 1:  .  D  D  D
 *   row 2:  .  D  D  .
 */
function syntheticPackage({ frameYears = [2015, 2020, 2025], thickness, bed, series, speed } = {}) {
  const nx = 4;
  const ny = 3;
  const mask = new Uint8Array(nx * ny);
  for (const index of [5, 6, 7, 9, 10]) mask[index] = 1;
  const count = 5;
  const frames = thickness || [
    [1000, 400, 0, 800, 100],
    [900, 300, 50, 800, 0],
    [600, 0, 100, 800, 0],
  ];
  const bedValues = bed || [-500, -500, 100, 200, -800];

  const thicknessBytes = frameYears.length * count * 2;
  // Ensemble packages put frame-major speed change between thickness and bed.
  const speedBytes = speed ? frameYears.length * count * 2 : 0;
  const bedOffset = thicknessBytes + speedBytes;
  const bedBytes = count * 2;
  const buffer = new ArrayBuffer(bedOffset + bedBytes + nx * ny);
  const view = new DataView(buffer);
  frames.flat().forEach((value, i) => view.setInt16(i * 2, value, true));
  if (speed) speed.flat().forEach((value, i) => view.setInt16(thicknessBytes + i * 2, value, true));
  bedValues.forEach((value, i) => view.setInt16(bedOffset + i * 2, value, true));
  new Uint8Array(buffer, bedOffset + bedBytes).set(mask);

  const meta = {
    geometry_type: ICE_PROJECTION_GEOMETRY_TYPE,
    grid: { nx, ny, x0_m: 0, y0_m: 0, dx_m: 10000, dy_m: -10000 },
    domain: { cell_count: count },
    frames: { count: frameYears.length, years: frameYears },
    quantization: { unit: "m", scale: 1, offset: 0, int16_fill_value: -32768 },
    physical_constants: { density_ratio: RATIO },
    fields: [
      { name: "thickness", dtype: "int16", byte_offset: 0, byte_length: thicknessBytes },
      { name: "bed", dtype: "int16", byte_offset: bedOffset, byte_length: bedBytes },
      { name: "domain_mask", dtype: "uint8", byte_offset: bedOffset + bedBytes, byte_length: nx * ny },
    ],
  };
  if (speed) {
    meta.fields.splice(1, 0, { name: "speed_change", dtype: "int16", byte_offset: thicknessBytes, byte_length: speedBytes });
  }
  if (series) meta.series = series;
  return { meta, buffer };
}

// ---------------------------------------------------------------- decoding

test("decodeIceProjection maps domain cells to grid cells in row-major order", () => {
  const { meta, buffer } = syntheticPackage();
  const projection = decodeIceProjection(meta, buffer);

  assert.equal(projection.domainCount, 5);
  assert.deepEqual([...projection.cellIndex], [5, 6, 7, 9, 10]);
  assert.equal(projection.vertexOfCell[0], -1);
  assert.equal(projection.vertexOfCell[7], 2);
  assert.equal(projection.vertexOfCell[10], 4);
  assert.deepEqual([...projection.bed], [-500, -500, 100, 200, -800]);
  assert.deepEqual([...projection.frameYears], [2015, 2020, 2025]);
  assert.equal(projection.firstYear, 2015);
  assert.equal(projection.lastYear, 2025);
  assert.ok(Math.abs(projection.densityRatio - RATIO) < 1e-12);
});

test("decodeIceProjection falls back to the standard density ratio", () => {
  const { meta, buffer } = syntheticPackage();
  delete meta.physical_constants;
  assert.equal(decodeIceProjection(meta, buffer).densityRatio, DEFAULT_DENSITY_RATIO);
});

test("decodeIceProjection reads the annual diagnostics when present", () => {
  const { meta, buffer } = syntheticPackage({
    series: {
      years: [2015, 2016, 2017],
      sea_level_contribution_m: [0, 0.1, 0.3],
      grounded_area_km2: [100, 90, 80],
      floating_area_km2: [10, 9, 8],
    },
  });
  const { series } = decodeIceProjection(meta, buffer);
  assert.deepEqual([...series.years], [2015, 2016, 2017]);
  assert.deepEqual([...series.seaLevel], [0, 0.1, 0.3]);
  assert.deepEqual([...series.groundedArea], [100, 90, 80]);
  assert.deepEqual([...series.floatingArea], [10, 9, 8]);
  assert.equal(decodeIceProjection(syntheticPackage().meta, buffer).series, null);
});

test("decodeIceProjection rejects packages that break the layout", () => {
  const good = syntheticPackage();
  assert.throws(
    () => decodeIceProjection({ ...good.meta, geometry_type: "streamlines_3d" }, good.buffer),
    /not an ice-sheet projection/,
  );
  assert.throws(
    () => decodeIceProjection({ ...good.meta, domain: { cell_count: 4 } }, good.buffer),
    /domain/,
  );
  assert.throws(
    () => decodeIceProjection({ ...good.meta, frames: { count: 3, years: [2015, 2015, 2025] } }, good.buffer),
    /increasing/,
  );
  assert.throws(
    () => decodeIceProjection({ ...good.meta, frames: { count: 2, years: [2015, 2020] } }, good.buffer),
    /thickness/,
  );
});

// ---------------------------------------------------------------- time axis

test("locateYear brackets a year between keyframes and clamps outside them", () => {
  const years = Float64Array.of(2015, 2020, 2025);
  assert.deepEqual(locateYear(years, 2015), { lower: 0, upper: 0, weight: 0, year: 2015 });
  assert.deepEqual(locateYear(years, 2017), { lower: 0, upper: 1, weight: 0.4, year: 2017 });
  assert.deepEqual(locateYear(years, 2020), { lower: 1, upper: 1, weight: 0, year: 2020 });
  assert.deepEqual(locateYear(years, 2024.5), { lower: 1, upper: 2, weight: 0.9, year: 2024.5 });
  assert.deepEqual(locateYear(years, 1990), { lower: 0, upper: 0, weight: 0, year: 2015 });
  assert.deepEqual(locateYear(years, 2400), { lower: 2, upper: 2, weight: 0, year: 2025 });
});

test("blendThicknessInto interpolates linearly between the bracketing keyframes", () => {
  const projection = decodeIceProjection(...Object.values(syntheticPackage()));
  const out = new Float32Array(projection.domainCount);

  blendThicknessInto(projection, 2015, out);
  assert.deepEqual([...out], [1000, 400, 0, 800, 100]);
  blendThicknessInto(projection, 2017, out);
  assert.deepEqual([...out], [960, 360, 20, 800, 60]);
  blendThicknessInto(projection, 2025, out);
  assert.deepEqual([...out], [600, 0, 100, 800, 0]);
});

test("sampleSeries interpolates annual values and clamps at the ends", () => {
  const years = Float64Array.of(2015, 2016, 2017);
  const values = Float64Array.of(0, 1, 3);
  assert.equal(sampleSeries(years, values, 2015), 0);
  assert.equal(sampleSeries(years, values, 2016.5), 2);
  assert.equal(sampleSeries(years, values, 1900), 0);
  assert.equal(sampleSeries(years, values, 2100), 3);
  assert.ok(Number.isNaN(sampleSeries(new Float64Array(0), new Float64Array(0), 2015)));
});

// ---------------------------------------------------------------- geometry

test("flotationGeometryInto grounds heavy ice, floats light ice, and collapses ice-free cells", () => {
  const bed = Float32Array.of(-500, -500, 200, -300);
  const thickness = Float32Array.of(1000, 400, 0, 0);
  const surface = new Float32Array(4);
  const base = new Float32Array(4);
  const grounded = new Uint8Array(4);
  flotationGeometryInto(bed, thickness, RATIO, surface, base, grounded);

  assert.deepEqual([...grounded], [1, 0, 0, 0]);
  assert.ok(Math.abs(surface[0] - 500) < 1e-3 && Math.abs(base[0] + 500) < 1e-3);
  assert.ok(Math.abs(base[1] + 400 * RATIO) < 1e-3);
  assert.ok(Math.abs(surface[1] - 400 * (1 - RATIO)) < 1e-3);
  assert.equal(surface[2], 200);
  assert.equal(base[2], 200);
  assert.equal(surface[3], 0);
  assert.equal(base[3], 0);
});

test("buildDomainTriangles keeps only triangles whose corners are all domain cells", () => {
  const projection = decodeIceProjection(...Object.values(syntheticPackage()));
  const triangles = buildDomainTriangles(projection);
  // Quads (1,1)-(2,2) yield both triangles; quad (2,1)-(3,2) keeps only its upper-left
  // triangle (cells 6, 10, 7); every other quad touches a non-domain cell.
  assert.deepEqual([...triangles], [0, 3, 1, 1, 3, 4, 1, 4, 2]);
});

test("collectIceTriangles drops any triangle with an ice-free corner", () => {
  const triangles = Uint32Array.of(0, 3, 1, 1, 3, 4, 1, 4, 2);
  const out = new Uint32Array(triangles.length);
  const count = collectIceTriangles(triangles, Uint8Array.of(1, 1, 0, 1, 1), out);
  assert.equal(count, 6);
  assert.deepEqual([...out.subarray(0, count)], [0, 3, 1, 1, 3, 4]);
});

test("heightfieldNormalsInto gives an upward unit normal tilted against the slope", () => {
  const projection = decodeIceProjection(...Object.values(syntheticPackage()));
  // Height rises by 1 unit per column: dh/dx = 1 / dx.
  const heights = Float32Array.of(0, 1, 2, 0, 1);
  const normals = new Float32Array(projection.domainCount * 3);
  heightfieldNormalsInto(projection, heights, 1, 1, normals, null);

  const [nx, ny, nz] = normals.subarray(3, 6); // cell 6: neighbours on both sides in x
  assert.ok(Math.abs(Math.hypot(nx, ny, nz) - 1) < 1e-6);
  assert.ok(Math.abs(nx + Math.SQRT1_2) < 1e-6);
  assert.ok(Math.abs(ny - Math.SQRT1_2) < 1e-6);
  assert.ok(Math.abs(nz) < 1e-6);
});

test("heightfieldNormalsInto ignores ice-free neighbours when a mask is given", () => {
  const projection = decodeIceProjection(...Object.values(syntheticPackage()));
  const heights = Float32Array.of(5, 5, -100, 5, 5);
  const normals = new Float32Array(projection.domainCount * 3);
  heightfieldNormalsInto(projection, heights, 1, 1, normals, Uint8Array.of(1, 1, 0, 1, 1));
  assert.deepEqual([...normals.subarray(3, 6)], [0, 1, 0]);
});

// ---------------------------------------------------------------- colour

test("thicknessChangeScaleT is centred on no change and saturates at the range", () => {
  assert.equal(thicknessChangeScaleT(0), 0.5);
  assert.ok(thicknessChangeScaleT(-10) < 0.5 && thicknessChangeScaleT(10) > 0.5);
  assert.ok(Math.abs(thicknessChangeScaleT(-50) + thicknessChangeScaleT(50) - 1) < 1e-12);
  assert.equal(thicknessChangeScaleT(-1000), 0);
  assert.equal(thicknessChangeScaleT(5000), 1);
  assert.ok(thicknessChangeScaleT(-100) < thicknessChangeScaleT(-10));
  assert.equal(thicknessChangeScaleT(Number.NaN), 0.5);
});

test("thicknessChangeColor is red for thinning, blue for thickening, near white at zero", () => {
  const [thinR, , thinB] = thicknessChangeColor(-500);
  const [thickR, , thickB] = thicknessChangeColor(500);
  const neutral = thicknessChangeColor(0);
  assert.ok(thinR > thinB);
  assert.ok(thickB > thickR);
  assert.ok(neutral.every((channel) => channel > 0.9));
});

test("thicknessChangeRampColor clamps its input and matches the per-value colour", () => {
  assert.deepEqual(thicknessChangeRampColor(-1), thicknessChangeRampColor(0));
  assert.deepEqual(thicknessChangeRampColor(2), thicknessChangeRampColor(1));
  assert.deepEqual(thicknessChangeRampColor(0.5), thicknessChangeColor(0));
  assert.deepEqual(thicknessChangeRampColor(thicknessChangeScaleT(-300)), thicknessChangeColor(-300));
});

// ---------------------------------------------------------------- ensemble packages

const SPEED = [
  [0, 0, 0, 0, 0],
  [100, -40, 0, 20, 0],
  [300, -80, -32768, 60, 0],
];

test("decodeIceProjection reads an optional speed-change field and the model spread", () => {
  const series = {
    years: [2015, 2016],
    sea_level_contribution_m: [0, 0.01],
    sea_level_contribution_min_m: [0, -0.02],
    sea_level_contribution_max_m: [0, 0.05],
  };
  const projection = decodeIceProjection(...Object.values(syntheticPackage({ speed: SPEED, series })));
  assert.equal(projection.speedChange.frames.length, 15);
  assert.deepEqual([...projection.series.seaLevelMin], [0, -0.02]);
  assert.deepEqual([...projection.series.seaLevelMax], [0, 0.05]);
  assert.equal(projection.series.groundedArea.length, 0);
  assert.equal(decodeIceProjection(...Object.values(syntheticPackage())).speedChange, null);
});

test("decodeIceProjection rejects speed-change frames of the wrong size", () => {
  const { meta, buffer } = syntheticPackage({ speed: SPEED });
  const field = meta.fields.find((f) => f.name === "speed_change");
  field.byte_length -= 2;
  assert.throws(() => decodeIceProjection(meta, buffer), /speed-change/);
});

test("blendSpeedChangeInto interpolates, reads the fill value as no change, and is zero without the field", () => {
  const projection = decodeIceProjection(...Object.values(syntheticPackage({ speed: SPEED })));
  const out = new Float32Array(5);
  const bracket = blendSpeedChangeInto(projection, 2017.5, out);
  assert.equal(bracket.lower, 0);
  assert.deepEqual([...out], [50, -20, 0, 10, 0]);
  blendSpeedChangeInto(projection, 2025, out);
  assert.deepEqual([...out], [300, -80, 0, 60, 0]);

  const plain = decodeIceProjection(...Object.values(syntheticPackage()));
  out.fill(7);
  blendSpeedChangeInto(plain, 2020, out);
  assert.deepEqual([...out], [0, 0, 0, 0, 0]);
});

test("sampleDomainInto blends four ice-covered corners and refuses partly ice-free cells", () => {
  const projection = decodeIceProjection(...Object.values(syntheticPackage()));
  // Domain vertices: cell 5 -> 0, 6 -> 1, 7 -> 2, 9 -> 3, 10 -> 4 (cols 1-3 of row 1, cols 1-2 of row 2).
  const sampler = buildDomainSampler(projection, [1.5, 2.5, 1, 0.5, 2], [1.5, 1.5, 2, 1, 1]);
  const values = Float32Array.from([10, 20, 30, 40, 50]);
  const out = new Float32Array(5);
  const covered = new Uint8Array(5);

  sampleDomainInto(sampler, values, Uint8Array.from([1, 1, 1, 1, 1]), out, covered);
  assert.ok(Math.abs(out[0] - 30) < 1e-6); // mean of 10, 20, 40, 50
  assert.equal(covered[1], 0); // cell 11 is outside the domain
  assert.ok(Number.isNaN(out[1]));
  assert.equal(out[2], 40); // exactly on a domain cell: zero-weight corners are ignored
  assert.equal(covered[3], 0); // half way to col 0, which is outside the domain
  assert.equal(out[4], 20);

  sampleDomainInto(sampler, values, Uint8Array.from([1, 0, 1, 1, 1]), out, covered);
  assert.equal(covered[0], 0); // one corner has lost its ice
  assert.equal(covered[2], 1);

  // With a majority rule the point keeps its three iced corners, renormalised.
  sampleDomainInto(sampler, values, Uint8Array.from([1, 0, 1, 1, 1]), out, covered, 0.5);
  assert.equal(covered[0], 1);
  assert.ok(Math.abs(out[0] - (10 + 40 + 50) / 3) < 1e-5);
  assert.equal(covered[3], 1); // half its weight sits on domain cell 5, which has ice
  assert.equal(out[3], 10);
  sampleDomainInto(sampler, values, Uint8Array.from([0, 0, 1, 1, 1]), out, covered, 0.5);
  assert.equal(covered[0], 1); // exactly half its weight left
  assert.equal(covered[3], 0);
});

test("rephaseInto keeps the pulse phase continuous across a rate change", () => {
  const scale = 3;
  const distance = 0.37;
  const time = 41.25;
  const oldRates = Float32Array.from([1.2, 0.8]);
  const newRates = Float32Array.from([2.0, 0.62]);
  const offsets = Float32Array.from([0.1, 0.9]);
  const phase = (rate, offset) => {
    const value = distance * scale - time * rate - offset;
    return value - Math.floor(value);
  };
  const before = [0, 1].map((i) => phase(oldRates[i], offsets[i]));
  rephaseInto(offsets, oldRates, newRates, time);
  const after = [0, 1].map((i) => phase(newRates[i], offsets[i]));
  for (let i = 0; i < 2; i += 1) {
    const gap = Math.abs(before[i] - after[i]);
    assert.ok(Math.min(gap, 1 - gap) < 1e-4, `phase ${i} jumped by ${gap}`);
    assert.ok(offsets[i] >= 0 && offsets[i] < 1);
  }
});
