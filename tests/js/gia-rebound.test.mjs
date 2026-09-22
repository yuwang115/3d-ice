import assert from "node:assert/strict";
import test from "node:test";

import {
  angularWavenumbers,
  createFft2dPlan,
  nextPowerOfTwo,
  transformFft2dInPlace,
} from "../../static/tools/js/fft2d.js";
import {
  bicubicUpsampleInto,
  blockMean,
  buildSubCellHypsometry,
  chooseCoarseningFactor,
  floodConnectedOcean,
  meanSubmergedDepth,
  polarStereographicCellAreaM2,
} from "../../static/tools/js/gia-grid.js";
import {
  ANTARCTIC_STANDARD_PARALLEL_DEGREES,
  applyReboundFraction,
  FRESHWATER_DENSITY_KG_M3,
  computeBedLoadPascals,
  DEFAULT_FLEXURAL_RIGIDITY_N_M,
  deriveReboundedIceSurfaces,
  farFieldUpliftMeters,
  flexuralLengthScaleMeters,
  GRAVITY_M_PER_S2,
  ICE_DENSITY_KG_M3,
  isFloatingIceMask,
  isGroundedIceMask,
  MANTLE_DENSITY_KG_M3,
  MASK_FLOATING_ICE,
  MASK_GROUNDED_ICE,
  MASK_ICE_FREE_LAND,
  MASK_OCEAN,
  MASK_SUBGLACIAL_LAKE,
  reboundElapsedYears,
  REBOUND_MODEL_FLEXURAL,
  REBOUND_MODEL_LOCAL,
  REBOUND_RELAXATION_TIME_YEARS,
  SEAWATER_DENSITY_KG_M3,
  solveIsostaticRebound,
} from "../../static/tools/js/gia-rebound.js";

// ---------------------------------------------------------------- fft2d

test("nextPowerOfTwo rounds up to a power of two", () => {
  assert.equal(nextPowerOfTwo(1), 1);
  assert.equal(nextPowerOfTwo(2), 2);
  assert.equal(nextPowerOfTwo(3), 4);
  assert.equal(nextPowerOfTwo(512), 512);
  assert.equal(nextPowerOfTwo(513), 1024);
});

test("createFft2dPlan rejects non-power-of-two axes", () => {
  assert.throws(() => createFft2dPlan(6, 8), /power of two/);
  assert.throws(() => createFft2dPlan(8, 0), /power of two/);
});

test("forward then inverse transform is the identity", () => {
  const plan = createFft2dPlan(16, 8);
  const original = new Float64Array(plan.real.length);
  for (let index = 0; index < original.length; index += 1) {
    original[index] = Math.sin(index * 0.7) * 3 + Math.cos(index * 0.13);
    plan.real[index] = original[index];
  }
  transformFft2dInPlace(plan, { inverse: false });
  transformFft2dInPlace(plan, { inverse: true });

  for (let index = 0; index < original.length; index += 1) {
    assert.ok(Math.abs(plan.real[index] - original[index]) < 1e-9);
    assert.ok(Math.abs(plan.imaginary[index]) < 1e-9);
  }
});

test("the zero wavenumber bin holds the sum of a real field", () => {
  const plan = createFft2dPlan(8, 8);
  let expected = 0;
  for (let index = 0; index < plan.real.length; index += 1) {
    plan.real[index] = index % 5;
    expected += index % 5;
  }
  transformFft2dInPlace(plan, { inverse: false });
  assert.ok(Math.abs(plan.real[0] - expected) < 1e-9);
});

test("a single cosine maps onto exactly two conjugate bins", () => {
  const size = 16;
  const plan = createFft2dPlan(size, size);
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column < size; column += 1) {
      plan.real[row * size + column] = Math.cos((2 * Math.PI * 3 * column) / size);
    }
  }
  transformFft2dInPlace(plan, { inverse: false });

  const magnitude = (row, column) => {
    const index = row * size + column;
    return Math.hypot(plan.real[index], plan.imaginary[index]);
  };
  // Row 0 (zero ky) holds the signal; +3 and -3 in kx each carry half the energy.
  assert.ok(Math.abs(magnitude(0, 3) - (size * size) / 2) < 1e-6);
  assert.ok(Math.abs(magnitude(0, size - 3) - (size * size) / 2) < 1e-6);
  assert.ok(magnitude(0, 0) < 1e-6);
  assert.ok(magnitude(1, 3) < 1e-6);
});

test("angularWavenumbers wraps negative frequencies after Nyquist", () => {
  const spacing = 1000;
  const wavenumbers = angularWavenumbers(8, spacing);
  assert.equal(wavenumbers[0], 0);
  assert.ok(Math.abs(wavenumbers[1] - (2 * Math.PI) / (8 * spacing)) < 1e-18);
  assert.ok(wavenumbers[5] < 0, "bin 5 of 8 is a negative frequency");
  assert.ok(Math.abs(wavenumbers[1] + wavenumbers[7]) < 1e-18);
});

// ---------------------------------------------------------------- parameters

test("the flexural length scale matches the ELRA default", () => {
  const lengthScale = flexuralLengthScaleMeters(DEFAULT_FLEXURAL_RIGIDITY_N_M);
  // (1e25 / (3300 * 9.81))^(1/4) = 132.6 km
  assert.ok(Math.abs(lengthScale / 1000 - 132.6) < 0.1);
  assert.equal(flexuralLengthScaleMeters(0), 0);
  assert.equal(flexuralLengthScaleMeters(Number.NaN), 0);
});

test("reboundElapsedYears inverts exponential relaxation", () => {
  assert.equal(reboundElapsedYears(0), 0);
  assert.equal(reboundElapsedYears(1), Number.POSITIVE_INFINITY);
  // One relaxation time reaches 1 - 1/e of the final uplift.
  const oneTau = reboundElapsedYears(1 - Math.exp(-1));
  assert.ok(Math.abs(oneTau - REBOUND_RELAXATION_TIME_YEARS) < 1e-6);
  assert.ok(Math.abs(reboundElapsedYears(0.95) - 3 * REBOUND_RELAXATION_TIME_YEARS) < 20);
});

test("farFieldUpliftMeters subsides the ocean floor under added meltwater", () => {
  assert.equal(farFieldUpliftMeters(0), 0);
  const expected = (-SEAWATER_DENSITY_KG_M3 * 58) / (MANTLE_DENSITY_KG_M3 - SEAWATER_DENSITY_KG_M3);
  assert.ok(Math.abs(farFieldUpliftMeters(58) - expected) < 1e-9);
  assert.ok(farFieldUpliftMeters(58) < 0);
});

test("mask helpers treat the subglacial lake as grounded ice", () => {
  assert.ok(isGroundedIceMask(MASK_GROUNDED_ICE));
  assert.ok(isGroundedIceMask(MASK_SUBGLACIAL_LAKE));
  assert.ok(!isGroundedIceMask(MASK_FLOATING_ICE));
  assert.ok(isFloatingIceMask(MASK_FLOATING_ICE));
  assert.ok(!isFloatingIceMask(MASK_GROUNDED_ICE));
});

// ---------------------------------------------------------------- load

test("computeBedLoadPascals case-splits the overburden by mask", () => {
  const cellCount = 5;
  const load = computeBedLoadPascals({
    cellCount,
    //            grounded  floating  ocean   dry land  no-data
    bedHeights: Float32Array.from([500, -800, -1200, 300, Number.NaN]),
    surfaceHeights: Float32Array.from([2500, 100, 0, 300, Number.NaN]),
    thickness: Float32Array.from([2000, 900, 0, 0, 500]),
    mask: Uint8Array.from([
      MASK_GROUNDED_ICE,
      MASK_FLOATING_ICE,
      MASK_OCEAN,
      MASK_ICE_FREE_LAND,
      MASK_OCEAN,
    ]),
  });

  // Grounded ice: rho_i g H.
  assert.ok(Math.abs(load[0] - ICE_DENSITY_KG_M3 * GRAVITY_M_PER_S2 * 2000) < 1e-6);
  // Floating ice: the hydrostatic-equivalent water column, i.e. the same load an open
  // ocean cell of the same depth would apply.
  assert.ok(Math.abs(load[1] - SEAWATER_DENSITY_KG_M3 * GRAVITY_M_PER_S2 * 800) < 1e-6);
  // Open ocean: water column only.
  assert.ok(Math.abs(load[2] - SEAWATER_DENSITY_KG_M3 * GRAVITY_M_PER_S2 * 1200) < 1e-6);
  // Dry land above sea level: nothing.
  assert.equal(load[3], 0);
  // Non-finite bed contributes nothing even though thickness is non-zero.
  assert.equal(load[4], 0);
});

test("a hydrostatically floating shelf loads the bed exactly like the water it displaces", () => {
  const bed = -900;
  const thickness = 500;
  const surface = thickness * (1 - ICE_DENSITY_KG_M3 / SEAWATER_DENSITY_KG_M3);
  const shelf = computeBedLoadPascals({
    cellCount: 1,
    bedHeights: Float32Array.from([bed]),
    surfaceHeights: Float32Array.from([surface]),
    thickness: Float32Array.from([thickness]),
    mask: Uint8Array.from([MASK_FLOATING_ICE]),
  });
  const openWater = computeBedLoadPascals({
    cellCount: 1,
    bedHeights: Float32Array.from([bed]),
    surfaceHeights: Float32Array.from([0]),
    thickness: Float32Array.from([0]),
    mask: Uint8Array.from([MASK_OCEAN]),
  });
  assert.equal(shelf[0], openWater[0]);
});

test("a subglacial lake adds its own water column to the bed load", () => {
  // Over Lake Vostok the ice floats on the lake and the lake rests on the bed, so the
  // bed carries both columns. The ice base sits 800 m above a bed at -1400 m.
  const thickness = 4000;
  const bed = -1400;
  const iceBase = -600;
  const load = computeBedLoadPascals({
    cellCount: 2,
    bedHeights: Float32Array.from([bed, bed]),
    surfaceHeights: Float32Array.from([iceBase + thickness, iceBase + thickness]),
    thickness: Float32Array.from([thickness, thickness]),
    mask: Uint8Array.from([MASK_SUBGLACIAL_LAKE, MASK_GROUNDED_ICE]),
  });

  const iceLoad = ICE_DENSITY_KG_M3 * GRAVITY_M_PER_S2 * thickness;
  const lakeLoad = FRESHWATER_DENSITY_KG_M3 * GRAVITY_M_PER_S2 * (iceBase - bed);
  assert.ok(Math.abs(load[0] - (iceLoad + lakeLoad)) < 1e-6);
  // Ordinary grounded ice carries no lake term even with the same geometry.
  assert.ok(Math.abs(load[1] - iceLoad) < 1e-6);
  assert.ok(load[0] > load[1]);
});

test("a lake with its ice base below the bed contributes no water column", () => {
  const load = computeBedLoadPascals({
    cellCount: 1,
    bedHeights: Float32Array.from([-500]),
    surfaceHeights: Float32Array.from([2400]),
    thickness: Float32Array.from([3000]),
    mask: Uint8Array.from([MASK_SUBGLACIAL_LAKE]),
  });
  assert.ok(Math.abs(load[0] - ICE_DENSITY_KG_M3 * GRAVITY_M_PER_S2 * 3000) < 1e-6);
});

test("floodConnectedOcean conducts through no-data cells instead of damming", () => {
  // A 5x5 grid whose entire border is no-data (Bedmap3 leaves 97 % of its domain edge
  // this way) with a single marine basin at the centre.
  const size = 5;
  const bed = new Float64Array(size * size).fill(500);
  const valid = new Uint8Array(size * size).fill(1);
  for (let index = 0; index < size * size; index += 1) {
    const row = (index / size) | 0;
    const column = index % size;
    if (row === 0 || column === 0 || row === size - 1 || column === size - 1) {
      valid[index] = 0;
      bed[index] = 0; // what blockMean reports for an all-gap cell
    }
  }
  const centre = 2 * size + 2;
  bed[centre] = -400;
  bed[2 * size + 1] = -100; // a channel out to the no-data ring
  const uplift = new Float64Array(size * size);

  // Without a validity mask the no-data ring reads as land at 0 m and seals the basin.
  const sealed = floodConnectedOcean({ coarseX: size, coarseY: size, bed, uplift, seaLevelMeters: 0 });
  assert.equal(sealed[centre], 0, "the fill cannot start when the border reads as land");

  // With it, the ring conducts and the basin is correctly marine.
  const open = floodConnectedOcean({
    coarseX: size,
    coarseY: size,
    bed,
    uplift,
    seaLevelMeters: 0,
    valid,
  });
  assert.equal(open[centre], 1, "a basin draining to no-data terrain is open ocean");
  assert.equal(open[2 * size + 3], 0, "dry ground stays dry");
});

test("blockMean reports which coarse cells have any data at all", () => {
  const source = Float32Array.from([
    1, 2, Number.NaN, Number.NaN,
    3, 4, Number.NaN, Number.NaN,
    Number.NaN, Number.NaN, 5, 6,
    Number.NaN, Number.NaN, 7, 8,
  ]);
  const { data, coverage } = blockMean(source, 4, 4, 2);
  assert.deepEqual(Array.from(coverage), [1, 0, 0, 1]);
  assert.ok(Math.abs(data[0] - 2.5) < 1e-9);
  assert.equal(data[1], 0, "an all-gap block reports zero, which coverage disambiguates");
});

test("sub-cell hypsometry integrates depth exactly over a coarse cell", () => {
  // One 2x2 coarse cell whose four sub-cells sit at -300, -100, 100 and 300 m.
  const bed = Float32Array.from([-300, -100, 100, 300]);
  const hypsometry = buildSubCellHypsometry(bed, 2, 2, 2);
  assert.equal(hypsometry.coarseCount, 1);

  const brute = (level) => {
    let total = 0;
    for (const value of bed) total += Math.max(0, level - value);
    return total / bed.length;
  };
  for (const level of [-500, -300, -250, -100, 0, 50, 100, 300, 400, 1000]) {
    const exact = brute(level);
    assert.ok(
      Math.abs(meanSubmergedDepth(hypsometry, 0, level) - exact) < 1e-6,
      `level ${level}: expected ${exact}`
    );
  }
});

test("sub-cell hypsometry beats the coarse-mean shortcut on straddling bathymetry", () => {
  const bed = Float32Array.from([-300, -100, 100, 300]);
  const hypsometry = buildSubCellHypsometry(bed, 2, 2, 2);
  const coarseMean = (-300 + -100 + 100 + 300) / 4; // exactly 0
  const level = 0;

  // The coarse mean says the cell is exactly at the waterline, so it carries no water.
  assert.equal(Math.max(0, level - coarseMean), 0);
  // In reality half the cell is submerged, averaging 100 m of water.
  assert.ok(Math.abs(meanSubmergedDepth(hypsometry, 0, level) - 100) < 1e-6);
});

test("sub-cell hypsometry skips no-data cells and handles empty coarse cells", () => {
  const bed = Float32Array.from([Number.NaN, -200, Number.NaN, Number.NaN]);
  const hypsometry = buildSubCellHypsometry(bed, 2, 2, 2);
  // Only the single finite value counts, so a waterline at 0 gives its full depth.
  assert.ok(Math.abs(meanSubmergedDepth(hypsometry, 0, 0) - 200) < 1e-6);

  const allGaps = buildSubCellHypsometry(new Float32Array(4).fill(Number.NaN), 2, 2, 2);
  assert.equal(meanSubmergedDepth(allGaps, 0, 0), 0);
});

test("sub-cell hypsometry sorts each coarse cell independently", () => {
  // 4x2 grid, factor 2 -> two coarse cells with deliberately interleaved values.
  const bed = Float32Array.from([50, -50, -400, -200, 150, 250, -600, -100]);
  const hypsometry = buildSubCellHypsometry(bed, 4, 2, 2);
  assert.equal(hypsometry.coarseCount, 2);

  const groups = [
    [50, -50, 150, 250],
    [-400, -200, -600, -100],
  ];
  for (let cell = 0; cell < 2; cell += 1) {
    const expected = groups[cell].reduce((sum, value) => sum + Math.max(0, 0 - value), 0) / 4;
    assert.ok(Math.abs(meanSubmergedDepth(hypsometry, cell, 0) - expected) < 1e-6, `cell ${cell}`);
  }
});

// ---------------------------------------------------------------- resampling

test("blockMean averages full blocks and ignores non-finite cells", () => {
  const source = Float32Array.from([1, 2, 3, 4, 5, Number.NaN, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
  const { data, coarseX, coarseY } = blockMean(source, 4, 4, 2);
  assert.equal(coarseX, 2);
  assert.equal(coarseY, 2);
  assert.ok(Math.abs(data[0] - (1 + 2 + 5) / 3) < 1e-9);
  assert.ok(Math.abs(data[1] - (3 + 4 + 7 + 8) / 4) < 1e-9);
  assert.ok(Math.abs(data[2] - (9 + 10 + 13 + 14) / 4) < 1e-9);
});

test("blockMean reports zero for an all-gap block rather than NaN", () => {
  const { data } = blockMean(Float32Array.from([Number.NaN, Number.NaN, Number.NaN, Number.NaN]), 2, 2, 2);
  assert.equal(data[0], 0);
});

test("bicubicUpsampleInto reproduces a constant field and stays within range", () => {
  const coarse = Float64Array.from([5, 5, 5, 5]);
  const out = new Float32Array(16);
  bicubicUpsampleInto(coarse, 2, 2, 4, 4, 2, out);
  for (const value of out) assert.ok(Math.abs(value - 5) < 1e-6);

  const ramp = Float64Array.from([0, 10, 0, 10]);
  bicubicUpsampleInto(ramp, 2, 2, 4, 4, 2, out);
  // Monotone across a row of the upsampled ramp.
  assert.ok(out[0] <= out[1] && out[1] <= out[2] && out[2] <= out[3]);
  // A cubic kernel rings at a step discontinuity, bounded at ~1.25x the jump. That is
  // acceptable here because the field being upsampled is the output of a low-pass
  // flexural operator, so neighbouring coarse values differ by metres, not by the full
  // range - the next assertion pins that down.
  for (const value of out) assert.ok(value >= -2.5 && value <= 12.5);
});

test("bicubicUpsampleInto barely overshoots a smooth field", () => {
  // Mimic the real input: a smooth, band-limited uplift field.
  const coarseX = 16;
  const coarseY = 16;
  const coarse = new Float64Array(coarseX * coarseY);
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (let row = 0; row < coarseY; row += 1) {
    for (let column = 0; column < coarseX; column += 1) {
      const value = 500 * Math.exp(-((column - 8) ** 2 + (row - 8) ** 2) / 40);
      coarse[row * coarseX + column] = value;
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
  }
  const factor = 5;
  const out = new Float32Array(coarseX * factor * coarseY * factor);
  bicubicUpsampleInto(coarse, coarseX, coarseY, coarseX * factor, coarseY * factor, factor, out);
  const range = maximum - minimum;
  for (const value of out) {
    assert.ok(value >= minimum - 0.005 * range && value <= maximum + 0.005 * range);
  }
});

test("polarStereographicCellAreaM2 corrects for the point scale factor", () => {
  const dxMeters = 10000;
  const dyMeters = -10000;
  const projected = 1e8;
  const atParallel = ({ xMeters, yMeters }) =>
    polarStereographicCellAreaM2({
      xMeters,
      yMeters,
      dxMeters,
      dyMeters,
      standardParallelDegrees: ANTARCTIC_STANDARD_PARALLEL_DEGREES,
    });

  // At the pole (radius 0) the scale factor is (1 + sin 71) / 2 < 1, so true area exceeds
  // the projected area.
  const pole = atParallel({ xMeters: 0, yMeters: 0 });
  const standardSine = Math.sin((71 * Math.PI) / 180);
  assert.ok(Math.abs(pole - projected / ((1 + standardSine) / 2) ** 2) < 1);
  assert.ok(pole > projected);

  // On the standard parallel the scale factor is exactly 1.
  const radiusAtStandardParallel =
    6371000 * (1 + standardSine) * Math.tan(Math.PI / 4 - (71 * Math.PI) / 360);
  const onParallel = atParallel({ xMeters: radiusAtStandardParallel, yMeters: 0 });
  assert.ok(Math.abs(onParallel - projected) / projected < 1e-9);

  // Further from the pole the cell shrinks relative to the projection.
  assert.ok(atParallel({ xMeters: 3000000, yMeters: 0 }) < projected);
});

test("floodConnectedOcean separates a closed basin from the open ocean", () => {
  const coarseX = 7;
  const coarseY = 7;
  const bed = new Float64Array(coarseX * coarseY).fill(500);
  // A ring of high ground at radius 2 encloses a deep hollow at the centre.
  bed[3 * coarseX + 3] = -800;
  // The whole outer edge is deep ocean.
  for (let index = 0; index < coarseX; index += 1) {
    bed[index] = -3000;
    bed[(coarseY - 1) * coarseX + index] = -3000;
  }
  for (let row = 0; row < coarseY; row += 1) {
    bed[row * coarseX] = -3000;
    bed[row * coarseX + coarseX - 1] = -3000;
  }

  const connected = floodConnectedOcean({
    coarseX,
    coarseY,
    bed,
    uplift: new Float64Array(coarseX * coarseY),
    seaLevelMeters: 0,
  });
  assert.equal(connected[0], 1, "the domain edge seeds the open ocean");
  assert.equal(connected[3 * coarseX + 3], 0, "a land-locked hollow is not marine");
  assert.equal(connected[2 * coarseX + 2], 0, "high ground is never marine");

  // Breach the ring and the hollow joins the ocean.
  bed[1 * coarseX + 3] = -100;
  bed[2 * coarseX + 3] = -100;
  const breached = floodConnectedOcean({
    coarseX,
    coarseY,
    bed,
    uplift: new Float64Array(coarseX * coarseY),
    seaLevelMeters: 0,
  });
  assert.equal(breached[3 * coarseX + 3], 1, "a breached basin drains to the ocean");
});

test("chooseCoarseningFactor keeps the padded transform inside the preferred axis", () => {
  const lengthScale = flexuralLengthScaleMeters(DEFAULT_FLEXURAL_RIGIDITY_N_M);
  // Antarctica 4 km: the 16 km target would need a 1024 transform, so it steps to 20 km.
  const antarcticaHd = chooseCoarseningFactor(1667, 1667, 4000, -4000, lengthScale);
  assert.ok(antarcticaHd >= 4 && antarcticaHd <= 7);
  assert.ok(Math.ceil(1667 / antarcticaHd) <= 512);

  // Antarctica 10 km lands on the 20 km solve grid.
  assert.equal(chooseCoarseningFactor(667, 667, 10000, -10000, lengthScale), 2);

  // Never coarsens below the native spacing when that is already coarse enough.
  assert.equal(chooseCoarseningFactor(64, 64, 20000, -20000, lengthScale), 1);
});

// ---------------------------------------------------------------- solver

function uniformSlab({ nx, ny, thickness, bed, mask }) {
  const cellCount = nx * ny;
  return {
    nx,
    ny,
    cellCount,
    grid: { nx, ny, x0_m: 0, y0_m: 0, dx_m: 20000, dy_m: -20000 },
    bedHeights: new Float32Array(cellCount).fill(bed),
    surfaceHeights: new Float32Array(cellCount).fill(bed + thickness),
    thickness: new Float32Array(cellCount).fill(thickness),
    mask: new Uint8Array(cellCount).fill(mask),
  };
}

test("local isostasy reproduces the analytic Airy uplift for a grounded slab above sea level", () => {
  const field = uniformSlab({ nx: 32, ny: 32, thickness: 2000, bed: 1500, mask: MASK_GROUNDED_ICE });
  const { uplift, stats } = solveIsostaticRebound({ ...field, model: REBOUND_MODEL_LOCAL });
  const expected = (ICE_DENSITY_KG_M3 / MANTLE_DENSITY_KG_M3) * 2000;
  for (const value of uplift) assert.ok(Math.abs(value - expected) < 1e-3);
  assert.equal(stats.model, REBOUND_MODEL_LOCAL);
  assert.equal(stats.flexuralLengthScaleKm, 0);
  assert.ok(Math.abs(stats.maxUpliftMeters - expected) < 1e-3);
});

test("flexural isostasy matches Airy in the interior of a wide uniform load", () => {
  // A slab many flexural length scales across has no curvature in its interior, so the
  // plate term vanishes and both models must agree there.
  const field = uniformSlab({ nx: 128, ny: 128, thickness: 2000, bed: 1500, mask: MASK_GROUNDED_ICE });
  const { uplift } = solveIsostaticRebound({ ...field, model: REBOUND_MODEL_FLEXURAL });
  const expected = (ICE_DENSITY_KG_M3 / MANTLE_DENSITY_KG_M3) * 2000;
  const centre = uplift[64 * 128 + 64];
  // This synthetic load runs right to the domain boundary, where zero padding truncates
  // the Kelvin kernel's outer tail; at 9.7 flexural length scales that leaves a deficit
  // of <0.2 %. Real ice sheets are ringed by ocean, so the effect does not arise there.
  assert.ok(Math.abs(centre - expected) / expected < 0.003, `centre ${centre} vs Airy ${expected}`);
});

test("removing a hydrostatically floating ice shelf produces no rebound", () => {
  const thickness = 600;
  const cellCount = 32 * 32;
  const surface = thickness * (1 - ICE_DENSITY_KG_M3 / SEAWATER_DENSITY_KG_M3);
  const { uplift, stats } = solveIsostaticRebound({
    nx: 32,
    ny: 32,
    cellCount,
    grid: { nx: 32, ny: 32, x0_m: 0, y0_m: 0, dx_m: 20000, dy_m: -20000 },
    bedHeights: new Float32Array(cellCount).fill(-1200),
    surfaceHeights: new Float32Array(cellCount).fill(surface),
    thickness: new Float32Array(cellCount).fill(thickness),
    mask: new Uint8Array(cellCount).fill(MASK_FLOATING_ICE),
    model: REBOUND_MODEL_LOCAL,
  });
  for (const value of uplift) assert.ok(Math.abs(value) < 1e-3, `expected ~0 uplift, saw ${value}`);
  assert.ok(Math.abs(stats.maxUpliftMeters) < 1e-3);
  // A shelf holds no volume above flotation, so it contributes no sea-level equivalent.
  assert.ok(Math.abs(stats.sleMeters) < 1e-9);
});

test("a grounded marine slab rebounds less than a dry one because the basin floods", () => {
  const dry = solveIsostaticRebound({
    ...uniformSlab({ nx: 32, ny: 32, thickness: 2000, bed: 1500, mask: MASK_GROUNDED_ICE }),
    model: REBOUND_MODEL_LOCAL,
  });
  const marine = solveIsostaticRebound({
    ...uniformSlab({ nx: 32, ny: 32, thickness: 2000, bed: -1500, mask: MASK_GROUNDED_ICE }),
    model: REBOUND_MODEL_LOCAL,
  });
  assert.ok(marine.stats.maxUpliftMeters < dry.stats.maxUpliftMeters);

  // Analytic check: rho_m u = rho_i H - rho_w (0 - (bed + u)).
  const bed = -1500;
  const expected =
    (ICE_DENSITY_KG_M3 * 2000 + SEAWATER_DENSITY_KG_M3 * bed) /
    (MANTLE_DENSITY_KG_M3 - SEAWATER_DENSITY_KG_M3);
  assert.ok(Math.abs(marine.stats.maxUpliftMeters - expected) < 1e-2);
});

test("raising the sea-level datum subsides open ocean by the analytic far-field amount", () => {
  const cellCount = 32 * 32;
  const datum = 58;
  const { uplift } = solveIsostaticRebound({
    nx: 32,
    ny: 32,
    cellCount,
    grid: { nx: 32, ny: 32, x0_m: 0, y0_m: 0, dx_m: 20000, dy_m: -20000 },
    bedHeights: new Float32Array(cellCount).fill(-4000),
    surfaceHeights: new Float32Array(cellCount).fill(0),
    thickness: new Float32Array(cellCount),
    mask: new Uint8Array(cellCount).fill(MASK_OCEAN),
    model: REBOUND_MODEL_LOCAL,
    seaLevelMeters: datum,
  });
  const expected = farFieldUpliftMeters(datum);
  for (const value of uplift) assert.ok(Math.abs(value - expected) < 1e-2);
});

test("the solver leaves no-data cells at zero uplift and never marks them emergent", () => {
  const cellCount = 16 * 16;
  const bedHeights = new Float32Array(cellCount).fill(-500);
  bedHeights[0] = Number.NaN;
  const { uplift, emergent } = solveIsostaticRebound({
    nx: 16,
    ny: 16,
    cellCount,
    grid: { nx: 16, ny: 16, x0_m: 0, y0_m: 0, dx_m: 20000, dy_m: -20000 },
    bedHeights,
    surfaceHeights: new Float32Array(cellCount).fill(1500),
    thickness: new Float32Array(cellCount).fill(2000),
    mask: new Uint8Array(cellCount).fill(MASK_GROUNDED_ICE),
    model: REBOUND_MODEL_LOCAL,
  });
  assert.equal(uplift[0], 0);
  assert.equal(emergent[0], 0);
  assert.equal(emergent[1], 1, "a flooded basin lifted above the datum is emergent");
});

test("the solver validates its inputs", () => {
  const base = uniformSlab({ nx: 8, ny: 8, thickness: 100, bed: 0, mask: MASK_GROUNDED_ICE });
  assert.throws(() => solveIsostaticRebound({ ...base, nx: 1 }), /at least 2x2/);
  assert.throws(() => solveIsostaticRebound({ ...base, cellCount: 63 }), /inconsistent cell count/);
  assert.throws(
    () => solveIsostaticRebound({ ...base, thickness: new Float32Array(3) }),
    /mismatched thickness field/
  );
});

test("a zero flexural rigidity falls back to the local model", () => {
  const field = uniformSlab({ nx: 16, ny: 16, thickness: 1000, bed: 800, mask: MASK_GROUNDED_ICE });
  const { stats } = solveIsostaticRebound({
    ...field,
    model: REBOUND_MODEL_FLEXURAL,
    flexuralRigidityNm: 0,
  });
  assert.equal(stats.model, REBOUND_MODEL_LOCAL);
});

test("the solver converges to centimetre residuals well inside its iteration budget", () => {
  const field = uniformSlab({ nx: 64, ny: 64, thickness: 3000, bed: -1000, mask: MASK_GROUNDED_ICE });
  const { stats } = solveIsostaticRebound({ ...field, model: REBOUND_MODEL_FLEXURAL });
  assert.ok(stats.residualMeters < 0.01, `residual ${stats.residualMeters}`);
  assert.ok(stats.iterations < 24, `iterations ${stats.iterations}`);
});

test("statistics account for emergence, ice volume and sea-level equivalent", () => {
  const nx = 16;
  const ny = 16;
  const cellCount = nx * ny;
  // Left half: grounded ice on a 400 m deep basin. Right half: dry rock at 200 m.
  const bedHeights = new Float32Array(cellCount);
  const thickness = new Float32Array(cellCount);
  const mask = new Uint8Array(cellCount);
  for (let row = 0; row < ny; row += 1) {
    for (let column = 0; column < nx; column += 1) {
      const index = row * nx + column;
      if (column < nx / 2) {
        bedHeights[index] = -400;
        thickness[index] = 2500;
        mask[index] = MASK_GROUNDED_ICE;
      } else {
        bedHeights[index] = 200;
        mask[index] = MASK_ICE_FREE_LAND;
      }
    }
  }
  const { stats } = solveIsostaticRebound({
    nx,
    ny,
    cellCount,
    grid: { nx, ny, x0_m: 0, y0_m: 0, dx_m: 10000, dy_m: -10000 },
    bedHeights,
    surfaceHeights: new Float32Array(cellCount),
    thickness,
    mask,
    model: REBOUND_MODEL_LOCAL,
  });

  // Areas carry the polar-stereographic scale correction, so assert the relationships
  // rather than a nominal cell count; polarStereographicCellAreaM2 is covered separately.
  assert.ok(stats.validBedAreaKm2 > 0);
  assert.ok(Math.abs(stats.landAreaAfterKm2 - stats.validBedAreaKm2) < 1e-6, "all bed emerges");
  assert.ok(Math.abs(stats.landAreaNowKm2 + stats.emergentAreaKm2 - stats.landAreaAfterKm2) < 1e-6);
  assert.ok(Math.abs(stats.emergentAreaKm2 - stats.iceFootprintAreaKm2) < 1e-6);
  assert.equal(stats.submergedAreaKm2, 0);
  assert.equal(stats.marineUnderIceAfterAreaKm2, 0);
  assert.equal(stats.closedBasinAreaKm2, 0, "nothing stays below the datum");
  // The ice covers exactly half the grid, and area weighting is symmetric about the pole.
  assert.ok(Math.abs(stats.iceFootprintAreaKm2 / stats.validBedAreaKm2 - 0.5) < 0.02);

  // Volume above flotation subtracts the flotation allowance for the 400 m basin.
  const aboveFlotation = 2500 - 400 * (SEAWATER_DENSITY_KG_M3 / ICE_DENSITY_KG_M3);
  assert.ok(
    Math.abs(stats.volumeAboveFlotationKm3 / stats.iceVolumeKm3 - aboveFlotation / 2500) < 1e-6
  );
  assert.ok(stats.sleMeters > 0 && stats.sleMeters < stats.iceVolumeKm3);
  assert.ok(stats.maxUpliftRow >= 0 && stats.maxUpliftColumn >= 0);
  // The rebounded bed is everywhere above the datum, so the deepest ice-covered point is
  // too. This statistic is scoped to the present ice footprint, not the whole domain.
  assert.ok(stats.deepestGroundedBedAfterMeters > 0);
});

// ---------------------------------------------------------------- scenario geometry

test("applyReboundFraction interpolates between the present and rebounded bed", () => {
  const bedHeights = Float32Array.from([-500, 100, Number.NaN]);
  const uplift = Float32Array.from([300, 40, 200]);

  const present = applyReboundFraction({ cellCount: 3, bedHeights, uplift, fraction: 0 });
  assert.equal(present[0], -500);
  assert.equal(present[1], 100);
  assert.ok(Number.isNaN(present[2]), "no-data cells stay no-data");

  const half = applyReboundFraction({ cellCount: 3, bedHeights, uplift, fraction: 0.5 });
  assert.ok(Math.abs(half[0] - -350) < 1e-4);
  assert.ok(Math.abs(half[1] - 120) < 1e-4);

  const full = applyReboundFraction({ cellCount: 3, bedHeights, uplift, fraction: 1 });
  assert.ok(Math.abs(full[0] - -200) < 1e-4);

  // Out-of-range fractions clamp rather than extrapolate.
  const over = applyReboundFraction({ cellCount: 3, bedHeights, uplift, fraction: 4 });
  assert.ok(Math.abs(over[0] - full[0]) < 1e-6);
  const under = applyReboundFraction({ cellCount: 3, bedHeights, uplift, fraction: -1 });
  assert.equal(under[0], -500);
});

test("deriveReboundedIceSurfaces thins ice and keeps grounded ice on the bed", () => {
  const reboundedBedHeights = Float32Array.from([-400, -400]);
  const thickness = Float32Array.from([2000, 2000]);
  const mask = Uint8Array.from([MASK_GROUNDED_ICE, MASK_FLOATING_ICE]);
  const iceValid = Uint8Array.from([1, 1]);

  const present = deriveReboundedIceSurfaces({
    cellCount: 2,
    reboundedBedHeights,
    thickness,
    mask,
    iceValid,
    fraction: 0,
  });
  assert.equal(present.bottom[0], -400, "grounded ice sits on the bed");
  assert.ok(Math.abs(present.surface[0] - 1600) < 1e-4);

  const half = deriveReboundedIceSurfaces({
    cellCount: 2,
    reboundedBedHeights,
    thickness,
    mask,
    iceValid,
    fraction: 0.5,
  });
  assert.ok(Math.abs(half.surface[0] - half.bottom[0] - 1000) < 1e-4, "thickness halves");

  const gone = deriveReboundedIceSurfaces({
    cellCount: 2,
    reboundedBedHeights,
    thickness,
    mask,
    iceValid,
    fraction: 1,
  });
  assert.ok(Math.abs(gone.surface[0] - gone.bottom[0]) < 1e-6, "no ice left at full deglaciation");
});

test("deriveReboundedIceSurfaces re-floats a shelf and grounds it on a rising bed", () => {
  const thickness = Float32Array.from([1000]);
  const mask = Uint8Array.from([MASK_FLOATING_ICE]);
  const iceValid = Uint8Array.from([1]);

  // Deep water: the shelf floats at its hydrostatic draft.
  const deep = deriveReboundedIceSurfaces({
    cellCount: 1,
    reboundedBedHeights: Float32Array.from([-2000]),
    thickness,
    mask,
    iceValid,
    fraction: 0,
  });
  const draft = -(ICE_DENSITY_KG_M3 * 1000) / SEAWATER_DENSITY_KG_M3;
  assert.ok(Math.abs(deep.bottom[0] - draft) < 1e-4);
  assert.ok(deep.bottom[0] > -2000, "a floating shelf does not touch a deep bed");

  // Shallow water: the bed overtakes the draft and the shelf grounds.
  const shallow = deriveReboundedIceSurfaces({
    cellCount: 1,
    reboundedBedHeights: Float32Array.from([-100]),
    thickness,
    mask,
    iceValid,
    fraction: 0,
  });
  assert.equal(shallow.bottom[0], -100, "a rising bed grounds the shelf");
});

test("deriveReboundedIceSurfaces marks ice-free cells as no-data", () => {
  const { surface, bottom } = deriveReboundedIceSurfaces({
    cellCount: 1,
    reboundedBedHeights: Float32Array.from([500]),
    thickness: Float32Array.from([0]),
    mask: Uint8Array.from([MASK_ICE_FREE_LAND]),
    iceValid: Uint8Array.from([0]),
    fraction: 0.5,
  });
  assert.ok(Number.isNaN(surface[0]));
  assert.ok(Number.isNaN(bottom[0]));
});

test("a raised datum lifts a re-floated shelf with the sea surface", () => {
  const datum = 58;
  const { bottom, surface } = deriveReboundedIceSurfaces({
    cellCount: 1,
    reboundedBedHeights: Float32Array.from([-2000]),
    thickness: Float32Array.from([1000]),
    mask: Uint8Array.from([MASK_FLOATING_ICE]),
    iceValid: Uint8Array.from([1]),
    fraction: 0,
    seaLevelMeters: datum,
  });
  const draft = datum - (ICE_DENSITY_KG_M3 * 1000) / SEAWATER_DENSITY_KG_M3;
  assert.ok(Math.abs(bottom[0] - draft) < 1e-4);
  assert.ok(Math.abs(surface[0] - (draft + 1000)) < 1e-4);
});
