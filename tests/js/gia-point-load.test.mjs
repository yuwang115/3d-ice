import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_FLEXURAL_RIGIDITY_N_M,
  flexuralLengthScaleMeters,
  GRAVITY_M_PER_S2,
  ICE_DENSITY_KG_M3,
  MASK_GROUNDED_ICE,
  MASK_ICE_FREE_LAND,
  REBOUND_MODEL_FLEXURAL,
  solveIsostaticRebound,
} from "../../static/tools/js/gia-rebound.js";

// The analytic benchmark for the spectral flexure solve. A point load V on a thin elastic
// plate over a fluid foundation deflects it by w(r) = -V L_r^2 / (2 pi D) kei(r / L_r)
// (Brotchie & Silvester 1969), so removing the load raises the bed by the same amount:
// u(0) = V L_r^2 / (8 D), with a peripheral forebulge beyond kei's first zero at 3.91 L_r.

const EULER_GAMMA = 0.5772156649015329;
const MAX_SERIES_TERMS = 100;

const GRID_CELLS = 128;
// Inside the 16-28 km range of the explorer's flexural solve grids, and coarse enough that
// the solver does not coarsen it further, so the benchmark sees the solve grid itself
// rather than the bicubic upsampling.
const CELL_METERS = 20000;
const LOAD_ICE_METERS = 1000;
// Dry land everywhere, so no basin floods and the problem stays linear.
const BED_METERS = 1000;

const LENGTH_SCALE_M = flexuralLengthScaleMeters(DEFAULT_FLEXURAL_RIGIDITY_N_M);
const POINT_LOAD_N = ICE_DENSITY_KG_M3 * GRAVITY_M_PER_S2 * LOAD_ICE_METERS * CELL_METERS ** 2;
const KELVIN_SCALE_M =
  (POINT_LOAD_N * LENGTH_SCALE_M ** 2) / (2 * Math.PI * DEFAULT_FLEXURAL_RIGIDITY_N_M);
const PEAK_UPLIFT_M = KELVIN_SCALE_M * (Math.PI / 4);

/**
 * Kelvin function kei(x) from its ascending series (Abramowitz & Stegun 1964, 9.9.10-9.9.13):
 * kei x = -ln(x/2) bei x - (pi/4) ber x + sum_k (-1)^k psi(2k+2) (x^2/4)^(2k+1) / ((2k+1)!)^2.
 * The terms (x^2/4)^m / (m!)^2 peak near m = x/2, and the cancellation they leave stays
 * far below the benchmark tolerance across the x < 14 this grid reaches.
 */
function kei(x) {
  if (x === 0) return -Math.PI / 4;
  const quarterSquare = (x * x) / 4;
  let ber = 0;
  let bei = 0;
  let digammaSeries = 0;
  let term = 1;
  let harmonic = 0;
  for (let m = 0; m < MAX_SERIES_TERMS; m += 1) {
    if (m > 0) {
      term *= quarterSquare / (m * m);
      harmonic += 1 / m;
    }
    const sign = Math.floor(m / 2) % 2 === 0 ? 1 : -1;
    if (m % 2 === 0) {
      ber += sign * term;
    } else {
      bei += sign * term;
      // psi(m + 1) = H_m - gamma, which for m = 2k + 1 is the series' psi(2k + 2).
      digammaSeries += sign * (harmonic - EULER_GAMMA) * term;
    }
    if (m > x && term < 1e-20) break;
  }
  return -Math.log(x / 2) * bei - (Math.PI / 4) * ber + digammaSeries;
}

function kelvinUpliftMeters(radiusMeters) {
  return -KELVIN_SCALE_M * kei(radiusMeters / LENGTH_SCALE_M);
}

function radiusFromLoadMeters(index) {
  const row = Math.floor(index / GRID_CELLS) - GRID_CELLS / 2;
  const column = (index % GRID_CELLS) - GRID_CELLS / 2;
  return Math.hypot(row, column) * CELL_METERS;
}

function solvePointLoad() {
  const cellCount = GRID_CELLS * GRID_CELLS;
  const loadIndex = (GRID_CELLS / 2) * GRID_CELLS + GRID_CELLS / 2;
  const thickness = new Float32Array(cellCount);
  thickness[loadIndex] = LOAD_ICE_METERS;
  const bedHeights = new Float32Array(cellCount).fill(BED_METERS);
  const surfaceHeights = bedHeights.map((bed, index) => bed + thickness[index]);
  const mask = new Uint8Array(cellCount).fill(MASK_ICE_FREE_LAND);
  mask[loadIndex] = MASK_GROUNDED_ICE;

  const { uplift, stats } = solveIsostaticRebound({
    nx: GRID_CELLS,
    ny: GRID_CELLS,
    cellCount,
    grid: { nx: GRID_CELLS, ny: GRID_CELLS, x0_m: 0, y0_m: 0, dx_m: CELL_METERS, dy_m: -CELL_METERS },
    bedHeights,
    surfaceHeights,
    thickness,
    mask,
    model: REBOUND_MODEL_FLEXURAL,
    seaLevelMeters: 0,
  });
  return { uplift, stats, loadIndex };
}

const pointLoad = solvePointLoad();

test("the kei series reproduces reference values of the Kelvin function", () => {
  // scipy.special.kei (SciPy 1.17), spanning the radii the benchmark grid reaches.
  const reference = [
    [0.5, -0.6715816950943676],
    [1, -0.49499463651872],
    [2, -0.20240006776470432],
    [3, -0.051121884045986665],
    [4, 0.002198399294972698],
    [5, 0.01118758650986929],
    [8, 0.00036958395614470973],
    [10, -0.00030752456908645856],
    [13, 5.387022189156348e-6],
  ];
  assert.equal(kei(0), -Math.PI / 4);
  for (const [x, expected] of reference) {
    assert.ok(Math.abs(kei(x) - expected) < 1e-10, `kei(${x}) = ${kei(x)}, expected ${expected}`);
  }
});

test("the benchmark solves on its own grid with the default flexural length scale", () => {
  assert.equal(pointLoad.stats.solveCellKm, CELL_METERS / 1000);
  assert.ok(Math.abs(LENGTH_SCALE_M - 132600) < 100, `L_r = ${LENGTH_SCALE_M} m`);
});

test("the flexural solver matches the Kelvin point-load solution beyond half a flexural length", () => {
  let worstMisfit = 0;
  for (let index = 0; index < pointLoad.uplift.length; index += 1) {
    const radius = radiusFromLoadMeters(index);
    if (radius < LENGTH_SCALE_M / 2) continue;
    const misfit = Math.abs(pointLoad.uplift[index] - kelvinUpliftMeters(radius));
    worstMisfit = Math.max(worstMisfit, misfit);
  }
  // Four significant figures of the peak, out to the corners at 13.6 L_r. A 1 % error in
  // the rigidity would miss by 3.6e-3 of the peak.
  assert.ok(
    worstMisfit < 1e-4 * PEAK_UPLIFT_M,
    `worst misfit ${(worstMisfit / PEAK_UPLIFT_M).toExponential(2)} of the peak`
  );
});

test("under the load the solver misses only the kernel's sub-grid tail", () => {
  // The solve grid holds no wavenumber above k_N = pi / dx, so at r = 0 it misses at most
  // the integral of 1 / (D k^4) beyond k_N: a fraction 2 / (pi (k_N L_r)^2) of the peak,
  // 1.5e-3 on this grid. That tail is positive, so the solve must fall short, not overshoot.
  const nyquistTimesLengthScale = (Math.PI / CELL_METERS) * LENGTH_SCALE_M;
  const tailBound = 2 / (Math.PI * nyquistTimesLengthScale ** 2);
  const deficit = (PEAK_UPLIFT_M - pointLoad.uplift[pointLoad.loadIndex]) / PEAK_UPLIFT_M;
  assert.ok(
    deficit > 0 && deficit < tailBound,
    `deficit ${deficit.toExponential(2)} of the peak, bound ${tailBound.toExponential(2)}`
  );
});

test("the solver puts the peripheral forebulge where the Kelvin solution does", () => {
  const firstNegativeStep = (upliftAtStep) => {
    for (let step = 1; step < GRID_CELLS / 2; step += 1) {
      if (upliftAtStep(step) < 0) return step;
    }
    return null;
  };
  const analytic = firstNegativeStep((step) => kelvinUpliftMeters(step * CELL_METERS));
  const solved = firstNegativeStep((step) => pointLoad.uplift[pointLoad.loadIndex + step]);
  // kei's first zero is at 3.9147, so the first depressed cell is 26 cells (3.92 L_r) out.
  assert.equal(analytic, 26);
  assert.equal(solved, analytic);
});
