/**
 * Glacial isostatic adjustment (GIA) for the 3D ICE cryosphere explorer.
 *
 * Answers the question "what would the bed look like with the ice gone and isostatic
 * rebound complete?" by solving for the EQUILIBRIUM deflection of the solid Earth
 * after the present ice load is removed. This is a steady-state (relaxed) calculation,
 * not a transient GIA simulation: it says where the bed ends up, not how it gets there.
 *
 * Physics
 * -------
 * Vertical displacement u (positive up) of a thin elastic plate over an inviscid
 * asthenosphere satisfies
 *
 *     D del^4 u + rho_m g u = -(sigma_after - sigma_now) = sigma_now - sigma_after
 *
 * where sigma is the vertical stress the overburden exerts on the bed, D is the
 * flexural rigidity and rho_m is mantle density. Setting D = 0 recovers local (Airy)
 * isostasy, u = (sigma_now - sigma_after) / (rho_m g).
 *
 * The present-day load distinguishes three cases, which matters:
 *   - grounded ice:  sigma_now = rho_i g H
 *   - floating ice:  sigma_now = rho_i g H + rho_w g (draft - bed), which for a
 *                    hydrostatic shelf collapses identically to rho_w g max(0, -bed) --
 *                    the very load an open water column would apply. Removing floating
 *                    ice therefore produces (correctly) no rebound whatsoever.
 *   - open ocean:    sigma_now = rho_w g max(0, -bed)
 *
 * After deglaciation the column is either dry or flooded to the chosen sea-level datum
 * zeta, so sigma_after = rho_w g (zeta - (bed + u)) wherever that is positive AND the
 * cell still drains to the open ocean; basins that rebound into closed hollows carry no
 * marine water column. Because both the flooded depth and the flooded footprint depend
 * on u, the system is non-linear and is solved by Picard iteration. In the flooded cells
 * the map's gain is rho_w / rho_m ~ 0.31 and the flexural operator damps it further, so
 * the iteration is a contraction: ~8 iterations reach centimetre residuals. A one-shot
 * solve that floods the un-rebounded bed is wrong by up to ~20 % of the peak signal, so
 * the iteration is required rather than a refinement.
 *
 * The far field is handled analytically rather than by truncation. A uniform sea-level
 * change zeta loads the whole ocean, which is a global rather than a regional effect;
 * its uniform response is u_inf = -rho_w zeta / (rho_m - rho_w). Subtracting the
 * matching far-field load before the transform leaves a residual that decays to zero
 * outside the domain, which is exactly what zero padding assumes.
 *
 * Numerics
 * --------
 * The flexural solve runs in the spectral domain, u(k) = R(k) / (rho_m g + D |k|^4),
 * which is the exact equilibrium of the plate equation including the peripheral
 * forebulge. Verified against the analytic point-load Kelvin-function solution
 * w(r) = V L_r^2 / (2 pi D) kei(r / L_r) to four significant figures, and against the
 * closed-form Airy limit exactly.
 *
 * Because the deflection is band-limited near the flexural length scale
 * L_r = (D / (rho_m g))^(1/4) ~ 133 km, the transform is taken on a ~16-28 km grid and
 * bicubically upsampled; against a native-resolution solve this shifts the peak uplift
 * by <0.1 % (RMS 1.6 m on a 1026 m peak). Local (Airy) isostasy has no intrinsic length
 * scale, so it is always solved pointwise at native resolution instead.
 *
 * Areas are integrated with the polar-stereographic point scale factor, which varies the
 * true area of a nominally constant grid cell by roughly -3 % to +8 % across Antarctica.
 *
 * References
 * ----------
 * Turcotte, D. L. & Schubert, G. (2002) Geodynamics, 2nd edn, CUP - thin-plate flexure.
 * Brotchie, J. F. & Silvester, R. (1969) J. Geophys. Res. 74, 5240-5252 - the Kelvin
 *   (kei) point-load Green's function the spectral solver is validated against.
 * Le Meur, E. & Huybrechts, P. (1996) Ann. Glaciol. 23, 309-317 - the canonical ELRA
 *   formulation and the parameter defaults used here (D = 1e25 N m, tau = 3000 yr).
 * Lingle, C. S. & Clark, J. A. (1985) J. Geophys. Res. 90, 1100-1114 - elastic
 *   lithosphere / viscous asthenosphere response to an ice load.
 * Bueler, E., Lingle, C. S. & Brown, J. (2007) Ann. Glaciol. 46, 97-105 - spectral
 *   (FFT) solution of the deformable-Earth response, the strategy adopted here.
 * Whitehouse, P. L., Gomez, N., King, M. A. & Wiens, D. A. (2019) Nat. Commun. 10, 503
 *   - present-day Antarctic uplift rates and lateral viscosity structure; the reason
 *   the present bed cannot be assumed fully relaxed (see the UI caveat text).
 */

import { angularWavenumbers, createFft2dPlan, transformFft2dInPlace } from "./fft2d.js";
import {
  bicubicUpsampleInto,
  blockMean,
  buildSubCellHypsometry,
  chooseCoarseningFactor,
  floodConnectedOcean,
  meanSubmergedDepth,
  paddedTransformSize,
  polarStereographicCellAreaM2,
} from "./gia-grid.js";

/** BedMachine's own hydrostatic constants, so the flotation algebra closes on its geometry. */
export const ICE_DENSITY_KG_M3 = 917;
export const SEAWATER_DENSITY_KG_M3 = 1027;
/** Subglacial lake water (Lake Vostok), which is fresh rather than marine. */
export const FRESHWATER_DENSITY_KG_M3 = 1000;
export const MANTLE_DENSITY_KG_M3 = 3300;
export const GRAVITY_M_PER_S2 = 9.81;

/** Le Meur & Huybrechts (1996) ELRA lithosphere rigidity. */
export const DEFAULT_FLEXURAL_RIGIDITY_N_M = 1e25;
/** Le Meur & Huybrechts (1996) asthenosphere relaxation time. */
export const REBOUND_RELAXATION_TIME_YEARS = 3000;
/** Present global ocean area used for sea-level-equivalent conversion (Gregory et al. 2019). */
export const GLOBAL_OCEAN_AREA_M2 = 3.625e14;

export const REBOUND_MODEL_FLEXURAL = "flexural";
export const REBOUND_MODEL_LOCAL = "local";
export const REBOUND_MODELS = [REBOUND_MODEL_FLEXURAL, REBOUND_MODEL_LOCAL];

/** EPSG:3031 standard parallel (Antarctic Polar Stereographic). */
export const ANTARCTIC_STANDARD_PARALLEL_DEGREES = -71;
/** EPSG:3413 standard parallel (NSIDC Sea Ice Polar Stereographic North). */
export const GREENLAND_STANDARD_PARALLEL_DEGREES = 70;

export const MASK_OCEAN = 0;
export const MASK_ICE_FREE_LAND = 1;
export const MASK_GROUNDED_ICE = 2;
export const MASK_FLOATING_ICE = 3;
export const MASK_SUBGLACIAL_LAKE = 4;

const SOLVER_MAX_ITERATIONS = 24;
const SOLVER_TOLERANCE_M = 0.01;
const MAX_FFT_AXIS = 2048;

function clampFraction(value) {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
}

export function isGroundedIceMask(maskValue) {
  return maskValue === MASK_GROUNDED_ICE || maskValue === MASK_SUBGLACIAL_LAKE;
}

export function isFloatingIceMask(maskValue) {
  return maskValue === MASK_FLOATING_ICE;
}

/** Flexural length scale L_r = (D / (rho_m g))^(1/4); zero for the local model. */
export function flexuralLengthScaleMeters(flexuralRigidityNm) {
  const rigidity = Number.isFinite(flexuralRigidityNm) ? Math.max(0, flexuralRigidityNm) : 0;
  if (rigidity <= 0) return 0;
  return Math.pow(rigidity / (MANTLE_DENSITY_KG_M3 * GRAVITY_M_PER_S2), 0.25);
}

/**
 * Elapsed time implied by a fraction of the equilibrium rebound under exponential
 * relaxation, u(t) = u_eq (1 - exp(-t / tau)). Returns Infinity at full relaxation.
 */
export function reboundElapsedYears(fraction, relaxationTimeYears = REBOUND_RELAXATION_TIME_YEARS) {
  const clamped = clampFraction(fraction);
  if (clamped <= 0) return 0;
  if (clamped >= 1) return Number.POSITIVE_INFINITY;
  return -relaxationTimeYears * Math.log(1 - clamped);
}

/** Uniform far-field response to a global sea-level change, from ocean loading alone. */
export function farFieldUpliftMeters(seaLevelMeters) {
  const zeta = Number.isFinite(seaLevelMeters) ? seaLevelMeters : 0;
  const uplift = (-SEAWATER_DENSITY_KG_M3 * zeta) / (MANTLE_DENSITY_KG_M3 - SEAWATER_DENSITY_KG_M3);
  // Normalise -0 so an unchanged datum never renders as "-0 m".
  return uplift === 0 ? 0 : uplift;
}

/**
 * Vertical stress (Pa) the present overburden exerts on the bed, case-split by mask so
 * that hydrostatically supported floating ice contributes no removable load.
 *
 * Floating ice uses the hydrostatic-equivalent water column rho_w g max(0, -bed) rather
 * than the grid-literal sum rho_i g H + rho_w g (draft - bed). The two are identical in
 * exact arithmetic; on BedMachine Antarctica v4 the grid values disagree by a few metres
 * of ice near grounding lines and ice rises, which would otherwise inject a speckled
 * halo of spurious uplift into a region that must rebound by exactly zero.
 */
export function computeBedLoadPascals({ cellCount, bedHeights, surfaceHeights, thickness, mask }) {
  const load = new Float64Array(cellCount);
  for (let index = 0; index < cellCount; index += 1) {
    const bed = bedHeights[index];
    if (!Number.isFinite(bed)) continue;

    const maskValue = mask[index];
    if (isGroundedIceMask(maskValue)) {
      const iceThickness = Number.isFinite(thickness[index]) ? Math.max(0, thickness[index]) : 0;
      load[index] = ICE_DENSITY_KG_M3 * GRAVITY_M_PER_S2 * iceThickness;
      if (maskValue === MASK_SUBGLACIAL_LAKE) {
        // Over Lake Vostok the ice floats on the lake and the lake rests on the bed, so
        // the bed carries both columns. Omitting the water understates the load by up to
        // ~900 m of fresh water, worth ~276 m of uplift.
        const surface = surfaceHeights[index];
        const iceBase = Number.isFinite(surface) ? surface - iceThickness : bed;
        load[index] +=
          FRESHWATER_DENSITY_KG_M3 * GRAVITY_M_PER_S2 * Math.max(0, iceBase - bed);
      }
    } else if (isFloatingIceMask(maskValue) || maskValue === MASK_OCEAN) {
      load[index] = SEAWATER_DENSITY_KG_M3 * GRAVITY_M_PER_S2 * Math.max(0, -bed);
    }
  }
  return load;
}

function buildFlexureDenominator(plan, cellX, cellY, flexuralRigidityNm) {
  const { sizeX, sizeY } = plan;
  const denominator = new Float64Array(sizeX * sizeY);
  const wavenumbersX = angularWavenumbers(sizeX, cellX);
  const wavenumbersY = angularWavenumbers(sizeY, cellY);
  const foundation = MANTLE_DENSITY_KG_M3 * GRAVITY_M_PER_S2;

  for (let row = 0; row < sizeY; row += 1) {
    const squaredY = wavenumbersY[row] * wavenumbersY[row];
    for (let column = 0; column < sizeX; column += 1) {
      const squared = wavenumbersX[column] * wavenumbersX[column] + squaredY;
      denominator[row * sizeX + column] = foundation + flexuralRigidityNm * squared * squared;
    }
  }
  return denominator;
}

function applyFlexureOperator(plan, denominator, rhs, coarseX, coarseY, out) {
  const { sizeX, sizeY, real, imaginary } = plan;
  real.fill(0);
  imaginary.fill(0);

  const offsetX = (sizeX - coarseX) >> 1;
  const offsetY = (sizeY - coarseY) >> 1;
  for (let row = 0; row < coarseY; row += 1) {
    const source = row * coarseX;
    const target = (row + offsetY) * sizeX + offsetX;
    for (let column = 0; column < coarseX; column += 1) real[target + column] = rhs[source + column];
  }

  transformFft2dInPlace(plan, { inverse: false });
  for (let index = 0; index < denominator.length; index += 1) {
    const scale = denominator[index];
    real[index] /= scale;
    imaginary[index] /= scale;
  }
  transformFft2dInPlace(plan, { inverse: true });

  for (let row = 0; row < coarseY; row += 1) {
    const target = row * coarseX;
    const source = (row + offsetY) * sizeX + offsetX;
    for (let column = 0; column < coarseX; column += 1) out[target + column] = real[source + column];
  }
  return out;
}

function solveLocalRebound({ nx, ny, cellCount, load, bedHeights, seaLevelMeters, onProgress }) {
  const uplift = new Float32Array(cellCount);
  const foundation = MANTLE_DENSITY_KG_M3 * GRAVITY_M_PER_S2;
  const waterWeight = SEAWATER_DENSITY_KG_M3 * GRAVITY_M_PER_S2;
  const filledBed = new Float64Array(cellCount);
  const bedCoverage = new Uint8Array(cellCount);
  for (let index = 0; index < cellCount; index += 1) {
    const finite = Number.isFinite(bedHeights[index]);
    filledBed[index] = finite ? bedHeights[index] : 0;
    bedCoverage[index] = finite ? 1 : 0;
  }

  const connected = new Uint8Array(cellCount);
  const queue = new Int32Array(cellCount);
  let iterations = 0;
  let residual = Number.POSITIVE_INFINITY;

  for (let step = 0; step < SOLVER_MAX_ITERATIONS; step += 1) {
    iterations = step + 1;
    floodConnectedOcean({
      coarseX: nx,
      coarseY: ny,
      bed: filledBed,
      uplift,
      seaLevelMeters,
      valid: bedCoverage,
      out: connected,
      queue,
    });

    residual = 0;
    for (let index = 0; index < cellCount; index += 1) {
      const bed = bedHeights[index];
      if (!Number.isFinite(bed)) continue;
      const rebounded = bed + uplift[index];
      const loadAfter =
        connected[index] && rebounded < seaLevelMeters
          ? waterWeight * (seaLevelMeters - rebounded)
          : 0;
      const next = (load[index] - loadAfter) / foundation;
      const change = Math.abs(next - uplift[index]);
      if (change > residual) residual = change;
      uplift[index] = next;
    }
    if (onProgress) onProgress(Math.min(1, iterations / 8));
    if (residual < SOLVER_TOLERANCE_M) break;
  }

  return { uplift, iterations, residual, solveCellMeters: 0, fftSizeX: 0, fftSizeY: 0 };
}

/** Build the coarse solve grid, the spectral operator and the sub-cell water-load table. */
function prepareFlexuralSolve({ nx, ny, cellCount, dxMeters, dyMeters, load, bedHeights, flexuralRigidityNm }) {
  const lengthScale = flexuralLengthScaleMeters(flexuralRigidityNm);
  const factor = chooseCoarseningFactor(nx, ny, dxMeters, dyMeters, lengthScale);
  const { sizeX, sizeY, cellX, cellY } = paddedTransformSize(nx, ny, dxMeters, dyMeters, factor, lengthScale);
  if (sizeX > MAX_FFT_AXIS || sizeY > MAX_FFT_AXIS) {
    throw new Error("Isostatic-rebound solver grid exceeds the supported transform size.");
  }

  const coarseLoad = blockMean(load, nx, ny, factor);
  const filledBed = new Float64Array(cellCount);
  for (let index = 0; index < cellCount; index += 1) {
    filledBed[index] = Number.isFinite(bedHeights[index]) ? bedHeights[index] : 0;
  }
  const plan = createFft2dPlan(sizeX, sizeY);
  return {
    factor,
    cellX,
    cellY,
    sizeX,
    sizeY,
    coarseX: coarseLoad.coarseX,
    coarseY: coarseLoad.coarseY,
    coarseLoad,
    coarseBed: blockMean(filledBed, nx, ny, factor),
    // The water load is evaluated against the sub-cell bathymetry rather than the coarse
    // mean bed, which removes a Jensen bias worth ~3 m RMS / ~17 m peak of uplift and
    // about half a percent of the emergent-area headline.
    hypsometry: buildSubCellHypsometry(bedHeights, nx, ny, factor),
    plan,
    denominator: buildFlexureDenominator(plan, cellX, cellY, flexuralRigidityNm),
  };
}

function solveFlexuralRebound({
  nx,
  ny,
  cellCount,
  dxMeters,
  dyMeters,
  load,
  bedHeights,
  seaLevelMeters,
  flexuralRigidityNm,
  onProgress,
}) {
  const grid = prepareFlexuralSolve({
    nx,
    ny,
    cellCount,
    dxMeters,
    dyMeters,
    load,
    bedHeights,
    flexuralRigidityNm,
  });
  const { factor, cellX, cellY, sizeX, sizeY, coarseX, coarseY, coarseLoad, coarseBed, hypsometry, plan, denominator } =
    grid;

  const farFieldUplift = farFieldUpliftMeters(seaLevelMeters);
  const farFieldLoad = SEAWATER_DENSITY_KG_M3 * GRAVITY_M_PER_S2 * (farFieldUplift - seaLevelMeters);
  const waterWeight = SEAWATER_DENSITY_KG_M3 * GRAVITY_M_PER_S2;

  const coarseCount = coarseX * coarseY;
  const current = new Float64Array(coarseCount).fill(farFieldUplift);
  const rhs = new Float64Array(coarseCount);
  const scratch = new Float64Array(coarseCount);
  const connected = new Uint8Array(coarseCount);
  const queue = new Int32Array(coarseCount);
  let iterations = 0;
  let residual = Number.POSITIVE_INFINITY;

  for (let step = 0; step < SOLVER_MAX_ITERATIONS; step += 1) {
    iterations = step + 1;
    floodConnectedOcean({
      coarseX,
      coarseY,
      bed: coarseBed.data,
      uplift: current,
      seaLevelMeters,
      valid: coarseBed.coverage,
      out: connected,
      queue,
    });

    for (let index = 0; index < coarseCount; index += 1) {
      // max(0, datum - (bed + uplift)) == max(0, (datum - uplift) - bed), so the whole
      // cell's mean depth is a function of this one waterline.
      const waterLevel = seaLevelMeters - current[index];
      const depth = connected[index] ? meanSubmergedDepth(hypsometry, index, waterLevel) : 0;
      rhs[index] = coarseLoad.data[index] - waterWeight * depth - farFieldLoad;
    }
    applyFlexureOperator(plan, denominator, rhs, coarseX, coarseY, scratch);

    residual = 0;
    for (let index = 0; index < coarseCount; index += 1) {
      const next = scratch[index] + farFieldUplift;
      const change = Math.abs(next - current[index]);
      if (change > residual) residual = change;
      current[index] = next;
    }
    if (onProgress) onProgress(Math.min(1, iterations / 8));
    if (residual < SOLVER_TOLERANCE_M) break;
  }

  const uplift = new Float32Array(cellCount);
  bicubicUpsampleInto(current, coarseX, coarseY, nx, ny, factor, uplift);
  return {
    uplift,
    iterations,
    residual,
    solveCellMeters: Math.max(cellX, cellY),
    fftSizeX: sizeX,
    fftSizeY: sizeY,
  };
}

/** One area-weighted pass over the grid, accumulating every reported quantity. */
function accumulateReboundTotals({
  nx,
  ny,
  cellCount,
  grid,
  standardParallelDegrees,
  bedHeights,
  thickness,
  mask,
  uplift,
  emergent,
  seaLevelMeters,
}) {
  // Precompute the per-row area weight: in a polar-stereographic grid the point scale
  // factor depends only on the radius from the pole, so it varies along a row too.
  const filledBed = new Float64Array(cellCount);
  const bedCoverage = new Uint8Array(cellCount);
  for (let index = 0; index < cellCount; index += 1) {
    const finite = Number.isFinite(bedHeights[index]);
    filledBed[index] = finite ? bedHeights[index] : 0;
    bedCoverage[index] = finite ? 1 : 0;
  }
  const connectedOcean = floodConnectedOcean({
    coarseX: nx,
    coarseY: ny,
    bed: filledBed,
    uplift,
    seaLevelMeters,
    valid: bedCoverage,
  });

  let maxUplift = Number.NEGATIVE_INFINITY;
  let maxUpliftIndex = -1;
  let groundedUpliftSum = 0;
  let groundedCells = 0;
  let landNowArea = 0;
  let landAfterArea = 0;
  let emergentArea = 0;
  let submergedArea = 0;
  let validArea = 0;
  let iceFootprintArea = 0;
  let emergentUnderIceArea = 0;
  let marineUnderIceArea = 0;
  let closedBasinArea = 0;
  let iceVolumeM3 = 0;
  let volumeAboveFlotationM3 = 0;
  let deepestAfter = Number.POSITIVE_INFINITY;
  let deepestAfterIndex = -1;

  for (let row = 0; row < ny; row += 1) {
    const yMeters = grid.y0_m + row * grid.dy_m;
    for (let column = 0; column < nx; column += 1) {
      const index = row * nx + column;
      const cellAreaM2 = polarStereographicCellAreaM2({
        xMeters: grid.x0_m + column * grid.dx_m,
        yMeters,
        dxMeters: grid.dx_m,
        dyMeters: grid.dy_m,
        standardParallelDegrees,
      });

      const bed = bedHeights[index];
      const maskValue = mask[index];
      const iceThickness = Number.isFinite(thickness[index]) ? Math.max(0, thickness[index]) : 0;

      if (iceThickness > 0) {
        iceVolumeM3 += iceThickness * cellAreaM2;
        if (isGroundedIceMask(maskValue) && Number.isFinite(bed)) {
          const aboveFlotation =
            iceThickness + Math.min(0, bed) * (SEAWATER_DENSITY_KG_M3 / ICE_DENSITY_KG_M3);
          if (aboveFlotation > 0) volumeAboveFlotationM3 += aboveFlotation * cellAreaM2;
        }
      }

      if (!Number.isFinite(bed)) continue;
      validArea += cellAreaM2;

      const grounded = isGroundedIceMask(maskValue);
      const underIce = grounded || isFloatingIceMask(maskValue);
      const cellUplift = uplift[index];
      if (cellUplift > maxUplift) {
        maxUplift = cellUplift;
        maxUpliftIndex = index;
      }
      if (grounded) {
        groundedUpliftSum += cellUplift;
        groundedCells += 1;
      }

      const reboundedBed = bed + cellUplift;
      // Restricted to grounded ice: the deepest open-ocean point is an abyssal-plain
      // figure that says nothing about the deglaciated continent, and a sub-shelf cell
      // can sit over the continental slope.
      if (grounded && reboundedBed < deepestAfter) {
        deepestAfter = reboundedBed;
        deepestAfterIndex = index;
      }

      const aboveNow = bed > 0;
      const aboveAfter = reboundedBed > seaLevelMeters;
      if (aboveNow) landNowArea += cellAreaM2;
      if (aboveAfter) landAfterArea += cellAreaM2;
      if (!aboveNow && aboveAfter) {
        emergentArea += cellAreaM2;
        emergent[index] = 1;
      }
      if (aboveNow && !aboveAfter) submergedArea += cellAreaM2;
      if (!aboveAfter && !connectedOcean[index]) closedBasinArea += cellAreaM2;

      if (underIce) {
        iceFootprintArea += cellAreaM2;
        if (aboveAfter) {
          if (!aboveNow) emergentUnderIceArea += cellAreaM2;
        } else {
          marineUnderIceArea += cellAreaM2;
        }
      }
    }
  }

  const toKm2 = (value) => value / 1e6;
  const maxRow = maxUpliftIndex >= 0 ? Math.floor(maxUpliftIndex / nx) : -1;
  const maxColumn = maxUpliftIndex >= 0 ? maxUpliftIndex % nx : -1;

  return {
    maxUpliftMeters: Number.isFinite(maxUplift) ? maxUplift : 0,
    maxUpliftRow: maxRow,
    maxUpliftColumn: maxColumn,
    meanGroundedUpliftMeters: groundedCells > 0 ? groundedUpliftSum / groundedCells : 0,
    deepestGroundedBedAfterMeters: Number.isFinite(deepestAfter) ? deepestAfter : 0,
    deepestAfterIndex,
    landAreaNowKm2: toKm2(landNowArea),
    landAreaAfterKm2: toKm2(landAfterArea),
    emergentAreaKm2: toKm2(emergentArea),
    submergedAreaKm2: toKm2(submergedArea),
    iceFootprintAreaKm2: toKm2(iceFootprintArea),
    emergentUnderIceAreaKm2: toKm2(emergentUnderIceArea),
    marineUnderIceAfterAreaKm2: toKm2(marineUnderIceArea),
    closedBasinAreaKm2: toKm2(closedBasinArea),
    validBedAreaKm2: toKm2(validArea),
    iceVolumeKm3: iceVolumeM3 / 1e9,
    volumeAboveFlotationKm3: volumeAboveFlotationM3 / 1e9,
    sleMeters:
      (volumeAboveFlotationM3 * ICE_DENSITY_KG_M3) /
      (SEAWATER_DENSITY_KG_M3 * GLOBAL_OCEAN_AREA_M2),
  };
}

/** Assemble the metadata-panel statistics from one accumulation pass plus solver metadata. */
function summarise({ nx, ny, cellCount, grid, standardParallelDegrees, bedHeights, thickness, mask, uplift, emergent, seaLevelMeters, model, flexuralRigidityNm, solved }) {
  const totals = accumulateReboundTotals({
    nx,
    ny,
    cellCount,
    grid,
    standardParallelDegrees,
    bedHeights,
    thickness,
    mask,
    uplift,
    emergent,
    seaLevelMeters,
  });
  // deepestAfterIndex is only needed to resolve the location below; keep it out of the
  // reported statistics.
  const { deepestAfterIndex, ...reported } = totals;

  return {
    ...reported,
    model,
    seaLevelMeters,
    flexuralRigidityNm: model === REBOUND_MODEL_FLEXURAL ? flexuralRigidityNm : 0,
    flexuralLengthScaleKm:
      model === REBOUND_MODEL_FLEXURAL ? flexuralLengthScaleMeters(flexuralRigidityNm) / 1000 : 0,
    relaxationTimeYears: REBOUND_RELAXATION_TIME_YEARS,
    maxUpliftXMeters:
      totals.maxUpliftColumn >= 0 ? grid.x0_m + totals.maxUpliftColumn * grid.dx_m : Number.NaN,
    maxUpliftYMeters:
      totals.maxUpliftRow >= 0 ? grid.y0_m + totals.maxUpliftRow * grid.dy_m : Number.NaN,
    deepestGroundedBedAfterXMeters:
      deepestAfterIndex >= 0 ? grid.x0_m + (deepestAfterIndex % nx) * grid.dx_m : Number.NaN,
    deepestGroundedBedAfterYMeters:
      deepestAfterIndex >= 0 ? grid.y0_m + Math.floor(deepestAfterIndex / nx) * grid.dy_m : Number.NaN,
    iterations: solved.iterations,
    residualMeters: solved.residual,
    solveCellKm: solved.solveCellMeters / 1000,
    fftSizeX: solved.fftSizeX,
    fftSizeY: solved.fftSizeY,
  };
}

/**
 * Solve for the equilibrium ice-free bed. Returns the uplift field (m, positive up),
 * a per-cell flag marking bed that crosses from below to above the sea-level datum,
 * and summary statistics for the metadata panel.
 */
export function solveIsostaticRebound(payload) {
  const {
    nx,
    ny,
    cellCount,
    grid,
    bedHeights,
    surfaceHeights,
    thickness,
    mask,
    model = REBOUND_MODEL_FLEXURAL,
    seaLevelMeters = 0,
    flexuralRigidityNm = DEFAULT_FLEXURAL_RIGIDITY_N_M,
    standardParallelDegrees = ANTARCTIC_STANDARD_PARALLEL_DEGREES,
    onProgress = null,
  } = payload;

  if (!Number.isInteger(nx) || !Number.isInteger(ny) || nx < 2 || ny < 2) {
    throw new Error("Isostatic-rebound solver needs a grid of at least 2x2 cells.");
  }
  if (cellCount !== nx * ny) {
    throw new Error("Isostatic-rebound solver received an inconsistent cell count.");
  }
  for (const [name, field] of [
    ["bed", bedHeights],
    ["surface", surfaceHeights],
    ["thickness", thickness],
    ["mask", mask],
  ]) {
    if (!field || field.length !== cellCount) {
      throw new Error(`Isostatic-rebound solver received a mismatched ${name} field.`);
    }
  }

  const resolvedModel = model === REBOUND_MODEL_LOCAL || flexuralRigidityNm <= 0
    ? REBOUND_MODEL_LOCAL
    : REBOUND_MODEL_FLEXURAL;
  const datum = Number.isFinite(seaLevelMeters) ? seaLevelMeters : 0;
  const load = computeBedLoadPascals({ cellCount, bedHeights, surfaceHeights, thickness, mask });

  const solved =
    resolvedModel === REBOUND_MODEL_FLEXURAL
      ? solveFlexuralRebound({
          nx,
          ny,
          cellCount,
          dxMeters: grid.dx_m,
          dyMeters: grid.dy_m,
          load,
          bedHeights,
          seaLevelMeters: datum,
          flexuralRigidityNm,
          onProgress,
        })
      : solveLocalRebound({ nx, ny, cellCount, load, bedHeights, seaLevelMeters: datum, onProgress });

  const emergent = new Uint8Array(cellCount);
  const stats = summarise({
    nx,
    ny,
    cellCount,
    grid,
    standardParallelDegrees,
    bedHeights,
    thickness,
    mask,
    uplift: solved.uplift,
    emergent,
    seaLevelMeters: datum,
    model: resolvedModel,
    flexuralRigidityNm,
    solved,
  });

  return { uplift: solved.uplift, emergent, stats };
}

/**
 * Bed elevation part-way through the scenario: bed + fraction * equilibrium uplift.
 * Non-finite bed cells are left untouched so mesh validity masks stay reusable.
 */
export function applyReboundFraction({ cellCount, bedHeights, uplift, fraction, out }) {
  const scale = clampFraction(fraction);
  const target = out || new Float32Array(cellCount);
  for (let index = 0; index < cellCount; index += 1) {
    const bed = bedHeights[index];
    target[index] = Number.isFinite(bed) ? bed + scale * uplift[index] : bed;
  }
  return target;
}

/**
 * Ice geometry part-way through the scenario. Ice thins uniformly by `fraction` while
 * the bed relaxes by the same fraction, so the two advance together as one illustrative
 * deglaciation path. Grounded ice rides the rebounding bed; floating ice re-floats to
 * the sea-level datum and grounds where the rising bed overtakes its draft.
 */
export function deriveReboundedIceSurfaces({
  cellCount,
  reboundedBedHeights,
  thickness,
  mask,
  iceValid,
  fraction,
  seaLevelMeters = 0,
  surfaceOut,
  bottomOut,
}) {
  const remaining = 1 - clampFraction(fraction);
  const datum = Number.isFinite(seaLevelMeters) ? seaLevelMeters : 0;
  const surface = surfaceOut || new Float32Array(cellCount);
  const bottom = bottomOut || new Float32Array(cellCount);

  for (let index = 0; index < cellCount; index += 1) {
    if (!iceValid[index]) {
      surface[index] = Number.NaN;
      bottom[index] = Number.NaN;
      continue;
    }
    const bed = reboundedBedHeights[index];
    const remainingThickness = Math.max(0, thickness[index]) * remaining;
    let base = bed;
    if (isFloatingIceMask(mask[index])) {
      const flotationBase = datum - (ICE_DENSITY_KG_M3 * remainingThickness) / SEAWATER_DENSITY_KG_M3;
      base = Math.max(bed, flotationBase);
    }
    bottom[index] = base;
    surface[index] = base + remainingThickness;
  }
  return { surface, bottom };
}
