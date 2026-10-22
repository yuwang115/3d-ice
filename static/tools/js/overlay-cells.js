/**
 * Which grid cells an overlay may be drawn on, shared by the geometry worker and the
 * explorer's main-thread fallbacks so that both apply the same rules.
 *
 * Terrain masks follow BedMachine: 0 ocean, 1 ice-free land, 2 grounded ice, 3 floating
 * ice and, in Antarctica, 4 for Lake Vostok, where the ice rests on the lake.
 *
 * No DOM or scene dependencies: this module runs unchanged under Node's test runner.
 */

export const MASK_OCEAN = 0;
export const MASK_ICE_FREE_LAND = 1;
export const MASK_GROUNDED_ICE = 2;
export const MASK_FLOATING_ICE = 3;
export const MASK_SUBGLACIAL_LAKE = 4;

/** Water kept between an ocean streamline and the bed, and between it and an ice shelf's base. */
export const OCEAN_STREAMLINE_BED_CLEARANCE_M = 20;
// Only a sliver beneath an ice shelf, where the shelf's boundary current runs.
export const OCEAN_STREAMLINE_ICE_CLEARANCE_M = 5;

/** Whether the cell lies beneath grounded ice, where subglacial water flows. */
export function isGroundedIceCell(maskValue) {
  return maskValue === MASK_GROUNDED_ICE || maskValue === MASK_SUBGLACIAL_LAKE;
}

/**
 * The stretch of a channel edge between cell centres a and b to draw, as fractions of the
 * way from a to b: the whole edge beneath grounded ice; from the grounded end to half-way,
 * where the grounded ice ends, when only one end is grounded (an outlet at the grounding
 * line, or a channel reaching the ice margin on land); nothing when neither end is.
 */
export function groundedChannelSpan(maskA, maskB) {
  const groundedA = isGroundedIceCell(maskA);
  const groundedB = isGroundedIceCell(maskB);
  if (groundedA && groundedB) return [0, 1];
  if (groundedA) return [0, 0.5];
  if (groundedB) return [0.5, 1];
  return null;
}

/**
 * Whether a point `depth` metres below sea level lies in open water: over the ocean, or in
 * an ice-shelf cavity, at least `bedClearance` metres above the bed and, beneath a shelf, at
 * least `iceClearance` metres (by default the bed's margin) below the ice base. A shelf
 * cell without a finite ice base is judged by its bed alone.
 */
export function isInOpenWater(maskValue, depth, bedHeight, iceBaseHeight, bedClearance, iceClearance = bedClearance) {
  if (maskValue !== MASK_OCEAN && maskValue !== MASK_FLOATING_ICE) return false;
  if (!Number.isFinite(bedHeight) || -bedHeight < depth + bedClearance) return false;
  if (maskValue === MASK_FLOATING_ICE && Number.isFinite(iceBaseHeight)) {
    return depth >= iceClearance - iceBaseHeight;
  }
  return true;
}
