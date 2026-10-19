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

/** Whether the cell lies beneath grounded ice, where subglacial water flows. */
export function isGroundedIceCell(maskValue) {
  return maskValue === MASK_GROUNDED_ICE || maskValue === MASK_SUBGLACIAL_LAKE;
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
