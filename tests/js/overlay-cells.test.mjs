/**
 * Which cells an overlay may be drawn on. In version 0.3.1 the subglacial-hydrology overlay
 * was also drawn on 522 floating and 359 ice-free cells of the 10 km grid (on the sea floor
 * beneath the shelves, and on bare rock), and 2.5 % of the sub-shelf ocean streamline
 * segments ran inside the ice shelves, because neither filter looked past the bed.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  isGroundedIceCell,
  isInOpenWater,
  MASK_FLOATING_ICE,
  MASK_GROUNDED_ICE,
  MASK_ICE_FREE_LAND,
  MASK_OCEAN,
  MASK_SUBGLACIAL_LAKE,
} from "../../static/tools/js/overlay-cells.js";

const BED_CLEARANCE = 20;
const ICE_CLEARANCE = 5;

test("the mask classes follow BedMachine", () => {
  assert.deepEqual(
    [MASK_OCEAN, MASK_ICE_FREE_LAND, MASK_GROUNDED_ICE, MASK_FLOATING_ICE, MASK_SUBGLACIAL_LAKE],
    [0, 1, 2, 3, 4]
  );
});

test("subglacial water belongs under grounded ice, including the ice over Lake Vostok", () => {
  assert.equal(isGroundedIceCell(MASK_GROUNDED_ICE), true);
  assert.equal(isGroundedIceCell(MASK_SUBGLACIAL_LAKE), true);
  assert.equal(isGroundedIceCell(MASK_FLOATING_ICE), false, "beneath a shelf lies the ocean");
  assert.equal(isGroundedIceCell(MASK_ICE_FREE_LAND), false);
  assert.equal(isGroundedIceCell(MASK_OCEAN), false);
});

test("open-ocean water must clear the bed", () => {
  assert.equal(isInOpenWater(MASK_OCEAN, 100, -500, Number.NaN, BED_CLEARANCE), true);
  assert.equal(isInOpenWater(MASK_OCEAN, 480, -500, Number.NaN, BED_CLEARANCE), true, "exactly the clearance");
  assert.equal(isInOpenWater(MASK_OCEAN, 490, -500, Number.NaN, BED_CLEARANCE), false);
  assert.equal(isInOpenWater(MASK_OCEAN, 100, Number.NaN, Number.NaN, BED_CLEARANCE), false, "no bed, no water");
});

test("cavity water must lie below the ice shelf as well as above the bed", () => {
  // A shelf with its base 300 m below sea level, over a cavity whose floor is at -800 m.
  const inCavity = (depth) => isInOpenWater(MASK_FLOATING_ICE, depth, -800, -300, BED_CLEARANCE, ICE_CLEARANCE);
  assert.equal(inCavity(100), false, "inside the shelf");
  assert.equal(inCavity(302), false, "grazing the ice base");
  assert.equal(inCavity(305), true, "just beneath the ice base, where the shelf's boundary current runs");
  assert.equal(inCavity(790), false, "grazing the cavity floor");
});

test("without its own margin the ice base keeps the bed's", () => {
  assert.equal(isInOpenWater(MASK_FLOATING_ICE, 310, -800, -300, BED_CLEARANCE), false);
  assert.equal(isInOpenWater(MASK_FLOATING_ICE, 320, -800, -300, BED_CLEARANCE), true);
});

test("a shelf cell with no ice base is judged by its bed alone", () => {
  assert.equal(isInOpenWater(MASK_FLOATING_ICE, 100, -800, Number.NaN, BED_CLEARANCE, ICE_CLEARANCE), true);
});

test("land, grounded ice and Lake Vostok hold no ocean", () => {
  for (const mask of [MASK_ICE_FREE_LAND, MASK_GROUNDED_ICE, MASK_SUBGLACIAL_LAKE]) {
    assert.equal(isInOpenWater(mask, 100, -2000, -1000, BED_CLEARANCE, ICE_CLEARANCE), false, `mask ${mask}`);
  }
});
