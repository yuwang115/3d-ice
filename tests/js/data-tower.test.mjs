/**
 * The data tower's geometry (static/js/3d-ice-data-tower-geometry.js): the orthographic view
 * of the stack, the canvas transform that draws each layer, the hit test and the label layout.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_PITCH,
  PITCH_MAX,
  PITCH_MIN,
  applyTransform,
  approach,
  clamp,
  easeInOut,
  farAndNearCorner,
  fitView,
  focusStyle,
  invertTransform,
  layerCorners,
  layerHalfExtents,
  layerHeights,
  layerTransform,
  pickLayer,
  projectPoint,
  rightmostCorner,
  spreadLabels,
  wrapYaw,
} from "../../static/js/3d-ice-data-tower-geometry.js";

const close = (actual, expected, tolerance = 1e-9, message = "") =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${message} ${actual} != ${expected}`);

const VIEW = { yaw: 0.7, pitch: 0.5, scale: 120, cx: 300, cy: 400 };

test("the default pitch lies inside the range a drag may tilt to", () => {
  assert.ok(PITCH_MIN < DEFAULT_PITCH && DEFAULT_PITCH < PITCH_MAX);
  assert.equal(clamp(2, PITCH_MIN, PITCH_MAX), PITCH_MAX);
  assert.equal(clamp(-1, PITCH_MIN, PITCH_MAX), PITCH_MIN);
});

test("yaw folds into one turn", () => {
  close(wrapYaw(2 * Math.PI + 0.25), 0.25);
  close(wrapYaw(-0.25), 2 * Math.PI - 0.25);
});

test("a layer's half extents keep its aspect and fit the same circle", () => {
  const square = layerHalfExtents(1);
  close(square.halfWidth, square.halfDepth);
  close(Math.hypot(square.halfWidth, square.halfDepth), 1);
  const tall = layerHalfExtents(0.5, 2);
  close(tall.halfWidth / tall.halfDepth, 0.5);
  close(Math.hypot(tall.halfWidth, tall.halfDepth), 2);
  assert.deepEqual(layerHalfExtents(0), layerHalfExtents(1), "a bad aspect falls back to a square");
});

test("the camera looks down: higher points sit higher on screen and nearer the camera", () => {
  const low = projectPoint(VIEW, 0.2, 0, -0.1);
  const high = projectPoint(VIEW, 0.2, 1, -0.1);
  close(high.x, low.x);
  close(low.y - high.y, VIEW.scale * Math.cos(VIEW.pitch));
  assert.ok(high.depth > low.depth);
  // At yaw 0 the near edge of a layer (z > 0) is drawn below its far edge.
  const ahead = { ...VIEW, yaw: 0 };
  assert.ok(projectPoint(ahead, 0, 0, 1).y > projectPoint(ahead, 0, 0, -1).y);
});

test("a layer image maps onto its rectangle, corner to corner", () => {
  const extents = layerHalfExtents(0.6);
  const [width, height] = [300, 500];
  const transform = layerTransform(VIEW, extents, 0.8, width, height);
  const corners = layerCorners(VIEW, extents, 0.8);
  const pixels = [
    [0, 0],
    [width, 0],
    [width, height],
    [0, height],
  ];
  pixels.forEach(([px, py], i) => {
    const point = applyTransform(transform, px, py);
    close(point.x, corners[i].x, 1e-6, `corner ${i} x`);
    close(point.y, corners[i].y, 1e-6, `corner ${i} y`);
  });
});

test("the inverse transform takes a screen point back to the image pixel under it", () => {
  const transform = layerTransform(VIEW, layerHalfExtents(1), 0.3, 600, 600);
  const inverse = invertTransform(transform);
  const screen = applyTransform(transform, 123.5, 456.25);
  const back = applyTransform(inverse, screen.x, screen.y);
  close(back.x, 123.5, 1e-6);
  close(back.y, 456.25, 1e-6);
  assert.equal(invertTransform([1, 2, 2, 4, 0, 0]), null, "a singular transform has no inverse");
});

test("the far and near corners and the rightmost one follow the yaw", () => {
  const extents = layerHalfExtents(1);
  for (const yaw of [0.3, 1.9, 3.5, 5.2]) {
    const corners = layerCorners({ ...VIEW, yaw }, extents, 0);
    const { far, near } = farAndNearCorner(corners);
    assert.notEqual(far, near);
    assert.equal((far + 2) % 4, near, "the far and near corners are opposite");
    const right = rightmostCorner(corners);
    assert.ok(corners.every((corner) => corner.x <= right.x));
  }
});

test("layers stand evenly from the top of the tower to its base", () => {
  assert.deepEqual(layerHeights(1, 2), [0]);
  const heights = layerHeights(5, 2);
  assert.deepEqual(heights, [2, 1.5, 1, 0.5, 0]);
});

test("the view fits the whole tower at every yaw, within the share of width it is given", () => {
  const options = { width: 1000, height: 700, pitch: 0.5, radius: 1, towerHeight: 2.2, baseDepth: 0.1, padding: 20, towerShare: 0.6, towerCentre: 0.3 };
  const fit = fitView(options);
  const extents = layerHalfExtents(1);
  for (let yaw = 0; yaw < 2 * Math.PI; yaw += 0.2) {
    const view = { ...fit, yaw, pitch: options.pitch };
    const points = [...layerCorners(view, extents, options.towerHeight), ...layerCorners(view, extents, -options.baseDepth)];
    for (const point of points) {
      assert.ok(point.y >= options.padding - 1e-6 && point.y <= options.height - options.padding + 1e-6, `y ${point.y} at yaw ${yaw}`);
      assert.ok(point.x >= options.padding - 1e-6 && point.x <= options.width * options.towerShare - options.padding + 1e-6, `x ${point.x}`);
    }
  }
  assert.equal(fit.cx, 300);
});

test("a pick finds the topmost layer opaque under the pointer", () => {
  const extents = layerHalfExtents(1);
  const make = (y, opaque) => ({
    transform: layerTransform(VIEW, extents, y, 100, 100),
    width: 100,
    height: 100,
    isOpaque: opaque,
  });
  const centre = projectPoint(VIEW, 0, 0.5, 0);
  const everywhere = () => true;
  const nowhere = () => false;
  // Three layers, the top one empty: a point over the middle of the stack picks layer 1.
  const layers = [make(1, nowhere), make(0.5, everywhere), make(0, everywhere)];
  assert.equal(pickLayer(layers, centre.x, centre.y), 1);
  assert.equal(pickLayer([{ ...layers[1], pickable: false }, layers[2]], centre.x, centre.y), 1);
  assert.equal(pickLayer(layers, -5000, -5000), -1, "nothing is under a point off the stack");
  assert.equal(pickLayer([null, { transform: null }], centre.x, centre.y), -1);
});

test("labels keep their order and their gap, and share the room when it is short", () => {
  assert.deepEqual(spreadLabels([], 10), []);
  assert.deepEqual(spreadLabels([0, 50, 100], 20), [0, 50, 100], "well apart, nothing moves");
  const crowded = spreadLabels([100, 101, 102], 30);
  assert.ok(crowded[1] - crowded[0] >= 30 - 1e-9 && crowded[2] - crowded[1] >= 30 - 1e-9);
  const bounded = spreadLabels([0, 1, 2], 30, 10, 200);
  assert.ok(bounded[0] >= 10 && bounded[2] <= 200);
  const squeezed = spreadLabels([0, 1, 2, 3], 100, 0, 90);
  assert.deepEqual(squeezed, [0, 30, 60, 90]);
});

test("motion helpers ease, approach and fade as the focus asks", () => {
  assert.equal(easeInOut(-1), 0);
  assert.equal(easeInOut(2), 1);
  close(easeInOut(0.5), 0.5);
  close(approach(0, 1, 1, 1), 1 - Math.exp(-1));
  assert.equal(approach(0, 1, 1, 0), 1);
  assert.deepEqual(focusStyle(3, -1), { opacity: 1, lift: 0 });
  assert.deepEqual(focusStyle(3, 3), { opacity: 1, lift: 1 });
  assert.ok(focusStyle(1, 3).opacity < focusStyle(5, 3).opacity, "layers above the focus are the most see-through");
});
