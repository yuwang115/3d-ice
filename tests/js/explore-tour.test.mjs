/**
 * Pure helpers behind the public edition's guided tour: where the camera goes for a
 * chapter, how it gets there, and the order in which a chapter flips layer toggles.
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  TOUR_CONTROL_DEFAULTS,
  clampChapterIndex,
  easeInOutCubic,
  flightDurationMs,
  framePose,
  interpolatePose,
  orbitPose,
  planToggleChanges,
  resolveChapterControls,
  viewShiftFor,
} from "../../static/tools/js/explore-tour.js";

const EPSILON = 1e-9;

function assertClose(actual, expected, tolerance = 1e-6, label = "") {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${label} ${actual} vs ${expected}`);
}

function assertPoseClose(actual, expected, tolerance = 1e-6) {
  for (let axis = 0; axis < 3; axis += 1) {
    assertClose(actual.position[axis], expected.position[axis], tolerance, `position[${axis}]`);
    assertClose(actual.target[axis], expected.target[axis], tolerance, `target[${axis}]`);
  }
  assertClose(actual.fov, expected.fov, tolerance, "fov");
}

function offsetOf(pose) {
  return pose.position.map((value, axis) => value - pose.target[axis]);
}

function azimuthOf(pose) {
  const [x, , z] = offsetOf(pose);
  return (Math.atan2(x, z) * 180) / Math.PI;
}

function radiusOf(pose) {
  return Math.hypot(...offsetOf(pose));
}

test("an orbit pose sits at the requested distance, azimuth and elevation from its target", () => {
  const target = [4, -2, 7];
  const rise = 10 * Math.sin(Math.PI / 6);
  const run = 10 * Math.cos(Math.PI / 6);

  // Azimuth 0 is the default view's side of the target: +z, the map's lower edge.
  const south = orbitPose({ target, distance: 10, azimuthDeg: 0, elevationDeg: 30, fov: 40 });
  assertPoseClose(south, { position: [4, -2 + rise, 7 + run], target, fov: 40 });

  const east = orbitPose({ target, distance: 10, azimuthDeg: 90, elevationDeg: 30, fov: 40 });
  assertPoseClose(east, { position: [4 + run, -2 + rise, 7], target, fov: 40 });

  const oblique = orbitPose({ target: [0, 0, 0], distance: 20, azimuthDeg: -35, elevationDeg: 40 });
  assertClose(radiusOf(oblique), 20);
  assertClose(azimuthOf(oblique), -35);
  assertClose((Math.asin(oblique.position[1] / 20) * 180) / Math.PI, 40);
});

test("an orbit pose keeps the camera above the horizon, off the pole and at a positive distance", () => {
  const low = orbitPose({ target: [0, 0, 0], distance: -5, azimuthDeg: 0, elevationDeg: -30 });
  assert.ok(low.position[1] > 0, "camera stays above its target");
  assert.ok(radiusOf(low) > 0);

  // Straight down is the orbit controls' singular direction, so the camera stops short of it.
  const overhead = orbitPose({ target: [0, 0, 0], distance: 10, azimuthDeg: 30, elevationDeg: 120 });
  assert.ok(Math.hypot(overhead.position[0], overhead.position[2]) > 1e-3, "camera keeps a horizontal offset");
  assert.ok(overhead.position[1] > 9.9);
});

/**
 * Where a world point lands on screen for a camera pose, in pixels from the top left,
 * including the pose's view shift (normalised device units, +x right and +y up).
 */
function projectToPixels(pose, point, width, height) {
  const forward = pose.target.map((value, axis) => value - pose.position[axis]);
  const length = Math.hypot(...forward);
  const f = forward.map((value) => value / length);
  const right = [-f[2], 0, f[0]].map((value, _axis, all) => value / Math.hypot(...all));
  const up = [
    right[1] * f[2] - right[2] * f[1],
    right[2] * f[0] - right[0] * f[2],
    right[0] * f[1] - right[1] * f[0],
  ];
  const v = point.map((value, axis) => value - pose.position[axis]);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const tanHalf = Math.tan((pose.fov * Math.PI) / 360);
  const depth = dot(v, f);
  const [shiftX, shiftY] = pose.shift || [0, 0];
  const xNdc = dot(v, right) / depth / tanHalf / (width / height) + shiftX;
  const yNdc = dot(v, up) / depth / tanHalf + shiftY;
  return [width / 2 + xNdc * (width / 2), height / 2 - yNdc * (height / 2)];
}

test("a framed pose keeps its pivot on the target and shifts the view to clear page UI", () => {
  const target = [12, 3, -8];
  const viewport = { width: 375, height: 812 };
  const uncovered = { top: 56, bottom: 460 };
  const pose = framePose({ target, diameter: 40, azimuthDeg: -30, elevationDeg: 50, fovDeg: 42, viewport, uncovered });
  assert.deepEqual(pose.target, target, "the orbit pivot stays on the point being shown");
  assertClose(pose.shift[0], 0, 1e-12, "no sideways shift");
  assertClose(pose.shift[1], (460 - 56) / 812, 1e-12, "shifted up by half the uncovered offset");
  const [x, y] = projectToPixels(pose, target, viewport.width, viewport.height);
  assertClose(x, 375 / 2, 1e-6, "x");
  assertClose(y, (56 + 812 - 460) / 2, 1e-6, "y");

  const side = framePose({
    target,
    diameter: 40,
    azimuthDeg: 20,
    elevationDeg: 40,
    fovDeg: 42,
    viewport: { width: 1200, height: 800 },
    uncovered: { left: 420 },
  });
  const [sx, sy] = projectToPixels(side, target, 1200, 800);
  assertClose(sx, (420 + 1200) / 2, 1e-6, "side x");
  assertClose(sy, 400, 1e-6, "side y");
});

test("a framed pose with nothing covered is centred", () => {
  const pose = framePose({ target: [0, 0, 0], diameter: 40, azimuthDeg: 0, elevationDeg: 45, viewport: { width: 800, height: 600 } });
  assert.deepEqual(pose.shift, [0, 0]);
  assert.deepEqual(viewShiftFor({ viewport: { width: 800, height: 600 } }), [0, 0]);
  assert.deepEqual(viewShiftFor({ viewport: { width: 800, height: 600 }, uncovered: { right: 200, top: 60 } }), [-0.25, -0.1]);
});

test("on a narrow screen a framed pose widens the view rather than crop the disc", () => {
  // A phone in portrait, with the tour card over the lower half of the view.
  const viewport = { width: 390, height: 844 };
  const uncovered = { top: 56, bottom: 422 };
  const base = { target: [0, 0, 0], diameter: 107.6, azimuthDeg: 0, elevationDeg: 55, viewport, uncovered, maxDistance: 150 };
  const cropped = framePose({ ...base, fovDeg: 42 });
  assert.equal(cropped.fov, 42, "without a wider limit the field of view stays put");
  const widened = framePose({ ...base, fovDeg: 42, maxFovDeg: 75 });
  assert.ok(widened.fov > 42 && widened.fov <= 75, `fov ${widened.fov}`);
  assert.ok(radiusOf(widened) <= 150 + 1e-9);
  const [x, y] = projectToPixels(widened, base.target, viewport.width, viewport.height);
  assertClose(x, 195, 1e-6, "x");
  assertClose(y, (56 + 844 - 422) / 2, 1e-6, "y");
  // A wide screen that fits already keeps its field of view.
  const desktop = framePose({ ...base, fovDeg: 42, maxFovDeg: 75, viewport: { width: 900, height: 760 }, uncovered: {} });
  assert.equal(desktop.fov, 42);
});

test("a framed pose backs off until the disc fits the tighter side of the uncovered part", () => {
  const base = { target: [0, 0, 0], diameter: 40, azimuthDeg: 0, elevationDeg: 45, fovDeg: 42 };
  const open = framePose({ ...base, viewport: { width: 800, height: 800 } });
  const half = framePose({ ...base, viewport: { width: 800, height: 800 }, uncovered: { bottom: 400 } });
  assertClose(radiusOf(half), 2 * radiusOf(open), 1e-6, "half the height doubles the distance");
  assertClose(radiusOf(open), 20 / Math.tan((21 * Math.PI) / 180), 1e-6, "an open square view");
  const capped = framePose({ ...base, viewport: { width: 800, height: 800 }, maxDistance: 10 });
  assertClose(radiusOf(capped), 10, 1e-6, "the distance cap holds");
  // A viewport covered completely still yields a finite pose.
  const covered = framePose({ ...base, viewport: { width: 800, height: 800 }, uncovered: { bottom: 900 } });
  assert.ok(covered.position.every(Number.isFinite) && covered.shift.every(Number.isFinite));
});

test("a flight moves the view shift smoothly, and a pose without one counts as centred", () => {
  const from = orbitPose({ target: [0, 0, 0], distance: 30, azimuthDeg: 0, elevationDeg: 40 });
  const to = { ...from, shift: [0, 0.4] };
  assertClose(interpolatePose(from, to, 0.5).shift[1], 0.2, 1e-12);
  assert.deepEqual(interpolatePose(from, to, 0).shift, [0, 0]);
  assert.deepEqual(interpolatePose(from, to, 1).shift, [0, 0.4]);
  assert.ok(flightDurationMs(from, to) > 0, "re-centring alone is still a flight");
});

test("a flight starts and ends exactly on its two poses", () => {
  const from = orbitPose({ target: [0, 0, 0], distance: 90, azimuthDeg: 0, elevationDeg: 40, fov: 45 });
  const to = orbitPose({ target: [-30, 1, 10], distance: 25, azimuthDeg: -70, elevationDeg: 35, fov: 40 });
  assertPoseClose(interpolatePose(from, to, 0), from);
  assertPoseClose(interpolatePose(from, to, 1), to);
});

test("a flight swings round the short way", () => {
  const from = orbitPose({ target: [0, 0, 0], distance: 30, azimuthDeg: 170, elevationDeg: 40 });
  const to = orbitPose({ target: [0, 0, 0], distance: 30, azimuthDeg: -170, elevationDeg: 40 });
  const midway = azimuthOf(interpolatePose(from, to, 0.5));
  assertClose(Math.abs(midway), 180, 1e-6, "midway azimuth");
});

test("a long flight lifts the camera out and back; a short one does not", () => {
  const from = orbitPose({ target: [0, 0, 0], distance: 30, azimuthDeg: 0, elevationDeg: 40 });
  const far = orbitPose({ target: [80, 0, 0], distance: 30, azimuthDeg: 0, elevationDeg: 40 });
  assert.ok(radiusOf(interpolatePose(from, far, 0.5)) > 30 + EPSILON, "long hop rises");

  const near = orbitPose({ target: [1, 0, 0], distance: 30, azimuthDeg: 0, elevationDeg: 40 });
  for (const t of [0.25, 0.5, 0.75]) {
    assertClose(radiusOf(interpolatePose(from, near, t)), 30, 1e-6, `short hop at ${t}`);
  }
});

test("a flight interpolates the field of view", () => {
  const from = orbitPose({ target: [0, 0, 0], distance: 30, azimuthDeg: 0, elevationDeg: 40, fov: 45 });
  const to = orbitPose({ target: [0, 0, 0], distance: 30, azimuthDeg: 0, elevationDeg: 40, fov: 35 });
  assertClose(interpolatePose(from, to, 0.5).fov, 40);
});

test("longer flights take longer, within fixed bounds", () => {
  const from = orbitPose({ target: [0, 0, 0], distance: 30, azimuthDeg: 0, elevationDeg: 40 });
  const near = orbitPose({ target: [2, 0, 0], distance: 30, azimuthDeg: 5, elevationDeg: 40 });
  const far = orbitPose({ target: [90, 0, 40], distance: 60, azimuthDeg: 120, elevationDeg: 40 });
  const short = flightDurationMs(from, near);
  const long = flightDurationMs(from, far);
  assert.ok(long > short);
  assert.ok(short >= 900 && long <= 3200, `${short}..${long}`);
  assert.equal(flightDurationMs(from, from), 0, "no flight when the camera is already there");
});

test("the easing curve starts slow, ends slow and is symmetric", () => {
  assert.equal(easeInOutCubic(0), 0);
  assert.equal(easeInOutCubic(1), 1);
  assertClose(easeInOutCubic(0.5), 0.5);
  assert.ok(easeInOutCubic(0.1) < 0.1);
  assert.ok(easeInOutCubic(0.9) > 0.9);
  assertClose(easeInOutCubic(0.3) + easeInOutCubic(0.7), 1);
  assert.equal(easeInOutCubic(-1), 0);
  assert.equal(easeInOutCubic(2), 1);
});

test("a chapter switches layers off before it switches any on", () => {
  const current = { showFlowline: true, showOceanCurrents: false, showSea: false, showBed: true };
  const desired = { showFlowline: false, showOceanCurrents: true, showSea: true, showBed: true };
  assert.deepEqual(planToggleChanges(current, desired), [
    { id: "showFlowline", checked: false },
    { id: "showOceanCurrents", checked: true },
    { id: "showSea", checked: true },
  ]);
});

test("the rebound layer is switched on last, because switching it on hides the flow layers", () => {
  const current = { showIsostaticRebound: false, showSea: false, showFlowline: true };
  const desired = { showIsostaticRebound: true, showSea: true, showFlowline: false };
  const plan = planToggleChanges(current, desired);
  assert.deepEqual(plan.at(-1), { id: "showIsostaticRebound", checked: true });
  assert.deepEqual(plan[0], { id: "showFlowline", checked: false });
});

test("the rebound layer is switched off before a flow layer comes back", () => {
  const plan = planToggleChanges(
    { showIsostaticRebound: true, showFlowline: false },
    { showIsostaticRebound: false, showFlowline: true }
  );
  assert.deepEqual(plan, [
    { id: "showIsostaticRebound", checked: false },
    { id: "showFlowline", checked: true },
  ]);
});

test("a chapter leaves unchanged toggles alone", () => {
  assert.deepEqual(planToggleChanges({ showBed: true }, { showBed: true }), []);
  assert.deepEqual(planToggleChanges({ showBed: true, showSea: true }, { showBed: true }), []);
});

test("a chapter's controls fill in every public toggle, so chapters can be visited in any order", () => {
  const controls = resolveChapterControls({ showFlowline: true });
  assert.deepEqual(Object.keys(controls).sort(), Object.keys(TOUR_CONTROL_DEFAULTS).sort());
  assert.equal(controls.showFlowline, true);
  assert.equal(controls.showIce, TOUR_CONTROL_DEFAULTS.showIce);
  assert.throws(() => resolveChapterControls({ showBasalFriction: true }), /showBasalFriction/);
});

test("chapter indices clamp to the chapters that exist", () => {
  assert.equal(clampChapterIndex(-3, 8), 0);
  assert.equal(clampChapterIndex(3, 8), 3);
  assert.equal(clampChapterIndex(12, 8), 7);
  assert.equal(clampChapterIndex(2.6, 8), 2);
  assert.equal(clampChapterIndex(Number.NaN, 8), 0);
  assert.equal(clampChapterIndex(0, 0), -1);
});
