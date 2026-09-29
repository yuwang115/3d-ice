/**
 * Pure helpers behind the public edition's guided tour: the camera pose a chapter asks
 * for, the flight between two poses, and the order in which a chapter flips layers.
 *
 * Poses are { position: [x, y, z], target: [x, y, z], fov, shift } in scene units, the
 * form the runtime's applyCameraPose() takes. `shift` is an optional [x, y] offset of the
 * image in normalised device units (+x right, +y up) that keeps the orbit target clear of
 * page UI covering part of the view; a pose without one is centred. No DOM or scene
 * dependencies: this module runs unchanged under Node's test runner.
 */

const DEG = Math.PI / 180;
const MIN_ORBIT_DISTANCE = 0.5;
const DEFAULT_ELEVATION_DEG = 45;
// Straight down is the orbit controls' singular direction, and a camera level with its
// target looks under the terrain, so elevations stay inside this range.
const MIN_ELEVATION_DEG = 1;
const MAX_ELEVATION_DEG = 89;
const DEFAULT_FOV = 42;

const MIN_FLIGHT_MS = 900;
const MAX_FLIGHT_MS = 3200;
const FLIGHT_MS_PER_UNIT_TRAVEL = 900;
// A flight whose target moves further than this share of the orbit radius pulls the
// camera back mid-way, so the viewer keeps the context of where it is going.
const HOP_THRESHOLD_RATIO = 0.5;
const HOP_PER_UNIT_TRAVEL = 0.35;
const MAX_HOP_RATIO = 0.8;

/**
 * The layer toggles the tour drives, and their state in a chapter that does not mention
 * them. Every chapter resolves to a full state, so chapters can be visited in any order.
 */
export const TOUR_CONTROL_DEFAULTS = Object.freeze({
  showIce: true,
  showBed: true,
  showFlowline: false,
  showOceanCurrents: false,
  showSea: false,
  showIsostaticRebound: false,
  showGeographicNames: false,
  showResearchStations: false,
});

// Switching the rebound layer on hides every overlay drawn on today's surfaces, so it
// must come after the toggles it would otherwise undo.
const SWITCHED_ON_LAST = new Set(["showIsostaticRebound"]);

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function lerp(from, to, t) {
  return from + (to - from) * t;
}

function subtract(left, right) {
  return [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
}

function distanceBetween(left, right) {
  return Math.hypot(...subtract(left, right));
}

function shiftOf(pose) {
  return Array.isArray(pose.shift) ? pose.shift : [0, 0];
}

function clonePose(pose) {
  return { position: [...pose.position], target: [...pose.target], fov: pose.fov, shift: [...shiftOf(pose)] };
}

/** Signed angle from `from` to `to`, the short way round, in (-pi, pi]. */
function shortestAngle(from, to) {
  let delta = (to - from) % (2 * Math.PI);
  if (delta > Math.PI) delta -= 2 * Math.PI;
  if (delta <= -Math.PI) delta += 2 * Math.PI;
  return delta;
}

function toSpherical(pose) {
  const [x, y, z] = subtract(pose.position, pose.target);
  const radius = Math.hypot(x, y, z) || MIN_ORBIT_DISTANCE;
  return {
    radius,
    azimuth: Math.atan2(x, z),
    elevation: Math.asin(clamp(y / radius, -1, 1)),
  };
}

function positionAround(target, radius, azimuth, elevation) {
  const horizontal = radius * Math.cos(elevation);
  return [
    target[0] + horizontal * Math.sin(azimuth),
    target[1] + radius * Math.sin(elevation),
    target[2] + horizontal * Math.cos(azimuth),
  ];
}

/**
 * A camera on a sphere around `target`. Azimuth 0 puts it on the +z side, the lower map
 * edge the default views look from, and 90 on the +x side; elevation is measured up from
 * the horizontal.
 */
export function orbitPose({ target, distance, azimuthDeg = 0, elevationDeg = DEFAULT_ELEVATION_DEG, fov = DEFAULT_FOV }) {
  const radius = Math.max(MIN_ORBIT_DISTANCE, Number(distance) || 0);
  const elevationValue = Number(elevationDeg);
  const elevation =
    clamp(Number.isFinite(elevationValue) ? elevationValue : DEFAULT_ELEVATION_DEG, MIN_ELEVATION_DEG, MAX_ELEVATION_DEG) * DEG;
  const azimuth = (Number(azimuthDeg) || 0) * DEG;
  const fovValue = Number(fov);
  return {
    position: positionAround(target, radius, azimuth, elevation),
    target: [...target],
    fov: Number.isFinite(fovValue) ? fovValue : DEFAULT_FOV,
  };
}

function coveredMargins(viewport, uncovered = {}) {
  const width = Math.max(1, Number(viewport?.width) || 1);
  const height = Math.max(1, Number(viewport?.height) || 1);
  const margin = (side, extent) => clamp(Number(uncovered[side]) || 0, 0, extent - 1);
  const left = margin("left", width);
  const right = margin("right", width - left);
  const top = margin("top", height);
  const bottom = margin("bottom", height - top);
  return { width, height, left, right, top, bottom };
}

/**
 * The view shift that moves the middle of a viewport (CSS pixels) to the middle of the
 * part `uncovered` (the covered margins { top, right, bottom, left }) leaves free.
 */
export function viewShiftFor({ viewport, uncovered = {} }) {
  const { width, height, left, right, top, bottom } = coveredMargins(viewport, uncovered);
  return [(left - right) / width, (bottom - top) / height];
}

/**
 * Frame a disc of `diameter` scene units round `target` inside the part of the viewport
 * that page UI leaves uncovered (see viewShiftFor). The camera backs off until the disc
 * fits that part, up to `maxDistance`; where that is not far enough it widens its field
 * of view instead, up to `maxFovDeg`. The orbit target stays on `target`, so dragging
 * afterwards turns the view about the point being shown, and the pose's view shift puts
 * that point in the middle of the uncovered part.
 */
export function framePose({
  target,
  diameter,
  azimuthDeg,
  elevationDeg,
  fovDeg = DEFAULT_FOV,
  maxFovDeg = fovDeg,
  viewport,
  uncovered = {},
  maxDistance = Infinity,
}) {
  const { width, height, left, right, top, bottom } = coveredMargins(viewport, uncovered);
  const freeExtent = Math.max(1, Math.min(width - left - right, height - top - bottom));
  const halfDiameter = Math.max(0, Number(diameter) || 0) / 2;
  const distanceFor = (tanHalf) => (halfDiameter * height) / (tanHalf * freeExtent);

  let fov = Number(fovDeg) || DEFAULT_FOV;
  let tanHalf = Math.tan((fov * DEG) / 2);
  const widest = Math.max(fov, Number(maxFovDeg) || fov);
  if (distanceFor(tanHalf) > maxDistance && widest > fov) {
    tanHalf = Math.min((halfDiameter * height) / (maxDistance * freeExtent), Math.tan((widest * DEG) / 2));
    fov = (2 * Math.atan(tanHalf)) / DEG;
  }
  const pose = orbitPose({ target, distance: Math.min(maxDistance, distanceFor(tanHalf)), azimuthDeg, elevationDeg, fov });
  return { ...pose, fov, shift: viewShiftFor({ viewport, uncovered }) };
}

function hopRatio(targetTravel, radius) {
  const excess = targetTravel / radius - HOP_THRESHOLD_RATIO;
  return excess > 0 ? Math.min(MAX_HOP_RATIO, excess * HOP_PER_UNIT_TRAVEL) : 0;
}

/**
 * The pose a fraction `t` of the way from one pose to another. The target moves in a
 * straight line, the camera orbits round it the short way, the zoom changes at a constant
 * rate, and a long flight lifts out and back. Callers apply their own easing to `t`.
 */
export function interpolatePose(from, to, t) {
  const s = clamp(Number(t) || 0, 0, 1);
  if (s <= 0) return clonePose(from);
  if (s >= 1) return clonePose(to);

  const start = toSpherical(from);
  const end = toSpherical(to);
  const fromShift = shiftOf(from);
  const toShift = shiftOf(to);
  const target = from.target.map((value, axis) => lerp(value, to.target[axis], s));
  const zoomedRadius = start.radius * (end.radius / start.radius) ** s;
  const hop = hopRatio(distanceBetween(from.target, to.target), Math.max(start.radius, end.radius));
  const radius = zoomedRadius * (1 + hop * Math.sin(Math.PI * s));
  const azimuth = start.azimuth + shortestAngle(start.azimuth, end.azimuth) * s;
  const elevation = lerp(start.elevation, end.elevation, s);
  return {
    position: positionAround(target, radius, azimuth, elevation),
    target,
    fov: lerp(from.fov, to.fov, s),
    shift: [lerp(fromShift[0], toShift[0], s), lerp(fromShift[1], toShift[1], s)],
  };
}

/** How long a flight between two poses should take; 0 when there is nowhere to go. */
export function flightDurationMs(from, to) {
  const start = toSpherical(from);
  const end = toSpherical(to);
  const meanRadius = (start.radius + end.radius) / 2;
  const travel =
    distanceBetween(from.target, to.target) / meanRadius +
    Math.abs(shortestAngle(start.azimuth, end.azimuth)) / Math.PI +
    Math.abs(end.elevation - start.elevation) / (Math.PI / 2) +
    Math.abs(Math.log(end.radius / start.radius)) +
    Math.abs((Number(to.fov) || 0) - (Number(from.fov) || 0)) / 45 +
    Math.abs(shiftOf(to)[0] - shiftOf(from)[0]) +
    Math.abs(shiftOf(to)[1] - shiftOf(from)[1]);
  if (travel < 1e-6) return 0;
  return Math.round(clamp(MIN_FLIGHT_MS + FLIGHT_MS_PER_UNIT_TRAVEL * travel, MIN_FLIGHT_MS, MAX_FLIGHT_MS));
}

export function easeInOutCubic(t) {
  const s = clamp(Number(t) || 0, 0, 1);
  return s < 0.5 ? 4 * s ** 3 : 1 - (-2 * s + 2) ** 3 / 2;
}

/** A chapter's full toggle state: its own settings over the defaults. */
export function resolveChapterControls(controls = {}) {
  for (const id of Object.keys(controls)) {
    if (!Object.prototype.hasOwnProperty.call(TOUR_CONTROL_DEFAULTS, id)) {
      throw new Error(`The tour cannot drive control: ${id}`);
    }
  }
  return { ...TOUR_CONTROL_DEFAULTS, ...controls };
}

/**
 * The toggle changes that take `current` to `desired`, in the order to make them:
 * everything switched off first, then everything switched on, with the rebound layer last.
 * Toggles `desired` does not mention are left alone.
 */
export function planToggleChanges(current, desired) {
  const rank = ({ id, checked }) => (!checked ? 0 : SWITCHED_ON_LAST.has(id) ? 2 : 1);
  return Object.entries(desired)
    .filter(([id, checked]) => Boolean(current[id]) !== Boolean(checked))
    .map(([id, checked], order) => ({ change: { id, checked: Boolean(checked) }, order }))
    .sort((left, right) => rank(left.change) - rank(right.change) || left.order - right.order)
    .map(({ change }) => change);
}

export function clampChapterIndex(index, count) {
  if (!(count > 0)) return -1;
  const value = Number(index);
  if (!Number.isFinite(value)) return 0;
  return clamp(Math.trunc(value), 0, count - 1);
}
