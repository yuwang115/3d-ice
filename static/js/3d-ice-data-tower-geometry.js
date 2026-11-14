/*
 * The geometry of the home page's data tower: a stack of horizontal map layers in a glass box,
 * seen from above through an orthographic camera that turns about the vertical axis.
 *
 * World axes: x to the right and z towards the viewer at yaw 0 (a layer image's columns and
 * rows), y up. A camera at `pitch` above the horizon sees a point at screen
 *   sx = cx + s (x cos(yaw) - z sin(yaw))
 *   sy = cy + s ((x sin(yaw) + z cos(yaw)) sin(pitch) - y cos(pitch)).
 * A horizontal layer therefore maps to the screen by one affine transform, which is how a 2D
 * canvas draws each layer image, and which inverts to tell which pixel of a layer is under
 * the pointer. Higher layers are always nearer the camera than lower ones under the same
 * pixel, so drawing bottom to top paints the stack correctly at any yaw.
 *
 * No DOM dependencies: tests/js/data-tower.test.mjs runs this module under Node.
 */

export const PITCH_MIN = 0.3;
export const PITCH_MAX = 0.95;
export const DEFAULT_PITCH = 0.5;

const TAU = Math.PI * 2;

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

/** The yaw folded into [0, 2π). */
export function wrapYaw(yaw) {
  return ((yaw % TAU) + TAU) % TAU;
}

/**
 * The turn a sideways drag of `dx` pixels gives the stack. A larger yaw carries the front of
 * the stack to the left (its near edge sits at x = -sin(yaw) on screen), so the turn has the
 * drag's opposite sign: the front follows the pointer.
 */
export function dragTurn(dx, radiansPerPx) {
  return -dx * radiansPerPx;
}

/**
 * Half-extents of a layer whose image is `aspect` (width / height) wide, scaled so its half
 * diagonal is `radius`: a square and a tall rectangle then turn inside the same circle.
 */
export function layerHalfExtents(aspect, radius = 1) {
  const safe = Number(aspect) > 0 ? Number(aspect) : 1;
  const halfDepth = radius / Math.sqrt(1 + safe * safe);
  return { halfWidth: safe * halfDepth, halfDepth };
}

/** Screen position of a world point. */
export function projectPoint(view, x, y, z) {
  const cos = Math.cos(view.yaw);
  const sin = Math.sin(view.yaw);
  const across = x * cos - z * sin;
  const toward = x * sin + z * cos;
  return {
    x: view.cx + view.scale * across,
    y: view.cy + view.scale * (toward * Math.sin(view.pitch) - y * Math.cos(view.pitch)),
    // Larger is nearer the camera.
    depth: toward * Math.cos(view.pitch) + y * Math.sin(view.pitch),
  };
}

/**
 * The canvas transform [a, b, c, d, e, f] (as setTransform takes it) that draws an image of
 * `imageWidth` x `imageHeight` pixels as the layer at height `y`.
 */
export function layerTransform(view, { halfWidth, halfDepth }, y, imageWidth, imageHeight) {
  const cos = Math.cos(view.yaw);
  const sin = Math.sin(view.yaw);
  const sinPitch = Math.sin(view.pitch);
  const s = view.scale;
  const ux = (2 * halfWidth) / imageWidth;
  const vz = (2 * halfDepth) / imageHeight;
  return [
    s * cos * ux,
    s * sin * sinPitch * ux,
    -s * sin * vz,
    s * cos * sinPitch * vz,
    view.cx + s * (-halfWidth * cos + halfDepth * sin),
    view.cy + s * (sinPitch * (-halfWidth * sin - halfDepth * cos) - y * Math.cos(view.pitch)),
  ];
}

/** The inverse of an affine transform, or null when it is singular. */
export function invertTransform([a, b, c, d, e, f]) {
  const det = a * d - b * c;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) return null;
  return [d / det, -b / det, -c / det, a / det, (c * f - d * e) / det, (b * e - a * f) / det];
}

export function applyTransform([a, b, c, d, e, f], x, y) {
  return { x: a * x + c * y + e, y: b * x + d * y + f };
}

/** The four corners of a layer's rectangle at height y, projected, in order around it. */
export function layerCorners(view, { halfWidth, halfDepth }, y) {
  return [
    [-halfWidth, -halfDepth],
    [halfWidth, -halfDepth],
    [halfWidth, halfDepth],
    [-halfWidth, halfDepth],
  ].map(([x, z]) => projectPoint(view, x, y, z));
}

/** Index of the corner farthest from the camera, and of the nearest one. */
export function farAndNearCorner(corners) {
  let far = 0;
  let near = 0;
  corners.forEach((corner, i) => {
    if (corner.depth < corners[far].depth) far = i;
    if (corner.depth > corners[near].depth) near = i;
  });
  return { far, near };
}

/** The corner furthest right on screen, where a layer's label line starts. */
export function rightmostCorner(corners) {
  return corners.reduce((best, corner) => (corner.x > best.x ? corner : best), corners[0]);
}

/**
 * Heights of `count` layers, top first: the top one at `height`, the last at 0, evenly apart.
 */
export function layerHeights(count, height) {
  if (count <= 1) return [0];
  return Array.from({ length: count }, (_, i) => height * (1 - i / (count - 1)));
}

/**
 * The scale and centre that fit the tower in a `width` x `height` box at any yaw: a footprint
 * of half diagonal `radius`, layers from 0 to `towerHeight`, and a base `baseDepth` thick.
 * `towerShare` is the fraction of the width the tower may use (the labels take the rest), and
 * `towerCentre` the fraction across the width at which it stands.
 */
export function fitView({ width, height, pitch, radius, towerHeight, baseDepth, padding = 16, towerShare = 1, towerCentre = 0.5 }) {
  const sinPitch = Math.sin(pitch);
  const cosPitch = Math.cos(pitch);
  const usableWidth = Math.max(1, width * towerShare - 2 * padding);
  const usableHeight = Math.max(1, height - 2 * padding);
  const span = towerHeight * cosPitch + 2 * radius * sinPitch + baseDepth * cosPitch;
  const scale = Math.max(1e-6, Math.min(usableWidth / (2 * radius), usableHeight / span));
  // The top of the box (its far corner at full height) and the base's near edge fit exactly.
  const top = -scale * (towerHeight * cosPitch + radius * sinPitch);
  const bottom = scale * (radius * sinPitch + baseDepth * cosPitch);
  return {
    scale,
    cx: width * towerCentre,
    cy: padding + (usableHeight - (bottom - top)) / 2 - top,
  };
}

/**
 * Hit test: the topmost layer (lowest index) whose opaque pixels lie under the screen point.
 * Each layer gives its canvas `transform` and an `isOpaque(px, py)` test in image pixels.
 * Returns the layer's index, or -1.
 */
export function pickLayer(layers, x, y) {
  for (let i = 0; i < layers.length; i += 1) {
    const layer = layers[i];
    if (!layer || !layer.transform || layer.pickable === false) continue;
    const inverse = invertTransform(layer.transform);
    if (!inverse) continue;
    const pixel = applyTransform(inverse, x, y);
    if (pixel.x < 0 || pixel.y < 0 || pixel.x >= layer.width || pixel.y >= layer.height) continue;
    if (layer.isOpaque(pixel.x, pixel.y)) return i;
  }
  return -1;
}

/**
 * Vertical positions for labels that want to sit at `targets` (sorted top to bottom), each
 * at least `gap` below the one above and all within [min, max]: the labels keep their order
 * and move as little as they can. Returns new positions.
 */
export function spreadLabels(targets, gap, min = -Infinity, max = Infinity) {
  const count = targets.length;
  if (!count) return [];
  const span = gap * (count - 1);
  // When the labels cannot all fit, they share the room evenly.
  if (Number.isFinite(min) && Number.isFinite(max) && max - min < span) {
    const step = count > 1 ? (max - min) / (count - 1) : 0;
    return targets.map((_, i) => min + i * step);
  }
  const placed = targets.map((target) => clamp(target, min, max));
  for (let pass = 0; pass < 4; pass += 1) {
    for (let i = 1; i < count; i += 1) placed[i] = Math.max(placed[i], placed[i - 1] + gap);
    if (placed[count - 1] > max) placed[count - 1] = max;
    for (let i = count - 2; i >= 0; i -= 1) placed[i] = Math.min(placed[i], placed[i + 1] - gap);
    if (placed[0] < min) placed[0] = min;
  }
  return placed;
}

/** Ease in and out, 0 to 1. */
export function easeInOut(t) {
  const x = clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
}

/** Exponential approach of `current` to `target` over `dt` seconds with time constant `tau`. */
export function approach(current, target, dt, tau) {
  if (tau <= 0) return target;
  return target + (current - target) * Math.exp(-dt / tau);
}

/**
 * Opacity and lift of each layer while one is in focus (`focus` >= 0): the focused layer at
 * full strength and raised a little, those above it see-through, those below dimmed.
 */
export function focusStyle(index, focus) {
  if (focus < 0) return { opacity: 1, lift: 0 };
  if (index === focus) return { opacity: 1, lift: 1 };
  return index < focus ? { opacity: 0.12, lift: 0 } : { opacity: 0.45, lift: 0 };
}
