/*
 * The data tower on the home pages (/ and /zh/): every dataset of a region drawn as one layer
 * of a slowly turning glass stack. Drag the stack to turn it; click a layer to open it.
 *
 * The page's markup is the source of truth and works without this script. Per region it holds
 * a list of links, one per layer from top to bottom, each naming its image in data-src
 * (drawn by scripts/build_home_data_tower.py). This script draws those images into the canvas,
 * sets the list beside the stack as its labels, and ties hover, focus and clicks between the
 * two. A click on a layer clicks its link, so it opens the same view and is counted the same.
 * emphasise("public") fades the layers only the research edition has (each link's data-opens
 * says which edition it opens), so a card for that edition can show what it holds.
 *
 * Loaded by 3d-ice-home.js with import() once the section nears the screen.
 */

import {
  DEFAULT_PITCH,
  PITCH_MAX,
  PITCH_MIN,
  approach,
  clamp,
  easeInOut,
  farAndNearCorner,
  fitView,
  focusStyle,
  layerCorners,
  layerHalfExtents,
  layerHeights,
  layerTransform,
  pickLayer,
  projectPoint,
  spreadLabels,
} from "./3d-ice-data-tower-geometry.js";

const TOWER_HEIGHT = 2.1;
const TOP_ROOM = 0.12;
const BASE_DEPTH = 0.09;
const SPIN_RADIANS_PER_SECOND = 0.11;
const DRAG_RADIANS_PER_PX = 0.0085;
const TILT_RADIANS_PER_PX = 0.004;
const DRAG_THRESHOLD_PX = 5;
const RETAP_RADIUS_PX = 24;
const RESUME_AFTER_MS = 2600;
const LIFT = 0.07;
// Below this stage width the labels go under the stack: beside it they would need cutting short.
const SIDE_LABELS_MIN_WIDTH = 840;
// Where the label column starts, as a share of the stage's width.
const LABEL_COLUMN = 0.62;
const LABEL_GAP_PX = 36;
const MAX_PIXEL_RATIO = 2;
const CRUISE_FRAME_MS = 30;
const MASK_SIZE = 192;
const MASK_ALPHA = 24;
const SWITCH_SECONDS = 0.95;
// How far a layer the emphasised edition lacks fades: to a seventh of its opacity.
const MUTED_FADE = 0.86;

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function readRegions(stage) {
  const regions = new Map();
  for (const list of stage.querySelectorAll("[data-tower-region]")) {
    const items = [...list.querySelectorAll("[data-tower-layer]")];
    regions.set(list.dataset.towerRegion, {
      key: list.dataset.towerRegion,
      list,
      layers: items.map((item) => ({
        key: item.dataset.towerLayer,
        item,
        link: item.querySelector("a[href]"),
        opens: item.querySelector("a[href]")?.dataset.opens || "public",
        muted: 0,
        src: item.dataset.src,
        image: null,
        mask: null,
        shown: 0,
        focus: { opacity: 1, lift: 0 },
      })),
      shadowSrc: list.dataset.shadowSrc,
      shadow: null,
      aspect: 1,
      loading: null,
    });
  }
  return regions;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`3D ICE data tower: could not load ${src}`));
    image.src = src;
  });
}

/** A coarse opacity map of an image, grown by a pixel so thin lines stay easy to hit. */
function buildMask(image) {
  const ratio = Math.min(1, MASK_SIZE / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(1, Math.round(image.naturalWidth * ratio));
  const height = Math.max(1, Math.round(image.naturalHeight * ratio));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0, width, height);
  const pixels = context.getImageData(0, 0, width, height).data;
  const solid = new Uint8Array(width * height);
  for (let i = 0; i < solid.length; i += 1) solid[i] = pixels[4 * i + 3] > MASK_ALPHA ? 1 : 0;
  const grown = new Uint8Array(solid.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let hit = 0;
      for (let dy = -1; dy <= 1 && !hit; dy += 1) {
        for (let dx = -1; dx <= 1 && !hit; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx >= 0 && ny >= 0 && nx < width && ny < height) hit = solid[ny * width + nx];
        }
      }
      grown[y * width + x] = hit;
    }
  }
  const scaleX = width / image.naturalWidth;
  const scaleY = height / image.naturalHeight;
  return (px, py) => grown[Math.floor(py * scaleY) * width + Math.floor(px * scaleX)] === 1;
}

async function loadRegion(region) {
  if (!region.loading) {
    region.loading = (async () => {
      const shadow = region.shadowSrc ? loadImage(region.shadowSrc).catch(() => null) : Promise.resolve(null);
      await Promise.all(
        region.layers.map(async (layer) => {
          try {
            layer.image = await loadImage(layer.src);
            layer.mask = buildMask(layer.image);
            if (layer.image.naturalHeight) region.aspect = layer.image.naturalWidth / layer.image.naturalHeight;
          } catch (error) {
            console.error(error);
          }
        })
      );
      region.shadow = await shadow;
    })();
  }
  return region.loading;
}

function readColours(section) {
  const style = getComputedStyle(section);
  const read = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
  return {
    glassFill: read("--tower-glass-fill", "rgba(255, 255, 255, 0.28)"),
    glassFront: read("--tower-glass-front", "rgba(255, 255, 255, 0.12)"),
    glassEdge: read("--tower-glass-edge", "rgba(120, 150, 170, 0.55)"),
    frame: read("--tower-frame", "rgba(120, 150, 170, 0.32)"),
    baseTop: read("--tower-base-top", "#d7e3ea"),
    baseLeft: read("--tower-base-left", "#b9cbd6"),
    baseRight: read("--tower-base-right", "#a5bac7"),
    leader: read("--tower-leader", "rgba(70, 100, 120, 0.55)"),
    accent: read("--tower-accent", "#0f7ea8"),
  };
}

function polygon(context, points) {
  context.beginPath();
  points.forEach((point, i) => (i ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y)));
  context.closePath();
}

function line(context, from, to) {
  context.beginPath();
  context.moveTo(from.x, from.y);
  context.lineTo(to.x, to.y);
  context.stroke();
}

export function mountDataTower(section) {
  const stage = section.querySelector("[data-tower-stage]");
  const canvas = stage?.querySelector("[data-tower-canvas]");
  const context = canvas?.getContext("2d");
  if (!stage || !canvas || !context) return null;
  const regions = readRegions(stage);
  if (!regions.size) return null;
  const buttons = [...section.querySelectorAll("[data-tower-region-button]")];
  const openChip = section.querySelector("[data-tower-open]");

  const state = {
    region: regions.get(buttons.find((b) => b.getAttribute("aria-pressed") === "true")?.dataset.towerRegionButton) || [...regions.values()][0],
    previous: null,
    switchedAt: 0,
    yaw: -0.62,
    pitch: DEFAULT_PITCH,
    velocity: 0,
    // The stack starts turning as soon as it is drawn; after a drag it waits RESUME_AFTER_MS.
    idleSince: -Infinity,
    hover: -1,
    labelFocus: -1,
    // The edition whose layers stand out, or null for all of them.
    edition: null,
    selected: -1,
    selectedAt: null,
    drag: null,
    width: 0,
    height: 0,
    side: false,
    visible: false,
    frame: 0,
    last: 0,
    drawnAt: 0,
    colours: readColours(section),
    extents: layerHalfExtents(1),
    labels: [],
  };

  const focusIndex = () => (state.labelFocus >= 0 ? state.labelFocus : state.selected >= 0 ? state.selected : state.hover);
  // A layer the emphasised edition does not have. Only the public edition lacks any.
  const isMuted = (layer) => state.edition === "public" && layer.opens !== "public";

  function resize() {
    const rect = stage.getBoundingClientRect();
    state.side = rect.width >= SIDE_LABELS_MIN_WIDTH;
    stage.classList.toggle("is-side-labels", state.side);
    const box = canvas.getBoundingClientRect();
    state.width = box.width;
    state.height = box.height;
    const ratio = Math.min(MAX_PIXEL_RATIO, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(box.width * ratio));
    canvas.height = Math.max(1, Math.round(box.height * ratio));
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    state.labelHeights = null;
    wake();
  }

  function view() {
    const fit = fitView({
      width: state.width,
      height: state.height,
      pitch: state.pitch,
      radius: 1,
      towerHeight: TOWER_HEIGHT + TOP_ROOM,
      baseDepth: BASE_DEPTH,
      padding: state.side ? 28 : 14,
      towerShare: state.side ? LABEL_COLUMN - 0.02 : 1,
      towerCentre: state.side ? (LABEL_COLUMN - 0.02) / 2 : 0.5,
    });
    return { ...fit, yaw: state.yaw, pitch: state.pitch };
  }

  // ---------------------------------------------------------------- drawing

  /** Each shown layer of a region: its height, opacity and the transform it is drawn with. */
  function placedLayers(region, v, extents, presence) {
    const heights = layerHeights(region.layers.length, TOWER_HEIGHT);
    const focus = region === state.region ? focusIndex() : -1;
    return region.layers.map((layer, i) => {
      const style = focusStyle(i, focus);
      layer.focus.opacity = approach(layer.focus.opacity, style.opacity, state.dt, 0.12);
      layer.focus.lift = approach(layer.focus.lift, style.lift, state.dt, 0.12);
      layer.muted = approach(layer.muted, region === state.region && isMuted(layer) ? 1 : 0, state.dt, 0.12);
      const arrival = presence(i);
      layer.shown = approach(layer.shown, layer.image ? 1 : 0, state.dt, 0.18);
      const y = heights[i] + LIFT * layer.focus.lift + (1 - arrival.offset) * 0.9;
      const opacity = layer.shown * layer.focus.opacity * arrival.opacity * (1 - MUTED_FADE * layer.muted);
      const image = layer.image;
      return {
        layer,
        index: i,
        y,
        opacity,
        transform: image ? layerTransform(v, extents, y, image.naturalWidth, image.naturalHeight) : null,
        width: image?.naturalWidth || 0,
        height: image?.naturalHeight || 0,
        isOpaque: (px, py) => Boolean(layer.mask?.(px, py)),
        pickable: region === state.region && layer.image !== null,
      };
    });
  }

  function drawBox(v, extents, part) {
    const { colours } = state;
    const bottom = layerCorners(v, extents, 0);
    const top = layerCorners(v, extents, TOWER_HEIGHT + TOP_ROOM);
    const { far, near } = farAndNearCorner(bottom);
    const around = (i) => (i + 4) % 4;
    context.lineWidth = 1;
    if (part === "back") {
      // The base slab, then the two glass walls behind the stack.
      const under = layerCorners(v, extents, -BASE_DEPTH);
      for (const [a, b, fill] of [
        [around(near - 1), near, colours.baseLeft],
        [near, around(near + 1), colours.baseRight],
      ]) {
        context.fillStyle = fill;
        polygon(context, [bottom[a], bottom[b], under[b], under[a]]);
        context.fill();
      }
      context.fillStyle = colours.baseTop;
      polygon(context, bottom);
      context.fill();
      context.fillStyle = colours.glassFill;
      for (const b of [around(far - 1), around(far + 1)]) {
        polygon(context, [bottom[far], bottom[b], top[b], top[far]]);
        context.fill();
      }
      // Edges the stack can stand in front of: the far upright and the floor edges beside it.
      context.strokeStyle = colours.glassEdge;
      line(context, bottom[far], top[far]);
      line(context, bottom[around(far - 1)], bottom[far]);
      line(context, bottom[far], bottom[around(far + 1)]);
      return;
    }
    context.fillStyle = colours.glassFront;
    for (const a of [around(near - 1), around(near + 1)]) {
      polygon(context, [bottom[a], bottom[near], top[near], top[a]]);
      context.fill();
    }
    context.strokeStyle = colours.glassEdge;
    for (let i = 0; i < 4; i += 1) {
      if (i !== far) line(context, bottom[i], top[i]);
      line(context, top[i], top[around(i + 1)]);
    }
    line(context, bottom[around(near - 1)], bottom[near]);
    line(context, bottom[near], bottom[around(near + 1)]);
  }

  function drawLayers(placed, extents, v, region) {
    const { colours } = state;
    for (let i = placed.length - 1; i >= 0; i -= 1) {
      const entry = placed[i];
      if (!entry.transform || entry.opacity < 0.01) continue;
      if (i === placed.length - 1 && region.shadow) {
        const [a, b, c, d, e, f] = layerTransform(v, extents, 0, region.shadow.naturalWidth, region.shadow.naturalHeight);
        context.save();
        context.globalAlpha = entry.opacity;
        context.transform(a, b, c, d, e, f);
        context.drawImage(region.shadow, 0, 0);
        context.restore();
      }
      context.save();
      context.globalAlpha = entry.opacity;
      context.transform(...entry.transform);
      context.imageSmoothingQuality = "high";
      context.drawImage(entry.layer.image, 0, 0);
      context.restore();
      const focused = region === state.region && entry.index === focusIndex();
      context.save();
      context.globalAlpha = Math.min(1, entry.opacity * (focused ? 1 : 0.9));
      context.strokeStyle = focused ? colours.accent : colours.frame;
      context.lineWidth = focused ? 1.5 : 1;
      polygon(context, layerCorners(v, extents, entry.y));
      context.stroke();
      context.restore();
    }
  }

  /** Where each label's line leaves the stack: the box's wall level with the layer's middle. */
  function anchorOf(v, extents, y) {
    const cos = Math.abs(Math.cos(v.yaw));
    const sin = Math.abs(Math.sin(v.yaw));
    const reach = Math.min(cos > 1e-6 ? extents.halfWidth / cos : Infinity, sin > 1e-6 ? extents.halfDepth / sin : Infinity);
    const centre = projectPoint(v, 0, y, 0);
    return { x: centre.x + v.scale * reach, y: centre.y };
  }

  function layoutLabels(placed, v, extents) {
    const items = state.region.layers.map((layer) => layer.item);
    if (!state.side) {
      for (const item of items) item.style.transform = "";
      state.labels = [];
      return;
    }
    if (!state.labelHeights || state.labelHeights.length !== items.length) {
      state.labelHeights = items.map((item) => item.offsetHeight);
    }
    const anchors = placed.map((entry) => anchorOf(v, extents, entry.y));
    // Two lines of text; the labels' padding may overlap, as only a focused label shows it.
    const gap = LABEL_GAP_PX;
    const positions = spreadLabels(anchors.map((a) => a.y), gap, gap / 2 + 4, state.height - gap / 2 - 4);
    const left = state.width * LABEL_COLUMN;
    state.labels = anchors.map((anchor, i) => ({ anchor, x: left, y: positions[i] }));
    items.forEach((item, i) => {
      const transform = `translate(${Math.round(left)}px, ${Math.round(positions[i] - state.labelHeights[i] / 2)}px)`;
      if (item.style.transform !== transform) item.style.transform = transform;
    });
  }

  function drawLeaders(placed) {
    if (!state.side) return;
    const { colours } = state;
    const focus = focusIndex();
    state.labels.forEach((label, i) => {
      const focused = i === focus;
      context.save();
      context.globalAlpha = placed[i].opacity > 0.05 ? (focus < 0 || focused ? 1 : 0.45) : 0.25;
      context.strokeStyle = focused ? colours.accent : colours.leader;
      context.fillStyle = focused ? colours.accent : colours.leader;
      context.lineWidth = focused ? 1.4 : 1;
      context.beginPath();
      context.moveTo(label.anchor.x, label.anchor.y);
      context.lineTo(label.x - 20, label.y);
      context.lineTo(label.x - 6, label.y);
      context.stroke();
      context.beginPath();
      context.arc(label.anchor.x, label.anchor.y, focused ? 3 : 2.2, 0, Math.PI * 2);
      context.fill();
      context.restore();
    });
  }

  function draw(now) {
    const v = view();
    const switching = state.previous && (now - state.switchedAt) / 1000 < SWITCH_SECONDS;
    const progress = switching ? (now - state.switchedAt) / 1000 / SWITCH_SECONDS : 1;
    const target = layerHalfExtents(state.region.aspect);
    const from = state.previous ? layerHalfExtents(state.previous.aspect) : target;
    const blend = easeInOut(progress * 1.4);
    const extents = {
      halfWidth: from.halfWidth + (target.halfWidth - from.halfWidth) * blend,
      halfDepth: from.halfDepth + (target.halfDepth - from.halfDepth) * blend,
    };
    state.extents = extents;
    context.clearRect(0, 0, state.width, state.height);
    drawBox(v, extents, "back");

    if (switching) {
      const leaving = placedLayers(state.previous, v, layerHalfExtents(state.previous.aspect), () => ({
        opacity: 1 - easeInOut(progress * 2.2),
        offset: 1,
      }));
      drawLayers(leaving, extents, v, state.previous);
    } else if (state.previous) {
      state.previous = null;
    }
    const count = state.region.layers.length;
    const placed = placedLayers(state.region, v, extents, (i) => {
      if (!switching) return { opacity: 1, offset: 1 };
      // The incoming layers settle one after another, the bottom one first.
      const start = 0.18 + 0.4 * ((count - 1 - i) / Math.max(1, count - 1));
      const t = easeInOut((progress - start) / 0.42);
      return { opacity: t, offset: t };
    });
    drawLayers(placed, extents, v, state.region);
    drawBox(v, extents, "front");
    layoutLabels(placed, v, extents);
    drawLeaders(placed);
    state.placed = placed;
    return switching;
  }

  // ---------------------------------------------------------------- the loop

  function step(now) {
    state.frame = 0;
    const dt = state.last ? Math.min(0.05, (now - state.last) / 1000) : 1 / 60;
    state.last = now;
    state.dt = dt;
    const idle = !state.drag && now - state.idleSince > RESUME_AFTER_MS;
    const holding = focusIndex() >= 0;
    // After a drag the stack coasts to a stop; after a pause it turns again, slowly.
    const desired = idle && !holding && !reducedMotion.matches ? SPIN_RADIANS_PER_SECOND : 0;
    if (!state.drag) {
      state.velocity = approach(state.velocity, desired, dt, idle ? 0.9 : 0.45);
      state.yaw += state.velocity * dt;
    }
    // A stack that is only turning slowly is drawn every other frame, which the eye does not
    // miss at this speed and the battery does.
    const cruising = desired > 0 && !holding && !state.previous && Math.abs(state.velocity - desired) < 1e-3;
    let animating = false;
    if (!cruising || now - state.drawnAt >= CRUISE_FRAME_MS) {
      animating = draw(now);
      state.drawnAt = now;
    }
    const moving = Boolean(state.drag) || Math.abs(state.velocity) > 1e-4 || desired > 0 || animating || isSettling() || !idle;
    if (state.visible && document.visibilityState === "visible" && moving) schedule();
    else state.last = 0;
  }

  /** Whether a fade or a lift is still on its way to where the focus puts it. */
  function isSettling() {
    const focus = focusIndex();
    return state.region.layers.some((layer, i) => {
      const target = focusStyle(i, focus);
      return (
        (layer.image && layer.shown < 0.997) ||
        Math.abs(layer.focus.opacity - target.opacity) > 0.003 ||
        Math.abs(layer.focus.lift - target.lift) > 0.003 ||
        Math.abs(layer.muted - (isMuted(layer) ? 1 : 0)) > 0.003
      );
    });
  }

  function schedule() {
    if (!state.frame) state.frame = window.requestAnimationFrame(step);
  }

  function wake() {
    state.drawnAt = 0;
    schedule();
  }

  // ---------------------------------------------------------------- interaction

  function setLabelState() {
    const focus = focusIndex();
    state.region.layers.forEach((layer, i) => {
      layer.item.classList.toggle("is-focused", i === focus);
      layer.item.classList.toggle("is-muted", isMuted(layer));
    });
    stage.classList.toggle("has-focus", focus >= 0);
  }

  /** Lets an edition's layers stand out ("public" or "research"), or all of them (null). */
  function emphasise(edition) {
    const next = edition === "public" || edition === "research" ? edition : null;
    if (state.edition === next) return;
    state.edition = next;
    // Not data-edition, which names the edition cards themselves.
    if (next) stage.dataset.emphasis = next;
    else delete stage.dataset.emphasis;
    setLabelState();
    wake();
  }

  function showOpenChip() {
    if (!openChip) return;
    const layer = state.region.layers[state.selected];
    openChip.hidden = !layer;
    if (!layer) return;
    const link = openChip.querySelector("a");
    link.href = layer.link.href;
    link.textContent = (openChip.dataset.template || "{layer}").replace("{layer}", layer.item.querySelector("[data-tower-name]")?.textContent.trim() || "");
  }

  function setHover(index) {
    if (state.hover === index) return;
    state.hover = index;
    canvas.style.cursor = index >= 0 ? "pointer" : "grab";
    setLabelState();
    wake();
  }

  function select(index) {
    state.selected = index;
    setLabelState();
    showOpenChip();
    wake();
  }

  function open(index, event) {
    const link = state.region.layers[index]?.link;
    if (!link) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button === 1) {
      window.open(link.href, "_blank", "noopener");
      return;
    }
    // A click on the link itself, so analytics count it as a click from this section.
    link.click();
  }

  function pointerAt(event) {
    const rect = canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  /**
   * The layer under the pointer: the one already in focus while the pointer stays on its
   * shape (the layers above it are see-through then), else the topmost one with a shape there.
   */
  function pick(event, keep = focusIndex()) {
    if (!state.placed) return -1;
    const { x, y } = pointerAt(event);
    const current = state.placed[keep];
    if (current && pickLayer([current], x, y) === 0) return keep;
    return pickLayer(state.placed, x, y);
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 && event.pointerType === "mouse") return;
    state.drag = {
      id: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      moved: false,
      lastX: event.clientX,
      lastY: event.clientY,
      lastT: performance.now(),
      velocity: 0,
    };
    canvas.setPointerCapture?.(event.pointerId);
    wake();
  });
  canvas.addEventListener("pointermove", (event) => {
    const drag = state.drag;
    if (!drag || drag.id !== event.pointerId) {
      if (event.pointerType === "mouse") setHover(pick(event));
      return;
    }
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) return;
    if (!drag.moved) {
      drag.moved = true;
      canvas.classList.add("is-dragging");
      setHover(-1);
    }
    const now = performance.now();
    const stepX = event.clientX - drag.lastX;
    state.yaw += stepX * DRAG_RADIANS_PER_PX;
    state.pitch = clamp(state.pitch + (event.clientY - drag.lastY) * TILT_RADIANS_PER_PX, PITCH_MIN, PITCH_MAX);
    const elapsed = Math.max(1, now - drag.lastT) / 1000;
    drag.velocity = 0.7 * drag.velocity + 0.3 * ((stepX * DRAG_RADIANS_PER_PX) / elapsed);
    drag.lastX = event.clientX;
    drag.lastY = event.clientY;
    drag.lastT = now;
    wake();
  });
  const endDrag = (event) => {
    const drag = state.drag;
    if (!drag || drag.id !== event.pointerId) return;
    state.drag = null;
    canvas.classList.remove("is-dragging");
    state.idleSince = performance.now();
    if (drag.moved) {
      state.velocity = clamp(drag.velocity, -6, 6);
    } else if (event.type === "pointerup") {
      const index = pick(event, event.pointerType === "mouse" ? state.hover : state.selected);
      const point = pointerAt(event);
      // The picked layer lifts a little, so a second tap where the first one landed may
      // fall just off its shape; it still means that layer.
      const again =
        state.selected >= 0 &&
        state.selectedAt &&
        Math.hypot(point.x - state.selectedAt.x, point.y - state.selectedAt.y) <= RETAP_RADIUS_PX;
      if (event.pointerType === "mouse") {
        if (index >= 0) open(index, event);
      } else if (again || (index >= 0 && index === state.selected)) {
        open(state.selected, event);
      } else {
        state.selectedAt = index >= 0 ? point : null;
        select(index);
      }
    }
    wake();
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointerleave", (event) => {
    if (event.pointerType === "mouse" && !state.drag) setHover(-1);
  });

  for (const region of regions.values()) {
    region.layers.forEach((layer, i) => {
      const enter = () => {
        if (region !== state.region) return;
        state.labelFocus = i;
        setLabelState();
        wake();
      };
      const leave = () => {
        if (state.labelFocus !== i) return;
        state.labelFocus = -1;
        setLabelState();
        wake();
      };
      layer.item.addEventListener("mouseenter", enter);
      layer.item.addEventListener("mouseleave", leave);
      layer.item.addEventListener("focusin", enter);
      layer.item.addEventListener("focusout", leave);
    });
  }

  function showRegion(key, { animate = true } = {}) {
    const next = regions.get(key);
    if (!next || next === state.region) return;
    state.previous = animate && !reducedMotion.matches ? state.region : null;
    state.switchedAt = performance.now();
    state.region = next;
    state.hover = -1;
    state.labelFocus = -1;
    state.selected = -1;
    for (const region of regions.values()) region.list.hidden = region !== next;
    for (const button of buttons) button.setAttribute("aria-pressed", String(button.dataset.towerRegionButton === key));
    for (const layer of next.layers) {
      layer.shown = layer.image ? 1 : 0;
      layer.muted = isMuted(layer) ? 1 : 0;
    }
    state.labelHeights = null;
    showOpenChip();
    setLabelState();
    loadRegion(next).then(wake);
    wake();
  }

  for (const button of buttons) {
    button.addEventListener("click", () => showRegion(button.dataset.towerRegionButton));
    // The other region's images start loading before the click lands.
    button.addEventListener("pointerenter", () => loadRegion(regions.get(button.dataset.towerRegionButton)));
  }

  // ---------------------------------------------------------------- start

  for (const region of regions.values()) region.list.hidden = region !== state.region;
  section.querySelectorAll("[data-tower-controls]").forEach((controls) => (controls.hidden = false));
  stage.classList.add("is-live");
  new ResizeObserver(resize).observe(stage);
  new MutationObserver(() => {
    state.colours = readColours(section);
    wake();
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
  new IntersectionObserver((entries) => {
    state.visible = entries.some((entry) => entry.isIntersecting);
    if (state.visible) wake();
  }).observe(stage);
  document.addEventListener("visibilitychange", wake);
  reducedMotion.addEventListener("change", wake);
  resize();
  loadRegion(state.region).then(() => {
    stage.classList.add("is-ready");
    wake();
  });
  const snapshot = () => ({
    region: state.region.key,
    yaw: state.yaw,
    pitch: state.pitch,
    velocity: state.velocity,
    focus: focusIndex() >= 0 ? state.region.layers[focusIndex()].key : null,
    edition: state.edition,
    muted: state.region.layers.filter(isMuted).map((layer) => layer.key),
    visible: state.visible,
    side: state.side,
    loaded: state.region.layers.every((layer) => layer.image),
  });
  // For tests and the console, as the explorer's window.render_game_to_text() is.
  Object.defineProperty(section, "dataTowerState", { value: snapshot, configurable: true });
  return { showRegion, snapshot, emphasise };
}
