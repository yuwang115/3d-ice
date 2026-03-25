"use strict";

const FLOWLINE_SURFACE_OFFSET_M = 18;
const VELOCITY_VISUALIZATION_MAX = 3000;
const VELOCITY_VISUALIZATION_KNEE = 20;
const BASAL_FRICTION_VISUALIZATION_MAX = 0.3;
const BASAL_FRICTION_VISUALIZATION_KNEE = 0.015;
const EFFECTIVE_PRESSURE_REFERENCE_PA = 5_000_000;
const CHANNEL_DISCHARGE_MIN = 1e-3;
const CHANNEL_DISCHARGE_MAX = 100;
const CHANNEL_STRIP_WIDTH_M_MIN = 1500;
const CHANNEL_STRIP_WIDTH_M_MAX = 5000;
const BASAL_FRICTION_SURFACE_OFFSET_M = 12;
const EFFECTIVE_PRESSURE_SURFACE_OFFSET_M = 10;
const SUBGLACIAL_CHANNEL_SURFACE_OFFSET_M = 14;
const OCEAN_CURRENT_BED_CLEARANCE_M = 20;
const OCEAN_CURRENT_ARROW_HEAD_RATIO = 0.4;
const OCEAN_CURRENT_ARROW_HEAD_MIN_UNITS = 0.16;
const OCEAN_CURRENT_ARROW_HEAD_MAX_UNITS = 1.0;
const OCEAN_CURRENT_ARROW_HEAD_WIDTH_RATIO = 0.65;
const OCEAN_CURRENT_LAYER_ORDER = ["surface", "upper", "mid", "lower"];

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpColor(from, to, t) {
  return [lerp(from[0], to[0], t), lerp(from[1], to[1], t), lerp(from[2], to[2], t)];
}

function sampleColorStops(stops, t) {
  if (!stops.length) return [1, 1, 1];
  const clamped = clamp01(t);
  if (clamped <= stops[0][0]) return stops[0][1];
  for (let i = 1; i < stops.length; i += 1) {
    if (clamped <= stops[i][0]) {
      const [t0, c0] = stops[i - 1];
      const [t1, c1] = stops[i];
      const localT = (clamped - t0) / Math.max(1e-6, t1 - t0);
      return lerpColor(c0, c1, localT);
    }
  }
  return stops[stops.length - 1][1];
}

function sampleColorLUT(lut, t) {
  if (!lut || !lut.length) return [0, 0, 0];
  const scaled = clamp01(t) * (lut.length - 1);
  const i0 = Math.floor(scaled);
  const i1 = Math.min(lut.length - 1, i0 + 1);
  return lerpColor(lut[i0], lut[i1], scaled - i0);
}

function velocityColor(speedMetersPerYear) {
  const speed = Number.isFinite(speedMetersPerYear) ? Math.max(0, speedMetersPerYear) : 0;
  const scaled =
    Math.log1p(speed / VELOCITY_VISUALIZATION_KNEE) /
    Math.log1p(VELOCITY_VISUALIZATION_MAX / VELOCITY_VISUALIZATION_KNEE);
  return sampleColorStops(
    [
      [0.0, [0.06, 0.2, 0.5]],
      [0.35, [0.08, 0.62, 0.86]],
      [0.65, [0.95, 0.9, 0.27]],
      [1.0, [0.9, 0.2, 0.12]],
    ],
    scaled
  );
}

function basalFrictionScaleT(valueMpa) {
  const value = Number.isFinite(valueMpa) ? Math.max(0, valueMpa) : 0;
  const clamped = Math.min(BASAL_FRICTION_VISUALIZATION_MAX, value);
  return (
    Math.log1p(clamped / BASAL_FRICTION_VISUALIZATION_KNEE) /
    Math.log1p(BASAL_FRICTION_VISUALIZATION_MAX / BASAL_FRICTION_VISUALIZATION_KNEE)
  );
}

function basalFrictionColor(valueMpa) {
  return sampleColorStops(
    [
      [0.0, [0.047, 0.118, 0.259]],
      [0.26, [0.125, 0.478, 0.678]],
      [0.52, [0.88, 0.91, 0.79]],
      [0.78, [0.851, 0.545, 0.188]],
      [1.0, [0.588, 0.094, 0.114]],
    ],
    basalFrictionScaleT(valueMpa)
  );
}

function effectivePressureColor(pressurePa, lut) {
  const pressure = Number.isFinite(pressurePa) ? Math.max(0, pressurePa) : 0;
  const scaled = clamp01(pressure / EFFECTIVE_PRESSURE_REFERENCE_PA);
  if (lut && lut.length) {
    return sampleColorLUT(lut, scaled);
  }
  return sampleColorStops(
    [
      [0.0, [0.9, 0.95, 0.95]],
      [1.0, [0.21, 0.05, 0.14]],
    ],
    scaled
  );
}

function channelDischargeNormalized(dischargeM3PerS) {
  if (!Number.isFinite(dischargeM3PerS) || dischargeM3PerS <= 0) return 0;
  const clamped = Math.min(CHANNEL_DISCHARGE_MAX, Math.max(CHANNEL_DISCHARGE_MIN, dischargeM3PerS));
  return (
    (Math.log10(clamped) - Math.log10(CHANNEL_DISCHARGE_MIN)) /
    (Math.log10(CHANNEL_DISCHARGE_MAX) - Math.log10(CHANNEL_DISCHARGE_MIN))
  );
}

function oceanCurrentColor(thetaC, salinityPsu, oceanMeta) {
  const visualization = oceanMeta?.visualization || {};
  const thetaRange = Array.isArray(visualization.temperature_range_c) ? visualization.temperature_range_c : [-2, 8];
  const salinityRange = Array.isArray(visualization.salinity_range_psu) ? visualization.salinity_range_psu : [30, 35];
  const thetaSpan = Math.max(1e-6, Number(thetaRange[1]) - Number(thetaRange[0]));
  const salinitySpan = Math.max(1e-6, Number(salinityRange[1]) - Number(salinityRange[0]));
  const thetaT = clamp01((Number(thetaC) - Number(thetaRange[0])) / thetaSpan);
  const salinityT = clamp01((Number(salinityPsu) - Number(salinityRange[0])) / salinitySpan);

  const coldFresh = [0.0, 0.87, 0.99];
  const coldSalty = [0.23, 0.11, 0.88];
  const warmFresh = [0.0, 0.76, 0.2];
  const warmSalty = [0.99, 0.28, 0.0];

  const freshBlend = lerpColor(coldFresh, warmFresh, thetaT);
  const saltyBlend = lerpColor(coldSalty, warmSalty, thetaT);
  const color = lerpColor(freshBlend, saltyBlend, salinityT);
  const brightnessLift = 0.022 + thetaT * 0.018;
  return [
    clamp01(color[0] + brightnessLift),
    clamp01(color[1] + brightnessLift * 0.65),
    clamp01(color[2] + brightnessLift * 0.18),
  ];
}

function subglacialChannelColor(dischargeM3PerS, lut) {
  const scaled = channelDischargeNormalized(dischargeM3PerS);
  if (lut && lut.length) {
    return sampleColorLUT(lut, scaled);
  }
  return sampleColorStops(
    [
      [0.0, [0.06, 0.16, 0.38]],
      [0.3, [0.05, 0.5, 0.74]],
      [0.62, [0.73, 0.85, 0.25]],
      [1.0, [0.86, 0.24, 0.1]],
    ],
    scaled
  );
}

function postProgress(id, enabled, progress, stageKey, stage) {
  if (!enabled) return;
  self.postMessage({
    id,
    kind: "progress",
    progress: clamp01(progress),
    stageKey: stageKey || "",
    stage: stage || "",
  });
}

function parseField(meta, arrayBuffer, name) {
  const field = meta.fields.find((item) => item.name === name);
  if (!field) {
    throw new Error(`Missing field: ${name}`);
  }

  if (field.dtype === "int16") {
    return new Int16Array(arrayBuffer, field.byte_offset, field.byte_length / 2);
  }
  if (field.dtype === "uint8") {
    return new Uint8Array(arrayBuffer, field.byte_offset, field.byte_length);
  }
  if (field.dtype === "uint16") {
    return new Uint16Array(arrayBuffer, field.byte_offset, field.byte_length / 2);
  }
  if (field.dtype === "int32") {
    return new Int32Array(arrayBuffer, field.byte_offset, field.byte_length / 4);
  }
  if (field.dtype === "float32") {
    if (field.byte_offset % 4 === 0) {
      return new Float32Array(arrayBuffer, field.byte_offset, field.byte_length / 4);
    }
    const count = field.byte_length / 4;
    const view = new DataView(arrayBuffer, field.byte_offset, field.byte_length);
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
      out[i] = view.getFloat32(i * 4, true);
    }
    return out;
  }
  throw new Error(`Unsupported dtype for ${name}: ${field.dtype}`);
}

function getOceanCurrentSeedBucketKeys(oceanMeta) {
  const samplingBuckets = Array.isArray(oceanMeta?.sampling?.seed_buckets) ? oceanMeta.sampling.seed_buckets : [];
  const bucketKeys = samplingBuckets
    .map((bucket) => (bucket && typeof bucket.key === "string" ? bucket.key : ""))
    .filter((key) => key.length > 0);
  if (bucketKeys.length) return bucketKeys;
  const bucketCounts = oceanMeta?.coverage?.streamlines_by_seed_bucket;
  return bucketCounts && typeof bucketCounts === "object" ? Object.keys(bucketCounts) : [];
}

function getOceanCurrentLayerFromBucketKey(bucketKey) {
  if (typeof bucketKey !== "string") return "";
  for (const layer of OCEAN_CURRENT_LAYER_ORDER) {
    if (bucketKey.endsWith(`_${layer}`) || bucketKey === layer) return layer;
  }
  return "";
}

function getOceanCurrentLayerSplitInfo(oceanMeta) {
  const availableByLayer = { surface: false, upper: false, mid: false, lower: false };
  const bucketCounts = oceanMeta?.coverage?.streamlines_by_seed_bucket;
  if (!bucketCounts || typeof bucketCounts !== "object") {
    return { enabled: false, availableByLayer, orderedBuckets: [] };
  }
  const orderedBuckets = [];
  for (const bucketKey of getOceanCurrentSeedBucketKeys(oceanMeta)) {
    const bucketCount = Number(bucketCounts[bucketKey] || 0);
    if (!(bucketCount > 0)) continue;
    const layer = getOceanCurrentLayerFromBucketKey(bucketKey);
    if (!layer) {
      return { enabled: false, availableByLayer: { surface: false, upper: false, mid: false, lower: false }, orderedBuckets: [] };
    }
    availableByLayer[layer] = true;
    orderedBuckets.push({ key: bucketKey, layer, count: bucketCount });
  }
  return { enabled: orderedBuckets.length > 0, availableByLayer, orderedBuckets };
}

function projectPsPointToGrid(grid, nx, ny, xMeters, yMeters) {
  if (!grid) return null;
  if (!Number.isFinite(xMeters) || !Number.isFinite(yMeters)) return null;
  if (!Number.isFinite(grid.x0_m) || !Number.isFinite(grid.y0_m)) return null;
  if (!Number.isFinite(grid.dx_m) || !Number.isFinite(grid.dy_m)) return null;
  if (Math.abs(grid.dx_m) < 1e-9 || Math.abs(grid.dy_m) < 1e-9) return null;
  const col = (xMeters - grid.x0_m) / grid.dx_m;
  const row = (yMeters - grid.y0_m) / grid.dy_m;
  if (!Number.isFinite(col) || !Number.isFinite(row)) return null;
  if (col < 0 || row < 0 || col > nx - 1 || row > ny - 1) return null;
  return { col, row };
}

function gridToSceneXZ(grid, nx, ny, horizontalMetersPerUnit, col, row) {
  const halfX = (nx - 1) / 2;
  const halfY = (ny - 1) / 2;
  return {
    x: ((col - halfX) * grid.dx_m) / horizontalMetersPerUnit,
    z: ((row - halfY) * Math.abs(grid.dy_m)) / horizontalMetersPerUnit,
  };
}

function sampleBedHeightNearest(nx, ny, bedHeights, bedValid, col, row) {
  const c = Math.round(col);
  const r = Math.round(row);
  if (c < 0 || r < 0 || c >= nx || r >= ny) return Number.NaN;
  const idx = r * nx + c;
  if (!bedValid[idx]) return Number.NaN;
  const value = bedHeights[idx];
  return Number.isFinite(value) ? value : Number.NaN;
}

function buildOceanCurrentTask(id, payload) {
  const {
    oceanMeta,
    oceanBuffer,
    mask,
    bedHeights,
    bedValid,
    nx,
    ny,
    grid,
    baseConfig,
    reportProgress,
  } = payload;
  if (!(oceanBuffer instanceof ArrayBuffer)) {
    throw new Error("Ocean-current payload is invalid.");
  }

  postProgress(id, reportProgress, 0.04, "oceanDecodingPackage", "Decoding ocean-current package...");
  const x0Ps = parseField(oceanMeta, oceanBuffer, "x0_ps_m");
  const y0Ps = parseField(oceanMeta, oceanBuffer, "y0_ps_m");
  const depth0M = parseField(oceanMeta, oceanBuffer, "depth0_m");
  const x1Ps = parseField(oceanMeta, oceanBuffer, "x1_ps_m");
  const y1Ps = parseField(oceanMeta, oceanBuffer, "y1_ps_m");
  const depth1M = parseField(oceanMeta, oceanBuffer, "depth1_m");
  const theta0C = parseField(oceanMeta, oceanBuffer, "theta0_c");
  const sal0Psu = parseField(oceanMeta, oceanBuffer, "sal0_psu");
  const theta1C = parseField(oceanMeta, oceanBuffer, "theta1_c");
  const sal1Psu = parseField(oceanMeta, oceanBuffer, "sal1_psu");
  const terminalFlag = parseField(oceanMeta, oceanBuffer, "terminal_flag");
  const count = x0Ps.length;
  const layerSplitInfo = getOceanCurrentLayerSplitInfo(oceanMeta);
  const useLayerSplit = layerSplitInfo.enabled;
  if (
    y0Ps.length !== count ||
    depth0M.length !== count ||
    x1Ps.length !== count ||
    y1Ps.length !== count ||
    depth1M.length !== count ||
    theta0C.length !== count ||
    sal0Psu.length !== count ||
    theta1C.length !== count ||
    sal1Psu.length !== count ||
    terminalFlag.length !== count
  ) {
    throw new Error("Ocean-current package fields are misaligned.");
  }

  const bucketBoundaries = [];
  if (useLayerSplit) {
    let cumulative = 0;
    for (const bucket of layerSplitInfo.orderedBuckets) {
      cumulative += bucket.count;
      bucketBoundaries.push({ layer: bucket.layer, endExclusive: cumulative });
    }
  }

  function getLayerName(streamlineIndexState) {
    if (!useLayerSplit) return "all";
    while (
      streamlineIndexState.bucketBoundaryIndex < bucketBoundaries.length - 1 &&
      streamlineIndexState.streamlineIndex >= bucketBoundaries[streamlineIndexState.bucketBoundaryIndex].endExclusive
    ) {
      streamlineIndexState.bucketBoundaryIndex += 1;
    }
    return bucketBoundaries[streamlineIndexState.bucketBoundaryIndex]?.layer || "mid";
  }

  const layerNames = useLayerSplit ? OCEAN_CURRENT_LAYER_ORDER.slice() : ["all"];
  const pairCounts = Object.fromEntries(layerNames.map((layer) => [layer, 0]));
  const segmentCounts = Object.fromEntries(layerNames.map((layer) => [layer, 0]));
  const firstPassState = { streamlineIndex: 0, bucketBoundaryIndex: 0 };
  const firstPassChunk = Math.max(20000, Math.floor(count / 20));

  for (let i = 0; i < count; i += 1) {
    const layer = getLayerName(firstPassState);
    const projected0 = projectPsPointToGrid(grid, nx, ny, x0Ps[i], y0Ps[i]);
    const projected1 = projectPsPointToGrid(grid, nx, ny, x1Ps[i], y1Ps[i]);
    let keep = Boolean(projected0 && projected1);
    if (keep) {
      const depth0 = Math.max(0, Number(depth0M[i]));
      const depth1 = Math.max(0, Number(depth1M[i]));
      const col0 = projected0.col;
      const row0 = projected0.row;
      const col1 = projected1.col;
      const row1 = projected1.row;
      const col0Nearest = Math.min(nx - 1, Math.max(0, Math.round(col0)));
      const row0Nearest = Math.min(ny - 1, Math.max(0, Math.round(row0)));
      const col1Nearest = Math.min(nx - 1, Math.max(0, Math.round(col1)));
      const row1Nearest = Math.min(ny - 1, Math.max(0, Math.round(row1)));
      const idx0 = row0Nearest * nx + col0Nearest;
      const idx1 = row1Nearest * nx + col1Nearest;
      const mask0 = Number(mask[idx0]);
      const mask1 = Number(mask[idx1]);
      const validOceanMask0 = mask0 === 0 || mask0 === 3;
      const validOceanMask1 = mask1 === 0 || mask1 === 3;
      const bedHeight0 = sampleBedHeightNearest(nx, ny, bedHeights, bedValid, col0, row0);
      const bedHeight1 = sampleBedHeightNearest(nx, ny, bedHeights, bedValid, col1, row1);
      keep =
        validOceanMask0 &&
        validOceanMask1 &&
        Number.isFinite(bedHeight0) &&
        Number.isFinite(bedHeight1) &&
        -bedHeight0 >= depth0 + OCEAN_CURRENT_BED_CLEARANCE_M &&
        -bedHeight1 >= depth1 + OCEAN_CURRENT_BED_CLEARANCE_M;
    }

    if (keep) {
      pairCounts[layer] += 1 + (terminalFlag[i] ? 2 : 0);
      segmentCounts[layer] += 1;
    }
    if (terminalFlag[i]) {
      firstPassState.streamlineIndex += 1;
    }
    if (i > 0 && i % firstPassChunk === 0) {
      const t = i / Math.max(1, count - 1);
      postProgress(id, reportProgress, 0.08 + t * 0.34, "oceanScanningSegments", "Scanning ocean-current segments...");
    }
  }

  const layerBuffers = {};
  for (const layer of layerNames) {
    layerBuffers[layer] = {
      positions: new Float32Array(pairCounts[layer] * 6),
      colors: new Float32Array(pairCounts[layer] * 6),
      cursor: 0,
      segmentCount: segmentCounts[layer],
    };
  }

  function writePair(buffer, x0, y0, z0, x1, y1, z1, c0, c1) {
    const base = buffer.cursor;
    buffer.positions[base] = x0;
    buffer.positions[base + 1] = y0;
    buffer.positions[base + 2] = z0;
    buffer.positions[base + 3] = x1;
    buffer.positions[base + 4] = y1;
    buffer.positions[base + 5] = z1;
    buffer.colors[base] = c0[0];
    buffer.colors[base + 1] = c0[1];
    buffer.colors[base + 2] = c0[2];
    buffer.colors[base + 3] = c1[0];
    buffer.colors[base + 4] = c1[1];
    buffer.colors[base + 5] = c1[2];
    buffer.cursor += 6;
  }

  function appendArrow(buffer, scenePoint1, y1, dx, dy, dz, color1) {
    const length = Math.hypot(dx, dy, dz);
    if (!(Number.isFinite(length) && length > 1e-5)) return;
    const headLength = clamp(
      length * OCEAN_CURRENT_ARROW_HEAD_RATIO,
      OCEAN_CURRENT_ARROW_HEAD_MIN_UNITS,
      OCEAN_CURRENT_ARROW_HEAD_MAX_UNITS
    );
    const headWidth = headLength * OCEAN_CURRENT_ARROW_HEAD_WIDTH_RATIO;
    const invLength = 1 / length;
    const dirX = dx * invLength;
    const dirY = dy * invLength;
    const dirZ = dz * invLength;

    let sideX = -dirZ;
    let sideY = 0;
    let sideZ = dirX;
    let sideLength = Math.hypot(sideX, sideY, sideZ);
    if (sideLength < 1e-6) {
      sideX = 0;
      sideY = dirZ;
      sideZ = -dirY;
      sideLength = Math.hypot(sideX, sideY, sideZ);
    }
    if (!(Number.isFinite(sideLength) && sideLength > 1e-6)) return;
    sideX = (sideX / sideLength) * headWidth;
    sideY = (sideY / sideLength) * headWidth;
    sideZ = (sideZ / sideLength) * headWidth;

    const tipX = scenePoint1.x;
    const tipY = y1;
    const tipZ = scenePoint1.z;
    const headBaseX = tipX - dirX * headLength;
    const headBaseY = tipY - dirY * headLength;
    const headBaseZ = tipZ - dirZ * headLength;
    const arrowColor = lerpColor(color1, [1, 1, 1], 0.35);

    writePair(
      buffer,
      tipX,
      tipY,
      tipZ,
      headBaseX + sideX,
      headBaseY + sideY,
      headBaseZ + sideZ,
      arrowColor,
      arrowColor
    );
    writePair(
      buffer,
      tipX,
      tipY,
      tipZ,
      headBaseX - sideX,
      headBaseY - sideY,
      headBaseZ - sideZ,
      arrowColor,
      arrowColor
    );
  }

  const secondPassState = { streamlineIndex: 0, bucketBoundaryIndex: 0 };
  const secondPassChunk = Math.max(20000, Math.floor(count / 20));
  for (let i = 0; i < count; i += 1) {
    const layer = getLayerName(secondPassState);
    const buffer = layerBuffers[layer];
    const projected0 = projectPsPointToGrid(grid, nx, ny, x0Ps[i], y0Ps[i]);
    const projected1 = projectPsPointToGrid(grid, nx, ny, x1Ps[i], y1Ps[i]);
    let keep = Boolean(projected0 && projected1);
    let scenePoint0;
    let scenePoint1;
    let y0;
    let y1;
    let color0;
    let color1;

    if (keep) {
      const depth0 = Math.max(0, Number(depth0M[i]));
      const depth1 = Math.max(0, Number(depth1M[i]));
      const col0 = projected0.col;
      const row0 = projected0.row;
      const col1 = projected1.col;
      const row1 = projected1.row;
      const col0Nearest = Math.min(nx - 1, Math.max(0, Math.round(col0)));
      const row0Nearest = Math.min(ny - 1, Math.max(0, Math.round(row0)));
      const col1Nearest = Math.min(nx - 1, Math.max(0, Math.round(col1)));
      const row1Nearest = Math.min(ny - 1, Math.max(0, Math.round(row1)));
      const idx0 = row0Nearest * nx + col0Nearest;
      const idx1 = row1Nearest * nx + col1Nearest;
      const mask0 = Number(mask[idx0]);
      const mask1 = Number(mask[idx1]);
      const validOceanMask0 = mask0 === 0 || mask0 === 3;
      const validOceanMask1 = mask1 === 0 || mask1 === 3;
      const bedHeight0 = sampleBedHeightNearest(nx, ny, bedHeights, bedValid, col0, row0);
      const bedHeight1 = sampleBedHeightNearest(nx, ny, bedHeights, bedValid, col1, row1);
      keep =
        validOceanMask0 &&
        validOceanMask1 &&
        Number.isFinite(bedHeight0) &&
        Number.isFinite(bedHeight1) &&
        -bedHeight0 >= depth0 + OCEAN_CURRENT_BED_CLEARANCE_M &&
        -bedHeight1 >= depth1 + OCEAN_CURRENT_BED_CLEARANCE_M;
      if (keep) {
        scenePoint0 = gridToSceneXZ(grid, nx, ny, baseConfig.horizontalMetersPerUnit, col0, row0);
        scenePoint1 = gridToSceneXZ(grid, nx, ny, baseConfig.horizontalMetersPerUnit, col1, row1);
        y0 = -depth0 / baseConfig.verticalMetersPerUnit;
        y1 = -depth1 / baseConfig.verticalMetersPerUnit;
        color0 = oceanCurrentColor(theta0C[i], sal0Psu[i], oceanMeta);
        color1 = oceanCurrentColor(theta1C[i], sal1Psu[i], oceanMeta);
        writePair(buffer, scenePoint0.x, y0, scenePoint0.z, scenePoint1.x, y1, scenePoint1.z, color0, color1);
        if (terminalFlag[i]) {
          appendArrow(buffer, scenePoint1, y1, scenePoint1.x - scenePoint0.x, y1 - y0, scenePoint1.z - scenePoint0.z, color1);
        }
      }
    }

    if (terminalFlag[i]) {
      secondPassState.streamlineIndex += 1;
    }
    if (i > 0 && i % secondPassChunk === 0) {
      const t = i / Math.max(1, count - 1);
      postProgress(id, reportProgress, 0.44 + t * 0.5, "oceanBuildingGeometry", "Building ocean streamline geometry...");
    }
  }

  const layers = {};
  let totalSegmentCount = 0;
  for (const layer of layerNames) {
    const buffer = layerBuffers[layer];
    totalSegmentCount += Number(buffer.segmentCount || 0);
    layers[layer] = {
      positions: buffer.positions,
      colors: buffer.colors,
      segmentCount: Number(buffer.segmentCount || 0),
    };
  }

  postProgress(id, reportProgress, 0.98, "oceanFinalizing", "Finalizing ocean streamlines...");
  return {
    useLayerSplit,
    flowlineCount: Number(oceanMeta.streamline_count || oceanMeta.flowline_count || 0),
    segmentCount: totalSegmentCount,
    layers,
  };
}

function finiteMedian(values, maxSampleSize = 180000) {
  let finiteCount = 0;
  for (let i = 0; i < values.length; i += 1) {
    if (Number.isFinite(values[i])) finiteCount += 1;
  }
  if (finiteCount === 0) return Number.NaN;

  // Sampling bounds the cost on very large grids while staying stable for metadata display.
  const stride = finiteCount > maxSampleSize ? Math.ceil(finiteCount / maxSampleSize) : 1;
  const sampleCount = Math.ceil(finiteCount / stride);
  let sample = new Float64Array(sampleCount);
  let cursor = 0;
  let seen = 0;
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    if (Number.isFinite(value)) {
      if (seen % stride === 0) {
        sample[cursor] = value;
        cursor += 1;
      }
      seen += 1;
    }
  }

  if (cursor === 0) return Number.NaN;
  if (cursor !== sample.length) {
    sample = sample.subarray(0, cursor);
  }

  sample.sort();
  const mid = Math.floor(sample.length / 2);
  if (sample.length % 2 === 1) return sample[mid];
  return (sample[mid - 1] + sample[mid]) / 2;
}

function buildSurfaceGeometryWithField({
  nx,
  ny,
  dxMeters,
  dyMeters,
  horizontalMetersPerUnit,
  verticalMetersPerUnit,
  heights,
  valid,
  fieldValues,
  colorFn,
  progress,
  progressStart,
  progressEnd,
  progressLabelVerticesKey,
  progressLabelVertices,
  progressLabelIndicesKey,
  progressLabelIndices,
  sampleStride = 1,
}) {
  const stride = Math.max(1, Math.floor(Number(sampleStride) || 1));
  const sampledNx = Math.floor((nx - 1) / stride) + 1;
  const sampledNy = Math.floor((ny - 1) / stride) + 1;
  const vertexCount = sampledNx * sampledNy;
  const positions = new Float32Array(vertexCount * 3);
  const colors = new Uint8Array(vertexCount * 3);
  const uvs = new Float32Array(vertexCount * 2);
  const sampledValid = new Uint8Array(vertexCount);
  const maxIndexCount = (sampledNx - 1) * (sampledNy - 1) * 6;
  const indices = new Uint32Array(maxIndexCount);
  let indexCursor = 0;
  const halfX = (nx - 1) / 2;
  const halfY = (ny - 1) / 2;
  const absDy = Math.abs(dyMeters);
  const rowStep = Math.max(4, Math.floor(sampledNy / 28));

  for (let sampledRow = 0; sampledRow < sampledNy; sampledRow += 1) {
    const row = Math.min(ny - 1, sampledRow * stride);
    for (let sampledCol = 0; sampledCol < sampledNx; sampledCol += 1) {
      const col = Math.min(nx - 1, sampledCol * stride);
      const sourceIndex = row * nx + col;
      const targetIndex = sampledRow * sampledNx + sampledCol;
      const px = ((col - halfX) * dxMeters) / horizontalMetersPerUnit;
      const pz = ((row - halfY) * absDy) / horizontalMetersPerUnit;
      const h = heights[sourceIndex];
      const isValid = Boolean(valid[sourceIndex]);
      sampledValid[targetIndex] = Number(isValid);
      positions[3 * targetIndex] = px;
      positions[3 * targetIndex + 1] = isValid ? h / verticalMetersPerUnit : 0;
      positions[3 * targetIndex + 2] = pz;
      uvs[2 * targetIndex] = col / Math.max(1, nx - 1);
      uvs[2 * targetIndex + 1] = row / Math.max(1, ny - 1);
      const rgb = colorFn(fieldValues[sourceIndex]);
      colors[3 * targetIndex] = Math.round(clamp01(rgb[0]) * 255);
      colors[3 * targetIndex + 1] = Math.round(clamp01(rgb[1]) * 255);
      colors[3 * targetIndex + 2] = Math.round(clamp01(rgb[2]) * 255);
    }

    if (sampledRow > 0 && sampledRow % rowStep === 0) {
      const t = sampledRow / Math.max(1, sampledNy - 1);
      const p = progressStart + (progressEnd - progressStart) * t * 0.5;
      progress(p, progressLabelVerticesKey, progressLabelVertices);
    }
  }

  for (let row = 0; row < sampledNy - 1; row += 1) {
    for (let col = 0; col < sampledNx - 1; col += 1) {
      const i0 = row * sampledNx + col;
      const i1 = i0 + 1;
      const i2 = i0 + sampledNx;
      const i3 = i2 + 1;

      if (sampledValid[i0] && sampledValid[i2] && sampledValid[i1]) {
        indices[indexCursor++] = i0;
        indices[indexCursor++] = i2;
        indices[indexCursor++] = i1;
      }
      if (sampledValid[i1] && sampledValid[i2] && sampledValid[i3]) {
        indices[indexCursor++] = i1;
        indices[indexCursor++] = i2;
        indices[indexCursor++] = i3;
      }
    }

    if (row > 0 && row % rowStep === 0) {
      const t = row / Math.max(1, sampledNy - 2);
      const p = progressStart + (progressEnd - progressStart) * (0.5 + t * 0.5);
      progress(p, progressLabelIndicesKey, progressLabelIndices);
    }
  }

  const trimmed = new Uint32Array(indexCursor);
  trimmed.set(indices.subarray(0, indexCursor));
  return { positions, colors, uvs, indices: trimmed };
}

function buildVelocityTask(id, payload) {
  const {
    velocityMeta,
    velocityBuffer,
    surfaceHeights,
    iceValid,
    nx,
    ny,
    cellCount,
    baseConfig,
    meshStride,
    reportProgress,
  } = payload;
  const safeStride = Math.max(1, Math.floor(Number(meshStride) || 1));
  // Decimated HD mesh can intersect full-res ice surface; add extra lift to keep overlay above ice.
  const velocitySurfaceOffsetMeters = FLOWLINE_SURFACE_OFFSET_M + (safeStride - 1) * 72;

  postProgress(id, reportProgress, 0.02, "velocityDecodingField", "Decoding velocity field...");
  const velocityXInt = parseField(velocityMeta, velocityBuffer, "vx");
  const velocityYInt = parseField(velocityMeta, velocityBuffer, "vy");
  if (velocityMeta.grid.nx !== nx || velocityMeta.grid.ny !== ny) {
    throw new Error("Velocity grid is not aligned to BedMachine grid.");
  }
  if (velocityXInt.length !== cellCount || velocityYInt.length !== cellCount) {
    throw new Error("Velocity field length mismatch.");
  }

  const fillValue = velocityMeta.quantization.int16_fill_value;
  const scale = Number(velocityMeta.quantization.scale ?? 1);
  const offset = Number(velocityMeta.quantization.offset ?? 0);
  const velocityX = new Float32Array(cellCount);
  const velocityY = new Float32Array(cellCount);
  const velocityValid = new Uint8Array(cellCount);
  const velocitySpeed = new Float32Array(cellCount);
  const velocitySurfaceValid = new Uint8Array(cellCount);
  const velocitySurfaceHeights = new Float32Array(cellCount);
  const chunk = 20000;

  for (let i = 0; i < cellCount; i += 1) {
    const rawX = velocityXInt[i];
    const rawY = velocityYInt[i];
    const vx = rawX === fillValue ? Number.NaN : rawX * scale + offset;
    const vy = rawY === fillValue ? Number.NaN : rawY * scale + offset;
    velocityX[i] = vx;
    velocityY[i] = vy;
    const hasVelocity = Number.isFinite(vx) && Number.isFinite(vy);
    velocityValid[i] = Number(hasVelocity);
    velocitySpeed[i] = hasVelocity ? Math.hypot(vx, vy) : Number.NaN;
    velocitySurfaceValid[i] = Number(iceValid[i] && hasVelocity);
    velocitySurfaceHeights[i] = velocitySurfaceValid[i] ? surfaceHeights[i] + velocitySurfaceOffsetMeters : Number.NaN;

    if (i > 0 && i % chunk === 0) {
      const t = i / Math.max(1, cellCount - 1);
      postProgress(id, reportProgress, 0.08 + t * 0.3, "velocityDecodingField", "Decoding velocity field...");
    }
  }

  const geometry = buildSurfaceGeometryWithField({
    nx,
    ny,
    dxMeters: baseConfig.dxMeters,
    dyMeters: baseConfig.dyMeters,
    horizontalMetersPerUnit: baseConfig.horizontalMetersPerUnit,
    verticalMetersPerUnit: baseConfig.verticalMetersPerUnit,
    heights: velocitySurfaceHeights,
    valid: velocitySurfaceValid,
    fieldValues: velocitySpeed,
    colorFn: (speed) => velocityColor(speed),
    progress: (p, stageKey, stage) => postProgress(id, reportProgress, p, stageKey, stage),
    progressStart: 0.4,
    progressEnd: 0.92,
    progressLabelVerticesKey: "velocityBuildingMesh",
    progressLabelVertices: "Building velocity mesh...",
    progressLabelIndicesKey: "velocityTriangulatingMesh",
    progressLabelIndices: "Triangulating velocity mesh...",
    sampleStride: meshStride,
  });

  const velocityMedianSpeed = finiteMedian(velocitySpeed);
  postProgress(id, reportProgress, 0.98, "velocityFinalizing", "Finalizing velocity layer...");
  return {
    positions: geometry.positions,
    colors: geometry.colors,
    uvs: geometry.uvs,
    indices: geometry.indices,
    velocityX,
    velocityY,
    velocitySpeed,
    velocityValid,
    velocityMedianSpeed,
  };
}

function buildHydrologyTask(id, payload) {
  const {
    hydrologyMeta,
    hydrologyBuffer,
    nx,
    ny,
    cellCount,
    baseConfig,
    bedHeights,
    bedValid,
    effectivePressureLut,
    channelDischargeLut,
    meshStride,
    reportProgress,
  } = payload;
  const safeStride = Math.max(1, Math.floor(Number(meshStride) || 1));
  // Decimated HD overlays can intersect bed; increase lift with stride.
  const effectivePressureSurfaceOffsetMeters =
    EFFECTIVE_PRESSURE_SURFACE_OFFSET_M + (safeStride - 1) * 64;
  const subglacialChannelSurfaceOffsetMeters = Math.max(
    effectivePressureSurfaceOffsetMeters + 24,
    SUBGLACIAL_CHANNEL_SURFACE_OFFSET_M + (safeStride - 1) * 76
  );

  if (hydrologyMeta.grid.nx !== nx || hydrologyMeta.grid.ny !== ny) {
    throw new Error("Hydrology grid is not aligned to BedMachine grid.");
  }

  postProgress(id, reportProgress, 0.04, "hydrologyProcessingField", "Processing hydrology field...");
  const pressureInt = parseField(hydrologyMeta, hydrologyBuffer, "effective_pressure");
  const channelCol1 = parseField(hydrologyMeta, hydrologyBuffer, "channel_col1");
  const channelRow1 = parseField(hydrologyMeta, hydrologyBuffer, "channel_row1");
  const channelCol2 = parseField(hydrologyMeta, hydrologyBuffer, "channel_col2");
  const channelRow2 = parseField(hydrologyMeta, hydrologyBuffer, "channel_row2");
  const channelDischarge = parseField(hydrologyMeta, hydrologyBuffer, "channel_discharge");
  if (pressureInt.length !== cellCount) {
    throw new Error("Hydrology effective-pressure field length mismatch.");
  }
  if (
    channelCol1.length !== channelRow1.length ||
    channelCol1.length !== channelCol2.length ||
    channelCol1.length !== channelRow2.length ||
    channelCol1.length !== channelDischarge.length
  ) {
    throw new Error("Hydrology channel fields are misaligned.");
  }

  const fillValue = hydrologyMeta.quantization.int16_fill_value;
  const pressureScale = Number(hydrologyMeta.quantization.effective_pressure_scale_pa_per_int16 ?? 1);
  const pressureOffset = Number(hydrologyMeta.quantization.effective_pressure_offset_pa ?? 0);
  const effectivePressure = new Float32Array(cellCount);
  const effectivePressureValid = new Uint8Array(cellCount);
  const effectivePressureHeights = new Float32Array(cellCount);
  const chunk = 20000;

  for (let i = 0; i < cellCount; i += 1) {
    const raw = pressureInt[i];
    const pressure = raw === fillValue ? Number.NaN : raw * pressureScale + pressureOffset;
    effectivePressure[i] = pressure;
    const hasPressure = bedValid[i] && Number.isFinite(pressure) && pressure > 0;
    effectivePressureValid[i] = Number(hasPressure);
    effectivePressureHeights[i] = hasPressure
      ? bedHeights[i] + effectivePressureSurfaceOffsetMeters
      : Number.NaN;

    if (i > 0 && i % chunk === 0) {
      const t = i / Math.max(1, cellCount - 1);
      postProgress(id, reportProgress, 0.12 + t * 0.2, "hydrologyProcessingField", "Processing hydrology field...");
    }
  }

  const effectivePressureGeometry = buildSurfaceGeometryWithField({
    nx,
    ny,
    dxMeters: baseConfig.dxMeters,
    dyMeters: baseConfig.dyMeters,
    horizontalMetersPerUnit: baseConfig.horizontalMetersPerUnit,
    verticalMetersPerUnit: baseConfig.verticalMetersPerUnit,
    heights: effectivePressureHeights,
    valid: effectivePressureValid,
    fieldValues: effectivePressure,
    colorFn: (pressure) => effectivePressureColor(pressure, effectivePressureLut),
    progress: (p, stageKey, stage) => postProgress(id, reportProgress, p, stageKey, stage),
    progressStart: 0.34,
    progressEnd: 0.72,
    progressLabelVerticesKey: "hydrologyBuildingMesh",
    progressLabelVertices: "Building hydrology mesh...",
    progressLabelIndicesKey: "hydrologyTriangulatingMesh",
    progressLabelIndices: "Triangulating hydrology mesh...",
    sampleStride: meshStride,
  });

  const halfX = (nx - 1) / 2;
  const halfY = (ny - 1) / 2;
  const absDy = Math.abs(baseConfig.dyMeters);
  const positions = [];
  const colors = [];
  let kept = 0;
  const channelCount = channelDischarge.length;
  const channelChunk = 8000;

  for (let i = 0; i < channelCount; i += 1) {
    const c0 = channelCol1[i];
    const r0 = channelRow1[i];
    const c1 = channelCol2[i];
    const r1 = channelRow2[i];
    if (c0 < 0 || c0 >= nx || c1 < 0 || c1 >= nx || r0 < 0 || r0 >= ny || r1 < 0 || r1 >= ny) continue;

    const idx0 = r0 * nx + c0;
    const idx1 = r1 * nx + c1;
    if (!bedValid[idx0] || !bedValid[idx1]) continue;

    const h0 = bedHeights[idx0];
    const h1 = bedHeights[idx1];
    if (!Number.isFinite(h0) || !Number.isFinite(h1)) continue;

    const discharge = channelDischarge[i];
    if (!Number.isFinite(discharge) || discharge <= 0) continue;

    const x0 = ((c0 - halfX) * baseConfig.dxMeters) / baseConfig.horizontalMetersPerUnit;
    const z0 = ((r0 - halfY) * absDy) / baseConfig.horizontalMetersPerUnit;
    const y0 = (h0 + subglacialChannelSurfaceOffsetMeters) / baseConfig.verticalMetersPerUnit;
    const x1 = ((c1 - halfX) * baseConfig.dxMeters) / baseConfig.horizontalMetersPerUnit;
    const z1 = ((r1 - halfY) * absDy) / baseConfig.horizontalMetersPerUnit;
    const y1 = (h1 + subglacialChannelSurfaceOffsetMeters) / baseConfig.verticalMetersPerUnit;

    const color = subglacialChannelColor(discharge, channelDischargeLut);
    const directionX = x1 - x0;
    const directionZ = z1 - z0;
    const length = Math.hypot(directionX, directionZ);
    if (!Number.isFinite(length) || length <= 1e-9) continue;

    const normalX = -directionZ / length;
    const normalZ = directionX / length;
    const dischargeT = channelDischargeNormalized(discharge);
    const widthMeters = lerp(CHANNEL_STRIP_WIDTH_M_MIN, CHANNEL_STRIP_WIDTH_M_MAX, dischargeT);
    const halfWidthUnits = (widthMeters / baseConfig.horizontalMetersPerUnit) * 0.5;

    const p0lx = x0 + normalX * halfWidthUnits;
    const p0lz = z0 + normalZ * halfWidthUnits;
    const p0rx = x0 - normalX * halfWidthUnits;
    const p0rz = z0 - normalZ * halfWidthUnits;
    const p1lx = x1 + normalX * halfWidthUnits;
    const p1lz = z1 + normalZ * halfWidthUnits;
    const p1rx = x1 - normalX * halfWidthUnits;
    const p1rz = z1 - normalZ * halfWidthUnits;

    positions.push(
      p0lx,
      y0,
      p0lz,
      p1lx,
      y1,
      p1lz,
      p0rx,
      y0,
      p0rz,
      p0rx,
      y0,
      p0rz,
      p1lx,
      y1,
      p1lz,
      p1rx,
      y1,
      p1rz
    );
    for (let v = 0; v < 6; v += 1) {
      colors.push(color[0], color[1], color[2]);
    }
    kept += 1;

    if (i > 0 && i % channelChunk === 0) {
      const t = i / Math.max(1, channelCount - 1);
      postProgress(id, reportProgress, 0.72 + t * 0.24, "hydrologyBuildingChannels", "Building channel ribbons...");
    }
  }

  postProgress(id, reportProgress, 0.98, "hydrologyFinalizing", "Finalizing hydrology layer...");
  return {
    effectivePressurePositions: effectivePressureGeometry.positions,
    effectivePressureColors: effectivePressureGeometry.colors,
    effectivePressureIndices: effectivePressureGeometry.indices,
    channelPositions: positions.length ? new Float32Array(positions) : null,
    channelColors: colors.length ? new Float32Array(colors) : null,
    channelCount: kept,
  };
}

function buildBasalFrictionTask(id, payload) {
  const {
    basalFrictionMeta,
    basalFrictionBuffer,
    nx,
    ny,
    cellCount,
    baseConfig,
    bedHeights,
    bedValid,
    mask,
    meshStride,
    reportProgress,
  } = payload;
  const safeStride = Math.max(1, Math.floor(Number(meshStride) || 1));
  const basalFrictionSurfaceOffsetMeters =
    BASAL_FRICTION_SURFACE_OFFSET_M + (safeStride - 1) * 64;

  if (basalFrictionMeta.grid.nx !== nx || basalFrictionMeta.grid.ny !== ny) {
    throw new Error("Basal-friction grid is not aligned to BedMachine grid.");
  }

  postProgress(id, reportProgress, 0.04, "basalFrictionProcessingField", "Processing basal friction field...");
  const basalFriction = parseField(basalFrictionMeta, basalFrictionBuffer, "basal_friction");
  if (basalFriction.length !== cellCount) {
    throw new Error("Basal-friction field length mismatch.");
  }
  if (!mask || mask.length !== cellCount) {
    throw new Error("Basal-friction mask length mismatch.");
  }

  const basalFrictionValid = new Uint8Array(cellCount);
  const basalFrictionHeights = new Float32Array(cellCount);
  const chunk = 20000;

  for (let i = 0; i < cellCount; i += 1) {
    const value = basalFriction[i];
    const hasValue = bedValid[i] && mask[i] === 2 && Number.isFinite(value) && value >= 0;
    basalFrictionValid[i] = Number(hasValue);
    basalFrictionHeights[i] = hasValue
      ? bedHeights[i] + basalFrictionSurfaceOffsetMeters
      : Number.NaN;

    if (i > 0 && i % chunk === 0) {
      const t = i / Math.max(1, cellCount - 1);
      postProgress(id, reportProgress, 0.12 + t * 0.2, "basalFrictionProcessingField", "Processing basal friction field...");
    }
  }

  const geometry = buildSurfaceGeometryWithField({
    nx,
    ny,
    dxMeters: baseConfig.dxMeters,
    dyMeters: baseConfig.dyMeters,
    horizontalMetersPerUnit: baseConfig.horizontalMetersPerUnit,
    verticalMetersPerUnit: baseConfig.verticalMetersPerUnit,
    heights: basalFrictionHeights,
    valid: basalFrictionValid,
    fieldValues: basalFriction,
    colorFn: (value) => basalFrictionColor(value),
    progress: (p, stageKey, stage) => postProgress(id, reportProgress, p, stageKey, stage),
    progressStart: 0.34,
    progressEnd: 0.96,
    progressLabelVerticesKey: "basalFrictionBuildingMesh",
    progressLabelVertices: "Building basal friction mesh...",
    progressLabelIndicesKey: "basalFrictionTriangulatingMesh",
    progressLabelIndices: "Triangulating basal friction mesh...",
    sampleStride: meshStride,
  });

  postProgress(id, reportProgress, 0.98, "basalFrictionFinalizing", "Finalizing basal friction layer...");
  return {
    positions: geometry.positions,
    colors: geometry.colors,
    indices: geometry.indices,
  };
}

function collectTransferables(result) {
  const list = [];
  const seen = new Set();
  function visit(value) {
    if (!value) return;
    if (ArrayBuffer.isView(value)) {
      const buffer = value.buffer;
      if (buffer && !seen.has(buffer)) {
        seen.add(buffer);
        list.push(buffer);
      }
      return;
    }
    if (value instanceof ArrayBuffer) {
      if (!seen.has(value)) {
        seen.add(value);
        list.push(value);
      }
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) visit(item);
      return;
    }
    if (typeof value === "object") {
      for (const item of Object.values(value)) visit(item);
    }
  }
  visit(result);
  return list;
}

self.addEventListener("message", (event) => {
  const { id, task, payload } = event.data || {};
  try {
    let result;
    if (task === "buildVelocity") {
      result = buildVelocityTask(id, payload);
    } else if (task === "buildBasalFriction") {
      result = buildBasalFrictionTask(id, payload);
    } else if (task === "buildHydrology") {
      result = buildHydrologyTask(id, payload);
    } else if (task === "buildOceanCurrents") {
      result = buildOceanCurrentTask(id, payload);
    } else {
      throw new Error(`Unknown worker task: ${String(task)}`);
    }
    const transferables = collectTransferables(result);
    self.postMessage({ id, kind: "result", ok: true, result }, transferables);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    self.postMessage({ id, kind: "result", ok: false, error: { message } });
  }
});
