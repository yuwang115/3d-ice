"use strict";

const FLOWLINE_SURFACE_OFFSET_M = 18;
const VELOCITY_REFERENCE_SPEED = 1800;
const EFFECTIVE_PRESSURE_REFERENCE_PA = 5_000_000;
const CHANNEL_DISCHARGE_MIN = 1e-3;
const CHANNEL_DISCHARGE_MAX = 100;
const CHANNEL_STRIP_WIDTH_M_MIN = 1500;
const CHANNEL_STRIP_WIDTH_M_MAX = 5000;
const EFFECTIVE_PRESSURE_SURFACE_OFFSET_M = 10;
const SUBGLACIAL_CHANNEL_SURFACE_OFFSET_M = 14;

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
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
  const scaled = Math.log1p(speed) / Math.log1p(VELOCITY_REFERENCE_SPEED);
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

function postProgress(id, enabled, progress, stage) {
  if (!enabled) return;
  self.postMessage({
    id,
    kind: "progress",
    progress: clamp01(progress),
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
  progressLabelVertices,
  progressLabelIndices,
  sampleStride = 1,
}) {
  const stride = Math.max(1, Math.floor(Number(sampleStride) || 1));
  const sampledNx = Math.floor((nx - 1) / stride) + 1;
  const sampledNy = Math.floor((ny - 1) / stride) + 1;
  const vertexCount = sampledNx * sampledNy;
  const positions = new Float32Array(vertexCount * 3);
  const colors = new Uint8Array(vertexCount * 3);
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
      const rgb = colorFn(fieldValues[sourceIndex]);
      colors[3 * targetIndex] = Math.round(clamp01(rgb[0]) * 255);
      colors[3 * targetIndex + 1] = Math.round(clamp01(rgb[1]) * 255);
      colors[3 * targetIndex + 2] = Math.round(clamp01(rgb[2]) * 255);
    }

    if (sampledRow > 0 && sampledRow % rowStep === 0) {
      const t = sampledRow / Math.max(1, sampledNy - 1);
      const p = progressStart + (progressEnd - progressStart) * t * 0.5;
      progress(p, progressLabelVertices);
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
      progress(p, progressLabelIndices);
    }
  }

  const trimmed = new Uint32Array(indexCursor);
  trimmed.set(indices.subarray(0, indexCursor));
  return { positions, colors, indices: trimmed };
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

  postProgress(id, reportProgress, 0.02, "Decoding velocity field...");
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
      postProgress(id, reportProgress, 0.08 + t * 0.3, "Decoding velocity field...");
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
    progress: (p, stage) => postProgress(id, reportProgress, p, stage),
    progressStart: 0.4,
    progressEnd: 0.92,
    progressLabelVertices: "Building velocity mesh...",
    progressLabelIndices: "Triangulating velocity mesh...",
    sampleStride: meshStride,
  });

  const velocityMedianSpeed = finiteMedian(velocitySpeed);
  postProgress(id, reportProgress, 0.98, "Finalizing velocity layer...");
  return {
    positions: geometry.positions,
    colors: geometry.colors,
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

  postProgress(id, reportProgress, 0.04, "Processing hydrology field...");
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
      postProgress(id, reportProgress, 0.12 + t * 0.2, "Processing hydrology field...");
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
    progress: (p, stage) => postProgress(id, reportProgress, p, stage),
    progressStart: 0.34,
    progressEnd: 0.72,
    progressLabelVertices: "Building hydrology mesh...",
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
      postProgress(id, reportProgress, 0.72 + t * 0.24, "Building channel ribbons...");
    }
  }

  postProgress(id, reportProgress, 0.98, "Finalizing hydrology layer...");
  return {
    effectivePressurePositions: effectivePressureGeometry.positions,
    effectivePressureColors: effectivePressureGeometry.colors,
    effectivePressureIndices: effectivePressureGeometry.indices,
    channelPositions: positions.length ? new Float32Array(positions) : null,
    channelColors: colors.length ? new Float32Array(colors) : null,
    channelCount: kept,
  };
}

function collectTransferables(result) {
  const list = [];
  const seen = new Set();
  for (const value of Object.values(result)) {
    if (ArrayBuffer.isView(value)) {
      const buffer = value.buffer;
      if (buffer && !seen.has(buffer)) {
        seen.add(buffer);
        list.push(buffer);
      }
    } else if (value instanceof ArrayBuffer) {
      if (!seen.has(value)) {
        seen.add(value);
        list.push(value);
      }
    }
  }
  return list;
}

self.addEventListener("message", (event) => {
  const { id, task, payload } = event.data || {};
  try {
    let result;
    if (task === "buildVelocity") {
      result = buildVelocityTask(id, payload);
    } else if (task === "buildHydrology") {
      result = buildHydrologyTask(id, payload);
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
