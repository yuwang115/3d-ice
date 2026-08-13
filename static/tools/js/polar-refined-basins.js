export const MAX_REFINED_BASIN_RESPONSE_BYTES = 8 * 1024 * 1024;

const MAX_BASINS = 300;
const MAX_SEGMENTS_PER_BASIN = 800;
const MAX_POINTS_PER_SEGMENT = 10_000;
const MAX_TOTAL_SEGMENTS = 1_200;
const MAX_TOTAL_POINTS = 100_000;
const MAX_NAME_LENGTH = 80;
const MAX_ID_LENGTH = 80;

function normalizedId(value) {
  return String(value ?? "")
    .trim()
    .toLocaleLowerCase("en-US")
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function hasFinitePair(value) {
  return Array.isArray(value) &&
    value.length >= 2 &&
    typeof value[0] === "number" &&
    typeof value[1] === "number" &&
    Number.isFinite(value[0]) &&
    Number.isFinite(value[1]);
}

export function validateRefinedBasinDataset(payload) {
  if (!payload || typeof payload !== "object" || !Array.isArray(payload.basins)) {
    throw new Error("Invalid refined basin dataset");
  }
  if (payload.basins.length > MAX_BASINS) {
    throw new Error(`Refined basin dataset exceeds ${MAX_BASINS} basins`);
  }
  if (!Number.isInteger(payload.basin_count) || payload.basin_count !== payload.basins.length) {
    throw new Error("Refined basin basin_count does not match the dataset");
  }

  const ids = new Set();
  let totalSegments = 0;
  let totalPoints = 0;
  for (const basin of payload.basins) {
    const idValue = basin?.id;
    const hasScalarId = typeof idValue === "string" || (typeof idValue === "number" && Number.isFinite(idValue));
    const idText = hasScalarId ? String(idValue).trim() : "";
    const basinId = normalizedId(idText);
    const name = typeof basin?.name === "string" ? basin.name.trim() : "";
    if (!basinId || idText.length > MAX_ID_LENGTH) throw new Error("Invalid refined basin ID");
    if (ids.has(basinId)) throw new Error(`Duplicate refined basin ID: ${idText}`);
    ids.add(basinId);
    if (!name || name.length > MAX_NAME_LENGTH) throw new Error(`Invalid refined basin name: ${idText}`);
    if (!hasFinitePair(basin?.label_xy_m)) throw new Error(`Invalid refined basin label coordinate: ${idText}`);
    if (basin.area_km2 != null &&
      (typeof basin.area_km2 !== "number" || !Number.isFinite(basin.area_km2) || basin.area_km2 < 0)) {
      throw new Error(`Invalid refined basin area: ${idText}`);
    }

    const segments = basin?.segments_xy_m;
    if (!Array.isArray(segments) || segments.length > MAX_SEGMENTS_PER_BASIN) {
      throw new Error(`Invalid refined basin segment count: ${idText}`);
    }
    totalSegments += segments.length;
    if (totalSegments > MAX_TOTAL_SEGMENTS) throw new Error("Refined basin dataset exceeds the total segment limit");
    for (const segment of segments) {
      if (!Array.isArray(segment) || segment.length > MAX_POINTS_PER_SEGMENT) {
        throw new Error(`Invalid refined basin segment size: ${idText}`);
      }
      totalPoints += segment.length;
      if (totalPoints > MAX_TOTAL_POINTS) throw new Error("Refined basin dataset exceeds the total point limit");
      for (const point of segment) {
        if (!hasFinitePair(point)) throw new Error(`Invalid refined basin coordinate: ${idText}`);
      }
    }
  }
  return payload;
}

async function readResponseBytes(response, maximumBytes) {
  const declaredSize = Number(response.headers?.get?.("content-length") || 0);
  if (declaredSize > maximumBytes) throw new Error("Refined basin response exceeds the size limit");

  if (!response.body?.getReader) throw new Error("Refined basin response requires bounded streaming");

  const reader = response.body.getReader();
  const chunks = [];
  let totalBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    totalBytes += value.byteLength;
    if (totalBytes > maximumBytes) {
      await reader.cancel();
      throw new Error("Refined basin response exceeds the size limit");
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

export async function fetchRefinedBasinJson(
  url,
  errorLabel,
  { fetchImpl = globalThis.fetch, maximumBytes = MAX_REFINED_BASIN_RESPONSE_BYTES } = {},
) {
  if (typeof fetchImpl !== "function") throw new Error("A fetch implementation is required");
  const response = await fetchImpl(url, { cache: "force-cache" });
  if (!response?.ok) throw new Error(`${errorLabel} (${response?.status || "network error"})`);
  const bytes = await readResponseBytes(response, maximumBytes);
  let payload;
  try {
    payload = JSON.parse(new TextDecoder().decode(bytes));
  } catch (_error) {
    throw new Error(`${errorLabel} (invalid JSON)`);
  }
  return validateRefinedBasinDataset(payload);
}
