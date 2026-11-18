/**
 * Where and when the ocean around Antarctica was measured, for the 3D ICE explorer: every ship
 * CTD, Argo float and instrumented-seal profile of the OCEAN ICE hydrography compilation (Zhou
 * et al., 2026) inside the explorer's Antarctic domain, 1970-2025.
 *
 * Two packages (docs/data-contract.md, "Ocean profiles"): the points, sorted by date, with
 * position, day, platform, the depth of the deepest measurement and the temperature and
 * salinity there; and, loaded on the first click, each profile's temperature and salinity at
 * the standard depths above its deepest measurement and at that measurement. Because the points
 * are sorted by date, a year is one contiguous range, which the explorer draws on its own.
 *
 * The module has no DOM or Three.js dependency and runs unchanged under Node's test runner.
 */

import { decodeFieldToFloat32, parseField } from "./data-contract.js";

export const PROFILES_KIND = "ocean_profiles";
export const CURVES_KIND = "ocean_profile_curves";
const EPOCH_1970_MS = Date.UTC(1970, 0, 1);
const DAY_MS = 86400000;
const INDEX_CELL_M = 25000;

/** Platform codes of the compilation, with the colours (sRGB 0-1) the points and chart share. */
export const PROFILE_PLATFORMS = Object.freeze([
  Object.freeze({ code: 1, key: "ship", color: Object.freeze([1.0, 0.71, 0.28]) }),
  Object.freeze({ code: 2, key: "argo", color: Object.freeze([0.35, 0.82, 1.0]) }),
  Object.freeze({ code: 3, key: "seal", color: Object.freeze([1.0, 0.44, 0.68]) }),
]);

function platformOf(code) {
  return PROFILE_PLATFORMS.find((platform) => platform.code === code) || null;
}

function uintField(meta, buffer, name, count) {
  const values = parseField(meta, buffer, name);
  if (values.length !== count) throw new Error(`Profile field ${name} does not hold one value per point.`);
  return values;
}

function floatField(meta, buffer, name, count) {
  const values = decodeFieldToFloat32(meta, buffer, name);
  if (values.length !== count) throw new Error(`Profile field ${name} does not hold one value per point.`);
  return values;
}

/**
 * Decode the points. Throws if the payload does not follow its declared layout, or if the
 * points are not sorted by date as the year ranges assume.
 */
export function decodeProfilePoints(meta, buffer) {
  if (meta?.geometry_type !== "points" || meta?.point_kind !== PROFILES_KIND) {
    throw new Error("Package is not an ocean-profile record.");
  }
  const count = Number(meta.points?.count);
  if (!Number.isInteger(count) || count < 1) throw new Error("Ocean-profile package has no points.");
  const day = uintField(meta, buffer, "day", count);
  for (let i = 1; i < count; i += 1) {
    if (day[i] < day[i - 1]) throw new Error("Ocean profiles are not sorted by date.");
  }
  const firstYear = Number(meta.years?.first);
  const lastYear = Number(meta.years?.last);
  const starts = Int32Array.from(meta.years?.starts || []);
  if (!Number.isInteger(firstYear) || !(lastYear >= firstYear) || starts.length !== lastYear - firstYear + 2 || starts[starts.length - 1] !== count) {
    throw new Error("Ocean-profile years do not index the points.");
  }
  for (let k = 1; k < starts.length; k += 1) {
    if (starts[k] < starts[k - 1] || starts[0] !== 0) throw new Error("Ocean-profile years do not index the points.");
  }
  const counts = {};
  for (const platform of PROFILE_PLATFORMS) {
    const values = meta.series?.counts_by_platform_and_year?.[String(platform.code)];
    counts[platform.code] = Int32Array.from(Array.isArray(values) ? values : new Array(lastYear - firstYear + 1).fill(0));
  }
  return {
    count,
    x: floatField(meta, buffer, "x", count),
    y: floatField(meta, buffer, "y", count),
    day,
    platform: uintField(meta, buffer, "platform", count),
    maxDepth: uintField(meta, buffer, "max_depth", count),
    bottomTemperature: floatField(meta, buffer, "bottom_ct", count),
    bottomSalinity: floatField(meta, buffer, "bottom_sa", count),
    firstYear,
    lastYear,
    starts,
    counts,
  };
}

/** The contiguous range of points of a year (empty outside the record), or of every year. */
export function yearRange(points, year = null) {
  if (year === null) return { start: 0, count: points.count };
  const k = Math.round(year) - points.firstYear;
  if (k < 0 || k > points.lastYear - points.firstYear) return { start: 0, count: 0 };
  return { start: points.starts[k], count: points.starts[k + 1] - points.starts[k] };
}

/**
 * The year the layer opens on: the last year of the record unless it is still filling up (fewer
 * than half the profiles of the year before), in which case the year before.
 */
export function defaultProfileYear(points) {
  const last = points.lastYear;
  if (last <= points.firstYear) return last;
  const count = (year) => yearRange(points, year).count;
  return count(last) < 0.5 * count(last - 1) ? last - 1 : last;
}

/** "YYYY-MM-DD" of a point's day. */
export function profileDate(points, index) {
  return new Date(EPOCH_1970_MS + points.day[index] * DAY_MS).toISOString().slice(0, 10);
}

/** Points per platform in a range, for the readout. */
export function countByPlatform(points, range) {
  const out = { total: range.count };
  for (const platform of PROFILE_PLATFORMS) out[platform.key] = 0;
  for (let i = range.start; i < range.start + range.count; i += 1) {
    const platform = platformOf(points.platform[i]);
    if (platform) out[platform.key] += 1;
  }
  return out;
}

/** A uniform grid over the points (projected metres) for nearest-point queries. */
export function buildProfileIndex(points, cellMeters = INDEX_CELL_M) {
  const buckets = new Map();
  for (let i = 0; i < points.count; i += 1) {
    const key = `${Math.floor(points.x[i] / cellMeters)},${Math.floor(points.y[i] / cellMeters)}`;
    let bucket = buckets.get(key);
    if (!bucket) {
      bucket = [];
      buckets.set(key, bucket);
    }
    bucket.push(i);
  }
  return { cellMeters, buckets };
}

/** The drawn point nearest (x, y) within radiusMeters, or -1. */
export function nearestProfile(points, index, x, y, radiusMeters, range = yearRange(points)) {
  const cell = index.cellMeters;
  const reach = Math.ceil(radiusMeters / cell);
  const cx = Math.floor(x / cell);
  const cy = Math.floor(y / cell);
  const end = range.start + range.count;
  let best = -1;
  let bestDistance = radiusMeters;
  for (let dy = -reach; dy <= reach; dy += 1) {
    for (let dx = -reach; dx <= reach; dx += 1) {
      const bucket = index.buckets.get(`${cx + dx},${cy + dy}`);
      if (!bucket) continue;
      for (const i of bucket) {
        if (i < range.start || i >= end) continue;
        const distance = Math.hypot(points.x[i] - x, points.y[i] - y);
        if (distance <= bestDistance) {
          best = i;
          bestDistance = distance;
        }
      }
    }
  }
  return best;
}

/** Decode the curves against the points they belong to. */
export function decodeProfileCurves(meta, buffer, points) {
  if (meta?.geometry_type !== "points" || meta?.point_kind !== CURVES_KIND) {
    throw new Error("Package is not an ocean-profile curve record.");
  }
  if (Number(meta.points?.count) !== points.count) throw new Error("Profile curves do not match the points.");
  const depths = Float64Array.from(meta.curves?.standard_depths_m || []);
  if (!depths.length) throw new Error("Profile curves name no standard depths.");
  const offsets = new Int32Array(points.count + 1);
  for (let i = 0; i < points.count; i += 1) {
    let length = 1;
    for (let k = 0; k < depths.length; k += 1) if (depths[k] < points.maxDepth[i]) length += 1;
    offsets[i + 1] = offsets[i] + length;
  }
  const temperature = decodeFieldToFloat32(meta, buffer, "ct");
  const salinity = decodeFieldToFloat32(meta, buffer, "sa");
  if (temperature.length !== offsets[points.count] || salinity.length !== offsets[points.count]) {
    throw new Error("Profile curves do not hold the values their points call for.");
  }
  return { depths, offsets, temperature, salinity };
}

/** One profile's curve: depth, temperature and salinity, shallowest first. */
export function profileCurve(curves, points, index) {
  const start = curves.offsets[index];
  const end = curves.offsets[index + 1];
  const out = [];
  for (let k = start; k < end; k += 1) {
    const depth = k === end - 1 ? points.maxDepth[index] : curves.depths[k - start];
    out.push({ depth, temperature: curves.temperature[k], salinity: curves.salinity[k] });
  }
  return out;
}

/** Stacked bars of the points per year and platform in a chart box, and the year's cursor. */
export function countsChartGeometry(points, box, year = null) {
  const years = points.lastYear - points.firstYear + 1;
  const totals = new Int32Array(years);
  for (const platform of PROFILE_PLATFORMS) {
    const values = points.counts[platform.code];
    for (let k = 0; k < years; k += 1) totals[k] += values[k] || 0;
  }
  const max = Math.max(1, ...totals);
  const width = (box.right - box.left) / years;
  const scale = (box.bottom - box.top) / max;
  const bars = [];
  for (let k = 0; k < years; k += 1) {
    let base = box.bottom;
    const segments = [];
    for (const platform of PROFILE_PLATFORMS) {
      const value = points.counts[platform.code][k] || 0;
      if (!value) continue;
      const height = value * scale;
      segments.push({ key: platform.key, y: base - height, height });
      base -= height;
    }
    bars.push({ year: points.firstYear + k, x: box.left + k * width, width, segments });
  }
  const cursor = year === null ? null : box.left + (Math.round(year) - points.firstYear + 0.5) * width;
  return { bars, max, cursor };
}

/** Polylines of a curve's temperature and salinity against depth in a chart box (depth down). */
export function curveChartGeometry(curve, box, { temperatureRange = null, salinityRange = null } = {}) {
  const finite = (key) => curve.map((point) => point[key]).filter(Number.isFinite);
  const range = (values, fallback, pad) => {
    if (!values.length) return fallback;
    const low = Math.min(...values);
    const high = Math.max(...values);
    return high - low < pad ? [low - pad / 2, high + pad / 2] : [low, high];
  };
  const tRange = temperatureRange || range(finite("temperature"), [-2, 2], 0.5);
  const sRange = salinityRange || range(finite("salinity"), [34, 35], 0.1);
  const maxDepth = Math.max(10, ...curve.map((point) => point.depth));
  const yOf = (depth) => box.top + (depth / maxDepth) * (box.bottom - box.top);
  const xOf = (value, [low, high]) => box.left + ((value - low) / (high - low)) * (box.right - box.left);
  const line = (key, valueRange) =>
    curve.filter((point) => Number.isFinite(point[key])).map((point) => [xOf(point[key], valueRange), yOf(point.depth)]);
  return { temperature: line("temperature", tRange), salinity: line("salinity", sRange), tRange, sRange, maxDepth };
}
