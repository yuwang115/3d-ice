/**
 * Observed sea ice for the 3D ICE explorer: NOAA/NSIDC monthly sea-ice concentration from
 * passive microwave radiometry (SMMR, SSM/I, SSMIS, AMSR2), November 1978 onwards, around
 * Antarctica and Greenland.
 *
 * One package per region (docs/data-contract.md, "Observed sea-ice concentration") holds
 * every month on a 25 km grid in the region's projection; for each calendar month, the share
 * of the 1981-2010 years in which each cell held ice, whose 50 % contour is the median ice
 * edge; and the extent and area of the whole hemisphere every month, with each calendar
 * month's 1981-2010 average and least-squares trend. The explorer quotes those hemisphere
 * totals rather than sums over the part of the ocean its terrain covers.
 *
 * The explorer picks a month with two sliders, the calendar month (the seasons) and the year
 * (the same month across the years), and plays the record month by month, blending between
 * months. The module has no DOM or Three.js dependency and runs unchanged under Node's test
 * runner.
 */

import { getFieldDefinition, parseField } from "./data-contract.js";

export const SEA_ICE_KIND = "observed_sea_ice_concentration";
/** Code of a cell in a month without data. */
export const NO_DATA = 255;
/** Concentration (percent) from which a cell counts towards the extent, and is drawn. */
export const EXTENT_THRESHOLD_PERCENT = 15;
/** The median ice edge: where the 1981-2010 share of years with ice crosses one half. */
export const MEDIAN_EDGE_PERCENT = 50;
export const MONTHS_PER_YEAR = 12;

/**
 * Colours of the ice, as sRGB 0-1: bluish at the threshold, white when the sea is covered,
 * and the share of the sea surface each hides. The legend and the shader read the same values.
 */
export const SEA_ICE_STYLE = Object.freeze({
  lowColor: Object.freeze([0.62, 0.75, 0.88]),
  highColor: Object.freeze([0.97, 0.98, 1.0]),
  lowAlpha: 0.5,
  highAlpha: 0.92,
  // NSIDC draws its median ice edge in orange; the same hue reads on the dark and light themes.
  edgeColor: Object.freeze([1.0, 0.55, 0.1]),
});

/** Legend ticks (percent), placed at their values on the 15-100 % ramp. */
export const SEA_ICE_LEGEND_TICKS = Object.freeze([15, 40, 70, 100]);

function gridSize(meta) {
  const nx = Number(meta?.grid?.nx);
  const ny = Number(meta?.grid?.ny);
  if (!Number.isInteger(nx) || !Number.isInteger(ny) || nx < 1 || ny < 1) {
    throw new Error("Sea-ice package has no valid grid.");
  }
  return { nx, ny };
}

function increasing(values) {
  for (let i = 1; i < values.length; i += 1) {
    if (!(values[i] > values[i - 1])) return false;
  }
  return values.length > 0;
}

/** "1987-12" to { year: 1987, month: 12 }; throws on anything else. */
export function parseMonthLabel(label) {
  const match = /^(\d{4})-(\d{2})$/.exec(String(label));
  const month = match ? Number(match[2]) : Number.NaN;
  if (!match || month < 1 || month > MONTHS_PER_YEAR) throw new Error(`Sea-ice month ${label} is not YYYY-MM.`);
  return { year: Number(match[1]), month };
}

function readSeries(values, count, name) {
  const array = Array.isArray(values) ? values : [];
  if (array.length !== count) throw new Error(`Sea-ice series ${name} does not have one value per month.`);
  return Float64Array.from(array, (value) => (value === null || value === undefined ? Number.NaN : Number(value)));
}

function readMonthlyStatistics(entries) {
  const out = new Array(MONTHS_PER_YEAR).fill(null);
  for (const entry of Array.isArray(entries) ? entries : []) {
    const month = Number(entry?.month);
    if (!Number.isInteger(month) || month < 1 || month > MONTHS_PER_YEAR) continue;
    out[month - 1] = {
      month,
      referenceMean: Number(entry.reference_mean_mkm2),
      slope: Number(entry.slope_mkm2_per_year),
      intercept: Number(entry.intercept_mkm2_at_year_0),
      percentPerDecade: Number(entry.percent_per_decade),
      firstYear: Number(entry.trend_first_year),
      lastYear: Number(entry.trend_last_year),
    };
  }
  if (out.some((entry) => !entry)) throw new Error("Sea-ice package does not describe every calendar month.");
  return out;
}

/**
 * Decode a region's package. Throws if the payload does not follow the declared layout, so a
 * truncated download fails loudly instead of drawing garbage.
 */
export function decodeSeaIceRecord(meta, buffer) {
  if (meta?.geometry_type !== "sparse_grid_time_series" || meta?.time_series_kind !== SEA_ICE_KIND) {
    throw new Error("Package is not an observed sea-ice record.");
  }
  const { nx, ny } = gridSize(meta);
  const mask = parseField(meta, buffer, "domain_mask");
  if (mask.length !== nx * ny) throw new Error("Sea-ice domain mask does not cover the grid.");
  const cells = [];
  for (let cell = 0; cell < mask.length; cell += 1) {
    if (mask[cell]) cells.push(cell);
  }
  const domainCount = cells.length;
  if (domainCount !== Number(meta.domain?.cell_count)) {
    throw new Error("Sea-ice domain size does not match its metadata.");
  }
  const labels = Array.isArray(meta.frames?.months) ? meta.frames.months : [];
  const frameCount = labels.length;
  const years = Float64Array.from(meta.frames?.years || []);
  if (!frameCount || years.length !== frameCount || frameCount !== Number(meta.frames?.count) || !increasing(years)) {
    throw new Error("Sea-ice frames must be consecutive months with increasing times.");
  }
  const yearOf = new Int16Array(frameCount);
  const monthOf = new Uint8Array(frameCount);
  labels.forEach((label, index) => {
    const parts = parseMonthLabel(label);
    yearOf[index] = parts.year;
    monthOf[index] = parts.month;
  });
  const firstKey = yearOf[0] * MONTHS_PER_YEAR + monthOf[0] - 1;
  for (let i = 1; i < frameCount; i += 1) {
    if (yearOf[i] * MONTHS_PER_YEAR + monthOf[i] - 1 !== firstKey + i) {
      throw new Error("Sea-ice frames must be consecutive months.");
    }
  }
  const frames = parseField(meta, buffer, "concentration");
  if (frames.length !== frameCount * domainCount) {
    throw new Error("Sea-ice concentration does not match the months and domain.");
  }
  const reference = parseField(meta, buffer, "reference_ice_frequency");
  if (reference.length !== MONTHS_PER_YEAR * domainCount) {
    throw new Error("Sea-ice reference frequency does not have twelve months of the domain.");
  }
  const fill = Number(getFieldDefinition(meta, "concentration").fill_value ?? NO_DATA);
  const empty = new Set(meta.frames?.empty_months || []);
  const hasData = new Uint8Array(frameCount);
  for (let i = 0; i < frameCount; i += 1) {
    hasData[i] = empty.has(labels[i]) ? 0 : 1;
    // A month without data is coded so throughout; a stray code elsewhere is a broken package.
    if (!hasData[i] && domainCount && frames[i * domainCount] !== fill) {
      throw new Error(`Sea-ice month ${labels[i]} is listed as empty but holds data.`);
    }
  }
  if (!hasData.some(Boolean)) throw new Error("Sea-ice package has no month with data.");
  const series = meta.series || {};
  return {
    region: String(meta.region || ""),
    hemisphere: series.hemisphere === "north" ? "north" : "south",
    grid: meta.grid,
    nx,
    ny,
    domainCount,
    cellIndex: Int32Array.from(cells),
    frameCount,
    frames,
    reference,
    fill,
    years,
    yearOf,
    monthOf,
    hasData,
    firstKey,
    firstYear: yearOf[0],
    lastYear: yearOf[frameCount - 1],
    extent: readSeries(series.extent_mkm2, frameCount, "extent_mkm2"),
    area: readSeries(series.area_mkm2, frameCount, "area_mkm2"),
    monthly: readMonthlyStatistics(series.monthly),
    thresholdPercent: Number(series.threshold_percent) || EXTENT_THRESHOLD_PERCENT,
    referencePeriod: Array.isArray(series.reference_period) ? series.reference_period.map(Number) : [1981, 2010],
    sensors: Array.isArray(meta.frames?.sensors) ? meta.frames.sensors : [],
  };
}

/** Index of the frame of a year and calendar month (1-12), or -1 outside the record. */
export function frameIndexOf(record, year, month) {
  const y = Number(year);
  const m = Number(month);
  if (!Number.isInteger(y) || !Number.isInteger(m) || m < 1 || m > MONTHS_PER_YEAR) return -1;
  const index = y * MONTHS_PER_YEAR + m - 1 - record.firstKey;
  return index >= 0 && index < record.frameCount ? index : -1;
}

/** The year and calendar month of a frame. */
export function frameDate(record, index) {
  const i = Math.min(record.frameCount - 1, Math.max(0, Math.round(Number(index) || 0)));
  return { year: record.yearOf[i], month: record.monthOf[i] };
}

/** The frame with data nearest `index` in `direction` (+1 or -1), `index` itself included; -1 if none. */
export function nearestFrameWithData(record, index, direction) {
  const step = direction < 0 ? -1 : 1;
  for (let i = Math.round(index); i >= 0 && i < record.frameCount; i += step) {
    if (record.hasData[i]) return i;
  }
  return -1;
}

/**
 * The two frames a fractional position blends and the weight of the later one. A whole
 * position on a month without data is shown as such (`empty`), unless `bridgeGaps` asks for
 * playback's behaviour: passing between the months with data on either side of an empty one.
 */
export function locateFrame(record, position, { bridgeGaps = false } = {}) {
  const last = record.frameCount - 1;
  const clamped = Math.min(last, Math.max(0, Number(position) || 0));
  const lower = Math.floor(clamped);
  if (clamped === lower && (record.hasData[lower] || !bridgeGaps)) {
    return { lower, upper: lower, weight: 0, empty: !record.hasData[lower] };
  }
  const before = nearestFrameWithData(record, lower, -1);
  const after = nearestFrameWithData(record, clamped === lower ? lower : lower + 1, +1);
  if (before < 0 || after < 0) {
    const only = before < 0 ? after : before;
    return { lower: only, upper: only, weight: 0, empty: only < 0 };
  }
  return { lower: before, upper: after, weight: (clamped - before) / (after - before), empty: false };
}

/**
 * Concentration (percent, 0-100) of every domain cell at a position of the record, written
 * into `out`; returns the bracket. A month without data writes zeros. `options` go to
 * `locateFrame`.
 */
export function blendConcentrationInto(record, position, out, options = {}) {
  const bracket = locateFrame(record, position, options);
  const { domainCount, frames } = record;
  if (bracket.empty) {
    out.fill(0, 0, domainCount);
    return bracket;
  }
  const a = bracket.lower * domainCount;
  const b = bracket.upper * domainCount;
  const w = bracket.weight;
  for (let i = 0; i < domainCount; i += 1) {
    out[i] = w === 0 ? frames[a + i] : (1 - w) * frames[a + i] + w * frames[b + i];
  }
  return bracket;
}

/**
 * Share of the 1981-2010 years with ice (percent) of every domain cell for the calendar
 * months of a bracket, blended with its weight, written into `out`.
 */
export function blendReferenceInto(record, bracket, out) {
  const { domainCount, reference } = record;
  const lower = Math.max(0, bracket.lower);
  const upper = Math.max(0, bracket.upper);
  const a = (record.monthOf[lower] - 1) * domainCount;
  const b = (record.monthOf[upper] - 1) * domainCount;
  const w = bracket.weight;
  for (let i = 0; i < domainCount; i += 1) {
    out[i] = w === 0 ? reference[a + i] : (1 - w) * reference[a + i] + w * reference[b + i];
  }
  return out;
}

/** One calendar month's extent in every year of the record (NaN where there is no value). */
export function monthSeries(record, month) {
  const years = [];
  const extent = [];
  for (let year = record.firstYear; year <= record.lastYear; year += 1) {
    const index = frameIndexOf(record, year, month);
    years.push(year);
    extent.push(index >= 0 && record.hasData[index] ? record.extent[index] : Number.NaN);
  }
  return { years: Float64Array.from(years), extent: Float64Array.from(extent) };
}

/** The trend line of a calendar month's extent: its value (million km2) in a year. */
export function trendAt(statistics, year) {
  return statistics.intercept + statistics.slope * Number(year);
}

/**
 * What the readout says of a frame: its extent and area (million km2), the 1981-2010
 * average of its calendar month, its departure from it, and where it ranks among the
 * years of that month (1 is the lowest). Null for a month without data.
 */
export function extentReadout(record, index) {
  if (index < 0 || index >= record.frameCount || !record.hasData[index]) return null;
  const month = record.monthOf[index];
  const statistics = record.monthly[month - 1];
  const extent = record.extent[index];
  const anomaly = extent - statistics.referenceMean;
  const values = Array.from(monthSeries(record, month).extent).filter(Number.isFinite);
  const rank = 1 + values.filter((value) => value < extent).length;
  return {
    year: record.yearOf[index],
    month,
    extent,
    area: record.area[index],
    referenceMean: statistics.referenceMean,
    anomaly,
    anomalyPercent: (anomaly / statistics.referenceMean) * 100,
    rank,
    rankCount: values.length,
    percentPerDecade: statistics.percentPerDecade,
  };
}

/**
 * How a rank reads: from the low end in the lower half of the years, from the high end in
 * the upper half; `place` counts from 1 at that end.
 */
export function rankFromNearestEnd(rank, count) {
  if (!(count > 0) || !(rank >= 1)) return null;
  const fromTop = count - rank + 1;
  return rank <= fromTop ? { end: "lowest", place: rank, count } : { end: "highest", place: fromTop, count };
}

// Round numbers an axis may step by, per power of ten.
const NICE_STEPS = [1, 2, 2.5, 5, 10];

function niceStep(span, ticks) {
  const raw = span / Math.max(1, ticks);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  return NICE_STEPS.find((step) => step * magnitude >= raw - 1e-9 * magnitude) * magnitude;
}

/**
 * Geometry of the extent chart in an SVG box { left, right, top, bottom }: the line of a
 * calendar month's extent across the years, its trend line, the axis range and its ticks.
 * The x axis spans the record's years, so the cursor sits under the year slider's thumb.
 */
export function extentChartGeometry(series, statistics, box, { firstYear, lastYear } = {}) {
  const first = Number.isFinite(firstYear) ? firstYear : series.years[0];
  const last = Number.isFinite(lastYear) ? lastYear : series.years[series.years.length - 1];
  const finite = [];
  for (let i = 0; i < series.years.length; i += 1) {
    if (Number.isFinite(series.extent[i])) finite.push(i);
  }
  const values = finite.map((i) => series.extent[i]);
  const trendFirst = statistics ? trendAt(statistics, statistics.firstYear) : Number.NaN;
  const trendLast = statistics ? trendAt(statistics, statistics.lastYear) : Number.NaN;
  for (const value of [trendFirst, trendLast]) if (Number.isFinite(value)) values.push(value);
  let low = values.length ? Math.min(...values) : 0;
  let high = values.length ? Math.max(...values) : 1;
  if (!(high > low)) {
    low -= 0.5;
    high += 0.5;
  }
  const step = niceStep(high - low, 3);
  const yMin = Math.floor(low / step) * step;
  const yMax = Math.ceil(high / step) * step;
  const range = yMax - yMin || 1;
  const xOf = (year) => box.left + ((year - first) / Math.max(1e-9, last - first)) * (box.right - box.left);
  const yOf = (value) => box.bottom - ((Math.min(yMax, Math.max(yMin, value)) - yMin) / range) * (box.bottom - box.top);
  const point = (year, value) => `${xOf(year).toFixed(1)},${yOf(value).toFixed(1)}`;
  // A gap in the years (a month without data) breaks the line rather than bridging it.
  let line = "";
  finite.forEach((i, n) => {
    const joined = n > 0 && finite[n - 1] === i - 1;
    line += `${joined ? "L" : "M"}${point(series.years[i], series.extent[i])}`;
  });
  const trend =
    statistics && Number.isFinite(trendFirst) && Number.isFinite(trendLast)
      ? `M${point(statistics.firstYear, trendFirst)}L${point(statistics.lastYear, trendLast)}`
      : "";
  const yTicks = [];
  for (let value = yMin; value <= yMax + step / 2; value += step) yTicks.push(Number(value.toFixed(6)));
  // The ends are labelled with their year; the decades between, if clear of them.
  const xTicks = [{ at: first, label: Math.floor(first) }];
  for (let decade = Math.ceil(first / 10) * 10; decade < last; decade += 10) {
    if (decade - first >= 4 && last - decade >= 4) xTicks.push({ at: decade, label: decade });
  }
  xTicks.push({ at: last, label: Math.floor(last) });
  return { first, last, yMin, yMax, step, yTicks, xTicks, line, trend, xOf, yOf };
}

/** Position (0-1) of a concentration (percent) on the legend's 15-100 % ramp. */
export function seaIceScaleT(percent) {
  const span = 100 - EXTENT_THRESHOLD_PERCENT;
  return Math.min(1, Math.max(0, (Number(percent) - EXTENT_THRESHOLD_PERCENT) / span));
}

function smoothstep(edge0, edge1, x) {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/** RGB (0-1) and opacity of the ice at a concentration (percent); none below the threshold. */
export function seaIceColor(percent) {
  const value = Number(percent);
  if (!(value >= EXTENT_THRESHOLD_PERCENT)) return { rgb: [0, 0, 0], alpha: 0 };
  const t = seaIceScaleT(value);
  const shade = smoothstep(0, 1, t);
  const { lowColor, highColor, lowAlpha, highAlpha } = SEA_ICE_STYLE;
  return {
    rgb: lowColor.map((channel, i) => channel + shade * (highColor[i] - channel)),
    alpha: lowAlpha + t * (highAlpha - lowAlpha),
  };
}

/** RGB (0-1) at position `t` of the legend's ramp, composited over a dark sea. */
export function seaIceRampColor(t, sea = [0.06, 0.16, 0.3]) {
  const { rgb, alpha } = seaIceColor(EXTENT_THRESHOLD_PERCENT + Math.min(1, Math.max(0, t)) * (100 - EXTENT_THRESHOLD_PERCENT));
  return rgb.map((channel, i) => sea[i] + alpha * (channel - sea[i]));
}

/** The sensor of a month, from the package's runs ("n07", "F13", "am2", ...), or null. */
export function sensorOfMonth(record, index) {
  if (index < 0 || index >= record.frameCount) return null;
  const key = record.yearOf[index] * MONTHS_PER_YEAR + record.monthOf[index] - 1;
  for (const run of record.sensors) {
    try {
      const first = parseMonthLabel(run.first);
      const last = parseMonthLabel(run.last);
      const from = first.year * MONTHS_PER_YEAR + first.month - 1;
      const to = last.year * MONTHS_PER_YEAR + last.month - 1;
      if (key >= from && key <= to) return { sensor: String(run.sensor), name: String(run.name || run.sensor) };
    } catch {
      // A run with an unreadable date names no month.
    }
  }
  return null;
}
