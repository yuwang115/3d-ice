/**
 * Observed surface elevation change for the 3D ICE explorer: CPOM's record of the Antarctic
 * Ice Sheet from multi-mission radar altimetry (ERS-2, Envisat, CryoSat-2), 1995-2026.
 *
 * Two packages feed it (docs/data-contract.md, "Observed surface elevation change"):
 *
 * - the rate map, `cpom_antarctica_dhdt_*`, a gridded overlay of the 1995-2026 trend and
 *   its uncertainty on the Balanced and HD terrain grids;
 * - the record, `cpom_antarctica_dh_480`, the cumulative change at every 140-day epoch on
 *   the 10 km grid, with CPOM's own mass-change series of the ice sheet, its regions and its
 *   18 IMBIE drainage basins, which the explorer quotes instead of summing the resampled grid.
 *
 * A basin, or a region's basins, can be picked from the rate map's `basin_id`: the drape
 * outlines it and a chart plots its mass change.
 *
 * Both are drawn as a colour on today's ice surface, on the explorer's thickness-change
 * ramp with a scale of their own. A change smaller than twice its uncertainty, and the
 * change near the pole that CPOM extrapolates rather than observes, are drawn at a quarter of
 * their colour. The module has no DOM or Three.js dependency and runs unchanged under
 * Node's test runner.
 */

import { getFieldDefinition, parseField, resolveInt16Quantization } from "./data-contract.js";
import { locateYear, thicknessChangeRampColor, thicknessChangeScaleT } from "./ice-projection.js";

export const SURFACE_CHANGE_KIND = "observed_surface_change";
export const SURFACE_CHANGE_VIEWS = Object.freeze({ cumulative: "cumulative", rate: "rate" });

/**
 * Colour scales: asinh, linear within `softness` and saturating at `saturation`. The ticks
 * are placed at their values on the legend. They are spread so that their labels fit on one
 * row of the 300 px legend, with the unit on the zero label only.
 */
export const SURFACE_CHANGE_SCALES = Object.freeze({
  cumulative: Object.freeze({
    softness: 1,
    saturation: 150,
    ticks: Object.freeze([-150, -20, -5, 0, 5, 20, 150]),
  }),
  rate: Object.freeze({
    softness: 0.05,
    saturation: 5,
    ticks: Object.freeze([-5, -1, -0.2, 0, 0.2, 1, 5]),
  }),
});

/** A change within this many standard deviations of zero is drawn faded. */
export const FADE_SIGMA = 2;
/** Share of its colour that a faded cell keeps, measured from the ramp's neutral middle. */
export const FADED_COLOUR_SHARE = 0.25;
export const REGION_KEYS = Object.freeze(["AIS", "WAIS", "EAIS", "APIS"]);
export const GT_PER_MM_SLE = 362.5;
/** The chart's subject when nothing is picked: the whole ice sheet. */
export const WHOLE_SHEET = "AIS";
const BASIN_KEY_PREFIX = "basin:";
/** IMBIE's drainage basins, as CPOM's basin_id names them, in the order their segments follow the coast. */
export const BASIN_COASTLINE_ORDER = Object.freeze([
  "A-Ap", "Ap-B", "B-C", "C-Cp", "Cp-D", "D-Dp", "Dp-E", "E-Ep", "Ep-F",
  "F-G", "G-H", "H-Hp", "Hp-I", "I-Ipp", "Ipp-J", "J-Jpp", "Jpp-K", "K-A",
]);

function int16AsFloat(meta, buffer, name) {
  const field = getFieldDefinition(meta, name);
  const codes = parseField(meta, buffer, name);
  const { scale, offset, fillValue } = resolveInt16Quantization(meta, field);
  const out = new Float32Array(codes.length);
  for (let i = 0; i < codes.length; i += 1) {
    out[i] = codes[i] === fillValue ? Number.NaN : codes[i] * scale + offset;
  }
  return out;
}

function gridSize(meta) {
  const nx = Number(meta?.grid?.nx);
  const ny = Number(meta?.grid?.ny);
  if (!Number.isInteger(nx) || !Number.isInteger(ny) || nx < 1 || ny < 1) {
    throw new Error("Surface-change package has no valid grid.");
  }
  return { nx, ny };
}

function scaleFromMeta(meta, view) {
  const fallback = SURFACE_CHANGE_SCALES[view];
  const visualization = meta?.visualization || {};
  const softness = Number(view === "rate" ? visualization.linear_core_m_per_yr : visualization.linear_core_m);
  const saturation = Number(view === "rate" ? visualization.saturation_m_per_yr : visualization.saturation_m);
  return {
    softness: softness > 0 ? softness : fallback.softness,
    saturation: saturation > 0 ? saturation : fallback.saturation,
    ticks: fallback.ticks,
  };
}

/**
 * Radii (m from the pole) of the polar caps CPOM extrapolates, and the first year CryoSat-2
 * data reach south of 81.5 S (NaN when the package does not say).
 */
function readPoleHole(meta) {
  const radius = meta?.pole_hole?.radius_m || {};
  return {
    earlyRadiusM: Number(radius.ers_envisat) || 0,
    lateRadiusM: Number(radius.cryosat2) || 0,
    observedFromYear: Number(meta?.pole_hole?.observed_from_year),
  };
}

/** Decode the rate map: rate and uncertainty (m/yr, NaN where not drawn) per grid cell. */
export function decodeSurfaceChangeRate(meta, buffer) {
  const { nx, ny } = gridSize(meta);
  const values = int16AsFloat(meta, buffer, "dhdt");
  const sigma = int16AsFloat(meta, buffer, "dhdt_uncertainty");
  const basin = parseField(meta, buffer, "basin_id");
  if (values.length !== nx * ny || sigma.length !== nx * ny || basin.length !== nx * ny) {
    throw new Error("Surface-change rate fields do not cover the grid.");
  }
  return {
    view: SURFACE_CHANGE_VIEWS.rate,
    grid: meta.grid,
    nx,
    ny,
    values,
    sigma,
    basin,
    period: { startYear: Number(meta.period?.start_year), endYear: Number(meta.period?.end_year) },
    poleHole: readPoleHole(meta),
    scale: scaleFromMeta(meta, "rate"),
  };
}

function increasing(values) {
  for (let i = 1; i < values.length; i += 1) {
    if (!(values[i] > values[i - 1])) return false;
  }
  return values.length > 0;
}

function readOneSeries(entry, frameCount, key) {
  const massChange = Float64Array.from(entry.mass_change_gt || []);
  const uncertainty = Float64Array.from(entry.uncertainty_gt || []);
  if (massChange.length !== frameCount || uncertainty.length !== frameCount) {
    throw new Error(`Surface-change series ${key} does not have one value per epoch.`);
  }
  return { massChange, uncertainty };
}

function readRegionSeries(meta, frameCount) {
  const regions = meta?.series?.regions || {};
  const out = {};
  for (const key of REGION_KEYS) {
    if (regions[key]) out[key] = readOneSeries(regions[key], frameCount, key);
  }
  return out;
}

/** CPOM's basins with their series, in coastline order (an unknown code goes last). */
function readBasinSeries(meta, frameCount) {
  const basins = meta?.series?.basins || {};
  const rank = (code) => {
    const index = BASIN_COASTLINE_ORDER.indexOf(code);
    return index < 0 ? BASIN_COASTLINE_ORDER.length : index;
  };
  return Object.entries(basins)
    .map(([id, entry]) => {
      const basinId = Number(id);
      if (!Number.isInteger(basinId) || basinId < 1 || basinId > 255) {
        throw new Error(`Surface-change basin id ${id} is not a basin_id value.`);
      }
      if (!REGION_KEYS.includes(entry.region) || entry.region === WHOLE_SHEET || !entry.code) {
        throw new Error(`Surface-change basin ${id} has no code or IMBIE region.`);
      }
      return { id: basinId, code: String(entry.code), region: entry.region, ...readOneSeries(entry, frameCount, id) };
    })
    .sort((a, b) => rank(a.code) - rank(b.code) || a.id - b.id);
}

/**
 * The mass-change series of a record's metadata: the epochs, the ice sheet's and regions'
 * series, and the basins'. The rate map's views read it from the record's metadata alone.
 */
export function decodeSurfaceChangeSeries(meta) {
  const years = Float64Array.from(meta?.series?.years || meta?.frames?.years || []);
  if (!increasing(years)) throw new Error("Surface-change series epochs must be strictly increasing.");
  return {
    years,
    regions: readRegionSeries(meta, years.length),
    basins: readBasinSeries(meta, years.length),
    gtPerMm: Number(meta?.series?.gt_per_mm_sle) > 0 ? Number(meta.series.gt_per_mm_sle) : GT_PER_MM_SLE,
  };
}

/**
 * Decode the record. Throws if the payload does not follow the declared layout, so a
 * truncated download fails loudly instead of drawing garbage.
 */
export function decodeSurfaceChangeRecord(meta, buffer) {
  if (meta?.geometry_type !== "sparse_grid_time_series" || meta?.time_series_kind !== SURFACE_CHANGE_KIND) {
    throw new Error("Package is not an observed surface-change record.");
  }
  const { nx, ny } = gridSize(meta);
  const mask = parseField(meta, buffer, "domain_mask");
  if (mask.length !== nx * ny) throw new Error("Surface-change domain mask does not cover the grid.");
  const cells = [];
  for (let cell = 0; cell < mask.length; cell += 1) {
    if (mask[cell]) cells.push(cell);
  }
  const domainCount = cells.length;
  if (domainCount !== Number(meta.domain?.cell_count)) {
    throw new Error("Surface-change domain size does not match its metadata.");
  }
  const years = Float64Array.from(meta.frames?.years || []);
  if (!increasing(years)) throw new Error("Surface-change epochs must be strictly increasing.");
  const field = getFieldDefinition(meta, "surface_change");
  const frames = parseField(meta, buffer, "surface_change");
  if (frames.length !== years.length * domainCount) {
    throw new Error("Surface-change frames do not match the epochs and domain.");
  }
  const finalSigma = int16AsFloat(meta, buffer, "final_uncertainty");
  if (finalSigma.length !== domainCount) throw new Error("Surface-change uncertainty does not match the domain.");
  const quant = resolveInt16Quantization(meta, field);
  return {
    view: SURFACE_CHANGE_VIEWS.cumulative,
    grid: meta.grid,
    nx,
    ny,
    domainCount,
    cellIndex: Int32Array.from(cells),
    frames,
    frameScale: quant.scale,
    frameOffset: quant.offset,
    frameFill: quant.fillValue,
    finalSigma,
    years,
    startYears: Float64Array.from(meta.frames?.start_years || []),
    endYears: Float64Array.from(meta.frames?.end_years || []),
    regions: readRegionSeries(meta, years.length),
    basins: readBasinSeries(meta, years.length),
    gtPerMm: Number(meta.series?.gt_per_mm_sle) > 0 ? Number(meta.series.gt_per_mm_sle) : GT_PER_MM_SLE,
    poleHole: readPoleHole(meta),
    scale: scaleFromMeta(meta, "cumulative"),
  };
}

/** Epoch time (decimal year) at a fractional epoch index, as the record's slider moves. */
export function yearAtEpoch(record, position) {
  const last = record.years.length - 1;
  const clamped = Math.min(last, Math.max(0, Number(position) || 0));
  const lower = Math.floor(clamped);
  if (lower >= last) return record.years[last];
  const weight = clamped - lower;
  return record.years[lower] + weight * (record.years[lower + 1] - record.years[lower]);
}

/** Change (m) of every domain cell at `year`, written into `out`; returns the bracket. */
export function blendRecordInto(record, year, out) {
  const bracket = locateYear(record.years, year);
  const { domainCount, frames, frameScale: scale, frameOffset: offset, frameFill: fill } = record;
  const lowerStart = bracket.lower * domainCount;
  const upperStart = bracket.upper * domainCount;
  const w = bracket.weight;
  for (let i = 0; i < domainCount; i += 1) {
    const a = frames[lowerStart + i];
    const b = frames[upperStart + i];
    if (a === fill || b === fill) {
      out[i] = Number.NaN;
      continue;
    }
    const va = a * scale + offset;
    out[i] = w === 0 ? va : (1 - w) * va + w * (b * scale + offset);
  }
  return bracket;
}

/**
 * Radius (m from the pole) inside which CPOM's change at `year` is extrapolated: the
 * ERS-2/Envisat limit (81.5 S) until CryoSat-2 data arrive, CryoSat-2's (88 S) afterwards.
 */
export function extrapolatedRadiusM(record, year) {
  const { earlyRadiusM, lateRadiusM, observedFromYear } = record.poleHole;
  if (!Number.isFinite(observedFromYear)) return lateRadiusM;
  return Number(year) < observedFromYear ? earlyRadiusM : lateRadiusM;
}

/** Whether a change stands out from its uncertainty (`k` standard deviations). */
export function isSignificant(value, sigma, k = FADE_SIGMA) {
  if (!Number.isFinite(value)) return false;
  if (!Number.isFinite(sigma)) return true;
  return Math.abs(value) >= k * sigma;
}

function interpolate(years, values, year) {
  const bracket = locateYear(years, year);
  const a = values[bracket.lower];
  const b = values[bracket.upper];
  return bracket.weight === 0 ? a : (1 - bracket.weight) * a + bracket.weight * b;
}

/**
 * CPOM's cumulative mass change at `year` for each region: Gt, its 1-sigma uncertainty, and
 * the sea-level equivalent in mm (positive is a rise).
 */
export function regionalChangeAt(record, year) {
  const out = {};
  for (const [key, series] of Object.entries(record.regions)) {
    out[key] = seriesChangeAt(record, series, year);
  }
  return out;
}

/** One series' mass change at `year`: Gt, its 1-sigma uncertainty, and the sea-level equivalent (mm). */
export function seriesChangeAt(seriesSet, series, year) {
  const massChangeGt = interpolate(seriesSet.years, series.massChange, year);
  return {
    massChangeGt,
    uncertaintyGt: interpolate(seriesSet.years, series.uncertainty, year),
    seaLevelMm: -massChangeGt / seriesSet.gtPerMm,
  };
}

/** The key that selects a basin; regions and the whole sheet are selected by their own keys. */
export function basinKey(id) {
  return `${BASIN_KEY_PREFIX}${id}`;
}

/**
 * What a selection key names: the whole sheet ("AIS"), a region ("WAIS", "EAIS", "APIS"), or
 * one basin ("basin:<id>"); with its series and, for a basin, its code and region. Null for a
 * key the series do not have.
 */
export function resolveSubject(seriesSet, key) {
  if (typeof key === "string" && key.startsWith(BASIN_KEY_PREFIX)) {
    const id = Number(key.slice(BASIN_KEY_PREFIX.length));
    const basin = seriesSet?.basins?.find((entry) => entry.id === id);
    return basin ? { key, kind: "basin", ...basin } : null;
  }
  const series = seriesSet?.regions?.[key];
  if (!series) return null;
  return { key, kind: key === WHOLE_SHEET ? "sheet" : "region", region: key, ...series };
}

/** Basin ids a selection outlines: none for the whole sheet, a region's basins, or one basin. */
export function selectionBasinIds(seriesSet, key) {
  const subject = resolveSubject(seriesSet, key);
  if (!subject || subject.kind === "sheet") return null;
  if (subject.kind === "basin") return new Set([subject.id]);
  return new Set(seriesSet.basins.filter((basin) => basin.region === subject.region).map((basin) => basin.id));
}

/** 1 for the grid cells in the selected basins, 0 elsewhere. */
export function selectionMask(basinIds, selected) {
  const mask = new Uint8Array(basinIds.length);
  if (!selected?.size) return mask;
  for (let cell = 0; cell < basinIds.length; cell += 1) {
    if (selected.has(basinIds[cell])) mask[cell] = 1;
  }
  return mask;
}

/**
 * The basin under a point of the drape, from its texture coordinate (u along the grid's
 * columns, v along its rows, 0-1 from the first cell centre to the last): that of the
 * nearest of the four cells around the point that has one, or 0.
 */
export function basinAtUv(basinIds, nx, ny, u, v) {
  if (!Number.isFinite(u) || !Number.isFinite(v) || nx < 1 || ny < 1) return 0;
  const col = Math.min(1, Math.max(0, u)) * (nx - 1);
  const row = Math.min(1, Math.max(0, v)) * (ny - 1);
  const corners = [];
  for (const c of new Set([Math.floor(col), Math.ceil(col)])) {
    for (const r of new Set([Math.floor(row), Math.ceil(row)])) {
      corners.push({ cell: r * nx + c, distance: Math.hypot(c - col, r - row) });
    }
  }
  corners.sort((a, b) => a.distance - b.distance);
  for (const { cell } of corners) {
    if (basinIds[cell] > 0) return basinIds[cell];
  }
  return 0;
}

// Round numbers an axis may end on, per power of ten.
const NICE_AXIS_STEPS = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];
// Closest an axis label may sit to the zero label, in viewBox units (labels are 7.5 tall).
const MIN_TICK_GAP = 9;

function niceCeiling(value) {
  if (!(value > 0)) return 0;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const step = NICE_AXIS_STEPS.find((candidate) => candidate * magnitude >= value - 1e-9 * magnitude);
  return step * magnitude;
}

/**
 * Geometry of a mass-change chart in an SVG box { left, right, top, bottom }: the line, the
 * 1-sigma band, the axis range and its ticks. An axis side with less than 4 % of the span is
 * dropped, so a series that barely leaves zero gets no tick for it; a short side keeps its
 * extent but loses its label when that would crowd the zero label.
 */
export function massChangeChartGeometry(years, massChange, uncertainty, box) {
  const first = years[0];
  const last = years[years.length - 1];
  let low = 0;
  let high = 0;
  for (let i = 0; i < years.length; i += 1) {
    const sigma = Number.isFinite(uncertainty[i]) ? uncertainty[i] : 0;
    if (!Number.isFinite(massChange[i])) continue;
    low = Math.min(low, massChange[i] - sigma);
    high = Math.max(high, massChange[i] + sigma);
  }
  const span = high - low;
  if (high < 0.04 * span) high = 0;
  if (-low < 0.04 * span) low = 0;
  const yMax = niceCeiling(high);
  const yMin = low < 0 ? -niceCeiling(-low) : 0;
  const range = yMax - yMin || 1;
  const xOf = (year) => box.left + ((year - first) / Math.max(1e-9, last - first)) * (box.right - box.left);
  const yOf = (value) => {
    const clamped = Math.min(yMax, Math.max(yMin, value));
    return box.bottom - ((clamped - yMin) / range) * (box.bottom - box.top);
  };
  const point = (year, value) => `${xOf(year).toFixed(1)},${yOf(value).toFixed(1)}`;
  const finite = [];
  for (let i = 0; i < years.length; i += 1) {
    if (Number.isFinite(massChange[i])) finite.push(i);
  }
  const line = finite.map((i, n) => `${n ? "L" : "M"}${point(years[i], massChange[i])}`).join("");
  const sigmaAt = (i) => (Number.isFinite(uncertainty[i]) ? uncertainty[i] : 0);
  const upper = finite.map((i) => point(years[i], massChange[i] + sigmaAt(i)));
  const lower = finite.map((i) => point(years[i], massChange[i] - sigmaAt(i))).reverse();
  const band = finite.length ? `M${upper.join("L")}L${lower.join("L")}Z` : "";
  const yTicks = [...new Set([yMin, 0, yMax])]
    .filter((value) => value === 0 || Math.abs(yOf(value) - yOf(0)) >= MIN_TICK_GAP)
    .sort((a, b) => a - b);
  // The ends are labelled with their calendar year; the decades between, if clear of them.
  const xTicks = [{ at: first, label: Math.floor(first) }];
  for (let decade = Math.ceil(first / 10) * 10; decade < last; decade += 10) {
    if (decade - first >= 4 && last - decade >= 4) xTicks.push({ at: decade, label: decade });
  }
  xTicks.push({ at: last, label: Math.floor(last) });
  return { first, last, yMin, yMax, yTicks, xTicks, line, band, xOf, yOf };
}

/** Position (0-1) of a change on the scale's ramp; 0.5 is no change. */
export function surfaceChangeScaleT(value, scale) {
  return thicknessChangeScaleT(Number(value), scale.softness, scale.saturation);
}

/** RGB (0-1) at position `t` (0-1) of the ramp, for the legend bar. */
export function surfaceChangeRampColor(t) {
  return thicknessChangeRampColor(t);
}

/** RGB (0-1) of a change, faded towards the neutral middle when `faded`. */
export function surfaceChangeColor(value, scale, { faded = false } = {}) {
  const rgb = thicknessChangeRampColor(surfaceChangeScaleT(value, scale));
  if (!faded) return rgb;
  const neutral = thicknessChangeRampColor(0.5);
  return rgb.map((channel, i) => neutral[i] + FADED_COLOUR_SHARE * (channel - neutral[i]));
}

/** RGBA (0-255) lookup table of the ramp from its thinning end to its thickening end. */
export function surfaceChangeRampLut(size = 256) {
  const lut = new Uint8Array(size * 4);
  for (let i = 0; i < size; i += 1) {
    const rgb = thicknessChangeRampColor(i / (size - 1));
    lut[4 * i] = Math.round(rgb[0] * 255);
    lut[4 * i + 1] = Math.round(rgb[1] * 255);
    lut[4 * i + 2] = Math.round(rgb[2] * 255);
    lut[4 * i + 3] = 255;
  }
  return lut;
}

/** Calendar year and month (0-11) of a decimal year, for the epoch label. */
export function decimalYearParts(year) {
  const value = Number(year);
  const whole = Math.floor(value);
  const month = Math.min(11, Math.max(0, Math.floor((value - whole) * 12)));
  return { year: whole, month };
}

/** Statistics of the values the layer draws: count, minimum, maximum, and the share significant. */
export function summariseDrawn(values, sigma) {
  let count = 0;
  let significant = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    if (!Number.isFinite(value)) continue;
    count += 1;
    if (value < min) min = value;
    if (value > max) max = value;
    if (isSignificant(value, sigma?.[i])) significant += 1;
  }
  return count
    ? { count, min, max, significantShare: significant / count }
    : { count: 0, min: Number.NaN, max: Number.NaN, significantShare: Number.NaN };
}
