import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  blendConcentrationInto,
  blendReferenceInto,
  decodeSeaIceRecord,
  EXTENT_THRESHOLD_PERCENT,
  extentChartGeometry,
  extentReadout,
  frameDate,
  frameIndexOf,
  locateFrame,
  monthSeries,
  nearestFrameWithData,
  NO_DATA,
  parseMonthLabel,
  rankFromNearestEnd,
  SEA_ICE_KIND,
  SEA_ICE_LEGEND_TICKS,
  SEA_ICE_STYLE,
  seaIceColor,
  seaIceRampColor,
  seaIceScaleT,
  sensorOfMonth,
  trendAt,
} from "../../static/tools/js/sea-ice.js";

const DATA = new URL("../../static/tools/data/", import.meta.url);
const BOX = { left: 36, right: 254, top: 16, bottom: 80 };

function loadPackage(name) {
  const meta = JSON.parse(readFileSync(new URL(`${name}.meta.json`, DATA), "utf8"));
  const bytes = readFileSync(new URL(`${name}.bin`, DATA));
  return { meta, buffer: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
}

function monthLabels(firstYear, firstMonth, count) {
  return Array.from({ length: count }, (_, i) => {
    const key = firstYear * 12 + firstMonth - 1 + i;
    return `${Math.floor(key / 12)}-${String((key % 12) + 1).padStart(2, "0")}`;
  });
}

function monthlyStatistics(referenceMean = 10) {
  return Array.from({ length: 12 }, (_, i) => ({
    month: i + 1,
    reference_mean_mkm2: referenceMean + i,
    slope_mkm2_per_year: -0.05,
    intercept_mkm2_at_year_0: 110 + i,
    percent_per_decade: -5,
    trend_first_year: 1979,
    trend_last_year: 1981,
  }));
}

/**
 * A 3 x 2 grid whose domain is cells 1, 2 and 4, over the 26 months from November 1978 to
 * December 1980, of which December 1979 has no data.
 */
function syntheticRecord({ kind = SEA_ICE_KIND, empty = ["1979-12"], labels = monthLabels(1978, 11, 26), extent = null } = {}) {
  const domain = [0, 1, 1, 0, 1, 0];
  const count = 3;
  const frameCount = labels.length;
  const frames = new Uint8Array(frameCount * count);
  labels.forEach((label, k) => {
    const values = empty.includes(label) ? [NO_DATA, NO_DATA, NO_DATA] : [k % 101, 100 - (k % 101), 50];
    frames.set(values, k * count);
  });
  const reference = new Uint8Array(12 * count);
  for (let month = 0; month < 12; month += 1) reference.set([month * 8, 100, 0], month * count);
  const buffer = new ArrayBuffer(frames.length + reference.length + domain.length);
  new Uint8Array(buffer, 0, frames.length).set(frames);
  new Uint8Array(buffer, frames.length, reference.length).set(reference);
  new Uint8Array(buffer, frames.length + reference.length).set(domain);
  const extents = extent || labels.map((label, k) => (empty.includes(label) ? null : 5 + (k % 12)));
  const meta = {
    geometry_type: "sparse_grid_time_series",
    time_series_kind: kind,
    region: "greenland",
    grid: { nx: 3, ny: 2, x0_m: 12500, y0_m: -12500, dx_m: 25000, dy_m: -25000 },
    domain: { cell_count: count },
    frames: {
      count: frameCount,
      months: labels,
      years: labels.map((label) => {
        const { year, month } = parseMonthLabel(label);
        return year + (month - 0.5) / 12;
      }),
      empty_months: empty,
      sensors: [
        { sensor: "n07", name: "SMMR (Nimbus-7)", first: labels[0], last: "1979-12" },
        { sensor: "F08", name: "SSM/I (DMSP F8)", first: "1980-01", last: labels[frameCount - 1] },
      ],
    },
    series: {
      hemisphere: "north",
      threshold_percent: 15,
      reference_period: [1981, 2010],
      extent_mkm2: extents,
      area_mkm2: extents.map((value) => (value === null ? null : value * 0.8)),
      monthly: monthlyStatistics(),
    },
    fields: [
      { name: "concentration", dtype: "uint8", byte_offset: 0, byte_length: frames.length, fill_value: NO_DATA },
      { name: "reference_ice_frequency", dtype: "uint8", byte_offset: frames.length, byte_length: reference.length },
      { name: "domain_mask", dtype: "uint8", byte_offset: frames.length + reference.length, byte_length: domain.length },
    ],
  };
  return { meta, buffer };
}

test("a record decodes its domain, its months and the hemisphere's series", () => {
  const { meta, buffer } = syntheticRecord();
  const record = decodeSeaIceRecord(meta, buffer);
  assert.equal(record.region, "greenland");
  assert.equal(record.hemisphere, "north");
  assert.deepEqual([record.nx, record.ny, record.domainCount], [3, 2, 3]);
  assert.deepEqual(Array.from(record.cellIndex), [1, 2, 4]);
  assert.equal(record.frameCount, 26);
  assert.deepEqual([record.firstYear, record.lastYear], [1978, 1980]);
  assert.deepEqual([record.yearOf[0], record.monthOf[0], record.yearOf[25], record.monthOf[25]], [1978, 11, 1980, 12]);
  assert.equal(record.hasData[13], 0, "December 1979 has no data");
  assert.equal(record.hasData[12], 1);
  assert.ok(Number.isNaN(record.extent[13]));
  assert.equal(record.extent[0], 5);
  assert.ok(Math.abs(record.area[0] - 4) < 1e-9);
  assert.equal(record.monthly[8].referenceMean, 18);
  assert.equal(record.monthly[0].firstYear, 1979);
  assert.equal(record.thresholdPercent, 15);
  assert.deepEqual(record.referencePeriod, [1981, 2010]);
});

test("a southern record defaults its hemisphere, threshold and reference period", () => {
  const { meta, buffer } = syntheticRecord();
  const southern = {
    ...meta,
    series: { ...meta.series, hemisphere: "south", threshold_percent: undefined, reference_period: undefined },
  };
  const record = decodeSeaIceRecord(southern, buffer);
  assert.equal(record.hemisphere, "south");
  assert.equal(record.thresholdPercent, EXTENT_THRESHOLD_PERCENT);
  assert.deepEqual(record.referencePeriod, [1981, 2010]);
});

test("a package that does not follow its layout fails loudly", () => {
  const good = syntheticRecord();
  assert.throws(() => decodeSeaIceRecord({ ...good.meta, time_series_kind: "observed_surface_change" }, good.buffer), /not an observed sea-ice/);
  assert.throws(() => decodeSeaIceRecord({ ...good.meta, grid: { nx: 0, ny: 2 } }, good.buffer), /no valid grid/);
  assert.throws(() => decodeSeaIceRecord({ ...good.meta, grid: { ...good.meta.grid, nx: 2 } }, good.buffer), /does not cover the grid/);
  assert.throws(() => decodeSeaIceRecord({ ...good.meta, domain: { cell_count: 4 } }, good.buffer), /domain size/);
  const shortMonths = { ...good.meta, frames: { ...good.meta.frames, count: 25 } };
  assert.throws(() => decodeSeaIceRecord(shortMonths, good.buffer), /consecutive months/);
  const skipped = good.meta.frames.months.slice();
  skipped[3] = "1979-03";
  assert.throws(() => decodeSeaIceRecord({ ...good.meta, frames: { ...good.meta.frames, months: skipped } }, good.buffer), /consecutive months/);
  const bad = good.meta.frames.months.slice();
  bad[0] = "1978-13";
  assert.throws(() => decodeSeaIceRecord({ ...good.meta, frames: { ...good.meta.frames, months: bad } }, good.buffer), /not YYYY-MM/);
  const series = { ...good.meta.series, extent_mkm2: [1, 2] };
  assert.throws(() => decodeSeaIceRecord({ ...good.meta, series }, good.buffer), /one value per month/);
  const monthly = { ...good.meta.series, monthly: monthlyStatistics().slice(1) };
  assert.throws(() => decodeSeaIceRecord({ ...good.meta, series: monthly }, good.buffer), /every calendar month/);
  const truncated = good.buffer.slice(0, good.buffer.byteLength - 1);
  assert.throws(() => decodeSeaIceRecord(good.meta, truncated));
});

test("the concentration and reference fields must match the months and the domain", () => {
  const good = syntheticRecord();
  const fields = good.meta.fields.map((field) => ({ ...field }));
  fields[0].byte_length -= 3;
  fields[1].byte_offset -= 3;
  assert.throws(() => decodeSeaIceRecord({ ...good.meta, fields }, good.buffer), /concentration does not match/);
  const shortReference = good.meta.fields.map((field) => ({ ...field }));
  shortReference[1].byte_length -= 3;
  assert.throws(() => decodeSeaIceRecord({ ...good.meta, fields: shortReference }, good.buffer), /twelve months/);
});

test("a month listed as empty must hold no data, and a record needs a month that does", () => {
  const { meta, buffer } = syntheticRecord();
  assert.throws(
    () => decodeSeaIceRecord({ ...meta, frames: { ...meta.frames, empty_months: ["1979-11"] } }, buffer),
    /listed as empty but holds data/
  );
  const labels = monthLabels(1978, 11, 2);
  const allEmpty = syntheticRecord({ labels, empty: labels });
  assert.throws(() => decodeSeaIceRecord(allEmpty.meta, allEmpty.buffer), /no month with data/);
});

test("months map to frames and back, and the record's ends bound them", () => {
  const record = decodeSeaIceRecord(syntheticRecord().meta, syntheticRecord().buffer);
  assert.equal(frameIndexOf(record, 1978, 11), 0);
  assert.equal(frameIndexOf(record, 1979, 9), 10);
  assert.equal(frameIndexOf(record, 1980, 12), 25);
  assert.equal(frameIndexOf(record, 1978, 10), -1);
  assert.equal(frameIndexOf(record, 1981, 1), -1);
  assert.equal(frameIndexOf(record, 1979.5, 1), -1);
  assert.equal(frameIndexOf(record, 1979, 13), -1);
  assert.deepEqual(frameDate(record, 10), { year: 1979, month: 9 });
  assert.deepEqual(frameDate(record, 10.4), { year: 1979, month: 9 });
  assert.deepEqual(frameDate(record, -3), { year: 1978, month: 11 });
  assert.deepEqual(frameDate(record, 99), { year: 1980, month: 12 });
  assert.deepEqual(parseMonthLabel("2026-09"), { year: 2026, month: 9 });
  assert.throws(() => parseMonthLabel("2026-9"), /not YYYY-MM/);
});

test("the nearest month with data skips the empty ones in either direction", () => {
  const record = decodeSeaIceRecord(syntheticRecord().meta, syntheticRecord().buffer);
  assert.equal(nearestFrameWithData(record, 13, -1), 12);
  assert.equal(nearestFrameWithData(record, 13, +1), 14);
  assert.equal(nearestFrameWithData(record, 12, +1), 12);
  assert.equal(nearestFrameWithData(record, 30, +1), -1);
});

test("a position between months blends the months with data on either side", () => {
  const record = decodeSeaIceRecord(syntheticRecord().meta, syntheticRecord().buffer);
  assert.deepEqual(locateFrame(record, 4), { lower: 4, upper: 4, weight: 0, empty: false });
  assert.deepEqual(locateFrame(record, 13), { lower: 13, upper: 13, weight: 0, empty: true });
  const across = locateFrame(record, 12.5);
  assert.deepEqual([across.lower, across.upper, across.empty], [12, 14, false]);
  assert.ok(Math.abs(across.weight - 0.25) < 1e-12);
  assert.deepEqual(locateFrame(record, -5), { lower: 0, upper: 0, weight: 0, empty: false });
  assert.deepEqual(locateFrame(record, 1e9), { lower: 25, upper: 25, weight: 0, empty: false });

  const out = new Float32Array(3);
  const bracket = blendConcentrationInto(record, 4.25, out);
  assert.deepEqual([bracket.lower, bracket.upper], [4, 5]);
  // Frame k holds (k, 100 - k, 50) on the three domain cells.
  assert.ok(Math.abs(out[0] - 4.25) < 1e-5);
  assert.ok(Math.abs(out[1] - 95.75) < 1e-5);
  assert.equal(out[2], 50);
  blendConcentrationInto(record, 7, out);
  assert.deepEqual(Array.from(out), [7, 93, 50]);
  blendConcentrationInto(record, 13, out);
  assert.deepEqual(Array.from(out), [0, 0, 0], "an empty month draws no ice");
});

test("playback bridges a gap even on a whole position inside it", () => {
  const record = decodeSeaIceRecord(syntheticRecord().meta, syntheticRecord().buffer);
  // Frame 13 (December 1979) is empty: a visitor's pick says so, playback blends across it.
  assert.equal(locateFrame(record, 13).empty, true);
  const bridged = locateFrame(record, 13, { bridgeGaps: true });
  assert.deepEqual([bridged.lower, bridged.upper, bridged.empty], [12, 14, false]);
  assert.ok(Math.abs(bridged.weight - 0.5) < 1e-12);
  assert.deepEqual(locateFrame(record, 12, { bridgeGaps: true }), { lower: 12, upper: 12, weight: 0, empty: false });
  const out = new Float32Array(3);
  blendConcentrationInto(record, 13, out, { bridgeGaps: true });
  assert.ok(Math.abs(out[0] - 13) < 1e-5, "halfway between frames 12 and 14");
});

test("a record whose ends are empty blends towards the only side with data", () => {
  const labels = monthLabels(1978, 11, 4);
  const { meta, buffer } = syntheticRecord({ labels, empty: ["1978-11", "1979-02"] });
  const record = decodeSeaIceRecord(meta, buffer);
  assert.deepEqual(locateFrame(record, 0.5), { lower: 1, upper: 1, weight: 0, empty: false });
  assert.deepEqual(locateFrame(record, 2.5), { lower: 2, upper: 2, weight: 0, empty: false });
});

test("the median edge blends the calendar months of a bracket", () => {
  const record = decodeSeaIceRecord(syntheticRecord().meta, syntheticRecord().buffer);
  const out = new Float32Array(3);
  // Frames 10 and 11 are September and October 1979: reference months 8 and 9 (zero-based).
  blendReferenceInto(record, { lower: 10, upper: 11, weight: 0.5 }, out);
  assert.deepEqual(Array.from(out), [68, 100, 0]);
  blendReferenceInto(record, { lower: 10, upper: 10, weight: 0 }, out);
  assert.deepEqual(Array.from(out), [64, 100, 0]);
  blendReferenceInto(record, { lower: -1, upper: -1, weight: 0 }, out);
  assert.deepEqual(Array.from(out), [80, 100, 0], "a bracket off the record falls back to its first frame");
});

test("a calendar month's series has one value per year, missing where there is none", () => {
  const record = decodeSeaIceRecord(syntheticRecord().meta, syntheticRecord().buffer);
  const december = monthSeries(record, 12);
  assert.deepEqual(Array.from(december.years), [1978, 1979, 1980]);
  assert.equal(december.extent[0], 6);
  assert.ok(Number.isNaN(december.extent[1]), "December 1979 is empty");
  const march = monthSeries(record, 3);
  assert.ok(Number.isNaN(march.extent[0]), "the record starts in November 1978");
  assert.equal(march.extent[1], 9);
});

test("the readout gives the extent, its departure from 1981-2010 and its rank", () => {
  const record = decodeSeaIceRecord(syntheticRecord().meta, syntheticRecord().buffer);
  const readout = extentReadout(record, frameIndexOf(record, 1980, 12));
  assert.equal(readout.year, 1980);
  assert.equal(readout.month, 12);
  assert.equal(readout.extent, 6);
  assert.ok(Math.abs(readout.area - 4.8) < 1e-9);
  assert.equal(readout.referenceMean, 21);
  assert.equal(readout.anomaly, -15);
  assert.ok(Math.abs(readout.anomalyPercent - (-15 / 21) * 100) < 1e-9);
  // Two Decembers have data (1978 and 1980), both 6 million km2: neither ranks below the other.
  assert.deepEqual([readout.rank, readout.rankCount], [1, 2]);
  assert.equal(readout.percentPerDecade, -5);
  assert.equal(extentReadout(record, frameIndexOf(record, 1979, 12)), null);
  assert.equal(extentReadout(record, -1), null);
  assert.equal(extentReadout(record, 26), null);
});

test("a rank reads from the nearer end of the years", () => {
  assert.deepEqual(rankFromNearestEnd(1, 48), { end: "lowest", place: 1, count: 48 });
  assert.deepEqual(rankFromNearestEnd(3, 48), { end: "lowest", place: 3, count: 48 });
  assert.deepEqual(rankFromNearestEnd(24, 48), { end: "lowest", place: 24, count: 48 });
  assert.deepEqual(rankFromNearestEnd(25, 48), { end: "highest", place: 24, count: 48 });
  assert.deepEqual(rankFromNearestEnd(48, 48), { end: "highest", place: 1, count: 48 });
  assert.equal(rankFromNearestEnd(1, 0), null);
  assert.equal(rankFromNearestEnd(0, 10), null);
});

test("the extent chart spans the record's years and breaks its line at a missing year", () => {
  const series = { years: Float64Array.of(1978, 1979, 1980, 1981), extent: Float64Array.of(Number.NaN, 7.2, Number.NaN, 6.1) };
  const statistics = { intercept: 140, slope: -0.0675, firstYear: 1979, lastYear: 1981 };
  const geometry = extentChartGeometry(series, statistics, BOX, { firstYear: 1978, lastYear: 1981 });
  assert.equal(geometry.xOf(1978), BOX.left);
  assert.equal(geometry.xOf(1981), BOX.right);
  // Two points with a gap between them: two separate moves, no line drawn across 1980.
  assert.equal((geometry.line.match(/M/g) || []).length, 2);
  assert.ok(!geometry.line.includes("L"));
  assert.ok(geometry.trend.startsWith("M") && geometry.trend.includes("L"));
  assert.ok(geometry.yMin <= 6.1 && geometry.yMax >= 7.2);
  assert.ok(geometry.yTicks.length >= 2 && geometry.yTicks.length <= 6);
  for (let i = 1; i < geometry.yTicks.length; i += 1) {
    assert.ok(Math.abs(geometry.yTicks[i] - geometry.yTicks[i - 1] - geometry.step) < 1e-9);
  }
  assert.equal(geometry.yOf(geometry.yMin), BOX.bottom);
  assert.equal(geometry.yOf(geometry.yMax), BOX.top);
  assert.equal(geometry.yOf(geometry.yMax + 100), BOX.top, "values clamp to the box");
  assert.deepEqual(geometry.xTicks.map((tick) => tick.label), [1978, 1981]);
});

test("consecutive years are joined, decades between the ends are labelled, and a flat series gets a range", () => {
  const years = Float64Array.from({ length: 48 }, (_, i) => 1979 + i);
  const extent = Float64Array.from(years, (year) => 8 - 0.08 * (year - 1979));
  const geometry = extentChartGeometry({ years, extent }, null, BOX);
  assert.equal((geometry.line.match(/M/g) || []).length, 1);
  assert.equal((geometry.line.match(/L/g) || []).length, 47);
  assert.equal(geometry.trend, "");
  assert.deepEqual(geometry.xTicks.map((tick) => tick.label), [1979, 1990, 2000, 2010, 2020, 2026]);
  const flat = extentChartGeometry({ years: Float64Array.of(2000, 2001), extent: Float64Array.of(3, 3) }, null, BOX);
  assert.ok(flat.yMax > flat.yMin);
  const none = extentChartGeometry({ years: Float64Array.of(2000, 2001), extent: Float64Array.of(Number.NaN, Number.NaN) }, null, BOX);
  assert.equal(none.line, "");
});

test("the trend line takes its value from the fit", () => {
  assert.ok(Math.abs(trendAt({ intercept: 171.5, slope: -0.0825 }, 2000) - (171.5 - 165)) < 1e-9);
});

test("the ice colour starts at the threshold, bluish and see-through, and whitens to full cover", () => {
  assert.deepEqual(seaIceColor(10), { rgb: [0, 0, 0], alpha: 0 });
  assert.deepEqual(seaIceColor(Number.NaN), { rgb: [0, 0, 0], alpha: 0 });
  const low = seaIceColor(15);
  const high = seaIceColor(100);
  assert.deepEqual(low.rgb, [...SEA_ICE_STYLE.lowColor]);
  assert.equal(low.alpha, SEA_ICE_STYLE.lowAlpha);
  for (let i = 0; i < 3; i += 1) assert.ok(Math.abs(high.rgb[i] - SEA_ICE_STYLE.highColor[i]) < 1e-12);
  assert.equal(high.alpha, SEA_ICE_STYLE.highAlpha);
  assert.ok(seaIceColor(60).alpha > low.alpha && seaIceColor(60).alpha < high.alpha);
  assert.equal(seaIceScaleT(15), 0);
  assert.equal(seaIceScaleT(100), 1);
  assert.equal(seaIceScaleT(0), 0);
  assert.ok(Math.abs(seaIceScaleT(57.5) - 0.5) < 1e-12);
});

test("the legend ramp is composited over the sea and brightens from its threshold end", () => {
  const sea = [0, 0, 0];
  const start = seaIceRampColor(0, sea);
  const end = seaIceRampColor(1, sea);
  assert.ok(end.every((channel, i) => channel > start[i]));
  assert.deepEqual(seaIceRampColor(-1, sea), start);
  assert.deepEqual(seaIceRampColor(2, sea), end);
  assert.ok(seaIceRampColor(0.5).every((channel) => channel >= 0 && channel <= 1));
});

test("the legend ticks sit on the ramp in order, from the threshold to full cover", () => {
  assert.equal(SEA_ICE_LEGEND_TICKS[0], EXTENT_THRESHOLD_PERCENT);
  assert.equal(SEA_ICE_LEGEND_TICKS[SEA_ICE_LEGEND_TICKS.length - 1], 100);
  const positions = SEA_ICE_LEGEND_TICKS.map(seaIceScaleT);
  for (let i = 1; i < positions.length; i += 1) {
    // At least a fifth of the 302 px legend apart, so the labels stay on one row.
    assert.ok(positions[i] - positions[i - 1] >= 0.2, `ticks ${SEA_ICE_LEGEND_TICKS[i - 1]} and ${SEA_ICE_LEGEND_TICKS[i]} crowd`);
  }
});

test("the sensor of a month comes from the package's runs", () => {
  const record = decodeSeaIceRecord(syntheticRecord().meta, syntheticRecord().buffer);
  assert.deepEqual(sensorOfMonth(record, 0), { sensor: "n07", name: "SMMR (Nimbus-7)" });
  assert.deepEqual(sensorOfMonth(record, 14), { sensor: "F08", name: "SSM/I (DMSP F8)" });
  assert.equal(sensorOfMonth(record, -1), null);
  record.sensors = [{ sensor: "x", first: "garbled", last: "2000-01" }];
  assert.equal(sensorOfMonth(record, 0), null);
});

// ---------------------------------------------------------------- committed packages

for (const [name, region, hemisphere] of [
  ["nsidc_sea_ice_antarctica", "antarctica", "south"],
  ["nsidc_sea_ice_greenland", "greenland", "north"],
]) {
  test(`${name} decodes to the full record of its hemisphere`, () => {
    const record = decodeSeaIceRecord(...Object.values(loadPackage(name)));
    assert.equal(record.region, region);
    assert.equal(record.hemisphere, hemisphere);
    assert.equal(frameIndexOf(record, 1978, 11), 0, "the record starts in November 1978");
    assert.ok(record.lastYear >= 2026);
    // The SSM/I gap of December 1987 and January 1988 is empty in both hemispheres.
    assert.equal(record.hasData[frameIndexOf(record, 1987, 12)], 0);
    assert.equal(record.hasData[frameIndexOf(record, 1988, 1)], 0);
    assert.equal(record.hasData[frameIndexOf(record, 1987, 11)], 1);
    for (let index = 0; index < record.frameCount; index += 1) {
      assert.equal(Number.isFinite(record.extent[index]), Boolean(record.hasData[index]), `month ${index}`);
    }
    for (const statistics of record.monthly) {
      assert.ok(statistics.referenceMean > 0);
      assert.equal(statistics.firstYear, 1979);
    }
  });
}

test("the Arctic's lowest September is 2012's, and its September trend is about -12 % per decade", () => {
  const record = decodeSeaIceRecord(...Object.values(loadPackage("nsidc_sea_ice_greenland")));
  const readout = extentReadout(record, frameIndexOf(record, 2012, 9));
  assert.equal(readout.rank, 1);
  assert.ok(readout.extent > 3.4 && readout.extent < 4.0, `September 2012: ${readout.extent}`);
  assert.ok(readout.percentPerDecade < -10 && readout.percentPerDecade > -14);
  // July 1984 is empty in the Northern Hemisphere file of this version.
  assert.equal(record.hasData[frameIndexOf(record, 1984, 7)], 0);
});

test("the Antarctic's lowest February is 2023's, and its September maximum peaked in 2014", () => {
  const record = decodeSeaIceRecord(...Object.values(loadPackage("nsidc_sea_ice_antarctica")));
  assert.equal(extentReadout(record, frameIndexOf(record, 2023, 2)).rank, 1);
  const september = extentReadout(record, frameIndexOf(record, 2014, 9));
  assert.equal(september.rank, september.rankCount, "September 2014 is the highest September");
});
