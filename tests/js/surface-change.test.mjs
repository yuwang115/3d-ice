import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  blendRecordInto,
  decimalYearParts,
  decodeSurfaceChangeRate,
  decodeSurfaceChangeRecord,
  extrapolatedRadiusM,
  FADE_SIGMA,
  FADED_COLOUR_SHARE,
  isSignificant,
  regionalChangeAt,
  REGION_KEYS,
  summariseDrawn,
  SURFACE_CHANGE_KIND,
  SURFACE_CHANGE_SCALES,
  surfaceChangeColor,
  surfaceChangeRampColor,
  surfaceChangeRampLut,
  surfaceChangeScaleT,
  yearAtEpoch,
} from "../../static/tools/js/surface-change.js";
import { thicknessChangeRampColor } from "../../static/tools/js/ice-projection.js";

const FILL = -32768;
const DATA = new URL("../../static/tools/data/", import.meta.url);

function loadPackage(name) {
  const meta = JSON.parse(readFileSync(new URL(`${name}.meta.json`, DATA), "utf8"));
  const bytes = readFileSync(new URL(`${name}.bin`, DATA));
  return { meta, buffer: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
}

function int16Field(name, offset, count, scale) {
  return { name, dtype: "int16", byte_offset: offset, byte_length: count * 2, scale, offset: 0, fill_value: FILL };
}

/** A 3 x 2 rate map; cell 4 is not drawn. */
function syntheticRate() {
  const cells = 6;
  const buffer = new ArrayBuffer(cells * 2 * 2 + cells);
  const view = new DataView(buffer);
  [100, -2500, 0, 40, FILL, 7].forEach((code, i) => view.setInt16(i * 2, code, true));
  [10, 100, 5, 30, FILL, 1].forEach((code, i) => view.setInt16(cells * 2 + i * 2, code, true));
  new Uint8Array(buffer, cells * 4).set([1, 2, 2, 3, 0, 18]);
  const meta = {
    grid: { nx: 3, ny: 2, x0_m: 0, y0_m: 0, dx_m: 10000, dy_m: -10000 },
    period: { start_year: 1995.41, end_year: 2026.07 },
    pole_hole: { radius_m: { ers_envisat: 925174.8, cryosat2: 217326.1 } },
    visualization: { linear_core_m_per_yr: 0.05, saturation_m_per_yr: 5 },
    fields: [
      int16Field("dhdt", 0, cells, 0.001),
      int16Field("dhdt_uncertainty", cells * 2, cells, 0.001),
      { name: "basin_id", dtype: "uint8", byte_offset: cells * 4, byte_length: cells },
    ],
  };
  return { meta, buffer };
}

/**
 * A 3 x 2 record whose domain is cells 1, 2 and 4, over three epochs; the third cell has no
 * value at the second epoch.
 */
function syntheticRecord({ years = [1995.4, 1995.8, 1996.2], kind = SURFACE_CHANGE_KIND } = {}) {
  const domain = [0, 1, 1, 0, 1, 0];
  const count = 3;
  const frames = [
    [0, 100, -200],
    [50, 300, FILL],
    [100, 500, -600],
  ];
  const sigma = [20, 1000, 50];
  const frameBytes = years.length * count * 2;
  const buffer = new ArrayBuffer(frameBytes + count * 2 + domain.length);
  const view = new DataView(buffer);
  frames.flat().forEach((code, i) => view.setInt16(i * 2, code, true));
  sigma.forEach((code, i) => view.setInt16(frameBytes + i * 2, code, true));
  new Uint8Array(buffer, frameBytes + count * 2).set(domain);
  const meta = {
    geometry_type: "sparse_grid_time_series",
    time_series_kind: kind,
    grid: { nx: 3, ny: 2, x0_m: 0, y0_m: 0, dx_m: 10000, dy_m: -10000 },
    domain: { cell_count: count },
    frames: { count: years.length, years, start_years: years.map((y) => y - 0.19), end_years: years.map((y) => y + 0.19) },
    series: {
      gt_per_mm_sle: 362.5,
      regions: Object.fromEntries(
        REGION_KEYS.map((key) => [key, { mass_change_gt: [0, -362.5, -725], uncertainty_gt: [1, 2, 3] }])
      ),
    },
    pole_hole: { radius_m: { ers_envisat: 925174.8, cryosat2: 217326.1 }, observed_from_year: 1995.6 },
    visualization: { linear_core_m: 1, saturation_m: 150 },
    fields: [
      int16Field("surface_change", 0, years.length * count, 0.01),
      int16Field("final_uncertainty", frameBytes, count, 0.01),
      { name: "domain_mask", dtype: "uint8", byte_offset: frameBytes + count * 2, byte_length: domain.length },
    ],
  };
  return { meta, buffer };
}

test("the rate map decodes rates, uncertainties and basins per grid cell", () => {
  const { meta, buffer } = syntheticRate();
  const rate = decodeSurfaceChangeRate(meta, buffer);
  assert.equal(rate.view, "rate");
  assert.deepEqual([rate.nx, rate.ny], [3, 2]);
  assert.ok(Math.abs(rate.values[1] - -2.5) < 1e-6);
  assert.ok(Number.isNaN(rate.values[4]));
  assert.ok(Math.abs(rate.sigma[3] - 0.03) < 1e-6);
  assert.deepEqual(Array.from(rate.basin), [1, 2, 2, 3, 0, 18]);
  assert.equal(rate.period.startYear, 1995.41);
  assert.deepEqual([rate.scale.softness, rate.scale.saturation], [0.05, 5]);
  // The rate is extrapolated throughout CryoSat-2's polar cap, which the view fades.
  assert.equal(rate.poleHole.lateRadiusM, 217326.1);
  assert.equal(rate.poleHole.earlyRadiusM, 925174.8);
});

test("a rate map without polar-cap radii fades no cap", () => {
  const { meta, buffer } = syntheticRate();
  delete meta.pole_hole;
  const rate = decodeSurfaceChangeRate(meta, buffer);
  assert.equal(rate.poleHole.lateRadiusM, 0);
  assert.ok(Number.isNaN(rate.poleHole.observedFromYear));
});

test("the rate map falls back to the default scale and refuses a grid it does not cover", () => {
  const { meta, buffer } = syntheticRate();
  delete meta.visualization;
  assert.deepEqual(decodeSurfaceChangeRate(meta, buffer).scale.ticks, SURFACE_CHANGE_SCALES.rate.ticks);
  assert.throws(() => decodeSurfaceChangeRate({ ...meta, grid: { ...meta.grid, nx: 4 } }, buffer), /cover the grid/);
  assert.throws(() => decodeSurfaceChangeRate({ ...meta, grid: { nx: 0, ny: 2 } }, buffer), /no valid grid/);
});

test("the record decodes its domain, epochs, final uncertainty and series", () => {
  const { meta, buffer } = syntheticRecord();
  const record = decodeSurfaceChangeRecord(meta, buffer);
  assert.equal(record.view, "cumulative");
  assert.deepEqual(Array.from(record.cellIndex), [1, 2, 4]);
  assert.equal(record.domainCount, 3);
  assert.deepEqual(Array.from(record.years), [1995.4, 1995.8, 1996.2]);
  assert.ok(Math.abs(record.finalSigma[1] - 10) < 1e-6);
  assert.deepEqual(Object.keys(record.regions), [...REGION_KEYS]);
  assert.equal(record.poleHole.observedFromYear, 1995.6);
  assert.equal(record.gtPerMm, 362.5);
});

test("the record refuses packages that are not observed records or do not match their metadata", () => {
  const good = syntheticRecord();
  assert.throws(() => decodeSurfaceChangeRecord({ ...good.meta, time_series_kind: undefined }, good.buffer), /not an observed/);
  assert.throws(
    () => decodeSurfaceChangeRecord({ ...good.meta, domain: { cell_count: 4 } }, good.buffer),
    /domain size/
  );
  const backwards = syntheticRecord({ years: [1996, 1995.5, 1997] });
  assert.throws(() => decodeSurfaceChangeRecord(backwards.meta, backwards.buffer), /strictly increasing/);
  const shortSeries = syntheticRecord();
  shortSeries.meta.series.regions.AIS.mass_change_gt = [0, 1];
  assert.throws(() => decodeSurfaceChangeRecord(shortSeries.meta, shortSeries.buffer), /one value per epoch/);
  const wrongFrames = syntheticRecord();
  wrongFrames.meta.frames.years = [1995.4, 1995.8];
  wrongFrames.meta.series.regions = {};
  assert.throws(() => decodeSurfaceChangeRecord(wrongFrames.meta, wrongFrames.buffer), /epochs and domain/);
});

test("blending interpolates between epochs and leaves cells without a value empty", () => {
  const { meta, buffer } = syntheticRecord();
  const record = decodeSurfaceChangeRecord(meta, buffer);
  const out = new Float32Array(record.domainCount);
  const exact = blendRecordInto(record, 1995.8, out);
  assert.equal(exact.weight, 0);
  assert.ok(Math.abs(out[0] - 0.5) < 1e-6 && Math.abs(out[1] - 3) < 1e-6);
  assert.ok(Number.isNaN(out[2]));
  blendRecordInto(record, 2000, out); // clamped to the last epoch
  assert.ok(Math.abs(out[2] - -6) < 1e-6);
  const halfway = blendRecordInto(record, 1996.0, out);
  assert.ok(Math.abs(halfway.weight - 0.5) < 1e-9);
  assert.ok(Math.abs(out[1] - 4) < 1e-6);
});

test("the slider position maps to the epoch times", () => {
  const record = decodeSurfaceChangeRecord(...Object.values(syntheticRecord()));
  assert.equal(yearAtEpoch(record, 0), 1995.4);
  assert.ok(Math.abs(yearAtEpoch(record, 1.5) - 1996.0) < 1e-9);
  assert.equal(yearAtEpoch(record, 9), 1996.2);
  assert.equal(yearAtEpoch(record, -3), 1995.4);
  assert.equal(yearAtEpoch(record, "x"), 1995.4);
});

test("the extrapolated polar cap shrinks once CryoSat-2 data arrive", () => {
  const record = decodeSurfaceChangeRecord(...Object.values(syntheticRecord()));
  assert.equal(extrapolatedRadiusM(record, 1995.5), 925174.8);
  assert.equal(extrapolatedRadiusM(record, 1995.6), 217326.1);
  assert.equal(extrapolatedRadiusM({ poleHole: { earlyRadiusM: 1, lateRadiusM: 2, observedFromYear: Number.NaN } }, 0), 2);
});

test("a change is significant from twice its uncertainty", () => {
  assert.equal(FADE_SIGMA, 2);
  assert.ok(isSignificant(0.4, 0.2));
  assert.ok(isSignificant(-0.4, 0.2));
  assert.ok(!isSignificant(0.39, 0.2));
  assert.ok(isSignificant(0.1, Number.NaN), "no uncertainty: nothing to fade against");
  assert.ok(!isSignificant(Number.NaN, 0.1));
  assert.ok(isSignificant(1, 0.4, 2.5));
});

test("the regional readout interpolates CPOM's series and converts it to sea level", () => {
  const record = decodeSurfaceChangeRecord(...Object.values(syntheticRecord()));
  const at = regionalChangeAt(record, 1996.0);
  assert.ok(Math.abs(at.AIS.massChangeGt - -543.75) < 1e-9);
  assert.ok(Math.abs(at.AIS.uncertaintyGt - 2.5) < 1e-9);
  assert.ok(Math.abs(at.WAIS.seaLevelMm - 1.5) < 1e-9);
});

test("colours sit on the explorer's thickness-change ramp and fade towards its middle", () => {
  const scale = SURFACE_CHANGE_SCALES.cumulative;
  assert.equal(surfaceChangeScaleT(0, scale), 0.5);
  assert.ok(surfaceChangeScaleT(-150, scale) === 0 && surfaceChangeScaleT(500, scale) === 1);
  // Linear inside the core, logarithmic beyond: 1 m is far from the middle, 150 m saturates.
  assert.ok(surfaceChangeScaleT(1, scale) > 0.57 && surfaceChangeScaleT(1, scale) < 0.59);
  assert.deepEqual(surfaceChangeColor(-150, scale), thicknessChangeRampColor(0));
  const full = surfaceChangeColor(30, scale);
  const faded = surfaceChangeColor(30, scale, { faded: true });
  const neutral = thicknessChangeRampColor(0.5);
  for (let i = 0; i < 3; i += 1) {
    assert.ok(Math.abs(faded[i] - (neutral[i] + FADED_COLOUR_SHARE * (full[i] - neutral[i]))) < 1e-12);
  }
});

test("the legend bar samples the same ramp", () => {
  assert.deepEqual(surfaceChangeRampColor(0.25), thicknessChangeRampColor(0.25));
});

test("the ramp lookup table runs from the thinning end to the thickening end", () => {
  const lut = surfaceChangeRampLut(5);
  assert.equal(lut.length, 20);
  const first = thicknessChangeRampColor(0).map((c) => Math.round(c * 255));
  const last = thicknessChangeRampColor(1).map((c) => Math.round(c * 255));
  assert.deepEqual(Array.from(lut.slice(0, 3)), first);
  assert.deepEqual(Array.from(lut.slice(16, 19)), last);
  assert.equal(lut[3], 255);
});

test("decimal years split into a calendar year and month", () => {
  assert.deepEqual(decimalYearParts(2010.9315), { year: 2010, month: 11 });
  assert.deepEqual(decimalYearParts(1995.41), { year: 1995, month: 4 });
  assert.deepEqual(decimalYearParts(2026.0), { year: 2026, month: 0 });
});

test("drawn values are summarised with the share that stands out from the uncertainty", () => {
  const summary = summariseDrawn(Float32Array.from([1, -3, Number.NaN, 0.1]), Float32Array.from([0.2, 1, 0.1, 0.1]));
  assert.equal(summary.count, 3);
  assert.equal(summary.min, -3);
  assert.equal(summary.max, 1);
  assert.ok(Math.abs(summary.significantShare - 2 / 3) < 1e-12);
  assert.ok(Number.isNaN(summariseDrawn(Float32Array.from([Number.NaN])).min));
});

test("the committed packages decode, and the record spans 1995 to 2026 with CPOM's totals", () => {
  const rate10 = loadPackage("cpom_antarctica_dhdt_480");
  const rate = decodeSurfaceChangeRate(rate10.meta, rate10.buffer);
  assert.equal(rate.values.length, 667 * 667);
  assert.ok(Math.abs(rate.poleHole.lateRadiusM - 217326.1) < 1, "the rate package carries the polar caps");
  const drawn = summariseDrawn(rate.values, rate.sigma);
  assert.equal(drawn.count, rate10.meta.coverage.drawn_cells);
  assert.ok(drawn.min < -5 && drawn.max > 1);

  const { meta, buffer } = loadPackage("cpom_antarctica_dh_480");
  const record = decodeSurfaceChangeRecord(meta, buffer);
  assert.ok(record.years[0] > 1995 && record.years[0] < 1996);
  assert.ok(record.years[record.years.length - 1] > 2026);
  const last = regionalChangeAt(record, record.years[record.years.length - 1]);
  assert.ok(last.AIS.massChangeGt < -1800 && last.AIS.massChangeGt > -2000, "CPOM's AIS total since 1995");
  assert.ok(last.WAIS.massChangeGt < last.AIS.massChangeGt, "West Antarctica loses more than the whole");
  assert.ok(last.EAIS.massChangeGt > 0, "East Antarctica gains");
  assert.ok(record.poleHole.observedFromYear > 2010 && record.poleHole.observedFromYear < 2011.5);
  const out = new Float32Array(record.domainCount);
  blendRecordInto(record, 2026.07, out);
  const summary = summariseDrawn(out, record.finalSigma);
  assert.ok(summary.min < -100, "Pine Island, Thwaites and Smith have thinned by more than 100 m");
  assert.ok(summary.significantShare > 0.4 && summary.significantShare < 0.8);
});
