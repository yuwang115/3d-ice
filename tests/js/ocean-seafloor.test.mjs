import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  decodeSeafloorRecord,
  sampleSeafloor,
  sampleSeafloorOnGrid,
  SEAFLOOR_FADE_BELOW,
  SEAFLOOR_KIND,
  SEAFLOOR_STYLE,
  seafloorColor,
  seafloorLegendGradient,
  seafloorLegendTicks,
  seafloorRampColor,
  seafloorRampGlsl,
  seafloorRampPosition,
} from "../../static/tools/js/ocean-seafloor.js";

const DATA = new URL("../../static/tools/data/", import.meta.url);

function loadPackage(name) {
  const meta = JSON.parse(readFileSync(new URL(`${name}.meta.json`, DATA), "utf8"));
  const bytes = readFileSync(new URL(`${name}.bin`, DATA));
  return { meta, buffer: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
}

/**
 * A 3 x 3 grid of 10 km cells centred on (0, 0) whose domain is every cell but the north-east
 * corner. Temperature is the column index (0, 1, 2 degC), salinity 34.5, the post-2000 period
 * 0.1 degC warmer, and support 0.4 in the first row, 1 elsewhere.
 */
function synthetic({ kind = SEAFLOOR_KIND, count = 8 } = {}) {
  const domain = [1, 1, 0, 1, 1, 1, 1, 1, 1];
  const cells = domain.flatMap((value, cell) => (value ? [cell] : []));
  const ct = cells.map((cell) => Math.round(((cell % 3) * 1) / 0.001));
  const ctPost = cells.map((cell) => Math.round(((cell % 3) + 0.1) / 0.001));
  const sa = cells.map(() => Math.round((34.5 - 34) / 0.0001));
  const pressure = cells.map((cell) => 500 + cell);
  const support = cells.map((cell) => (cell < 3 ? 40 : 100));
  const int16 = (values) => new Int16Array(values);
  const blobs = [
    ["ct_full", int16(ct), { scale: 0.001, offset: 0 }],
    ["sa_full", int16(sa), { scale: 0.0001, offset: 34 }],
    ["ct_post2000", int16(ctPost), { scale: 0.001, offset: 0 }],
    ["sa_post2000", int16(sa), { scale: 0.0001, offset: 34 }],
    ["bottom_pressure", int16(pressure), { scale: 1, offset: 0 }],
    ["support", new Uint8Array(support), {}],
    ["domain_mask", new Uint8Array(domain), {}],
  ];
  let offset = 0;
  const fields = blobs.map(([name, array, quantization]) => {
    const field = {
      name,
      dtype: array instanceof Int16Array ? "int16" : "uint8",
      byte_offset: offset,
      byte_length: array.byteLength,
      ...(array instanceof Int16Array ? { ...quantization, fill_value: -32768 } : {}),
    };
    offset += array.byteLength;
    return field;
  });
  const bytes = new Uint8Array(offset);
  blobs.forEach(([, array], i) => bytes.set(new Uint8Array(array.buffer), fields[i].byte_offset));
  const meta = {
    geometry_type: "sparse_grid",
    field_kind: kind,
    grid: { nx: 3, ny: 3, x0_m: -10000, y0_m: 10000, dx_m: 10000, dy_m: -10000 },
    domain: { cell_count: count },
    periods: [
      { key: "full", label: "1972-2023", fields: { ct: "ct_full", sa: "sa_full" } },
      { key: "post2000", label: "2000-2023", fields: { ct: "ct_post2000", sa: "sa_post2000" } },
    ],
    support: { field: "support", fade_below_percent: 50 },
    fields,
  };
  return { meta, buffer: bytes.buffer };
}

test("decodeSeafloorRecord reads both periods, the support and the seafloor pressure", () => {
  const { meta, buffer } = synthetic();
  const record = decodeSeafloorRecord(meta, buffer);
  assert.equal(record.domainCount, 8);
  assert.equal(record.cellIndex[2], -1);
  assert.equal(record.cellIndex[3], 2);
  assert.deepEqual([...record.periods.full.temperature].map((v) => Math.round(v * 1000) / 1000), [0, 1, 0, 1, 2, 0, 1, 2]);
  assert.ok(Math.abs(record.periods.post2000.temperature[1] - 1.1) < 1e-6);
  assert.ok(Math.abs(record.periods.full.salinity[0] - 34.5) < 1e-6);
  assert.deepEqual([...record.support].map((v) => Math.round(v * 100) / 100), [0.4, 0.4, 1, 1, 1, 1, 1, 1]);
  assert.equal(record.bottomPressure[2], 503);
  assert.equal(record.fadeBelow, SEAFLOOR_FADE_BELOW);
});

test("decodeSeafloorRecord refuses other packages and inconsistent layouts", () => {
  const other = synthetic({ kind: "observed_sea_ice_concentration" });
  assert.throws(() => decodeSeafloorRecord(other.meta, other.buffer), /not an observed seafloor/);
  const wrongCount = synthetic({ count: 9 });
  assert.throws(() => decodeSeafloorRecord(wrongCount.meta, wrongCount.buffer), /domain size/);
  const noPeriod = synthetic();
  noPeriod.meta.periods = noPeriod.meta.periods.slice(0, 1);
  assert.throws(() => decodeSeafloorRecord(noPeriod.meta, noPeriod.buffer), /no post2000 period/);
  const badGrid = synthetic();
  badGrid.meta.grid = { ...badGrid.meta.grid, dx_m: 0 };
  assert.throws(() => decodeSeafloorRecord(badGrid.meta, badGrid.buffer), /origin and step/);
});

test("sampleSeafloor interpolates bilinearly and renormalises over cells with a value", () => {
  const { meta, buffer } = synthetic();
  const record = decodeSeafloorRecord(meta, buffer);
  // Half-way between the centre column (1 degC) and the east column (2 degC), on the middle row.
  const mid = sampleSeafloor(record, 5000, 0);
  assert.ok(Math.abs(mid.value - 1.5) < 1e-6);
  assert.equal(mid.support, 1);
  // Next to the missing north-east corner the remaining corners share the weight.
  const nearGap = sampleSeafloor(record, 5000, 5000);
  assert.ok(Math.abs(nearGap.value - (1 * 0.25 + 1 * 0.25 + 2 * 0.25) / 0.75) < 1e-6);
  assert.ok(Math.abs(nearGap.support - (0.4 * 0.25 + 1 * 0.25 + 1 * 0.25) / 0.75) < 1e-6);
  const post = sampleSeafloor(record, 0, 0, { period: "post2000" });
  assert.ok(Math.abs(post.value - 1.1) < 1e-6);
  const salt = sampleSeafloor(record, 0, 0, { variable: "salinity" });
  assert.ok(Math.abs(salt.value - 34.5) < 1e-5);
  assert.equal(sampleSeafloor(record, 10000, 10000), null);
  assert.equal(sampleSeafloor(record, 90000, 0), null);
  assert.throws(() => sampleSeafloor(record, 0, 0, { variable: "oxygen" }), /Unknown seafloor view/);
});

test("sampleSeafloorOnGrid fills only the wanted terrain nodes", () => {
  const { meta, buffer } = synthetic();
  const record = decodeSeafloorRecord(meta, buffer);
  const terrain = { nx: 2, ny: 1, x0_m: 0, y0_m: 0, dx_m: 5000, dy_m: -5000 };
  const out = sampleSeafloorOnGrid(record, terrain, new Uint8Array([1, 0]));
  assert.equal(out.valid[0], 1);
  assert.equal(out.valid[1], 0);
  assert.ok(Math.abs(out.values[0] - 1) < 1e-6);
  assert.ok(Number.isNaN(out.values[1]));
  assert.throws(() => sampleSeafloorOnGrid(record, terrain, new Uint8Array(3)), /does not cover/);
});

test("ramps clamp, match their stops and agree with the legend and the GLSL", () => {
  assert.equal(seafloorRampPosition("temperature", -5), 0);
  assert.equal(seafloorRampPosition("temperature", 0), 0.5);
  assert.equal(seafloorRampPosition("salinity", 35.5), 1);
  assert.deepEqual(seafloorRampColor("temperature", 0.5), [...SEAFLOOR_STYLE.temperature.stops[3][1]]);
  assert.deepEqual(seafloorColor("salinity", 34.2), [...SEAFLOOR_STYLE.salinity.stops[0][1]]);
  const quarter = seafloorRampColor("salinity", 0.125);
  const [a, b] = [SEAFLOOR_STYLE.salinity.stops[0][1], SEAFLOOR_STYLE.salinity.stops[1][1]];
  quarter.forEach((channel, c) => assert.ok(Math.abs(channel - (a[c] + b[c]) / 2) < 1e-9));
  assert.match(seafloorLegendGradient("temperature"), /^linear-gradient\(90deg, rgb\(27, 51, 120\) 0%/);
  assert.deepEqual(seafloorLegendTicks("temperature").map((tick) => tick.percent), [0, 25, 50, 75, 100]);
  const glsl = seafloorRampGlsl("temperature", "tempRamp");
  assert.match(glsl, /^vec3 tempRamp\(float t\) \{/);
  assert.equal((glsl.match(/mix\(/g) || []).length, SEAFLOOR_STYLE.temperature.stops.length - 1);
  assert.throws(() => seafloorRampGlsl("oxygen", "x"), /Unknown seafloor variable/);
  assert.throws(() => seafloorLegendTicks("oxygen"), /Unknown seafloor variable/);
});

test("the committed package decodes and holds warm Amundsen and cold Ross and Weddell shelves", () => {
  const { meta, buffer } = loadPackage("ocean_ice_seafloor_antarctica");
  const record = decodeSeafloorRecord(meta, buffer);
  assert.equal(record.domainCount, meta.domain.cell_count);
  // EPSG:3031 positions of the packer's check sites.
  const project = (lon, lat) => {
    const a = 6378137;
    const e = Math.sqrt(1 - (6356752.314245179 / a) ** 2);
    const t = (phi) => Math.tan(Math.PI / 4 - phi / 2) / ((1 - e * Math.sin(phi)) / (1 + e * Math.sin(phi))) ** (e / 2);
    const m = (phi) => Math.cos(phi) / Math.sqrt(1 - (e * Math.sin(phi)) ** 2);
    const phic = (71 * Math.PI) / 180;
    const rho = ((a * m(phic)) / t(phic)) * t((-lat * Math.PI) / 180);
    const lam = (lon * Math.PI) / 180;
    return [rho * Math.sin(lam), rho * Math.cos(lam)];
  };
  for (const site of meta.validation.sites.filter((entry) => entry.in_domain)) {
    const [x, y] = project(site.lon, site.lat);
    const sample = sampleSeafloor(record, x, y);
    assert.ok(sample, site.name);
    assert.ok(Math.abs(sample.value - site.packed_ct_full_degc) < 0.25, `${site.name}: ${sample.value}`);
  }
  const pineIsland = sampleSeafloor(record, ...project(-102.5, -74.9));
  const ross = sampleSeafloor(record, ...project(172, -75.5));
  assert.ok(pineIsland.value > 0.8);
  assert.ok(ross.value < -1.7);
  assert.ok(sampleSeafloor(record, ...project(172, -75.5), { variable: "salinity" }).value > 34.8);
});
