import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

import {
  buildProfileIndex,
  countByPlatform,
  countsChartGeometry,
  defaultProfileYear,
  curveChartGeometry,
  decodeProfileCurves,
  decodeProfilePoints,
  nearestProfile,
  PROFILE_PLATFORMS,
  profileCurve,
  profileDate,
  yearRange,
} from "../../static/tools/js/ocean-profiles.js";

const DATA = new URL("../../static/tools/data/", import.meta.url);

function pack(blobs) {
  let offset = 0;
  const fields = blobs.map(([name, array, extra]) => {
    const dtype = array instanceof Int16Array ? "int16" : array instanceof Uint16Array ? "uint16" : "uint8";
    const field = { name, dtype, byte_offset: offset, byte_length: array.byteLength, ...(extra || {}) };
    offset += array.byteLength;
    return field;
  });
  const bytes = new Uint8Array(offset);
  blobs.forEach(([, array], i) => bytes.set(new Uint8Array(array.buffer), fields[i].byte_offset));
  return { fields, buffer: bytes.buffer };
}

/**
 * Four profiles: two in 1970 (a ship CTD at the origin, 1000 m deep; an Argo float 30 km east),
 * none in 1971, a seal in 1972 at (0, 50 km) to 95 m, and a ship CTD in 1972.
 */
function synthetic({ days = [0, 100, 760, 800] } = {}) {
  const xy = { scale: 250, offset: 0, fill_value: -32768 };
  const { fields, buffer } = pack([
    ["x", new Int16Array([0, 120, 0, 400]), xy],
    ["y", new Int16Array([0, 0, 200, 400]), xy],
    ["bottom_ct", new Int16Array([500, 1000, -1800, -32768]), { scale: 0.001, offset: 0, fill_value: -32768 }],
    ["bottom_sa", new Int16Array([1800, 1600, 1000, -32768]), { scale: 0.0005, offset: 34, fill_value: -32768 }],
    ["day", new Uint16Array(days)],
    ["max_depth", new Uint16Array([1000, 2000, 95, 300])],
    ["platform", new Uint8Array([1, 2, 3, 1])],
  ]);
  const meta = {
    geometry_type: "points",
    point_kind: "ocean_profiles",
    points: { count: 4 },
    years: { first: 1970, last: 1972, starts: [0, 2, 2, 4] },
    series: { counts_by_platform_and_year: { 1: [1, 0, 1], 2: [1, 0, 0], 3: [0, 0, 1] } },
    fields,
  };
  return { meta, buffer };
}

function syntheticCurves(points) {
  const depths = [10, 100, 500, 1000];
  // 1000 m: 10, 100, 500, bottom; 2000 m: 10, 100, 500, 1000, bottom; 95 m: 10, bottom; 300 m: 10, 100, bottom.
  const lengths = [4, 5, 2, 3];
  const total = lengths.reduce((a, b) => a + b, 0);
  const ct = new Int16Array(total).map((_, k) => k * 100);
  const sa = new Int16Array(total).map((_, k) => 1000 + k);
  const { fields, buffer } = pack([
    ["ct", ct, { scale: 0.001, offset: 0, fill_value: -32768 }],
    ["sa", sa, { scale: 0.0005, offset: 34, fill_value: -32768 }],
  ]);
  return {
    meta: { geometry_type: "points", point_kind: "ocean_profile_curves", points: { count: points.count }, curves: { standard_depths_m: depths }, fields },
    buffer,
  };
}

test("decodeProfilePoints reads every field and indexes the years", () => {
  const { meta, buffer } = synthetic();
  const points = decodeProfilePoints(meta, buffer);
  assert.equal(points.count, 4);
  assert.deepEqual([...points.x], [0, 30000, 0, 100000]);
  assert.ok(Math.abs(points.bottomTemperature[0] - 0.5) < 1e-6);
  assert.ok(Number.isNaN(points.bottomTemperature[3]));
  assert.ok(Math.abs(points.bottomSalinity[0] - 34.9) < 1e-5); // float32
  assert.deepEqual(yearRange(points, 1970), { start: 0, count: 2 });
  assert.deepEqual(yearRange(points, 1971), { start: 2, count: 0 });
  assert.deepEqual(yearRange(points, 1972), { start: 2, count: 2 });
  assert.deepEqual(yearRange(points, 1999), { start: 0, count: 0 });
  assert.deepEqual(yearRange(points), { start: 0, count: 4 });
  assert.equal(profileDate(points, 2), "1972-01-31");
  assert.deepEqual(countByPlatform(points, yearRange(points, 1972)), { total: 2, ship: 1, argo: 0, seal: 1 });
});

test("decodeProfilePoints refuses other packages, unsorted dates and a broken year index", () => {
  const other = synthetic();
  other.meta.point_kind = "ocean_profile_curves";
  assert.throws(() => decodeProfilePoints(other.meta, other.buffer), /not an ocean-profile record/);
  const unsorted = synthetic({ days: [0, 100, 50, 800] });
  assert.throws(() => decodeProfilePoints(unsorted.meta, unsorted.buffer), /not sorted by date/);
  const broken = synthetic();
  broken.meta.years.starts = [0, 2, 4];
  assert.throws(() => decodeProfilePoints(broken.meta, broken.buffer), /do not index the points/);
  const decreasing = synthetic();
  decreasing.meta.years.starts = [0, 3, 2, 4];
  assert.throws(() => decodeProfilePoints(decreasing.meta, decreasing.buffer), /do not index the points/);
  const empty = synthetic();
  empty.meta.points.count = 0;
  assert.throws(() => decodeProfilePoints(empty.meta, empty.buffer), /has no points/);
});

test("nearestProfile searches the grid within the drawn range", () => {
  const { meta, buffer } = synthetic();
  const points = decodeProfilePoints(meta, buffer);
  const index = buildProfileIndex(points);
  assert.equal(nearestProfile(points, index, 26000, 1000, 10000), 1);
  assert.equal(nearestProfile(points, index, 26000, 1000, 10000, yearRange(points, 1972)), -1);
  assert.equal(nearestProfile(points, index, 1000, 48000, 5000, yearRange(points, 1972)), 2);
  assert.equal(nearestProfile(points, index, -900000, 0, 5000), -1);
});

test("decodeProfileCurves derives each curve's length from its depth", () => {
  const { meta, buffer } = synthetic();
  const points = decodeProfilePoints(meta, buffer);
  const curvesPackage = syntheticCurves(points);
  const curves = decodeProfileCurves(curvesPackage.meta, curvesPackage.buffer, points);
  assert.deepEqual([...curves.offsets], [0, 4, 9, 11, 14]);
  const deep = profileCurve(curves, points, 1);
  assert.deepEqual(deep.map((p) => p.depth), [10, 100, 500, 1000, 2000]);
  assert.ok(Math.abs(deep[4].temperature - 0.8) < 1e-6);
  const seal = profileCurve(curves, points, 2);
  assert.deepEqual(seal.map((p) => p.depth), [10, 95]);
  assert.ok(Math.abs(seal[0].salinity - (34 + 1009 * 0.0005)) < 1e-5);

  const mismatched = syntheticCurves({ count: 5 });
  assert.throws(() => decodeProfileCurves(mismatched.meta, mismatched.buffer, points), /do not match the points/);
  const short = syntheticCurves(points);
  short.meta.curves.standard_depths_m = [10, 100, 500, 1000, 1500];
  assert.throws(() => decodeProfileCurves(short.meta, short.buffer, points), /values their points call for/);
});

test("chart geometry stacks the platforms and places the curve against depth", () => {
  const { meta, buffer } = synthetic();
  const points = decodeProfilePoints(meta, buffer);
  const box = { left: 30, right: 250, top: 10, bottom: 80 };
  const chart = countsChartGeometry(points, box, 1972);
  assert.equal(chart.bars.length, 3);
  assert.equal(chart.max, 2);
  assert.deepEqual(chart.bars[0].segments.map((s) => s.key), ["ship", "argo"]);
  assert.equal(chart.bars[1].segments.length, 0);
  assert.ok(Math.abs(chart.bars[2].segments[1].y - box.top) < 1e-9);
  assert.ok(Math.abs(chart.cursor - (box.left + 2.5 * ((box.right - box.left) / 3))) < 1e-9);
  assert.equal(countsChartGeometry(points, box).cursor, null);

  const curve = [
    { depth: 10, temperature: -1.8, salinity: 34.0 },
    { depth: 500, temperature: 1.0, salinity: 34.7 },
    { depth: 1000, temperature: Number.NaN, salinity: 34.72 },
  ];
  const geometry = curveChartGeometry(curve, box);
  assert.equal(geometry.temperature.length, 2);
  assert.equal(geometry.salinity.length, 3);
  assert.deepEqual(geometry.temperature[0], [box.left, box.top + 0.01 * (box.bottom - box.top)]);
  assert.deepEqual(geometry.salinity[2], [box.right, box.bottom]);
  const flat = curveChartGeometry([{ depth: 50, temperature: 0, salinity: Number.NaN }], box);
  assert.deepEqual(flat.tRange, [-0.25, 0.25]);
  assert.deepEqual(flat.sRange, [34, 35]);
});

test("platform colours are distinct", () => {
  const keys = new Set(PROFILE_PLATFORMS.map((p) => p.color.join(",")));
  assert.equal(keys.size, PROFILE_PLATFORMS.length);
});

test("the committed packages decode together", { skip: !existsSync(new URL("ocean_ice_profiles_antarctica.meta.json", DATA)) }, () => {
  const load = (name) => {
    const meta = JSON.parse(readFileSync(new URL(`${name}.meta.json`, DATA), "utf8"));
    const bytes = readFileSync(new URL(`${name}.bin`, DATA));
    return { meta, buffer: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
  };
  const pointsPackage = load("ocean_ice_profiles_antarctica");
  const points = decodeProfilePoints(pointsPackage.meta, pointsPackage.buffer);
  assert.ok(points.count > 400000);
  const curvesPackage = load("ocean_ice_profiles_antarctica_curves");
  const curves = decodeProfileCurves(curvesPackage.meta, curvesPackage.buffer, points);
  const last = profileCurve(curves, points, points.count - 1);
  assert.equal(last[last.length - 1].depth, points.maxDepth[points.count - 1]);
  const totals = countByPlatform(points, yearRange(points));
  assert.equal(totals.ship + totals.argo + totals.seal, points.count);
});

test("defaultProfileYear skips a last year that is still filling up", () => {
  const { meta, buffer } = synthetic();
  const points = decodeProfilePoints(meta, buffer);
  // 1972 holds two profiles, as many as 1970 and more than half of 1971's none.
  assert.equal(defaultProfileYear(points), 1972);
  const thin = synthetic({ days: [0, 100, 400, 800] });
  thin.meta.years.starts = [0, 2, 3, 4];
  // 1971 holds one profile and 1972 one: not fewer than half, so the last year stands.
  assert.equal(defaultProfileYear(decodeProfilePoints(thin.meta, thin.buffer)), 1972);
  const partial = synthetic({ days: [400, 410, 420, 800] });
  partial.meta.years.starts = [0, 0, 3, 4];
  // 1971 holds three, 1972 one: the record is still filling 1972.
  assert.equal(defaultProfileYear(decodeProfilePoints(partial.meta, partial.buffer)), 1971);
});
