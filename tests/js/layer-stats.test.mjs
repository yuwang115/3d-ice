/**
 * The Data Snapshot summarises a field over the cells its layer draws. Version 0.3.0 read
 * statistics that the packers computed over every grid cell, ocean included, so Antarctica's
 * "mean ice thickness" read 590 m and its median ice speed 0 m/yr.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { summariseMasked } from "../../static/tools/js/layer-stats.js";

test("statistics cover only the cells the mask selects", () => {
  const values = Float32Array.from([0, 0, 1000, 2000, 3000, 0]);
  const ice = Uint8Array.from([0, 0, 1, 1, 1, 0]);
  const stats = summariseMasked(values, ice);
  assert.equal(stats.count, 3);
  assert.equal(stats.min, 1000);
  assert.equal(stats.max, 3000);
  assert.equal(stats.mean, 2000);
  assert.equal(stats.quantiles.median, 2000);
});

test("a predicate works as the mask, and non-finite values are skipped", () => {
  const values = Float32Array.from([5, Number.NaN, 7, 9, Number.POSITIVE_INFINITY]);
  const stats = summariseMasked(values, (index) => index !== 3);
  assert.equal(stats.count, 2);
  assert.equal(stats.min, 5);
  assert.equal(stats.max, 7);
  assert.equal(stats.mean, 6);
});

test("quantiles interpolate linearly between ranked values", () => {
  const values = Float32Array.from(Array.from({ length: 101 }, (_, index) => index));
  const stats = summariseMasked(values, () => true);
  assert.equal(stats.quantiles.median, 50);
  assert.equal(stats.quantiles.q90, 90);
  assert.equal(stats.quantiles.q95, 95);
  assert.equal(stats.quantiles.q99, 99);
  const even = summariseMasked(Float32Array.from([1, 2, 3, 4]), () => true);
  assert.equal(even.quantiles.median, 2.5);
});

test("quantiles can be skipped when only the mean is wanted", () => {
  const stats = summariseMasked(Float32Array.from([1, 2, 3]), () => true, { quantiles: false });
  assert.equal(stats.mean, 2);
  assert.equal(stats.quantiles, null);
});

test("an empty selection gives null rather than NaN or Infinity", () => {
  assert.equal(summariseMasked(Float32Array.from([1, 2]), Uint8Array.from([0, 0])), null);
  assert.equal(summariseMasked(null, () => true), null);
});
