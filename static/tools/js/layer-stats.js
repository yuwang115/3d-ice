/**
 * Statistics of a gridded field over the cells a layer draws.
 *
 * Package metadata records statistics over every valid grid cell, which for terrain and
 * velocity packages includes open ocean at 0. The Data Snapshot needs them over the cells
 * the layer actually shows, ice-covered cells for thickness and speed.
 *
 * No DOM or scene dependencies: this module runs unchanged under Node's test runner.
 */

const QUANTILES = Object.freeze({ median: 0.5, q90: 0.9, q95: 0.95, q99: 0.99 });

function quantileOfSorted(sorted, probability) {
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

/**
 * Summarise `values` over the cells `mask` selects. `mask` is an array of 0/1 flags or a
 * predicate on the cell index. Returns null when no finite value is selected.
 */
export function summariseMasked(values, mask, { quantiles = true } = {}) {
  if (!values || !values.length) return null;
  const selects = typeof mask === "function" ? mask : (index) => Boolean(mask[index]);
  const selected = new Float64Array(values.length);
  let count = 0;
  let sum = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value) || !selects(index)) continue;
    selected[count] = value;
    count += 1;
    sum += value;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (count === 0) return null;
  let quantileValues = null;
  if (quantiles) {
    const sorted = selected.subarray(0, count).sort();
    quantileValues = Object.fromEntries(
      Object.entries(QUANTILES).map(([key, probability]) => [key, quantileOfSorted(sorted, probability)])
    );
  }
  return { count, min, max, mean: sum / count, quantiles: quantileValues };
}
