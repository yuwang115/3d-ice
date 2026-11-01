/**
 * Decisions of the research edition's panel that need no DOM: which terrain and grid a
 * dataset is, why a layer is off on the terrain shown, and which groups start open.
 *
 * The panel offers a region's datasets as two choices, a terrain (BedMachine v4, Bedmap3)
 * and a grid (10 km, 4 km), and a layer the terrain lacks names where it can be found
 * ("Antarctica only"). Both read the runtime's region registry, whose datasets carry
 * `terrain`, `terrainLabel`, `grid` and `gridLabel`.
 *
 * No DOM or scene dependencies: this module runs unchanged under Node's test runner.
 */

/**
 * @typedef {{ id: string, label?: string, terrain?: string, terrainLabel?: string, grid?: string, gridLabel?: string }} DatasetEntry
 * @typedef {{ terrain: string, grid: string }} TerrainChoice
 * @typedef {{ key: string, label: string }} ChoiceOption
 * @typedef {Record<string, { key?: string, datasets?: Record<string, DatasetEntry> }>} RegionRegistry
 * @typedef {null
 *   | { kind: "region", region: string }
 *   | { kind: "terrain", terrain: string }
 *   | { kind: "dataset", terrain: string, grid: string }
 *   | { kind: "grid", grid: string }
 *   | { kind: "none" }} Unavailability
 */

/** A dataset without a terrain or grid of its own is a terrain and grid of its own id. */
const terrainOf = (dataset) => dataset.terrain || dataset.id;
const gridOf = (dataset) => dataset.grid || dataset.id;
const terrainLabelOf = (dataset) => dataset.terrainLabel || dataset.label || terrainOf(dataset);
const gridLabelOf = (dataset) => dataset.gridLabel || gridOf(dataset);

function datasetList(datasets) {
  return Object.values(datasets || {});
}

/**
 * The terrains and grids a region's datasets offer, each once, in registry order.
 * @param {Record<string, DatasetEntry> | undefined} datasets
 * @returns {{ terrains: ChoiceOption[], grids: ChoiceOption[] }}
 */
export function listTerrainChoices(datasets) {
  const terrains = new Map();
  const grids = new Map();
  for (const dataset of datasetList(datasets)) {
    if (!terrains.has(terrainOf(dataset))) terrains.set(terrainOf(dataset), terrainLabelOf(dataset));
    if (!grids.has(gridOf(dataset))) grids.set(gridOf(dataset), gridLabelOf(dataset));
  }
  const toOptions = (entries) => [...entries].map(([key, label]) => ({ key, label }));
  return { terrains: toOptions(terrains), grids: toOptions(grids) };
}

/**
 * The terrain and grid a dataset stands for, or null for a dataset the region lacks.
 * @param {Record<string, DatasetEntry> | undefined} datasets
 * @param {string} datasetKey
 * @returns {TerrainChoice | null}
 */
export function getTerrainChoice(datasets, datasetKey) {
  const dataset = datasets?.[datasetKey];
  return dataset ? { terrain: terrainOf(dataset), grid: gridOf(dataset) } : null;
}

/**
 * Whether the region has a dataset for this terrain on this grid.
 * @param {Record<string, DatasetEntry> | undefined} datasets
 * @param {TerrainChoice} choice
 */
export function offersChoice(datasets, { terrain, grid }) {
  return datasetList(datasets).some((dataset) => terrainOf(dataset) === terrain && gridOf(dataset) === grid);
}

/**
 * The dataset of a terrain and grid. For a pair the region lacks the terrain wins, on its
 * first grid; for a terrain it lacks, the region's first dataset.
 * @param {Record<string, DatasetEntry> | undefined} datasets
 * @param {TerrainChoice} choice
 * @returns {string | null}
 */
export function findDatasetForChoice(datasets, { terrain, grid }) {
  const entries = datasetList(datasets);
  const match =
    entries.find((dataset) => terrainOf(dataset) === terrain && gridOf(dataset) === grid) ||
    entries.find((dataset) => terrainOf(dataset) === terrain) ||
    entries[0];
  return match ? match.id : null;
}

/**
 * Why a layer is off on the dataset shown: where it can be found instead, or null when
 * this dataset offers it.
 *
 * - No dataset of this region offers it, and one other region does: that region.
 * - One dataset offers it: that terrain and grid ("BedMachine v4 10 km").
 * - Several datasets of one terrain offer it: that terrain.
 * - Several terrains offer it on one grid: that grid.
 * - Otherwise there is no single place to point to: `{ kind: "none" }`.
 *
 * @param {RegionRegistry} regions
 * @param {string} regionKey
 * @param {string} datasetKey
 * @param {(regionKey: string, datasetKey: string) => boolean} offers
 * @returns {Unavailability}
 */
export function describeUnavailability(regions, regionKey, datasetKey, offers) {
  if (offers(regionKey, datasetKey)) return null;
  const offering = datasetList(regions?.[regionKey]?.datasets).filter((dataset) => offers(regionKey, dataset.id));
  if (!offering.length) {
    const elsewhere = Object.entries(regions || {}).filter(
      ([key, region]) => key !== regionKey && datasetList(region.datasets).some((dataset) => offers(key, dataset.id))
    );
    return elsewhere.length === 1 ? { kind: "region", region: elsewhere[0][0] } : { kind: "none" };
  }
  if (offering.length === 1) {
    return { kind: "dataset", terrain: terrainLabelOf(offering[0]), grid: gridLabelOf(offering[0]) };
  }
  if (new Set(offering.map(terrainOf)).size === 1) return { kind: "terrain", terrain: terrainLabelOf(offering[0]) };
  if (new Set(offering.map(gridOf)).size === 1) return { kind: "grid", grid: gridLabelOf(offering[0]) };
  return { kind: "none" };
}

/** The group choices in a stored value: its boolean entries, if it is a JSON object. */
function parseGroupChoices(stored) {
  let saved = null;
  try {
    saved = JSON.parse(stored);
  } catch (_error) {
    saved = null;
  }
  if (!saved || typeof saved !== "object" || Array.isArray(saved)) return {};
  return Object.fromEntries(Object.entries(saved).filter(([, open]) => typeof open === "boolean"));
}

/**
 * Which groups start open: the stored choice for each group that has one, else what the
 * page marks open. A stored value that is not a map of group names to booleans is ignored.
 * @param {string | null} stored
 * @param {readonly string[]} groupKeys
 * @param {readonly string[]} defaultOpen
 * @returns {Record<string, boolean>}
 */
export function readGroupState(stored, groupKeys, defaultOpen) {
  const choices = parseGroupChoices(stored);
  return Object.fromEntries(groupKeys.map((key) => [key, choices[key] ?? defaultOpen.includes(key)]));
}

/**
 * The value to store once the reader has opened or closed a group: the stored choices with
 * this one added. Only groups the reader has touched are stored, so a group the page
 * opened by itself, or a new default, is never pinned.
 * @param {string | null} stored
 * @param {string} groupKey
 * @param {boolean} open
 * @returns {string}
 */
export function recordGroupState(stored, groupKey, open) {
  return JSON.stringify({ ...parseGroupChoices(stored), [groupKey]: open });
}

/**
 * The groups holding a layer that is on now and was not before, in the order the layers
 * are listed.
 * @param {ReadonlySet<string>} before
 * @param {ReadonlySet<string>} after
 * @param {(layerId: string) => string | null} groupOf
 * @returns {Set<string>}
 */
export function groupsToOpen(before, after, groupOf) {
  const groups = new Set();
  for (const layerId of after) {
    const group = before.has(layerId) ? null : groupOf(layerId);
    if (group) groups.add(group);
  }
  return groups;
}
