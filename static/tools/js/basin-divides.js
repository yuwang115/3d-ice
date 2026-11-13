/**
 * Drainage divides as scripts/prepare_mankoff_basins.py packages them: a basin set too large
 * to ship basin by basin (Mankoff et al., 2020: 23,906 outlet catchments on a 100 m grid),
 * stored as the divides between its large basins, each divide once.
 *
 * Each arc is [x0, y0, dx1, dy1, ...] in lattice units (`lattice_m` metres) of the
 * package's projection. No DOM or scene dependencies, so it runs under Node's test runner.
 */

export const BASIN_DIVIDE_SCHEMA = "3d-ice/basin-divides";

const MAX_ARCS = 60_000;
const MAX_POINTS = 600_000;
const MAX_POINTS_PER_ARC = 50_000;
const MAX_LATTICE_M = 10_000;
// Greenland's EPSG:3413 extent with room to spare, in metres.
const MAX_ABS_COORDINATE_M = 6_000_000;

export function isBasinDivideDataset(payload) {
  return Boolean(payload && typeof payload === "object" && payload.schema === BASIN_DIVIDE_SCHEMA);
}

function isCount(value) {
  return Number.isInteger(value) && value >= 0;
}

function validateArc(arc, latticeM) {
  if (!Array.isArray(arc) || arc.length < 4 || arc.length % 2 !== 0 || arc.length / 2 > MAX_POINTS_PER_ARC) {
    throw new Error("Basin divides: invalid arc");
  }
  let x = 0;
  let y = 0;
  for (let i = 0; i < arc.length; i += 2) {
    if (!Number.isInteger(arc[i]) || !Number.isInteger(arc[i + 1])) {
      throw new Error("Basin divides: coordinates must be integers");
    }
    x += arc[i];
    y += arc[i + 1];
    if (Math.abs(x * latticeM) > MAX_ABS_COORDINATE_M || Math.abs(y * latticeM) > MAX_ABS_COORDINATE_M) {
      throw new Error("Basin divides: coordinate out of range");
    }
  }
  return arc.length / 2;
}

/** The payload, if it is a well-formed divide package within the limits; throws otherwise. */
export function validateBasinDivideDataset(payload) {
  if (!isBasinDivideDataset(payload)) throw new Error("Invalid basin divide dataset");
  if (payload.schema_version !== 1) throw new Error(`Unsupported basin divide schema version ${payload.schema_version}`);
  const latticeM = payload.lattice_m;
  if (!Number.isInteger(latticeM) || latticeM <= 0 || latticeM > MAX_LATTICE_M) {
    throw new Error("Basin divides: invalid lattice spacing");
  }
  for (const key of ["basin_count", "large_basin_count", "divide_count", "point_count"]) {
    if (!isCount(payload[key])) throw new Error(`Basin divides: invalid ${key}`);
  }
  if (payload.large_basin_count > payload.basin_count) {
    throw new Error("Basin divides: more large basins than the set holds");
  }
  const fraction = payload.large_area_fraction;
  if (!Number.isFinite(fraction) || fraction < 0 || fraction > 1) {
    throw new Error("Basin divides: invalid large_area_fraction");
  }
  if (!Number.isFinite(payload.min_area_km2) || payload.min_area_km2 < 0) {
    throw new Error("Basin divides: invalid min_area_km2");
  }
  const { divides } = payload;
  if (!Array.isArray(divides)) throw new Error("Basin divides: divides must be a list");
  if (divides.length > MAX_ARCS) throw new Error("Basin divides exceed the arc limit");
  if (divides.length !== payload.divide_count) throw new Error("Basin divides: divide_count does not match the divides");
  let points = 0;
  for (const arc of divides) {
    points += validateArc(arc, latticeM);
    if (points > MAX_POINTS) throw new Error("Basin divides exceed the point limit");
  }
  if (points !== payload.point_count) throw new Error("Basin divides: point_count does not match the arcs");
  return payload;
}

/** One delta-encoded arc as [[x, y], ...] in metres. */
export function decodeBasinDivideArc(arc, latticeM) {
  const points = new Array(arc.length / 2);
  let x = 0;
  let y = 0;
  for (let i = 0; i < arc.length; i += 2) {
    x += arc[i];
    y += arc[i + 1];
    points[i / 2] = [x * latticeM, y * latticeM];
  }
  return points;
}

/** A validated package's divides as polylines of [x, y] metres. */
export function decodeBasinDivides(payload) {
  return payload.divides.map((arc) => decodeBasinDivideArc(arc, payload.lattice_m));
}
