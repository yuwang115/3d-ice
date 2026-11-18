/**
 * Polar stereographic projections of the explorer's two grids, WGS84 ellipsoid:
 * EPSG:3031 (Antarctica, true scale at 71 deg S) and EPSG:3413 (Greenland, true scale at
 * 70 deg N, central meridian 45 deg W). A port of project_polar() in
 * scripts/prepare_polar_features.py, which wrote the x/y of every place catalogue entry.
 *
 * No DOM or scene dependencies: this module runs unchanged under Node's test runner.
 */

const SEMI_MAJOR_AXIS_M = 6_378_137.0;
const ECCENTRICITY = Math.sqrt(0.0066943799901413165);
const DEG = Math.PI / 180;

export const POLAR_PROJECTIONS = Object.freeze({
  antarctica: Object.freeze({
    epsg: "EPSG:3031",
    hemisphere: "south",
    standardParallelDeg: 71,
    centralMeridianDeg: 0,
  }),
  greenland: Object.freeze({
    epsg: "EPSG:3413",
    hemisphere: "north",
    standardParallelDeg: 70,
    centralMeridianDeg: -45,
  }),
});

function isometricT(phi) {
  const sinPhi = Math.sin(phi);
  const correction = ((1 - ECCENTRICITY * sinPhi) / (1 + ECCENTRICITY * sinPhi)) ** (ECCENTRICITY / 2);
  return Math.tan(Math.PI / 4 - phi / 2) / correction;
}

/**
 * Project a WGS84 latitude and longitude onto a region's polar stereographic grid.
 *
 * @param {number} latDeg
 * @param {number} lonDeg
 * @param {"antarctica" | "greenland"} regionKey
 * @returns {{ x: number, y: number }} projected metres
 */
export function projectLatLon(latDeg, lonDeg, regionKey) {
  const projection = POLAR_PROJECTIONS[regionKey];
  if (!projection) throw new RangeError(`No polar projection for region: ${String(regionKey)}`);
  if (!Number.isFinite(latDeg) || !Number.isFinite(lonDeg) || Math.abs(latDeg) > 90 || Math.abs(lonDeg) > 180) {
    throw new RangeError(`Invalid coordinate: ${latDeg}, ${lonDeg}`);
  }
  const north = projection.hemisphere === "north";
  if (north ? latDeg <= 0 : latDeg >= 0) {
    throw new RangeError(`${latDeg} deg lies outside the ${projection.hemisphere}ern polar grid`);
  }

  const standardParallel = projection.standardParallelDeg * DEG;
  const sinStandard = Math.sin(standardParallel);
  const mStandard = Math.cos(standardParallel) / Math.sqrt(1 - ECCENTRICITY ** 2 * sinStandard ** 2);
  const radius = (SEMI_MAJOR_AXIS_M * mStandard * isometricT(Math.abs(latDeg) * DEG)) / isometricT(standardParallel);
  const delta = (lonDeg - projection.centralMeridianDeg) * DEG;
  return {
    x: radius * Math.sin(delta),
    y: (north ? -1 : 1) * radius * Math.cos(delta),
  };
}

/**
 * Latitude and longitude of a point on a region's polar stereographic grid: the inverse of
 * projectLatLon (Snyder 1987, eqs. 7-9 and 21-38 to 21-40), iterated to below 1e-12 rad.
 *
 * @param {number} x projected metres
 * @param {number} y projected metres
 * @param {"antarctica" | "greenland"} regionKey
 * @returns {{ lat: number, lon: number }} degrees, longitude in [-180, 180)
 */
export function unprojectXY(x, y, regionKey) {
  const projection = POLAR_PROJECTIONS[regionKey];
  if (!projection) throw new RangeError(`No polar projection for region: ${String(regionKey)}`);
  if (!Number.isFinite(x) || !Number.isFinite(y)) throw new RangeError(`Invalid point: ${x}, ${y}`);
  const north = projection.hemisphere === "north";
  const standardParallel = projection.standardParallelDeg * DEG;
  const sinStandard = Math.sin(standardParallel);
  const mStandard = Math.cos(standardParallel) / Math.sqrt(1 - ECCENTRICITY ** 2 * sinStandard ** 2);
  const t = (Math.hypot(x, y) * isometricT(standardParallel)) / (SEMI_MAJOR_AXIS_M * mStandard);
  let phi = Math.PI / 2 - 2 * Math.atan(t);
  for (let i = 0; i < 20; i += 1) {
    const sinPhi = Math.sin(phi);
    const next = Math.PI / 2 - 2 * Math.atan(t * ((1 - ECCENTRICITY * sinPhi) / (1 + ECCENTRICITY * sinPhi)) ** (ECCENTRICITY / 2));
    const done = Math.abs(next - phi) < 1e-12;
    phi = next;
    if (done) break;
  }
  const delta = Math.atan2(x, north ? -y : y) / DEG;
  const lon = ((((delta + projection.centralMeridianDeg + 180) % 360) + 360) % 360) - 180;
  return { lat: (north ? 1 : -1) * (phi / DEG), lon };
}
