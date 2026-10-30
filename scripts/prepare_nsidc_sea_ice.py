#!/usr/bin/env python3
"""Pack NOAA/NSIDC monthly sea-ice concentration (passive microwave, 1978-2026) for 3D ICE.

Input: the NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice Concentration,
Version 6 (G02202 v6, doi:10.7265/b18j-z797), monthly files
`sic_ps{n,s}25_<YYYYMM>_<sensor>_v06r00.nc` from
https://noaadata.apps.nsidc.org/NOAA/G02202_V6/{north,south}/monthly/, with the `.mnf`
manifest NSIDC publishes beside each file (file name, MD5, size) and the grid's ancillary
file. The files state "No constraints on data access or use"; NSIDC asks that the data set
be cited as a condition of use.

The record runs from November 1978: SMMR to July 1987, SSM/I and SSMIS to December 2024,
AMSR2 from January 2025. December 1987 and January 1988 are empty (the SSM/I gap of
3 December 1987 to 12 January 1988) and are kept as frames without data.

Each region's package holds every month on a 25 km grid of its own, in the explorer's
projection (EPSG:3031 for Antarctica, EPSG:3413 for Greenland). NSIDC's grids are polar
stereographic on the Hughes 1980 ellipsoid (EPSG:3412 and EPSG:3411), with another latitude
of true scale in the south, so every cell centre of the package grid is projected into the
NSIDC grid and takes the bilinear mean of the ocean cells around it. A cell that NSIDC's
land mask leaves without ocean neighbours but that the explorer's terrain calls ocean (a
coast drawn differently at 25 km, an ice front that has moved) takes the mean of its nearest
filled neighbours. The grid is coarser than the terrain grids, so the explorer draws it as a
texture on the sea surface and clips it to the terrain's own ocean.

Each package also carries

- for every calendar month, the share of the 1981-2010 years in which each cell held at
  least 15 % ice, whose 50 % contour is the median ice edge NSIDC's Sea Ice Index draws;
- the sea-ice extent and area of the whole hemisphere every month, summed on NSIDC's own
  grid, with the 1981-2010 average and the least-squares trend of each calendar month, so the
  explorer quotes hemisphere totals rather than sums over the part of the ocean its terrain
  covers. The extent counts the cells with at least 15 % ice on at least half of the month's
  days, from the files' QA flag: NSIDC's Sea Ice Index averages daily extents, and thresholding
  the monthly mean concentration instead would add 5 % in the Arctic and 10 % in the Antarctic.

Outputs: `nsidc_sea_ice_antarctica` and `nsidc_sea_ice_greenland`, sparse time series
(`concentration`, `reference_ice_frequency`, `domain_mask`).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NO_DATA = 255
EXTENT_THRESHOLD_PERCENT = 15
REFERENCE_YEARS = (1981, 2010)
TREND_FIRST_YEAR = 1979
# NSIDC's median ice edge: the cells with at least 15 % ice in at least half the years.
MEDIAN_EDGE_PERCENT = 50
CELL_M = 25000.0
# Coastal cells without an NSIDC ocean neighbour are filled from filled neighbours, one ring
# per pass; 40 passes reach 1,000 km, further than any terrain ocean lies from NSIDC's.
MAX_FILL_PASSES = 40
MONTH_FILE = re.compile(r"sic_ps([ns])25_(\d{4})(\d{2})_([A-Za-z0-9]+)_v06r00\.nc$")
# The QA flag that marks cells with at least 15 % ice on at least half of the month's days.
MAJORITY_FLAG = "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15"
SENSOR_NAMES = {
    "n07": "SMMR (Nimbus-7)",
    "F08": "SSM/I (DMSP F8)",
    "F11": "SSM/I (DMSP F11)",
    "F13": "SSM/I (DMSP F13)",
    "F17": "SSMIS (DMSP F17)",
    "am2": "AMSR2 (GCOM-W1)",
}
DATA_SET = {
    "title": "NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice Concentration, Version 6",
    "id": "G02202",
    "version": "v06r00",
    "doi": "10.7265/b18j-z797",
    "url": "https://nsidc.org/data/g02202/versions/6",
    "citation": (
        "Meier, W. N., Fetterer, F., Windnagel, A. K., Stewart, J. S. & Stafford, T. (2026). "
        "NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice Concentration (G02202, Version 6). "
        "Boulder, Colorado USA. National Snow and Ice Data Center. https://doi.org/10.7265/b18j-z797"
    ),
    "license": "No constraints on data access or use (the files' license attribute); NSIDC asks that the data set be cited as a condition of use",
    "files_url": "https://noaadata.apps.nsidc.org/NOAA/G02202_V6/",
}


# ---------------------------------------------------------------------------- projections


@dataclass(frozen=True)
class PolarStereographic:
    """Polar stereographic projection of an ellipsoid with a latitude of true scale.

    The formulas are Snyder (1987), Map Projections: A Working Manual, eqs. 15-9, 21-33 to
    21-40 and 7-9; the south polar aspect reverses the signs of the latitudes and of y.
    """

    code: str
    semi_major_m: float
    semi_minor_m: float
    true_scale_lat_deg: float
    central_meridian_deg: float

    @property
    def is_south(self) -> bool:
        return self.true_scale_lat_deg < 0

    @property
    def eccentricity(self) -> float:
        return float(np.sqrt(1.0 - (self.semi_minor_m / self.semi_major_m) ** 2))

    def _t(self, phi: np.ndarray) -> np.ndarray:
        e = self.eccentricity
        s = np.sin(phi)
        return np.tan(np.pi / 4 - phi / 2) / ((1 - e * s) / (1 + e * s)) ** (e / 2)

    def _m(self, phi: np.ndarray) -> np.ndarray:
        return np.cos(phi) / np.sqrt(1 - (self.eccentricity * np.sin(phi)) ** 2)

    def _rho_per_t(self) -> float:
        phic = np.deg2rad(abs(self.true_scale_lat_deg))
        return float(self.semi_major_m * self._m(phic) / self._t(phic))

    def _sign(self) -> float:
        return -1.0 if self.is_south else 1.0

    def forward(self, lon_deg, lat_deg) -> tuple[np.ndarray, np.ndarray]:
        """Projected x and y (m) of longitudes and latitudes (degrees)."""
        sign = self._sign()
        phi = np.deg2rad(np.asarray(lat_deg, dtype=np.float64)) * sign
        dlam = np.deg2rad(np.asarray(lon_deg, dtype=np.float64) - self.central_meridian_deg)
        rho = self._rho_per_t() * self._t(phi)
        return rho * np.sin(dlam), -sign * rho * np.cos(dlam)

    def inverse(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Longitudes (-180 to 180) and latitudes (degrees) of projected points."""
        sign = self._sign()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        t = np.hypot(x, y) / self._rho_per_t()
        e = self.eccentricity
        phi = np.pi / 2 - 2 * np.arctan(t)
        for _ in range(15):
            s = np.sin(phi)
            phi = np.pi / 2 - 2 * np.arctan(t * ((1 - e * s) / (1 + e * s)) ** (e / 2))
        lon = self.central_meridian_deg + np.rad2deg(np.arctan2(x, -sign * y))
        return (lon + 180.0) % 360.0 - 180.0, np.rad2deg(phi) * sign

    def scale_factor(self, lat_deg) -> np.ndarray:
        """Point scale factor k at a latitude; the projection is conformal, so a small
        projected square of side d covers d**2 / k**2 of the ellipsoid."""
        phi = np.deg2rad(np.abs(np.asarray(lat_deg, dtype=np.float64)))
        return self._rho_per_t() * self._t(phi) / (self.semi_major_m * self._m(phi))


HUGHES_1980 = (6378273.0, 6356889.449)
WGS84 = (6378137.0, 6356752.314245179)
PROJECTIONS = {
    "EPSG:3411": PolarStereographic("EPSG:3411", *HUGHES_1980, 70.0, -45.0),
    "EPSG:3412": PolarStereographic("EPSG:3412", *HUGHES_1980, -70.0, 0.0),
    "EPSG:3413": PolarStereographic("EPSG:3413", *WGS84, 70.0, -45.0),
    "EPSG:3031": PolarStereographic("EPSG:3031", *WGS84, -71.0, 0.0),
}

REGIONS = {
    "antarctica": {
        "hemisphere": "south",
        "file_prefix": "s",
        "source_crs": "EPSG:3412",
        "explorer_crs": "EPSG:3031",
        "terrains": (
            "bedmachine_antarctica_v4_480",
            "bedmachine_antarctica_v4_741",
            "bedmap3_antarctica_10km",
            "bedmap3_antarctica_4km",
        ),
        "basename": "nsidc_sea_ice_antarctica",
        "hemisphere_name": "Southern Hemisphere (the Southern Ocean around Antarctica)",
        "index_prefix": "S",
    },
    "greenland": {
        "hemisphere": "north",
        "file_prefix": "n",
        "source_crs": "EPSG:3411",
        "explorer_crs": "EPSG:3413",
        "terrains": (
            "bedmachine_greenland_v6_3km",
            "bedmachine_greenland_v6_1km",
            "greenland_qrf_2025_3km",
            "greenland_qrf_2025_1km",
        ),
        "basename": "nsidc_sea_ice_greenland",
        "hemisphere_name": "Northern Hemisphere (the whole Arctic, of which the Greenland grid holds a part)",
        "index_prefix": "N",
    },
}


# ---------------------------------------------------------------------------- inputs


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict:
    """NSIDC's `.mnf` line: `name,md5,size`."""
    name, md5, size = Path(path).read_text(encoding="utf-8").strip().split(",")
    return {"name": name, "md5": md5, "bytes": int(size)}


def list_months(directory: Path, file_prefix: str) -> list[dict]:
    """Monthly files of one hemisphere in date order, with their year, month and sensor."""
    months = []
    for path in sorted(Path(directory).glob(f"sic_ps{file_prefix}25_*_v06r00.nc")):
        match = MONTH_FILE.search(path.name)
        if not match:
            continue
        months.append({"path": path, "year": int(match.group(2)), "month": int(match.group(3)), "sensor": match.group(4)})
    if not months:
        raise FileNotFoundError(f"no sic_ps{file_prefix}25 monthly files in {directory}")
    keys = [12 * m["year"] + m["month"] - 1 for m in months]
    if np.any(np.diff(keys) != 1):
        raise ValueError("the monthly files must run without a gap (the empty months are files too)")
    return months


def majority_mask(qa_var) -> np.ndarray:
    """Cells the QA flag marks as holding at least 15 % ice on at least half of the month's days."""
    meanings = qa_var.getncattr("flag_meanings").split()
    masks = np.atleast_1d(qa_var.getncattr("flag_masks"))
    if MAJORITY_FLAG not in meanings or len(masks) != len(meanings):
        raise ValueError(f"the QA flag does not define {MAJORITY_FLAG}")
    bit = int(masks[meanings.index(MAJORITY_FLAG)])
    return (np.asarray(qa_var[0], dtype=np.uint8) & bit) != 0


def read_month(path: Path) -> dict:
    """One monthly file: concentration codes (uint8 percent, NO_DATA where missing), the cells
    with ice on at least half of the days (from the QA flag), the grid axes and the provenance
    attributes."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        var = ds["cdr_seaice_conc_monthly"]
        codes = np.asarray(var[0], dtype=np.uint8)
        fill = int(var.getncattr("_FillValue"))
        valid_max = int(np.max(var.getncattr("valid_range")))
        majority = majority_mask(ds["cdr_seaice_conc_monthly_qa_flag"])
        attrs = {key: str(ds.getncattr(key)) for key in ("date_created", "product_version", "sensor") if key in ds.ncattrs()}
        x = np.asarray(ds["x"][:], dtype=np.float64)
        y = np.asarray(ds["y"][:], dtype=np.float64)
    codes = np.where((codes == fill) | (codes > valid_max), NO_DATA, codes).astype(np.uint8)
    return {"codes": codes, "majority": majority & (codes != NO_DATA), "x": x, "y": y, "attrs": attrs}


def read_ancillary(path: Path) -> dict:
    """The grid's surface type (50 ocean, 75 lake, 200 coast, 250 land) and cell latitudes."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        return {
            "surface_type": np.asarray(ds["surface_type"][:], dtype=np.uint8),
            "latitude": np.asarray(ds["latitude"][:], dtype=np.float64),
            "longitude": np.asarray(ds["longitude"][:], dtype=np.float64),
            "x": np.asarray(ds["x"][:], dtype=np.float64),
            "y": np.asarray(ds["y"][:], dtype=np.float64),
        }


def read_terrain_ocean(meta_path: Path) -> tuple[dict, np.ndarray]:
    """Grid and ocean cells (mask class 0) of an explorer terrain package."""
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    raw = Path(meta_path).with_name(Path(meta_path).name.replace(".meta.json", ".bin")).read_bytes()
    grid = {key: meta["grid"][key] for key in ("nx", "ny", "x0_m", "y0_m", "dx_m", "dy_m")}
    field = next(f for f in meta["fields"] if f["name"] == "mask")
    mask = np.frombuffer(raw, dtype=np.uint8, count=field["byte_length"], offset=field["byte_offset"])
    return grid, mask.reshape(grid["ny"], grid["nx"]) == 0


def read_sea_ice_index(directory: Path, prefix: str) -> dict[tuple[int, int], float]:
    """NSIDC Sea Ice Index v4 monthly extents (million km2) keyed by (year, month), for validation."""
    out = {}
    for path in sorted(Path(directory).glob(f"{prefix}_*_extent_v4.0.csv")):
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle, skipinitialspace=True):
                extent = float(row["extent"])
                if extent > 0:
                    out[(int(row["year"]), int(row["mo"]))] = extent
    return out


# ---------------------------------------------------------------------------- grids


def axes(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    x = grid["x0_m"] + grid["dx_m"] * np.arange(grid["nx"], dtype=np.float64)
    y = grid["y0_m"] + grid["dy_m"] * np.arange(grid["ny"], dtype=np.float64)
    return x, y


def sea_ice_grid(terrain_grids: list[dict], cell_m: float = CELL_M) -> dict:
    """A grid of cell centres at (k + 1/2) * cell_m, like NSIDC's, whose cells cover every
    terrain grid's first to last node, rows north to south."""
    x_min = min(g["x0_m"] for g in terrain_grids)
    x_max = max(g["x0_m"] + g["dx_m"] * (g["nx"] - 1) for g in terrain_grids)
    y_max = max(g["y0_m"] for g in terrain_grids)
    y_min = min(g["y0_m"] + g["dy_m"] * (g["ny"] - 1) for g in terrain_grids)
    half = cell_m / 2
    first_x = np.floor((x_min - half) / cell_m) * cell_m + half
    last_x = np.ceil((x_max - half) / cell_m) * cell_m + half
    top_y = np.ceil((y_max - half) / cell_m) * cell_m + half
    bottom_y = np.floor((y_min - half) / cell_m) * cell_m + half
    return {
        "nx": int(round((last_x - first_x) / cell_m)) + 1,
        "ny": int(round((top_y - bottom_y) / cell_m)) + 1,
        "x0_m": float(first_x),
        "y0_m": float(top_y),
        "dx_m": cell_m,
        "dy_m": -cell_m,
    }


@dataclass
class BilinearLookup:
    """The four source cells around each target point and their bilinear weights."""

    rows: np.ndarray  # (4, n) int
    cols: np.ndarray  # (4, n) int
    weights: np.ndarray  # (4, n) float, zero for corners outside the source grid

    @classmethod
    def build(cls, src_x: np.ndarray, src_y: np.ndarray, x: np.ndarray, y: np.ndarray) -> "BilinearLookup":
        dx = src_x[1] - src_x[0]
        dy = src_y[1] - src_y[0]
        u = (np.ravel(x) - src_x[0]) / dx
        v = (np.ravel(y) - src_y[0]) / dy
        c0 = np.floor(u).astype(np.int64)
        r0 = np.floor(v).astype(np.int64)
        fu = u - c0
        fv = v - r0
        rows = np.stack([r0, r0, r0 + 1, r0 + 1])
        cols = np.stack([c0, c0 + 1, c0, c0 + 1])
        weights = np.stack([(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv])
        inside = (rows >= 0) & (rows < src_y.size) & (cols >= 0) & (cols < src_x.size)
        return cls(np.clip(rows, 0, src_y.size - 1), np.clip(cols, 0, src_x.size - 1), np.where(inside, weights, 0.0))

    def mean(self, field: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Weighted mean of the valid corners (NaN where no corner is valid)."""
        w = self.weights * valid[self.rows, self.cols]
        total = w.sum(axis=0)
        values = (w * field[self.rows, self.cols]).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total > 1e-9, values / total, np.nan)


def target_lookup(source: dict, target_grid: dict, region: dict) -> BilinearLookup:
    """Where each package cell centre falls in NSIDC's grid. Latitude and longitude are
    taken as the same on the Hughes 1980 ellipsoid and WGS84, as NSIDC's grids assume."""
    tx, ty = axes(target_grid)
    gx, gy = np.meshgrid(tx, ty)
    lon, lat = PROJECTIONS[region["explorer_crs"]].inverse(gx, gy)
    sx, sy = PROJECTIONS[region["source_crs"]].forward(lon, lat)
    return BilinearLookup.build(source["x"], source["y"], sx, sy)


def terrain_reach(target_grid: dict, terrains: list[tuple[dict, np.ndarray]]) -> np.ndarray:
    """Package cells a texture lookup on the terrain's ocean can read: the four cells around
    each terrain ocean node, and one ring more for the pixels between nodes."""
    tx, ty = axes(target_grid)
    reach = np.zeros((target_grid["ny"], target_grid["nx"]), dtype=bool)
    for grid, ocean in terrains:
        gx, gy = axes(grid)
        cols = (gx - tx[0]) / target_grid["dx_m"]
        rows = (gy - ty[0]) / target_grid["dy_m"]
        r_idx, c_idx = np.nonzero(ocean)
        c0 = np.floor(cols[c_idx]).astype(np.int64)
        r0 = np.floor(rows[r_idx]).astype(np.int64)
        for dr in (-1, 0, 1, 2):
            for dc in (-1, 0, 1, 2):
                rr = np.clip(r0 + dr, 0, target_grid["ny"] - 1)
                cc = np.clip(c0 + dc, 0, target_grid["nx"] - 1)
                reach[rr, cc] = True
    return reach


def fill_from_neighbours(values: np.ndarray, needed: np.ndarray, max_passes: int = MAX_FILL_PASSES) -> tuple[np.ndarray, int]:
    """Give empty cells the mean of their filled 8-neighbours, one ring per pass, until every
    needed cell has a value.

    The rings grow through every empty cell, so a needed cell is reached even across cells
    nobody needs. Returns the field and the number of needed cells filled; a needed cell
    still empty after `max_passes` (none in practice) stays NaN.
    """
    out = np.array(values, dtype=np.float64)
    was_empty = needed & ~np.isfinite(out)
    ny, nx = out.shape
    for _ in range(max_passes):
        if not (needed & ~np.isfinite(out)).any():
            break
        finite = np.isfinite(out)
        padded = np.pad(np.where(finite, out, 0.0), 1)
        weight = np.pad(finite.astype(np.float64), 1)
        total = np.zeros_like(out)
        count = np.zeros_like(out)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                total += padded[1 + dr : 1 + dr + ny, 1 + dc : 1 + dc + nx]
                count += weight[1 + dr : 1 + dr + ny, 1 + dc : 1 + dc + nx]
        grow = ~finite & (count > 0)
        if not grow.any():
            break
        out[grow] = total[grow] / count[grow]
    return out, int((was_empty & np.isfinite(out)).sum())


# ---------------------------------------------------------------------------- hemisphere totals


def cell_areas_km2(source: dict, region: dict) -> np.ndarray:
    """Area (km2) of every NSIDC grid cell, from the projection's scale factor at its centre."""
    projection = PROJECTIONS[region["source_crs"]]
    gx, gy = np.meshgrid(source["x"], source["y"])
    _, lat = projection.inverse(gx, gy)
    side = abs(source["x"][1] - source["x"][0])
    return (side / projection.scale_factor(lat)) ** 2 / 1e6


def extent_and_area(codes: np.ndarray, areas_km2: np.ndarray, counted: np.ndarray | None = None) -> tuple[float, float]:
    """Extent and area (million km2) of a month: the area of the counted cells, and that area
    weighted by their monthly mean concentration; NaN for a month without data.

    By default the counted cells are those whose monthly mean concentration is at least 15 %,
    the ice the map draws. The packages count instead the cells the QA flag marks as holding at
    least 15 % ice on at least half of the days, which is close to the Sea Ice Index's monthly
    extent, the average of its daily extents.
    """
    valid = codes != NO_DATA
    if not valid.any():
        return float("nan"), float("nan")
    ice = valid & (codes >= EXTENT_THRESHOLD_PERCENT) if counted is None else valid & counted
    extent = float(areas_km2[ice].sum()) / 1e6
    area = float((areas_km2[ice] * codes[ice] / 100.0).sum()) / 1e6
    return extent, area


def reference_frequency(stack: np.ndarray, years: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Percent of the 1981-2010 years with at least 15 % ice, per calendar month and cell
    (NaN where a cell has no data in those years)."""
    out = np.full((12,) + stack.shape[1:], np.nan)
    in_period = (years >= REFERENCE_YEARS[0]) & (years <= REFERENCE_YEARS[1])
    for month in range(1, 13):
        frames = stack[in_period & (months == month)]
        valid = frames != NO_DATA
        count = valid.sum(axis=0)
        ice = (valid & (frames >= EXTENT_THRESHOLD_PERCENT)).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[month - 1] = np.where(count > 0, 100.0 * ice / count, np.nan)
    return out


def monthly_statistics(years: np.ndarray, months: np.ndarray, extent: np.ndarray) -> list[dict]:
    """The 1981-2010 average extent of each calendar month and its least-squares trend from
    1979, in million km2 per year and in percent of that average per decade, as NSIDC
    reports trends."""
    stats = []
    for month in range(1, 13):
        sel = (months == month) & np.isfinite(extent)
        ref = sel & (years >= REFERENCE_YEARS[0]) & (years <= REFERENCE_YEARS[1])
        fit = sel & (years >= TREND_FIRST_YEAR)
        reference_mean = float(extent[ref].mean())
        x = years[fit].astype(np.float64)
        slope, intercept = np.polyfit(x, extent[fit], 1)
        stats.append({
            "month": month,
            "reference_mean_mkm2": round(reference_mean, 4),
            "reference_years": int(ref.sum()),
            "trend_first_year": int(x.min()),
            "trend_last_year": int(x.max()),
            "trend_years": int(x.size),
            "slope_mkm2_per_year": round(float(slope), 6),
            "intercept_mkm2_at_year_0": round(float(intercept), 4),
            "percent_per_decade": round(float(10 * slope / reference_mean * 100), 2),
        })
    return stats


# ---------------------------------------------------------------------------- frames


def build_frames(months: list[dict], region: dict, terrains: list[tuple[dict, np.ndarray]], ancillary: dict) -> dict:
    """Read every month, resample it onto the package grid, and total the hemisphere."""
    first = read_month(months[0]["path"])
    source = {"x": first["x"], "y": first["y"]}
    if not (np.allclose(source["x"], ancillary["x"]) and np.allclose(source["y"], ancillary["y"])):
        raise ValueError("the monthly files and the ancillary file are not on the same grid")
    target_grid = sea_ice_grid([grid for grid, _ in terrains])
    lookup = target_lookup(source, target_grid, region)
    reach = terrain_reach(target_grid, terrains)
    areas = cell_areas_km2(source, region)
    ocean = ancillary["surface_type"] == 50

    stack = np.full((len(months),) + first["codes"].shape, NO_DATA, dtype=np.uint8)
    files, extents, ice_areas, mean_field_extents, missing_ocean = [], [], [], [], []
    for index, entry in enumerate(months):
        data = first if index == 0 else read_month(entry["path"])
        if not (np.array_equal(data["x"], source["x"]) and np.array_equal(data["y"], source["y"])):
            raise ValueError(f"{entry['path'].name} is not on the grid of the first file")
        stack[index] = data["codes"]
        extent, area = extent_and_area(data["codes"], areas, data["majority"])
        extents.append(extent)
        ice_areas.append(area)
        mean_field_extents.append(extent_and_area(data["codes"], areas)[0])
        valid = data["codes"] != NO_DATA
        missing_ocean.append(int((ocean & ~valid).sum()) if valid.any() else None)
        manifest_path = entry["path"].parent.parent / "checksums" / f"{entry['path'].name}.mnf"
        record = {"name": entry["path"].name, "md5": md5_of(entry["path"]), "sensor": entry["sensor"], **data["attrs"]}
        if manifest_path.exists():
            manifest = read_manifest(manifest_path)
            if manifest["md5"] != record["md5"]:
                raise ValueError(f"{entry['path'].name}: MD5 does not match NSIDC's manifest")
            record["md5_matches_manifest"] = True
        files.append(record)

    years = np.array([m["year"] for m in months])
    month_numbers = np.array([m["month"] for m in months])
    frequency = reference_frequency(stack, years, month_numbers)

    shape = (target_grid["ny"], target_grid["nx"])
    frames = np.full((len(months),) + shape, np.nan)
    filled_cells = []
    for index in range(len(months)):
        valid = stack[index] != NO_DATA
        if not valid.any():
            filled_cells.append(0)
            continue
        resampled = lookup.mean(stack[index].astype(np.float64), valid).reshape(shape)
        resampled, filled = fill_from_neighbours(resampled, reach)
        frames[index] = resampled
        filled_cells.append(filled)
    reference = np.full((12,) + shape, np.nan)
    for month in range(12):
        valid = np.isfinite(frequency[month])
        resampled = lookup.mean(np.nan_to_num(frequency[month]), valid).reshape(shape)
        reference[month], _ = fill_from_neighbours(resampled, reach)

    has_data = np.array([np.isfinite(frame).any() for frame in frames])
    ever_ice = np.nanmax(np.where(np.isfinite(frames[has_data]), frames[has_data], 0.0), axis=0) > 0
    domain = reach & ever_ice
    return {
        "grid": target_grid,
        "source": source,
        "domain": domain,
        "reach": reach,
        "frames": frames,
        "reference": reference,
        "has_data": has_data,
        "years": years,
        "months": month_numbers,
        "sensors": [m["sensor"] for m in months],
        "extent": np.array(extents),
        "area": np.array(ice_areas),
        "mean_field_extent": np.array(mean_field_extents),
        "files": files,
        "filled_cells": np.array(filled_cells),
        "missing_ocean_cells": missing_ocean,
        "lookup": lookup,
    }


# ---------------------------------------------------------------------------- packaging


def to_percent_codes(values: np.ndarray) -> np.ndarray:
    """Whole percent (0-100) as uint8, NO_DATA where a value is missing."""
    array = np.asarray(values, dtype=np.float64)
    out = np.full(array.shape, NO_DATA, dtype=np.uint8)
    valid = np.isfinite(array)
    out[valid] = np.clip(np.rint(array[valid]), 0, 100).astype(np.uint8)
    return out


def percent_stats(codes: np.ndarray) -> dict[str, float]:
    valid = codes[codes != NO_DATA].astype(np.float64)
    return {"min": float(valid.min()), "max": float(valid.max()), "mean": round(float(valid.mean()), 4)}


def month_label(year: int, month: int) -> str:
    return f"{int(year):04d}-{int(month):02d}"


def sensor_runs(labels: list[str], sensors: list[str]) -> list[dict]:
    """Consecutive months from one sensor, as runs."""
    runs: list[dict] = []
    for label, sensor in zip(labels, sensors):
        if runs and runs[-1]["sensor"] == sensor:
            runs[-1]["last"] = label
        else:
            runs.append({"sensor": sensor, "name": SENSOR_NAMES.get(sensor, sensor), "first": label, "last": label})
    return runs


def rounded_or_none(values: np.ndarray, digits: int) -> list:
    return [round(float(v), digits) if np.isfinite(v) else None for v in values]


def compare_with_index(labels_years, labels_months, extent: np.ndarray, index: dict, mean_field: np.ndarray | None = None) -> dict | None:
    """Mean and range of the packaged extent's departure from the Sea Ice Index's, by calendar
    month, and of the monthly mean field's extent, if given."""
    mean_field = np.full(len(extent), np.nan) if mean_field is None else np.asarray(mean_field, dtype=np.float64)
    pairs = [
        (m, e, index[(y, m)], h)
        for y, m, e, h in zip(labels_years, labels_months, extent, mean_field)
        if (y, m) in index and np.isfinite(e)
    ]
    if not pairs:
        return None
    months = np.array([p[0] for p in pairs])
    ours = np.array([p[1] for p in pairs])
    theirs = np.array([p[2] for p in pairs])
    halves = np.array([p[3] for p in pairs])
    relative = (ours - theirs) / theirs * 100
    mean_field_part = {}
    if np.isfinite(halves).all():
        field_relative = (halves - theirs) / theirs * 100
        mean_field_part = {
            "mean_field_mean_relative_difference_percent": round(float(np.mean(field_relative)), 2),
            "mean_field_mean_relative_difference_percent_by_month": [
                round(float(np.mean(field_relative[months == month])), 2) if np.any(months == month) else None
                for month in range(1, 13)
            ],
        }
    return {
        "product": "NSIDC Sea Ice Index, Version 4 (G02135, doi:10.7265/a98x-0f50), monthly extent CSV files",
        "months_compared": int(len(pairs)),
        "mean_difference_mkm2": round(float(np.mean(ours - theirs)), 3),
        "mean_relative_difference_percent": round(float(np.mean(relative)), 2),
        "relative_difference_range_percent": [round(float(relative.min()), 2), round(float(relative.max()), 2)],
        "mean_relative_difference_percent_by_month": [
            round(float(np.mean(relative[months == month])), 2) if np.any(months == month) else None
            for month in range(1, 13)
        ],
        **mean_field_part,
        "note": "the packaged extent counts the cells with at least 15 % ice on at least half of the month's days (the files' QA flag), close to the Sea Ice Index's average of daily extents; what difference remains is mostly the algorithm, the Sea Ice Index taking the NASA Team concentration and the CDR the higher of the NASA Team and Bootstrap concentrations in each cell. The extent of the monthly mean field (mean_field_ keys), every cell whose monthly mean concentration reaches 15 %, departs further",
    }


def build_package(out: dict, region_key: str, ancillary_check: dict, index: dict | None) -> tuple[bytes, dict]:
    region = REGIONS[region_key]
    domain = out["domain"]
    count = int(domain.sum())
    if count == 0:
        raise ValueError("no package cell both reaches the terrain's ocean and ever holds ice")
    frames = out["frames"][:, domain]
    frames[~out["has_data"]] = np.nan
    concentration = to_percent_codes(frames)
    reference = to_percent_codes(out["reference"][:, domain])
    if np.any(reference == NO_DATA):
        raise ValueError("a domain cell has no 1981-2010 reference frequency")
    if np.any(concentration[out["has_data"]] == NO_DATA):
        raise ValueError("a domain cell is empty in a month with data")

    labels = [month_label(y, m) for y, m in zip(out["years"], out["months"])]
    decimal_years = out["years"] + (out["months"] - 0.5) / 12.0
    empty = [label for label, has in zip(labels, out["has_data"]) if not has]
    payload, fields = pack_fields([
        {"name": "concentration", "dtype": "uint8", "unit": "percent", "fill_value": NO_DATA,
         "stats_percent": percent_stats(concentration),
         "note": "monthly mean sea-ice concentration, whole percent, frame-major (concentration[frame * cell_count + cell]); 255 throughout a month without data",
         "blob": concentration.tobytes()},
        {"name": "reference_ice_frequency", "dtype": "uint8", "unit": "percent", "fill_value": NO_DATA,
         "stats_percent": percent_stats(reference),
         "note": "percent of the 1981-2010 years in which the cell held at least 15 % ice, one frame per calendar month (January first), month-major like concentration; the 50 % contour is the median ice edge",
         "blob": reference.tobytes()},
        {"name": "domain_mask", "dtype": "uint8",
         "flags": {"0": "outside the domain (no ice in any month, or out of reach of the terrain's ocean)", "1": "domain cell"},
         "blob": domain.astype(np.uint8).tobytes()},
    ])
    stats = monthly_statistics(out["years"], out["months"], out["extent"])
    filled = out["filled_cells"][out["has_data"]]
    meta = {
        "title": f"NOAA/NSIDC monthly sea-ice concentration around {region_key.capitalize()}, "
                 f"{labels[0]} to {labels[-1]} (passive microwave climate data record, Version 6)",
        "product_version": f"{DATA_SET['title']} ({DATA_SET['id']}, {DATA_SET['version']})",
        "reference": DATA_SET["citation"],
        "license": DATA_SET["license"],
        "source_url": DATA_SET["url"],
        "source_dataset": {**DATA_SET, "hemisphere": region["hemisphere"]},
        "geometry_type": "sparse_grid_time_series",
        "time_series_kind": "observed_sea_ice_concentration",
        "region": region_key,
        "crs": region["explorer_crs"],
        "grid": dict(out["grid"]),
        "source_grid": {
            "crs": region["source_crs"],
            "description": "NSIDC Sea Ice Polar Stereographic, Hughes 1980 ellipsoid, 25 km",
            "nx": int(out["source"]["x"].size),
            "ny": int(out["source"]["y"].size),
            "x0_m": float(out["source"]["x"][0]),
            "y0_m": float(out["source"]["y"][0]),
            "dx_m": float(out["source"]["x"][1] - out["source"]["x"][0]),
            "dy_m": float(out["source"]["y"][1] - out["source"]["y"][0]),
        },
        "source_package": {"terrains": [f"{name}.meta.json" for name in region["terrains"]]},
        "resampling": {
            "method": f"each cell centre is projected from {region['explorer_crs']} into {region['source_crs']} and takes the bilinear mean of the NSIDC ocean cells among its four neighbours, weights renormalised over those cells",
            "coast": "a cell with no NSIDC ocean neighbour that a terrain lookup can read takes the mean of its filled 8-neighbours, one ring per pass, so coastal water NSIDC masks as land or coast carries the nearest observed concentration",
            "reason": "NSIDC's grids are on the Hughes 1980 ellipsoid with their own latitude of true scale; the explorer draws this coarser grid as a texture on the sea surface and clips it to each terrain's ocean, so the grid need not match a terrain grid",
            "latitude_longitude": "taken as the same on the Hughes 1980 ellipsoid and WGS84, as NSIDC's grids assume",
        },
        "domain": {
            "cell_count": count,
            "order": "row-major order of the non-zero cells of domain_mask",
            "definition": "cells a texture lookup on any terrain's ocean can read (the four cells around each ocean node and one ring more) that hold ice in at least one month; every other cell is open water throughout",
        },
        "frames": {
            "count": len(labels),
            "years": [round(float(v), 4) for v in decimal_years],
            "months": labels,
            "empty_months": empty,
            "empty_note": "months whose NSIDC file holds no data: December 1987 and January 1988 (the SSM/I gap of 3 December 1987 to 12 January 1988) in both hemispheres; the Northern Hemisphere file for July 1984 is empty too",
            "sensors": sensor_runs(labels, out["sensors"]),
            "layout": "frame-major: concentration[frame * cell_count + cell]; the frames are consecutive months, each placed at mid-month",
        },
        "series": {
            "hemisphere": region["hemisphere"],
            "scope": region["hemisphere_name"],
            "threshold_percent": EXTENT_THRESHOLD_PERCENT,
            "reference_period": list(REFERENCE_YEARS),
            "extent_mkm2": rounded_or_none(out["extent"], 4),
            "area_mkm2": rounded_or_none(out["area"], 4),
            "mean_field_extent_mkm2": rounded_or_none(out["mean_field_extent"], 4),
            "monthly": stats,
            "note": "extent is the area of the hemisphere's ocean cells with at least 15 % ice on at least half of the month's days (the files' QA flag), close to the Sea Ice Index's monthly extent, which averages daily extents; area weights the same cells by their monthly mean concentration; both are summed on NSIDC's grid with each cell's area from the projection's scale factor. The map draws the monthly mean concentration, as NSIDC's monthly images do: the area of its cells at 15 % or more, mean_field_extent, is larger, mostly in the marginal ice zone. The trend is the least-squares slope of a calendar month's extent from 1979, also given as a share of its 1981-2010 average per decade",
        },
        "reference_edge": {
            "field": "reference_ice_frequency",
            "period": list(REFERENCE_YEARS),
            "threshold_percent": EXTENT_THRESHOLD_PERCENT,
            "edge_percent": MEDIAN_EDGE_PERCENT,
            "note": "the median ice edge of a calendar month: where the cells that held at least 15 % ice in at least half of the 1981-2010 years give way to those that did not",
        },
        "visualization": {"min_drawn_percent": EXTENT_THRESHOLD_PERCENT},
        "validation": {
            "projection": ancillary_check,
            "filled_cells": {
                "per_month_max": int(filled.max()),
                "per_month_median": round(float(np.median(filled)), 1),
                "share_of_domain_max": round(float(filled.max() / count), 4),
            },
            **({"sea_ice_index": index} if index else {}),
        },
        "source_files": out["files"],
        "fields": fields,
    }
    return payload, meta


def pack_fields(specs: list[dict]) -> tuple[bytes, list[dict]]:
    """Concatenate field blobs in order and describe each (offsets tile the payload)."""
    blobs, fields, offset = [], [], 0
    for spec in specs:
        blob = spec.pop("blob")
        fields.append({**spec, "byte_offset": offset, "byte_length": len(blob)})
        blobs.append(blob)
        offset += len(blob)
    return b"".join(blobs), fields


def write_package(output_dir: Path, basename: str, payload: bytes, meta: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{basename}.bin").write_bytes(payload)
    (output_dir / f"{basename}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def check_projection(ancillary: dict, region: dict) -> dict:
    """Largest departure of the projection's inverse from NSIDC's own cell latitudes and
    longitudes (degrees), away from the pole where longitude is undefined."""
    projection = PROJECTIONS[region["source_crs"]]
    gx, gy = np.meshgrid(ancillary["x"], ancillary["y"])
    lon, lat = projection.inverse(gx, gy)
    away = np.abs(ancillary["latitude"]) < 89.5
    dlon = (lon - ancillary["longitude"] + 180.0) % 360.0 - 180.0
    worst_lat = float(np.max(np.abs(lat - ancillary["latitude"])))
    worst_lon = float(np.max(np.abs(dlon[away])))
    if worst_lat > 1e-6 or worst_lon > 1e-6:
        raise ValueError(f"{region['source_crs']} does not reproduce NSIDC's cell positions ({worst_lat}, {worst_lon} deg)")
    return {"max_latitude_error_deg": float(f"{worst_lat:.1e}"), "max_longitude_error_deg": float(f"{worst_lon:.1e}"),
            "against": "the latitude and longitude of every cell in NSIDC's ancillary file"}


# ---------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cdr-dir", type=Path, required=True,
                        help="directory with north/monthly, south/monthly (and their checksums/) and ancillary/")
    parser.add_argument("--sea-ice-index-dir", type=Path, default=None,
                        help="optional: directory with north/N_MM_extent_v4.0.csv and south/S_MM_extent_v4.0.csv, for validation")
    parser.add_argument("--terrain-dir", type=Path, default=Path("static/tools/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("static/tools/data"))
    parser.add_argument("--region", choices=sorted(REGIONS), action="append", help="default: both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for region_key in args.region or sorted(REGIONS):
        region = REGIONS[region_key]
        ancillary = read_ancillary(args.cdr_dir / "ancillary" / f"G02202-ancillary-ps{region['file_prefix']}25-v06r00.nc")
        projection_check = check_projection(ancillary, region)
        terrains = [read_terrain_ocean(args.terrain_dir / f"{name}.meta.json") for name in region["terrains"]]
        months = list_months(args.cdr_dir / region["hemisphere"] / "monthly", region["file_prefix"])
        out = build_frames(months, region, terrains, ancillary)
        index = None
        if args.sea_ice_index_dir:
            extents = read_sea_ice_index(args.sea_ice_index_dir / region["hemisphere"], region["index_prefix"])
            index = compare_with_index(out["years"], out["months"], out["extent"], extents, out["mean_field_extent"])
        payload, meta = build_package(out, region_key, projection_check, index)
        write_package(args.output_dir, region["basename"], payload, meta)
        print(f"{region['basename']}: {meta['frames']['count']} months x {meta['domain']['cell_count']} cells "
              f"on {meta['grid']['nx']} x {meta['grid']['ny']}, {len(payload) / 1e6:.1f} MB; "
              f"empty {', '.join(meta['frames']['empty_months'])}")


if __name__ == "__main__":
    main()
