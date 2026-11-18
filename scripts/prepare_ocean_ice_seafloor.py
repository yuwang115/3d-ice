#!/usr/bin/env python3
"""Pack the seafloor water of the OCEAN ICE Southern Ocean climatology for 3D ICE.

Input: the OCEAN ICE Southern Ocean Climatology (Zhou, Dutrieux & Giulivi, 2024; SEANOE,
doi:10.17882/103946; CC BY 4.0), version 2: annual-mean Conservative Temperature (CT) and
Absolute Salinity (SA) south of 45 deg S on a 0.2 deg x 0.1 deg (longitude x latitude) grid at
10 dbar from 5 to 6005 dbar, objectively mapped from about 819,000 ship CTD, Argo and
instrumented-seal profiles (Zhou et al., 2026, Earth System Science Data Discussions,
doi:10.5194/essd-2025-727). Two of its three products are packed:

- the climatology of every profile, 1972-2023 (`OI_Climatology.nc`), and
- the climatology of the profiles since 2000 (`OI_2000_Climatology.nc`), when Argo floats and
  seals took over the sampling.

The seafloor water of a column is the mean of its lowest 150 dbar, the definition of
Schmidtko et al. (2014) that Zhou et al. use for the shelf's bottom water: the deepest level
with data and the levels less than 150 dbar above it (15 levels at 10 dbar). The climatology's
seafloor is its own (RTopo-2.0.4 bathymetry), not the explorer's.

The full climatology writes a modified standard error of 1e10 where no profile lies inside the
mapping ellipse at a level, and its value there comes from profiles further along the isobaths
(the fast-marching search). `support` is the share of a column's seafloor levels that had a
profile inside the ellipse; the explorer fades the cells where it is below one half. The
post-2000 climatology does not mark such levels (its error is finite everywhere), so both periods
are faded by the full record's support: where no profile of any year lies nearby. For the
post-2000 view that is the least it should fade, since its profiles are a subset.

The fields are resampled onto the explorer's 10 km BedMachine grid, which every Antarctic terrain
samples bilinearly: each cell centre is projected from EPSG:3031 to latitude and longitude and
takes the bilinear mean of the climatology columns among its four neighbours, weights
renormalised over the columns with data. Only cells that a lookup from some terrain's open ocean
can read are kept. Ice-shelf cavities are left out on purpose: the climatology extrapolates into
some of them, but no profile was taken there.

Output: `ocean_ice_seafloor_antarctica`, a sparse grid (`ct_full`, `sa_full`, `ct_post2000`,
`sa_post2000`, `bottom_pressure`, `support`, `domain_mask`).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from prepare_nsidc_sea_ice import (
    PROJECTIONS,
    BilinearLookup,
    axes,
    pack_fields,
    read_terrain_ocean,
    terrain_reach,
    write_package,
)

WINDOW_DBAR = 150.0
NO_SUPPORT_MSE = 1e9
FADE_BELOW_SUPPORT_PERCENT = 50
CT_SCALE = 0.001
SA_SCALE = 0.0001
SA_OFFSET = 34.0
INT16_FILL = -32768
UINT8_FILL = 255
CHUNK_LEVELS = 25
BASENAME = "ocean_ice_seafloor_antarctica"
GRID_PACKAGE = "bedmachine_antarctica_v4_480"
TERRAINS = (
    "bedmachine_antarctica_v4_480",
    "bedmachine_antarctica_v4_741",
    "bedmap3_antarctica_10km",
    "bedmap3_antarctica_4km",
)
PERIODS = (
    {"key": "full", "label": "1972-2023", "description": "every profile, 1972-2023 (the OCEAN ICE full climatology)"},
    {"key": "post2000", "label": "2000-2023", "description": "the profiles since 2000 (the OCEAN ICE post-2000 climatology), when Argo floats and instrumented seals took over the sampling"},
)
DATA_SET = {
    "title": "The OCEAN ICE Southern Ocean Climatology",
    "creators": ["Zhou, Shenjie", "Dutrieux, Pierre", "Giulivi, Claudia"],
    "publisher": "SEANOE",
    "year": 2024,
    "version": "2 (annual mean, created 16 October 2025)",
    "doi": "10.17882/103946",
    "url": "https://doi.org/10.17882/103946",
    "license": "CC BY 4.0",
    "citation": "Zhou, S., Dutrieux, P. and Giulivi, C. (2024). The OCEAN ICE Southern Ocean Climatology. SEANOE. https://doi.org/10.17882/103946",
    "article": "Zhou, S., Dutrieux, P., Giulivi, C., Meijers, A., Lee, W. S., Kim, T.-W., Hattermann, T. and Janout, M. (2026). The OCEAN ICE hydrography profiles compilation and climatology. Earth System Science Data Discussions. https://doi.org/10.5194/essd-2025-727",
    "profiles": "Zhou, S., Dutrieux, P., Giulivi, C., Lee, W. S., Kim, T.-W., Hattermann, T. and Janout, M. (2026). Southern Ocean (90S-45S) conservative temperature and absolute salinity profiles compilation (OCEAN ICE D1.1). SEANOE. https://doi.org/10.17882/99787",
}
# Named places whose packed seafloor water the metadata records, for readers and tests.
CHECK_SITES = (
    ("Pine Island Bay", -74.9, -102.5),
    ("Pine Island Trough, mid-shelf", -73.5, -106.5),
    ("Dotson Ice Shelf front", -74.2, -112.5),
    ("Belgica Trough", -70.5, -86.0),
    ("Ross Sea shelf", -75.5, 172.0),
    ("Filchner Trough", -76.5, -33.0),
    ("Prydz Bay", -67.5, 75.0),
    ("Southern Ocean abyss, 60 S 140 E", -60.0, 140.0),
)


# ----------------------------------------------------------------------------- source


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def bottom_pressures(pressure: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Pressure of each column's deepest level with data (`bottom` is its index, -1 for none)."""
    return np.where(bottom >= 0, pressure[np.clip(bottom, 0, None)], np.nan)


def window_levels(level_pressure: np.ndarray, p_bottom: np.ndarray, window_dbar: float = WINDOW_DBAR) -> np.ndarray:
    """Whether each level (axis 0, at `level_pressure`) lies in its column's seafloor window: at
    or above the column's deepest level with data, at `p_bottom` (NaN for none), and less than
    `window_dbar` above it."""
    levels = np.asarray(level_pressure, dtype=np.float64).reshape((-1,) + (1,) * p_bottom.ndim)
    with np.errstate(invalid="ignore"):
        return (levels <= p_bottom) & (levels > p_bottom - window_dbar)


def deepest_levels(valid_chunks) -> np.ndarray:
    """Index of the deepest level with data in each column, -1 where there is none, from an
    iterable of (first level index, boolean block of shape (levels, ...))."""
    bottom = None
    for first, valid in valid_chunks:
        if bottom is None:
            bottom = np.full(valid.shape[1:], -1, dtype=np.int32)
        any_valid = valid.any(axis=0)
        last = first + valid.shape[0] - 1 - np.argmax(valid[::-1], axis=0)
        bottom = np.where(any_valid, last.astype(np.int32), bottom)
    if bottom is None:
        raise ValueError("the climatology has no levels")
    return bottom


def seafloor_means(pressure, bottom, chunks, window_dbar: float = WINDOW_DBAR) -> dict:
    """Mean CT and SA over each column's seafloor window, and the share of the window's levels
    with a profile inside the mapping ellipse, from an iterable of (first level index, ct, sa,
    ct_mse) blocks of shape (levels, ...)."""
    shape = bottom.shape
    p_bottom = bottom_pressures(pressure, bottom)
    sum_ct = np.zeros(shape)
    sum_sa = np.zeros(shape)
    count = np.zeros(shape, dtype=np.int32)
    supported = np.zeros(shape, dtype=np.int32)
    for first, ct, sa, mse in chunks:
        inside = window_levels(pressure[first:first + ct.shape[0]], p_bottom, window_dbar)
        use = inside & np.isfinite(ct) & np.isfinite(sa)
        sum_ct += np.where(use, ct, 0.0).sum(axis=0)
        sum_sa += np.where(use, sa, 0.0).sum(axis=0)
        count += use.sum(axis=0)
        with np.errstate(invalid="ignore"):
            supported += (use & np.isfinite(mse) & (mse < NO_SUPPORT_MSE)).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        has = count > 0
        return {
            "ct": np.where(has, sum_ct / np.maximum(count, 1), np.nan),
            "sa": np.where(has, sum_sa / np.maximum(count, 1), np.nan),
            "support": np.where(has, supported / np.maximum(count, 1), np.nan),
            "levels": count,
            "bottom_pressure": p_bottom,
        }


def read_climatology(path: Path, window_dbar: float = WINDOW_DBAR, chunk: int = CHUNK_LEVELS) -> dict:
    """Seafloor CT, SA, support and pressure of every column of an OCEAN ICE climatology file,
    as (latitude, longitude) arrays with increasing axes."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        lat = np.asarray(ds["latitude"][:], dtype=np.float64).ravel()
        lon = np.asarray(ds["longitude"][:], dtype=np.float64).ravel()
        pressure = np.asarray(ds["pressure"][:], dtype=np.float64).ravel()
        ct_var, sa_var, mse_var = ds["ct"], ds["sa"], ds["ct_mse"]
        if ct_var.dimensions[0] != "nz" or ct_var.shape != (pressure.size, lon.size, lat.size):
            raise ValueError(f"unexpected climatology layout {ct_var.dimensions} {ct_var.shape}")
        nz = pressure.size
        starts = range(0, nz, chunk)
        bottom = deepest_levels((k, np.isfinite(ct_var[k:k + chunk])) for k in starts)
        means = seafloor_means(
            pressure,
            bottom,
            ((k, ct_var[k:k + chunk], sa_var[k:k + chunk], mse_var[k:k + chunk]) for k in starts),
            window_dbar,
        )
        attrs = {name: str(getattr(ds, name)) for name in ds.ncattrs()}
    # (nx = longitude, ny = latitude) to rows of latitude and columns of longitude.
    out = {key: np.ascontiguousarray(value.T) for key, value in means.items()}
    out.update({"lat": lat, "lon": lon, "pressure": pressure, "attrs": attrs})
    if np.any(np.diff(lat) <= 0) or np.any(np.diff(lon) <= 0):
        raise ValueError("the climatology's axes must increase")
    return out


# ---------------------------------------------------------------------------- resampling


def explorer_lookup(columns: dict, grid: dict) -> BilinearLookup:
    """Where each explorer cell centre falls among the climatology's columns."""
    tx, ty = axes(grid)
    gx, gy = np.meshgrid(tx, ty)
    lon, lat = PROJECTIONS["EPSG:3031"].inverse(gx, gy)
    return BilinearLookup.build(columns["lon"], columns["lat"], lon, lat)


def resample(columns: dict, lookup: BilinearLookup, grid: dict) -> dict:
    """The seafloor fields on the explorer grid (NaN where no neighbouring column has data)."""
    valid = np.isfinite(columns["ct"]) & np.isfinite(columns["sa"])
    shape = (grid["ny"], grid["nx"])
    return {
        key: lookup.mean(np.where(valid, columns[key], 0.0), valid).reshape(shape)
        for key in ("ct", "sa", "support", "bottom_pressure")
    }


def sample_columns(columns: dict, lat: float, lon: float) -> dict:
    """The seafloor water of the column nearest a latitude and longitude."""
    row = int(np.argmin(np.abs(columns["lat"] - lat)))
    col = int(np.argmin(np.abs(columns["lon"] - lon)))
    return {key: float(columns[key][row, col]) for key in ("ct", "sa", "support", "bottom_pressure")}


# ----------------------------------------------------------------------------- packing


def quantize(values: np.ndarray, scale: float, offset: float = 0.0) -> np.ndarray:
    """int16 codes of values (round half to even), INT16_FILL where a value is missing."""
    array = np.asarray(values, dtype=np.float64)
    codes = np.full(array.shape, INT16_FILL, dtype=np.int16)
    good = np.isfinite(array)
    scaled = np.rint((array[good] - offset) / scale)
    if scaled.size and (scaled.min() < -32767 or scaled.max() > 32767):
        raise ValueError("values exceed the int16 code range")
    codes[good] = scaled.astype(np.int16)
    return codes


def percent_codes(fraction: np.ndarray) -> np.ndarray:
    array = np.asarray(fraction, dtype=np.float64)
    codes = np.full(array.shape, UINT8_FILL, dtype=np.uint8)
    good = np.isfinite(array)
    codes[good] = np.clip(np.rint(array[good] * 100.0), 0, 100).astype(np.uint8)
    return codes


def value_stats(values: np.ndarray, digits: int = 4) -> dict:
    good = np.asarray(values, dtype=np.float64)
    good = good[np.isfinite(good)]
    return {"min": round(float(good.min()), digits), "max": round(float(good.max()), digits), "mean": round(float(good.mean()), digits)}


def int16_field(name: str, values: np.ndarray, unit: str, scale: float, offset: float, note: str) -> dict:
    key = "stats_" + {"degC": "degc", "g/kg": "g_per_kg", "dbar": "dbar"}[unit]
    return {
        "name": name,
        "dtype": "int16",
        "unit": unit,
        "scale": scale,
        "offset": offset,
        "fill_value": INT16_FILL,
        key: value_stats(values),
        "note": note,
        "blob": quantize(values, scale, offset).tobytes(),
    }


def percent_field(name: str, fraction: np.ndarray, note: str) -> dict:
    codes = percent_codes(fraction)
    present = codes[codes != UINT8_FILL].astype(np.float64)
    return {"name": name, "dtype": "uint8", "unit": "percent", "fill_value": UINT8_FILL,
            "stats_percent": value_stats(present, 2), "note": note, "blob": codes.tobytes()}


def domain_cells(fields: dict, reach: np.ndarray) -> np.ndarray:
    return reach & np.isfinite(fields["ct"]) & np.isfinite(fields["sa"])


def build_package(periods: dict, domain: np.ndarray, grid: dict, checks: list, sources: list) -> tuple[bytes, dict]:
    count = int(domain.sum())
    if count == 0:
        raise ValueError("no explorer ocean cell has seafloor water")
    full, post = periods["full"], periods["post2000"]
    if not np.all(np.isfinite(full["ct"][domain])):
        raise ValueError("a domain cell has no full-record value")
    pressure = full["bottom_pressure"][domain]
    specs = [
        int16_field("ct_full", full["ct"][domain], "degC", CT_SCALE, 0.0,
                    "Conservative Temperature of the seafloor water (mean of the column's lowest 150 dbar), every profile 1972-2023, one value per domain cell"),
        int16_field("sa_full", full["sa"][domain], "g/kg", SA_SCALE, SA_OFFSET,
                    "Absolute Salinity of the seafloor water, every profile 1972-2023, one value per domain cell"),
        int16_field("ct_post2000", post["ct"][domain], "degC", CT_SCALE, 0.0,
                    "as ct_full, from the profiles since 2000; fill where that climatology has no column"),
        int16_field("sa_post2000", post["sa"][domain], "g/kg", SA_SCALE, SA_OFFSET,
                    "as sa_full, from the profiles since 2000; fill where that climatology has no column"),
        int16_field("bottom_pressure", pressure, "dbar", 1.0, 0.0,
                    "pressure of the climatology's deepest level with data (its own RTopo-2.0.4 seafloor), full record"),
        percent_field("support", full["support"][domain],
                      "percent of the seafloor levels with a profile of any year inside the mapping ellipse (full record's modified standard error below 1e9); the explorer fades cells below 50 in both periods"),
        {"name": "domain_mask", "dtype": "uint8",
         "flags": {"0": "outside the domain (no seafloor water, or out of reach of every terrain's open ocean)", "1": "domain cell"},
         "blob": domain.astype(np.uint8).tobytes()},
    ]
    payload, fields = pack_fields(specs)
    support = full["support"][domain]
    low_support = round(float(np.mean(support[np.isfinite(support)] < FADE_BELOW_SUPPORT_PERCENT / 100)), 4)
    differs = np.abs(post["ct"][domain] - full["ct"][domain])
    differs = differs[np.isfinite(differs)]
    period_difference = {
        "median_abs_ct_degc": round(float(np.median(differs)), 4),
        "share_abs_ct_above_0_05_degc": round(float(np.mean(differs > 0.05)), 4),
        "share_abs_ct_above_0_2_degc": round(float(np.mean(differs > 0.2)), 4),
        "note": "post-2000 minus full record, seafloor Conservative Temperature over the domain; the two products differ by their sampling as much as by any change, so the explorer offers no difference map",
    }
    meta = {
        "title": "Seafloor water around Antarctica from the OCEAN ICE Southern Ocean Climatology (annual mean; every profile 1972-2023, and the profiles since 2000)",
        "product_version": f"{DATA_SET['title']}, version {DATA_SET['version']}",
        "reference": DATA_SET["citation"],
        "license": DATA_SET["license"],
        "source_url": DATA_SET["url"],
        "source_dataset": DATA_SET,
        "geometry_type": "sparse_grid",
        "field_kind": "observed_ocean_seafloor",
        "region": "antarctica",
        "crs": "EPSG:3031",
        "grid": dict(grid),
        "source_grid": {
            "description": "OCEAN ICE climatology: 0.2 deg longitude x 0.1 deg latitude, 180 W-180 E, 90 S-45 S, 601 levels at 10 dbar from 5 to 6005 dbar",
            "longitude_step_deg": 0.2,
            "latitude_step_deg": 0.1,
        },
        "source_package": {"grid": f"{GRID_PACKAGE}.meta.json", "terrains": [f"{name}.meta.json" for name in TERRAINS]},
        "seafloor": {
            "definition": "mean of each climatology column's lowest 150 dbar: the deepest level with data and the levels less than 150 dbar above it (15 levels at 10 dbar)",
            "after": "Schmidtko et al. (2014), Science 346, 1227-1231, doi:10.1126/science.1256117; Zhou et al. (2026), Sect. 4.2",
            "window_dbar": WINDOW_DBAR,
            "bathymetry": "the climatology's own seafloor (RTopo-2.0.4), not the explorer's",
        },
        "support": {
            "field": "support",
            "definition": "share of the seafloor levels whose modified standard error in the full climatology is below 1e9: it writes 1e10 where no profile lies inside the mapping ellipse, and fills the value from profiles further along the isobaths",
            "post2000": "the post-2000 climatology's error is finite everywhere and marks no such levels, so its view is faded by the full record's support, where no profile of any year lies nearby: the least it should fade",
            "fade_below_percent": FADE_BELOW_SUPPORT_PERCENT,
            "share_of_domain_below_fade": low_support,
        },
        "periods": [
            {**period, "fields": {"ct": f"ct_{period['key']}", "sa": f"sa_{period['key']}"}}
            for period in PERIODS
        ],
        "period_difference": period_difference,
        "resampling": {
            "method": "each cell centre is projected from EPSG:3031 to latitude and longitude (WGS84) and takes the bilinear mean of the climatology columns among its four neighbours, weights renormalised over the columns with data",
            "reason": "one package on the 10 km BedMachine grid serves every Antarctic terrain: the explorer samples it bilinearly at each terrain's own cells",
            "cavities": "only cells a lookup from some terrain's open ocean (mask class 0) can read are kept; ice-shelf cavities are left out because no profile was taken in them",
        },
        "domain": {
            "cell_count": count,
            "order": "row-major order of the non-zero cells of domain_mask",
            "definition": "cells that a bilinear lookup from the open ocean of any Antarctic terrain can read (the four cells around each ocean node and one ring more) and whose full-record seafloor water is defined",
        },
        "visualization": {
            "temperature_range_degc": [-2.0, 2.0],
            "salinity_range_g_per_kg": [34.2, 35.0],
        },
        "validation": {"sites": checks},
        "source_files": sources,
        "fields": fields,
    }
    return payload, meta


# --------------------------------------------------------------------------------- main


def site_checks(periods: dict, columns: dict, domain: np.ndarray, grid: dict) -> list:
    """The packed seafloor water at the named sites next to the climatology's own column."""
    out = []
    projection = PROJECTIONS["EPSG:3031"]
    tx, ty = axes(grid)
    for name, lat, lon in CHECK_SITES:
        x, y = projection.forward(lon, lat)
        col = int(np.argmin(np.abs(tx - x)))
        row = int(np.argmin(np.abs(ty - y)))
        column = sample_columns(columns, lat, lon)
        entry = {"name": name, "lat": lat, "lon": lon, "column_ct_degc": round(column["ct"], 3),
                 "column_sa_g_per_kg": round(column["sa"], 4), "column_bottom_dbar": round(column["bottom_pressure"], 1),
                 "in_domain": bool(domain[row, col])}
        if domain[row, col]:
            entry.update({"packed_ct_full_degc": round(float(periods["full"]["ct"][row, col]), 3),
                          "packed_sa_full_g_per_kg": round(float(periods["full"]["sa"][row, col]), 4),
                          "packed_ct_post2000_degc": round(float(periods["post2000"]["ct"][row, col]), 3)})
        out.append(entry)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--full", type=Path, required=True, help="OI_Climatology.nc (every profile)")
    parser.add_argument("--post2000", type=Path, required=True, help="OI_2000_Climatology.nc (profiles since 2000)")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "static" / "tools" / "data")
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to --data-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or args.data_dir
    grid, _ = read_terrain_ocean(args.data_dir / f"{GRID_PACKAGE}.meta.json")
    terrains = [read_terrain_ocean(args.data_dir / f"{name}.meta.json") for name in TERRAINS]
    reach = terrain_reach(grid, terrains)
    periods, sources, full_columns, lookup = {}, [], None, None
    for period, path in (("full", args.full), ("post2000", args.post2000)):
        print(f"reading {path.name} ...", flush=True)
        columns = read_climatology(path)
        lookup = lookup or explorer_lookup(columns, grid)
        periods[period] = resample(columns, lookup, grid)
        full_columns = full_columns or columns
        sources.append({"period": period, "name": path.name, "sha256": sha256_of(path),
                        "description": columns["attrs"].get("Description", ""),
                        "created": columns["attrs"].get("Creation Date", "")})
    domain = domain_cells(periods["full"], reach)
    checks = site_checks(periods, full_columns, domain, grid)
    payload, meta = build_package(periods, domain, grid, checks, sources)
    write_package(out_dir, BASENAME, payload, meta)
    print(f"wrote {BASENAME}: {meta['domain']['cell_count']} cells, {len(payload) / 1e6:.2f} MB")
    for entry in checks:
        print("  ", entry)


if __name__ == "__main__":
    main()
