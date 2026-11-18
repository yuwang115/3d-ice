#!/usr/bin/env python3
"""Pack the OCEAN ICE hydrography profile compilation (where and when the ocean was measured)
for 3D ICE.

Input: the Southern Ocean (90S-45S) Conservative Temperature and Absolute Salinity profiles
compilation (OCEAN ICE D1.1; Zhou et al., 2026; SEANOE, doi:10.17882/99787; CC BY 4.0), 1970-2025:
quality-controlled, de-duplicated ship CTD, Argo float and instrumented-seal (MEOP) profiles,
bin-averaged at 10 m from 0 to 6000 m, in two files of the same profiles, one of Conservative
Temperature and one of Absolute Salinity (Zhou et al., 2026, Earth System Science Data
Discussions, doi:10.5194/essd-2025-727).

The explorer draws every profile inside its Antarctic domain as a point on the sea surface, one
year at a time or all at once, coloured by platform, and shows a profile's measurements when it
is clicked. Two packages are written:

- `ocean_ice_profiles_antarctica`, the points, sorted by date: position, day, platform, the
  depth of the deepest measurement and the temperature and salinity there. Profiles the
  compilation flags as bad (CODC-QC) or that hold no temperature are left out.
- `ocean_ice_profiles_antarctica_curves`, loaded on the first click: each profile's temperature
  and salinity at the standard depths above its deepest measurement, then at that measurement.
  A standard depth takes the nearest bin with data within 20 m.

Outputs: the two packages above (`geometry_type` "points").
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path

import numpy as np

from prepare_nsidc_sea_ice import PROJECTIONS, pack_fields, write_package

DOMAIN_HALF_WIDTH_M = 3333000.0
XY_SCALE_M = 250.0
CT_SCALE = 0.001
SA_SCALE = 0.0005
SA_OFFSET = 34.0
INT16_FILL = -32768
STANDARD_DEPTHS_M = (10, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 4000, 5000)
SEARCH_LEVELS = 2  # a standard depth takes the nearest bin with data within 2 bins (20 m)
CHUNK_LEVELS = 40
EPOCH_1950 = dt.date(1950, 1, 1)
EPOCH_1970 = dt.date(1970, 1, 1)
DAYS_1950_TO_1970 = (EPOCH_1970 - EPOCH_1950).days
PLATFORMS = {1: "ship CTD", 2: "Argo float", 3: "instrumented seal (MEOP)"}
BASENAME = "ocean_ice_profiles_antarctica"
DATA_SET = {
    "title": "Southern Ocean (90S-45S) conservative temperature and absolute salinity profiles compilation (OCEAN ICE D1.1)",
    "creators": ["Zhou, Shenjie", "Dutrieux, Pierre", "Giulivi, Claudia", "Lee, Won Sang", "Kim, Tae-Wan", "Hattermann, Tore", "Janout, Markus"],
    "publisher": "SEANOE",
    "year": 2026,
    "version": "1970-2025 (published August 2026)",
    "doi": "10.17882/99787",
    "url": "https://doi.org/10.17882/99787",
    "license": "CC BY 4.0",
    "citation": "Zhou, S., Dutrieux, P., Giulivi, C., Lee, W. S., Kim, T.-W., Hattermann, T. and Janout, M. (2026). Southern Ocean (90S-45S) conservative temperature and absolute salinity profiles compilation (OCEAN ICE D1.1). SEANOE. https://doi.org/10.17882/99787",
    "article": "Zhou, S., Dutrieux, P., Giulivi, C., Meijers, A., Lee, W. S., Kim, T.-W., Hattermann, T. and Janout, M. (2026). The OCEAN ICE hydrography profiles compilation and climatology. Earth System Science Data Discussions. https://doi.org/10.5194/essd-2025-727",
}


# ------------------------------------------------------------------------------ reading


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def deepest(chunks, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Index of each profile's deepest bin with data (-1 for none) and the value there, from an
    iterable of (first level, block of shape (levels, profiles))."""
    index = np.full(count, -1, dtype=np.int32)
    value = np.full(count, np.nan)
    for first, block in chunks:
        valid = np.isfinite(block)
        any_valid = valid.any(axis=0)
        last = valid.shape[0] - 1 - np.argmax(valid[::-1], axis=0)
        index = np.where(any_valid, first + last, index)
        picked = block[last, np.arange(count)]
        value = np.where(any_valid, picked, value)
    return index, value


def nearest_with_data(rows: np.ndarray, centre: int) -> np.ndarray:
    """Per profile, the value of the row nearest `centre` that has data (NaN for none); rows is
    (levels, profiles) around the standard depth."""
    out = np.full(rows.shape[1], np.nan)
    order = sorted(range(rows.shape[0]), key=lambda k: (abs(k - centre), k))
    for k in reversed(order):
        out = np.where(np.isfinite(rows[k]), rows[k], out)
    return out


def standard_values(variable, levels: np.ndarray, count: int, search: int = SEARCH_LEVELS) -> np.ndarray:
    """(standard depths, profiles): the value at each standard depth, from the nearest bin with
    data within `search` bins."""
    out = np.full((len(STANDARD_DEPTHS_M), count), np.nan)
    for row, depth in enumerate(STANDARD_DEPTHS_M):
        centre = int(np.argmin(np.abs(levels - depth)))
        if abs(levels[centre] - depth) > 5.0:
            continue  # the bins do not reach this depth
        first = max(0, centre - search)
        last = min(levels.size, centre + search + 1)
        out[row] = nearest_with_data(np.asarray(variable[first:last, :], dtype=np.float64), centre - first)
    return out


# Values outside these are a handful of spurious bins (Conservative Temperature from -12 to
# 72 degC and Absolute Salinity from 0 g/kg in the 2026 release) and are read as missing.
PLAUSIBLE = {"ct": (-3.0, 30.0), "sa": (20.0, 42.0)}


class PlausibleBins:
    """A (levels, profiles) variable that reads implausible values as missing."""

    def __init__(self, variable, bounds: tuple[float, float]):
        self.variable = variable
        self.low, self.high = bounds
        self.shape = variable.shape
        self.masked = 0

    def __getitem__(self, item):
        block = np.asarray(self.variable[item], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            bad = np.isfinite(block) & ((block < self.low) | (block > self.high))
        self.masked += int(bad.sum())
        return np.where(bad, np.nan, block)


def read_file(path: Path, value_name: str, chunk: int = CHUNK_LEVELS) -> dict:
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        lat = np.asarray(ds["lat"][:], dtype=np.float64).ravel()
        lon = np.asarray(ds["lon"][:], dtype=np.float64).ravel()
        days = np.asarray(ds["dyr"][:], dtype=np.float64).ravel()
        levels = np.asarray(ds["pres"][:], dtype=np.float64).ravel()
        raw = ds[value_name]
        if raw.shape != (levels.size, lat.size):
            raise ValueError(f"unexpected {value_name} layout {raw.dimensions} {raw.shape}")
        count = lat.size
        variable = PlausibleBins(raw, PLAUSIBLE[value_name])
        index, bottom = deepest(((k, variable[k:k + chunk, :]) for k in range(0, levels.size, chunk)), count)
        masked = variable.masked
        standard = standard_values(variable, levels, count)
        out = {"lat": lat, "lon": lon, "days": days, "levels": levels, "deepest": index, "bottom": bottom,
               "standard": standard, "masked_bins": masked}
        for name in ("CODC_QCflag", "instr_flag"):
            if name in ds.variables:
                out[name] = np.asarray(ds[name][:], dtype=np.float64).ravel()
    return out


# ------------------------------------------------------------------------------ packing


def select_profiles(t: dict, s: dict | None = None) -> np.ndarray:
    """Indices of the profiles to pack, sorted by date: inside the explorer's domain, flagged bad
    in neither file, with temperature data and a known platform."""
    x, y = PROJECTIONS["EPSG:3031"].forward(t["lon"], t["lat"])
    inside = (np.abs(x) <= DOMAIN_HALF_WIDTH_M) & (np.abs(y) <= DOMAIN_HALF_WIDTH_M)
    good = t.get("CODC_QCflag", np.zeros_like(t["lat"])) == 0
    if s is not None and "CODC_QCflag" in s:
        good &= s["CODC_QCflag"] == 0
    platform = t.get("instr_flag", np.zeros_like(t["lat"]))
    known = np.isin(platform, list(PLATFORMS))
    has = t["deepest"] >= 0
    dated = np.isfinite(t["days"]) & (t["days"] >= DAYS_1950_TO_1970) & (t["days"] < DAYS_1950_TO_1970 + 65535)
    keep = np.flatnonzero(inside & good & known & has & dated)
    return keep[np.argsort(t["days"][keep], kind="stable")]


def quantize(values: np.ndarray, scale: float, offset: float = 0.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    codes = np.full(array.shape, INT16_FILL, dtype=np.int16)
    good = np.isfinite(array)
    scaled = np.rint((array[good] - offset) / scale)
    if scaled.size and (scaled.min() < -32767 or scaled.max() > 32767):
        raise ValueError("values exceed the int16 code range")
    codes[good] = scaled.astype(np.int16)
    return codes


def curve_lengths(max_depth: np.ndarray) -> np.ndarray:
    """Values in each profile's curve: the standard depths above its deepest measurement, plus it."""
    standard = np.asarray(STANDARD_DEPTHS_M, dtype=np.float64)
    return (standard[None, :] < max_depth[:, None]).sum(axis=1) + 1


def build_curves(standard: np.ndarray, bottom: np.ndarray, max_depth: np.ndarray) -> np.ndarray:
    """Concatenated curves: for each profile, its standard-depth values above its deepest
    measurement, then that measurement. `standard` is (standard depths, profiles)."""
    lengths = curve_lengths(max_depth)
    out = np.empty(int(lengths.sum()))
    cursor = 0
    for i, length in enumerate(lengths):
        out[cursor:cursor + length - 1] = standard[: length - 1, i]
        out[cursor + length - 1] = bottom[i]
        cursor += length
    return out


def int16_field(name, values, unit, scale, offset, note, key):
    good = np.asarray(values, dtype=np.float64)
    good = good[np.isfinite(good)]
    stats = {"min": round(float(good.min()), 4), "max": round(float(good.max()), 4), "mean": round(float(good.mean()), 4)} if good.size else None
    field = {"name": name, "dtype": "int16", "unit": unit, "scale": scale, "offset": offset, "fill_value": INT16_FILL,
             "note": note, "blob": quantize(values, scale, offset).tobytes()}
    if stats:
        field[key] = stats
    return field


def year_index(days_since_1970: np.ndarray) -> dict:
    """First point of each year (points are sorted by date), from the first year to the last."""
    years = np.array([(EPOCH_1970 + dt.timedelta(days=int(d))).year for d in days_since_1970])
    first, last = int(years.min()), int(years.max())
    starts = [int(np.searchsorted(years, year, side="left")) for year in range(first, last + 2)]
    return {"first_year": first, "last_year": last, "starts": starts, "years": years}


def counts_by_year(years: np.ndarray, platform: np.ndarray, first: int, last: int) -> dict:
    return {
        str(code): [int(np.sum((years == year) & (platform == code))) for year in range(first, last + 1)]
        for code in PLATFORMS
    }


def build_point_package(t: dict, s: dict, keep: np.ndarray) -> tuple[bytes, dict, dict]:
    x, y = PROJECTIONS["EPSG:3031"].forward(t["lon"][keep], t["lat"][keep])
    day = np.floor(t["days"][keep] - DAYS_1950_TO_1970).astype(np.int64)
    platform = t["instr_flag"][keep].astype(np.uint8)
    max_depth = t["levels"][t["deepest"][keep]]
    bottom_sa = np.where(s["deepest"][keep] == t["deepest"][keep], s["bottom"][keep], np.nan)
    index = year_index(day)
    specs = [
        int16_field("x", x, "m", XY_SCALE_M, 0.0, "EPSG:3031 x of the profile", "stats_m"),
        int16_field("y", y, "m", XY_SCALE_M, 0.0, "EPSG:3031 y of the profile", "stats_m"),
        int16_field("bottom_ct", t["bottom"][keep], "degC", CT_SCALE, 0.0, "Conservative Temperature of the deepest bin with data", "stats_degc"),
        int16_field("bottom_sa", bottom_sa, "g/kg", SA_SCALE, SA_OFFSET, "Absolute Salinity at that bin (fill where the salinity profile ends elsewhere)", "stats_g_per_kg"),
        {"name": "day", "dtype": "uint16", "unit": "days since 1970-01-01",
         "stats_days": {"min": int(day.min()), "max": int(day.max()), "mean": round(float(day.mean()), 1)},
         "note": "day of the profile; points are sorted by it", "blob": day.astype("<u2").tobytes()},
        {"name": "max_depth", "dtype": "uint16", "unit": "m",
         "stats_m": {"min": float(max_depth.min()), "max": float(max_depth.max()), "mean": round(float(max_depth.mean()), 1)},
         "note": "depth of the deepest bin with temperature (10 m bins)", "blob": max_depth.astype("<u2").tobytes()},
        {"name": "platform", "dtype": "uint8", "flags": {str(code): name for code, name in PLATFORMS.items()},
         "blob": platform.tobytes()},
    ]
    payload, fields = pack_fields(specs)
    counts = counts_by_year(index["years"], platform, index["first_year"], index["last_year"])
    meta = {
        "title": f"Ocean temperature and salinity profiles around Antarctica, {index['first_year']}-{index['last_year']} (OCEAN ICE profile compilation)",
        "product_version": f"{DATA_SET['title']}, {DATA_SET['version']}",
        "reference": DATA_SET["citation"],
        "license": DATA_SET["license"],
        "source_url": DATA_SET["url"],
        "source_dataset": DATA_SET,
        "geometry_type": "points",
        "point_kind": "ocean_profiles",
        "region": "antarctica",
        "crs": "EPSG:3031",
        "points": {
            "count": int(keep.size),
            "order": "by date",
            "selection": f"profiles within {DOMAIN_HALF_WIDTH_M / 1000:.0f} km of the pole in EPSG:3031 x and y (the explorer's Antarctic domain), flagged bad by CODC-QC in neither the temperature nor the salinity file, with temperature data",
        },
        "years": {"first": index["first_year"], "last": index["last_year"],
                  "starts": index["starts"],
                  "note": "starts[k] is the first point of year first + k; starts[-1] is the point count"},
        "series": {"counts_by_platform_and_year": counts, "platforms": {str(k): v for k, v in PLATFORMS.items()}},
        "curves_package": f"{BASENAME}_curves",
        "fields": fields,
    }
    return payload, meta, {"max_depth": max_depth}


def build_curve_package(t: dict, s: dict, keep: np.ndarray, max_depth: np.ndarray) -> tuple[bytes, dict]:
    ct = build_curves(t["standard"][:, keep], t["bottom"][keep], max_depth)
    bottom_sa = np.where(s["deepest"][keep] == t["deepest"][keep], s["bottom"][keep], np.nan)
    sa = build_curves(s["standard"][:, keep], bottom_sa, max_depth)
    payload, fields = pack_fields([
        int16_field("ct", ct, "degC", CT_SCALE, 0.0, "Conservative Temperature along each curve, curves concatenated in point order", "stats_degc"),
        int16_field("sa", sa, "g/kg", SA_SCALE, SA_OFFSET, "Absolute Salinity along each curve, in the same order", "stats_g_per_kg"),
    ])
    meta = {
        "title": "Temperature and salinity of each ocean profile around Antarctica at standard depths (OCEAN ICE profile compilation)",
        "product_version": f"{DATA_SET['title']}, {DATA_SET['version']}",
        "reference": DATA_SET["citation"],
        "license": DATA_SET["license"],
        "source_url": DATA_SET["url"],
        "geometry_type": "points",
        "point_kind": "ocean_profile_curves",
        "region": "antarctica",
        "points_package": BASENAME,
        "points": {"count": int(keep.size), "value_count": int(ct.size)},
        "curves": {
            "standard_depths_m": list(STANDARD_DEPTHS_M),
            "layout": "for each point in the points package's order: the values at the standard depths shallower than its max_depth, then the value at max_depth; a curve's length follows from max_depth, so curves need no offsets",
            "search": f"a standard depth takes the nearest 10 m bin with data within {SEARCH_LEVELS * 10} m (fill where none)",
        },
        "fields": fields,
    }
    return payload, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--temperature", type=Path, required=True, help="the compilation's Conservative Temperature file")
    parser.add_argument("--salinity", type=Path, required=True, help="the compilation's Absolute Salinity file")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "static" / "tools" / "data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("reading temperature ...", flush=True)
    t = read_file(args.temperature, "ct")
    print("reading salinity ...", flush=True)
    s = read_file(args.salinity, "sa")
    for key in ("lat", "lon", "days"):
        if not np.array_equal(t[key], s[key], equal_nan=True):
            raise ValueError(f"the temperature and salinity files list different profiles ({key})")
    keep = select_profiles(t, s)
    payload, meta, extra = build_point_package(t, s, keep)
    sources = [{"name": path.name, "sha256": sha256_of(path)} for path in (args.temperature, args.salinity)]
    meta["source_files"] = sources
    meta["validation"] = {
        "implausible_bins_read_as_missing": {"ct": t["masked_bins"], "sa": s["masked_bins"]},
        "plausible_ranges": {"ct_degc": list(PLAUSIBLE["ct"]), "sa_g_per_kg": list(PLAUSIBLE["sa"])},
    }
    write_package(args.output_dir, BASENAME, payload, meta)
    curve_payload, curve_meta = build_curve_package(t, s, keep, extra["max_depth"])
    curve_meta["source_files"] = sources
    write_package(args.output_dir, f"{BASENAME}_curves", curve_payload, curve_meta)
    print(f"wrote {BASENAME}: {keep.size} points, {len(payload) / 1e6:.2f} MB; curves {len(curve_payload) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
