#!/usr/bin/env python3
"""Resample one ISMIP6 Antarctica 2300 run onto the 3D ICE 10 km grid at keyframe years.

For every keyframe year the output holds, on the explorer grid:

  thickness     cell-mean ice thickness (ice volume / cell area), m
  ice_fraction  fraction of the cell covered by ice
  speed         ice-area-weighted mean of the depth-averaged speed |(xvelmean, yvelmean)|,
                m/yr, where at least `--min-cover` of the cell is ice with a valid velocity;
                NaN elsewhere

Resampling is conservative: every target value is an area-weighted sum over the exact overlap
of source and target cells, so ice volume is preserved (checked and stored in the output).
Records are matched to years by position, as ISMIP6's own process_scalars.m does: record k is
the end of year 2015 + k. A run with 287 records (LSCE) starts with its initial state, which is
skipped. Depth-averaged velocity is used because three of the eight models did not submit
surface velocity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4
import numpy as np

FILL_LIMIT = 1e19  # ISMIP6 missing value is 1e20
SECONDS_PER_YEAR = 31_556_926.0  # value used throughout the ISMIP6 protocol
FIRST_YEAR = 2015
MAX_RELATIVE_VOLUME_ERROR = 1e-6
ISMIP6_HALF_WIDTH_M = 3_040_000.0
VARIABLES = ("lithk", "sftgif", "xvelmean", "yvelmean")


def overlap_weights(src_centers: np.ndarray, src_width: float, dst_centers: np.ndarray, dst_width: float) -> np.ndarray:
    """Fraction of each target cell's width covered by each source cell, shape (n_dst, n_src)."""
    src = np.asarray(src_centers, dtype=np.float64)
    dst = np.asarray(dst_centers, dtype=np.float64)
    lo = np.maximum((dst - dst_width / 2)[:, None], (src - src_width / 2)[None, :])
    hi = np.minimum((dst + dst_width / 2)[:, None], (src + src_width / 2)[None, :])
    return np.clip(hi - lo, 0.0, None) / dst_width


def area_sum(field: np.ndarray, weight_y: np.ndarray, weight_x: np.ndarray) -> np.ndarray:
    """Sum of a source field over each target cell, in units of target-cell area."""
    return weight_y @ field @ weight_x.T


def explorer_axes(grid: dict) -> tuple[np.ndarray, np.ndarray]:
    x = grid["x0_m"] + grid["dx_m"] * np.arange(grid["nx"], dtype=np.float64)
    y = grid["y0_m"] + grid["dy_m"] * np.arange(grid["ny"], dtype=np.float64)
    return x, y


def usable_axis(values: np.ndarray, n: int) -> bool:
    """A real coordinate axis: n finite, evenly spaced values inside the ISMIP6 domain."""
    if values.size != n or not np.all(np.isfinite(values)) or np.any(np.abs(values) > 2 * ISMIP6_HALF_WIDTH_M):
        return False
    steps = np.diff(values)
    return bool(np.all(steps != 0) and np.allclose(steps, steps[0], rtol=1e-6))


def source_axes(ds: netCDF4.Dataset, ny: int, nx: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Cell-centre coordinates of the source grid.

    Falls back to the standard ISMIP6 grid (centres from -3040 to +3040 km, y increasing) when
    the file has no usable axes: IMAU declares x and y but never wrote them, so they hold the
    netCDF default fill value.
    """
    if "x" in ds.variables and "y" in ds.variables:
        x = np.ma.filled(np.ma.asarray(ds["x"][:], dtype=np.float64), np.nan)
        y = np.ma.filled(np.ma.asarray(ds["y"][:], dtype=np.float64), np.nan)
        if usable_axis(x, nx) and usable_axis(y, ny):
            return x, y, False
    x = np.linspace(-ISMIP6_HALF_WIDTH_M, ISMIP6_HALF_WIDTH_M, nx)
    y = np.linspace(-ISMIP6_HALF_WIDTH_M, ISMIP6_HALF_WIDTH_M, ny)
    return x, y, True


def record_index(year: int, n_records: int) -> int:
    if n_records not in (285, 286, 287):
        raise ValueError(f"unexpected record count {n_records}")
    index = year - FIRST_YEAR + (1 if n_records == 287 else 0)
    if not 0 <= index < n_records:
        raise ValueError(f"year {year} is outside a run of {n_records} records")
    return index


def read_record(ds: netCDF4.Dataset, name: str, index: int) -> np.ndarray:
    values = np.ma.filled(np.ma.asarray(ds[name][index], dtype=np.float64), np.nan)
    values[np.abs(values) > FILL_LIMIT] = np.nan
    return values


def resample_record(
    thickness: np.ndarray, fraction: np.ndarray, u: np.ndarray, v: np.ndarray,
    weight_y: np.ndarray, weight_x: np.ndarray, min_cover: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Cell-mean thickness, ice fraction and ice-weighted speed on the target grid.

    Returns those three fields and the source ice volume per unit source-cell area summed over
    the grid (for the conservation check). Thickness is weighted by sftgif, as process_scalars.m
    weights it when it integrates volume, and ice counts only where sftgif > 0 and lithk > 0.
    """
    f = np.clip(np.nan_to_num(fraction, nan=0.0), 0.0, 1.0)
    h = np.nan_to_num(thickness, nan=0.0)
    f_ice = np.where(h > 0, f, 0.0)
    volume = h * f_ice
    speed = np.hypot(u, v) * SECONDS_PER_YEAR
    valid = np.isfinite(speed) & (f_ice > 0)
    weight = np.where(valid, f_ice, 0.0)

    h10 = area_sum(volume, weight_y, weight_x)
    f10 = area_sum(f_ice, weight_y, weight_x)
    cover = area_sum(weight, weight_y, weight_x)
    numerator = area_sum(np.where(valid, speed, 0.0) * weight, weight_y, weight_x)
    s10 = np.full(h10.shape, np.nan)
    keep = cover >= min_cover
    s10[keep] = numerator[keep] / cover[keep]
    return h10, f10, s10, float(volume.sum())


def run_files(run_dir: Path) -> dict[str, Path]:
    files = {}
    for name in VARIABLES:
        matches = sorted(run_dir.glob(f"{name}_AIS_*.nc"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one {name}_AIS_*.nc in {run_dir}, found {len(matches)}")
        files[name] = matches[0]
    return files


def write_output(path: Path, years: list[int], frames: dict[str, np.ndarray], grid: dict, attrs: dict) -> None:
    x, y = explorer_axes(grid)
    with netCDF4.Dataset(path, "w") as out:
        out.createDimension("year", len(years))
        out.createDimension("y", grid["ny"])
        out.createDimension("x", grid["nx"])
        out.createVariable("year", "i4", ("year",))[:] = years
        out.createVariable("x", "f8", ("x",))[:] = x
        out.createVariable("y", "f8", ("y",))[:] = y
        units = {"thickness": "m", "ice_fraction": "1", "speed": "m yr-1"}
        for name, data in frames.items():
            var = out.createVariable(
                name, "f4", ("year", "y", "x"), zlib=True, complevel=4,
                chunksizes=(1, grid["ny"], grid["nx"]), fill_value=np.float32(np.nan),
            )
            var.units = units[name]
            var[:] = data.astype(np.float32)
        for key, value in attrs.items():
            out.setncattr(key, value if isinstance(value, str) else json.dumps(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="experiment folder of one configuration")
    parser.add_argument("--model", required=True, help="public model name, e.g. UCM_Yelmo")
    parser.add_argument("--experiment", required=True, help="e.g. expAE05")
    parser.add_argument("--grid-meta", type=Path, required=True, help="3D ICE package meta.json with the target grid")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval-years", type=int, default=5)
    parser.add_argument("--last-year", type=int, default=2300)
    parser.add_argument("--years", type=int, nargs="*", help="explicit years instead of the keyframe series")
    parser.add_argument("--min-cover", type=float, default=0.5)
    args = parser.parse_args()

    grid = json.loads(args.grid_meta.read_text())["grid"]
    xt, yt = explorer_axes(grid)
    files = run_files(args.run_dir)
    datasets = {name: netCDF4.Dataset(path) for name, path in files.items()}
    try:
        lithk = datasets["lithk"]["lithk"]
        n_records, ny, nx = lithk.shape
        for name in VARIABLES[1:]:
            shape = datasets[name][name].shape
            if shape != lithk.shape:
                raise ValueError(f"{name} has shape {shape}, lithk {lithk.shape}: records would not line up")
        xs, ys, constructed = source_axes(datasets["lithk"], ny, nx)
        dxs = float(abs(xs[1] - xs[0]))
        weight_x = overlap_weights(xs, dxs, xt, abs(grid["dx_m"]))
        weight_y = overlap_weights(ys, float(abs(ys[1] - ys[0])), yt, abs(grid["dy_m"]))
        years = args.years or list(range(FIRST_YEAR, args.last_year + 1, args.interval_years))
        frames = {name: np.zeros((len(years), grid["ny"], grid["nx"])) for name in ("thickness", "ice_fraction", "speed")}
        conservation = []
        for k, year in enumerate(years):
            index = record_index(year, n_records)
            fields = [read_record(datasets[name], name, index) for name in VARIABLES]
            h10, f10, s10, source_volume = resample_record(*fields, weight_y, weight_x, args.min_cover)
            frames["thickness"][k], frames["ice_fraction"][k], frames["speed"][k] = h10, f10, s10
            native = source_volume * dxs**2
            regridded = float(h10.sum()) * abs(grid["dx_m"] * grid["dy_m"])
            conservation.append(abs(regridded - native) / max(native, 1.0))
            print(f"{args.model} {args.experiment} {year}: volume {native / 1e9:.1f} km3, "
                  f"ice cells {int((h10 > 0).sum())}, speed cells {int(np.isfinite(s10).sum())}", flush=True)
    finally:
        for ds in datasets.values():
            ds.close()

    if max(conservation) > MAX_RELATIVE_VOLUME_ERROR:
        # The weights lose or invent ice: the source grid does not sit where its axes say.
        raise ValueError(f"resampling changed the ice volume by up to {max(conservation):.2e}")
    attrs = {
        "title": f"ISMIP6 Antarctica 2300, {args.model} {args.experiment}, resampled to the 3D ICE 10 km grid",
        "source_files": [str(path) for path in files.values()],
        "source_grid": {"nx": int(nx), "ny": int(ny), "dx_m": dxs, "coordinates_constructed": constructed},
        "records_in_source": int(n_records),
        "time_convention": "record k is the end of year 2015 + k; a 287-record run skips its initial state",
        "max_relative_volume_error": max(conservation),
        "min_cover": args.min_cover,
        "velocity": "depth-averaged (xvelmean, yvelmean), converted with 31556926 s/yr",
    }
    write_output(args.out, years, frames, grid, attrs)
    print(f"wrote {args.out}; max relative volume error {max(conservation):.2e}", flush=True)


if __name__ == "__main__":
    main()
