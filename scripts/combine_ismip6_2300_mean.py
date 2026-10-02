#!/usr/bin/env python3
"""Multi-model mean of the thickness and speed change for one ISMIP6 2300 experiment.

Reads the per-run files written by regrid_ismip6_2300_run.py (all on the 3D ICE 10 km grid, same
keyframe years) and writes, for every keyframe year:

  dthickness_mean  mean over the models of (thickness(t) - thickness(2015)), m.
                   Thickness is the cell-mean (volume / area), so this conserves the
                   ensemble-mean volume change. Every model contributes everywhere
                   (zero change where it has no ice at either time).
  dthickness_std   standard deviation of that change across the models (ddof = 1), m
  dspeed_mean      mean over the models of (speed(t) - speed(2015)), m/yr, using only the
                   models with a usable speed in the cell at both times; NaN if none.
                   A speed is usable where the model's cell-mean thickness is at least
                   --min-speed-thickness and the speed itself at most --max-speed: thin
                   remnant ice can carry unphysical speeds (UCSD_ISSM keeps a ~1 m film at
                   a fixed front that reaches millions of m/yr).
  n_speed          number of models behind dspeed_mean
  n_speed_rejected number of models whose speed was rejected by those two tests

plus the per-model and mean ice-volume change, for checking against the published scalars.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4
import numpy as np


def create(out: netCDF4.Dataset, name: str, dtype: str, dims: tuple[str, ...], units: str, fill) -> netCDF4.Variable:
    chunks = tuple(1 if d == "year" else len(out.dimensions[d]) for d in dims)
    var = out.createVariable(name, dtype, dims, zlib=True, complevel=4, chunksizes=chunks, fill_value=fill)
    var.units = units
    return var


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--in-dir", type=Path, default=Path("regridded"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-speed-thickness", type=float, default=10.0, help="m of cell-mean thickness")
    parser.add_argument("--max-speed", type=float, default=20000.0, help="m/yr")
    args = parser.parse_args()

    sources = [netCDF4.Dataset(args.in_dir / f"{m}_{args.experiment}.nc") for m in args.models]
    try:
        years = sources[0]["year"][:].tolist()
        for model, ds in zip(args.models, sources):
            if ds["year"][:].tolist() != years:
                raise ValueError(f"{model} has different keyframe years")
        ny, nx = sources[0]["thickness"].shape[1:]
        cell_area = abs(float(np.diff(sources[0]["x"][:2])[0]) * float(np.diff(sources[0]["y"][:2])[0]))
        h0 = np.stack([np.ma.filled(ds["thickness"][0].astype(np.float64), 0.0) for ds in sources])
        s0 = np.stack([np.ma.filled(ds["speed"][0].astype(np.float64), np.nan) for ds in sources])

        def usable(speed: np.ndarray, thickness: np.ndarray) -> np.ndarray:
            return np.isfinite(speed) & (thickness >= args.min_speed_thickness) & (speed <= args.max_speed)

        usable0 = usable(s0, h0)

        with netCDF4.Dataset(args.out, "w") as out:
            out.createDimension("year", len(years))
            out.createDimension("model", len(args.models))
            out.createDimension("y", ny)
            out.createDimension("x", nx)
            out.createVariable("year", "i4", ("year",))[:] = years
            out.createVariable("x", "f8", ("x",))[:] = sources[0]["x"][:]
            out.createVariable("y", "f8", ("y",))[:] = sources[0]["y"][:]
            nan32 = np.float32(np.nan)
            dh_mean = create(out, "dthickness_mean", "f4", ("year", "y", "x"), "m", nan32)
            dh_std = create(out, "dthickness_std", "f4", ("year", "y", "x"), "m", nan32)
            ds_mean = create(out, "dspeed_mean", "f4", ("year", "y", "x"), "m yr-1", nan32)
            n_speed = create(out, "n_speed", "i1", ("year", "y", "x"), "1", np.int8(-1))
            n_rejected = create(out, "n_speed_rejected", "i1", ("year", "y", "x"), "1", np.int8(-1))
            create(out, "thickness_2015", "f4", ("model", "y", "x"), "m", nan32)[:] = h0.astype(np.float32)
            create(out, "speed_2015", "f4", ("model", "y", "x"), "m yr-1", nan32)[:] = s0.astype(np.float32)
            volume = out.createVariable("volume_change_km3", "f8", ("model", "year"))
            volume.units = "km3"
            volume_mean = out.createVariable("volume_change_mean_km3", "f8", ("year",))
            volume_mean.units = "km3"

            for k in range(len(years)):
                h = np.stack([np.ma.filled(ds["thickness"][k].astype(np.float64), 0.0) for ds in sources])
                s = np.stack([np.ma.filled(ds["speed"][k].astype(np.float64), np.nan) for ds in sources])
                dh = h - h0
                finite_both = np.isfinite(s) & np.isfinite(s0)
                keep = usable0 & usable(s, h)
                ds_models = np.where(keep, s - s0, np.nan)
                count = keep.sum(axis=0)
                rejected_here = (finite_both & ~keep).sum(axis=0)
                with np.errstate(invalid="ignore"):
                    mean_speed = np.where(count > 0, np.nansum(ds_models, axis=0) / np.maximum(count, 1), np.nan)
                dh_mean[k] = dh.mean(axis=0).astype(np.float32)
                dh_std[k] = dh.std(axis=0, ddof=1).astype(np.float32)
                ds_mean[k] = mean_speed.astype(np.float32)
                n_speed[k] = count.astype(np.int8)
                n_rejected[k] = rejected_here.astype(np.int8)
                change = dh.sum(axis=(1, 2)) * cell_area / 1e9
                volume[:, k] = change
                volume_mean[k] = float(change.mean())
                print(f"{args.experiment} {years[k]}: mean volume change {change.mean():.0f} km3", flush=True)

            out.title = f"ISMIP6 Antarctica 2300 {args.experiment}: mean change of {len(args.models)} main submissions on the 3D ICE 10 km grid"
            out.models = json.dumps(args.models)
            out.speed_filter = json.dumps({"min_cell_mean_thickness_m": args.min_speed_thickness, "max_speed_m_per_yr": args.max_speed})
            out.method = (
                "Each model resampled conservatively to the explorer grid (regrid_ismip6_2300_run.py); "
                "change relative to the model's own 2015 keyframe; equal weight per model."
            )
    finally:
        for ds in sources:
            ds.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
