#!/usr/bin/env python3
"""Pack an ISMIP6 Antarctica 2300 multi-model mean change into a 3D ICE projection package.

Input is the ensemble-mean file written on Gadi by `combine_ismip6_2300_mean.py`: the mean
change of ice thickness and of depth-averaged ice speed since 2015, on the explorer's 10 km
grid at keyframe years. The change is applied to the explorer's present-day ice (BedMachine
Antarctica v4, the `balanced` terrain package), so the 2015 frame is exactly the ice the
explorer already draws:

    thickness(t)    = H_bedmachine + mean dH(t)                      where mean dH(t) >= 0
                    = H_bedmachine * (H_2015 + mean dH(t)) / H_2015  where it thins
                      on present-day ice, 0 elsewhere; H_2015 is the models' mean 2015
                      thickness, so thinning keeps the fraction of ice the models keep.
                      0 where the models' mean thickness drops below 10 m; at least 1 m
                      where ice remains
    speed_change(t) = mean dSpeed(t)                      where at least `--min-speed-models`
                                                          models have ice with a valid speed

The package uses the same sparse time-series layout as the ISMIP7 projection packages
(docs/data-contract.md, "Ice-sheet projections"), with one extra frame-major field,
`speed_change`. The browser rebuilds surface and base from thickness and the BedMachine bed
by flotation.

The sea-level series stored with the package is the mean of the published scalars of the same
models (Seroussi & Pelle 2024, doi:10.5281/zenodo.10528582), so the number on screen is the
ensemble's own. The package also records how far the volume above flotation of the packed
geometry departs from it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INT16_FILL = -32768
GT_PER_MM_SLE = 362.5
RHO_ICE = 917.0
RHO_SEAWATER = 1028.0
PRESENT_DAY_ICE_FLAGS = (2, 3, 4)  # BedMachine: grounded ice, floating ice, Lake Vostok
DEFAULT_MIN_SPEED_MODELS = 4
# Well above any modelled or observed ice speed change; only guards the int16 storage.
SPEED_CHANGE_LIMIT_M_PER_YR = 30000.0
# Ice counts as lost where the models' mean thickness falls below this (the thickness below
# which the ensemble step already ignores a model's speed), and comes back only once it
# regrows past the second value, so cells near the threshold do not blink during playback.
LOST_ICE_THICKNESS_M = 10.0
REGROWN_ICE_THICKNESS_M = 20.0
# Thickness is packed in whole metres; remaining ice is kept at least this thick so that
# rounding never removes it.
MIN_ICE_THICKNESS_M = 1.0

EXPERIMENTS = {
    "expAE05": {
        "climate_model": "UKESM1-0-LL",
        "scenario": "SSP5-8.5",
        "label": "High emissions (SSP5-8.5)",
        "forcing": "UKESM1-0-LL atmosphere and ocean forcing extended to 2300",
        "shelf_collapse": False,
    },
    "expAE10": {
        "climate_model": "UKESM1-0-LL",
        "scenario": "SSP1-2.6",
        "label": "Low emissions (SSP1-2.6)",
        "forcing": "UKESM1-0-LL atmosphere and ocean forcing extended to 2300",
        "shelf_collapse": False,
    },
    "expAE14": {
        "climate_model": "UKESM1-0-LL",
        "scenario": "SSP5-8.5",
        "label": "High emissions with ice-shelf collapse (SSP5-8.5)",
        "forcing": "UKESM1-0-LL forcing extended to 2300, with ice-shelf collapse prescribed from surface melt",
        "shelf_collapse": True,
    },
}

SOURCES = {
    "model_output": {
        "citation": "Nowicki, S. and ISMIP6 Team (2024): ISMIP6 23rd Century Projections [Dataset]. Zenodo.",
        "doi": "10.5281/zenodo.13135599",
        "license": "CC-BY-4.0",
    },
    "scalars": {
        "citation": "Seroussi, H., Pelle, T. and ISMIP6 Team (2024): Dataset and scripts for 'Evolution of the Antarctic Ice Sheet over the next three centuries from an ISMIP6 model ensemble' [Dataset]. Zenodo.",
        "doi": "10.5281/zenodo.10528582",
        "license": "CC-BY-4.0",
    },
    "paper": {
        "citation": "Seroussi, H. et al. (2024): Evolution of the Antarctic Ice Sheet over the next three centuries from an ISMIP6 model ensemble. Earth's Future 12, e2024EF004561.",
        "doi": "10.1029/2024EF004561",
    },
    "protocol": {
        "citation": "Nowicki, S. et al. (2020): Experimental protocol for sea level projections from ISMIP6 stand-alone ice sheet models. The Cryosphere 14, 2331-2368.",
        "doi": "10.5194/tc-14-2331-2020",
    },
    "present_day_geometry": {
        "citation": "Morlighem, M. et al. (2020): Deep glacial troughs and stabilizing ridges unveiled beneath the margins of the Antarctic ice sheet. Nat. Geosci. 13, 132-137 (BedMachine Antarctica v4).",
        "doi": "10.1038/s41561-019-0510-8",
    },
    "also_cite": [
        "Barthel, A. et al. (2020), The Cryosphere 14, 855-879, doi:10.5194/tc-14-855-2020 (CMIP model selection)",
        "Jourdain, N. C. et al. (2020), The Cryosphere 14, 3111-3134, doi:10.5194/tc-14-3111-2020 (basal melt)",
    ],
    "acknowledgement": "We acknowledge the World Climate Research Programme, which coordinated CMIP6, the climate modelling groups, ESGF, ISMIP6 and the ice sheet modelling groups whose simulations are averaged here, and Ghub for distributing the data.",
}


# ---------------------------------------------------------------------------- inputs


def read_terrain(meta_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Grid and decoded fields (float, NaN for fill) of an explorer terrain package."""
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    raw = Path(meta_path).with_name(Path(meta_path).name.replace(".meta.json", ".bin")).read_bytes()
    grid = meta["grid"]
    shape = (grid["ny"], grid["nx"])
    quant = meta.get("quantization", {})
    fill = quant.get("int16_fill_value", INT16_FILL)
    fields = {}
    for field in meta["fields"]:
        start, length = field["byte_offset"], field["byte_length"]
        if field["dtype"] == "int16":
            codes = np.frombuffer(raw, dtype="<i2", count=length // 2, offset=start).reshape(shape)
            values = codes.astype(np.float64) * quant.get("scale", 1.0) + quant.get("offset", 0.0)
            fields[field["name"]] = np.where(codes == fill, np.nan, values)
        elif field["dtype"] == "uint8":
            fields[field["name"]] = np.frombuffer(raw, dtype=np.uint8, count=length, offset=start).reshape(shape)
    return meta, fields


def read_ensemble(path: Path) -> dict:
    import netCDF4  # noqa: PLC0415 - only the packing run needs it

    with netCDF4.Dataset(path) as ds:
        return {
            "years": [int(v) for v in ds["year"][:]],
            "x": np.asarray(ds["x"][:], dtype=np.float64),
            "y": np.asarray(ds["y"][:], dtype=np.float64),
            "dthickness": np.ma.filled(ds["dthickness_mean"][:].astype(np.float64), 0.0),
            "start_thickness": np.ma.filled(ds["thickness_2015"][:].astype(np.float64), 0.0).mean(axis=0),
            "dspeed": np.ma.filled(ds["dspeed_mean"][:].astype(np.float64), np.nan),
            "n_speed": np.ma.filled(ds["n_speed"][:].astype(np.int64), 0),
            "volume_change_mean_km3": np.asarray(ds["volume_change_mean_km3"][:], dtype=np.float64),
            "models": json.loads(ds.getncattr("models")),
            "title": ds.getncattr("title"),
            "speed_filter": json.loads(ds.getncattr("speed_filter")) if "speed_filter" in ds.ncattrs() else None,
        }


def check_same_grid(grid: dict, x: np.ndarray, y: np.ndarray) -> None:
    expected_x = grid["x0_m"] + grid["dx_m"] * np.arange(grid["nx"])
    expected_y = grid["y0_m"] + grid["dy_m"] * np.arange(grid["ny"])
    if x.shape != expected_x.shape or y.shape != expected_y.shape:
        raise ValueError("ensemble grid size differs from the terrain package")
    if not (np.allclose(x, expected_x) and np.allclose(y, expected_y)):
        raise ValueError("ensemble grid coordinates differ from the terrain package")


def published_mean_sle(scalars_dir: Path, models: list[str], experiment: str) -> dict:
    """Annual mean, min and max of the models' published `sle` change since their 2015 record.

    A 287-record series (LSCE) starts with the initial state, which is dropped, matching how
    the 2D fields were read; every series then has its first value at 2015.
    """
    import netCDF4  # noqa: PLC0415

    curves = []
    for model in models:
        path = scalars_dir / experiment / "sle" / f"computed_sle_AIS_{model}_{experiment}.nc"
        with netCDF4.Dataset(path) as ds:
            values = np.asarray(ds["sle"][:], dtype=np.float64)
        if values.size == 287:
            values = values[1:]
        curves.append(values - values[0])
    length = min(c.size for c in curves)
    stack = np.stack([c[:length] for c in curves])
    return {
        "years": list(range(2015, 2015 + length)),
        "mean": stack.mean(axis=0),
        "min": stack.min(axis=0),
        "max": stack.max(axis=0),
        "per_model_end": {model: float(c[length - 1]) for model, c in zip(models, curves)},
    }


# ---------------------------------------------------------------------------- frames


def present_day_ice(thickness: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (np.nan_to_num(thickness, nan=0.0) > 0) & np.isin(mask, PRESENT_DAY_ICE_FLAGS)


def thickness_frames(
    h_now: np.ndarray, domain: np.ndarray, dthickness: np.ndarray, start_thickness: np.ndarray
) -> np.ndarray:
    """Present-day thickness carrying the models' mean change, for every domain cell.

    `start_thickness` is the models' mean 2015 thickness, the reference of `dthickness`.
    Thickening is added as it is. Thinning is scaled by today's thickness over the models'
    2015 thickness, so a cell keeps the fraction of its ice that the models keep of theirs:
    ice the models lose entirely is lost entirely here, and never more than that. Adding the
    raw mean thinning instead removes too much where the models start thicker than BedMachine
    and too little where they start thinner; on ice shelves, which most models thin by
    nearly their whole thickness, that leaves them riddled with holes.

    Where the models' mean thickness falls below LOST_ICE_THICKNESS_M from at least that in
    2015, the ice is gone: what scaling would leave there is a film of a few metres that the
    models have all but lost, and drawing it would show a collapsed shelf as still there.
    Remaining ice is at least MIN_ICE_THICKNESS_M, so packing in whole metres cannot open
    holes either.
    """
    base = np.nan_to_num(h_now, nan=0.0)[domain][None, :]
    change = dthickness[:, domain]
    start = np.nan_to_num(start_thickness, nan=0.0)[domain][None, :]
    models_now = start + change
    kept = np.divide(np.maximum(models_now, 0.0), start, out=np.ones(change.shape), where=start > 0)
    frames = np.where(change < 0, base * np.minimum(kept, 1.0), base + change)
    lost = lost_ice(models_now, start[0] >= LOST_ICE_THICKNESS_M)
    return np.where(lost | (frames <= 0), 0.0, np.maximum(frames, MIN_ICE_THICKNESS_M))


def lost_ice(models_now: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Frames x cells: where the models' mean thickness has dropped below LOST_ICE_THICKNESS_M.

    A cell that has lost its ice keeps it lost until the mean regrows past
    REGROWN_ICE_THICKNESS_M. Only `eligible` cells (models had the ice in 2015) can be lost.
    """
    lost = np.zeros(models_now.shape, dtype=bool)
    previous = np.zeros(models_now.shape[1], dtype=bool)
    for k in range(models_now.shape[0]):
        limit = np.where(previous, REGROWN_ICE_THICKNESS_M, LOST_ICE_THICKNESS_M)
        previous = eligible & (models_now[k] < limit)
        lost[k] = previous
    return lost


def speed_change_frames(
    dspeed: np.ndarray, n_speed: np.ndarray, domain: np.ndarray, frames: np.ndarray, min_models: int
) -> np.ndarray:
    """Mean speed change where enough models agree there is moving ice and ice remains; else 0."""
    values = dspeed[:, domain]
    usable = (n_speed[:, domain] >= min_models) & np.isfinite(values) & (frames > 0)
    return np.clip(np.where(usable, values, 0.0), -SPEED_CHANGE_LIMIT_M_PER_YR, SPEED_CHANGE_LIMIT_M_PER_YR)


def flotation_vaf_m3(bed: np.ndarray, frames: np.ndarray, density_ratio: float, cell_area_m2: float) -> np.ndarray:
    """Ice volume above flotation of every frame (grid area, no map-scale correction)."""
    flotation = np.maximum(0.0, -np.nan_to_num(bed, nan=0.0)) / density_ratio
    excess = np.clip(frames - flotation[None, :], 0.0, None)
    return excess.sum(axis=1) * cell_area_m2


def sle_from_vaf(vaf_m3: np.ndarray, rho_ice: float) -> np.ndarray:
    """Sea-level contribution relative to the first frame, at 362.5 Gt of ice per mm."""
    return -(vaf_m3 - vaf_m3[0]) * rho_ice / 1e12 / GT_PER_MM_SLE / 1000.0


# ---------------------------------------------------------------------------- packaging


def quantize_int16(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    out = np.full(array.shape, INT16_FILL, dtype=np.int16)
    valid = np.isfinite(array)
    out[valid] = np.rint(np.clip(array[valid], -32767, 32767)).astype(np.int16)
    return out


def finite_stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {"min": float(finite.min()), "max": float(finite.max()), "mean": float(finite.mean())}


def write_package(
    output_dir: Path,
    basename: str,
    *,
    grid: dict,
    domain: np.ndarray,
    bed: np.ndarray,
    thickness: np.ndarray,
    speed_change: np.ndarray,
    years: list[int],
    extra_meta: dict,
) -> dict:
    """Write `<basename>.bin` and `.meta.json`.

    Layout: `thickness` (int16, frame-major), `speed_change` (int16, frame-major), `bed` (int16,
    domain cells), then `domain_mask` (uint8, full grid). Domain cells are the mask's non-zero
    cells in row-major order. int16 fields come first so they start on even offsets.
    """
    mask = np.asarray(domain, dtype=bool)
    count = int(mask.sum())
    if mask.shape != (grid["ny"], grid["nx"]):
        raise ValueError(f"domain shape {mask.shape} does not match the grid")
    for name, frames in (("thickness", thickness), ("speed_change", speed_change)):
        if frames.shape != (len(years), count):
            raise ValueError(f"{name} frames {frames.shape} do not match {len(years)} years x {count} cells")
    if bed.shape != (count,):
        raise ValueError(f"bed has {bed.shape} values for {count} domain cells")

    blobs = [
        quantize_int16(thickness).astype("<i2").tobytes(),
        quantize_int16(speed_change).astype("<i2").tobytes(),
        quantize_int16(bed).astype("<i2").tobytes(),
        mask.astype(np.uint8).tobytes(),
    ]
    offsets = np.concatenate([[0], np.cumsum([len(blob) for blob in blobs])]).tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{basename}.bin").write_bytes(b"".join(blobs))

    def field(name: str, index: int, dtype: str, **extra) -> dict:
        return {"name": name, "dtype": dtype, "byte_offset": offsets[index],
                "byte_length": offsets[index + 1] - offsets[index], **extra}

    meta = {
        **extra_meta,
        "geometry_type": "sparse_grid_time_series",
        "grid": dict(grid),
        "domain": {"cell_count": count, "order": "row-major order of the non-zero cells of domain_mask"},
        "frames": {
            "count": len(years),
            "years": [int(y) for y in years],
            "layout": "frame-major: field[frame * cell_count + cell] for thickness and speed_change",
        },
        "quantization": {"unit": "m", "scale": 1.0, "offset": 0.0, "int16_fill_value": INT16_FILL},
        "fields": [
            field("thickness", 0, "int16", unit="m", stats_m=finite_stats(thickness)),
            field("speed_change", 1, "int16", unit="m/year", stats_m_per_year=finite_stats(speed_change)),
            field("bed", 2, "int16", unit="m", stats_m=finite_stats(bed)),
            field("domain_mask", 3, "uint8", flags={"0": "outside_projection_domain", "1": "projection_domain"}),
        ],
    }
    (output_dir / f"{basename}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ensemble", type=Path, required=True, help="ismip6_2300_mean8_<exp>_10km.nc from Gadi")
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--scalars", type=Path, required=True, help="unzipped ComputedScalars (Zenodo 10528582)")
    parser.add_argument("--terrain-meta", type=Path, default=Path("static/tools/data/bedmachine_antarctica_v4_480.meta.json"))
    parser.add_argument("--min-speed-models", type=int, default=DEFAULT_MIN_SPEED_MODELS)
    parser.add_argument("--output-dir", type=Path, default=Path("static/tools/data"))
    parser.add_argument("--basename", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    density_ratio = RHO_ICE / RHO_SEAWATER
    terrain_meta, terrain = read_terrain(args.terrain_meta)
    grid = {key: terrain_meta["grid"][key] for key in ("nx", "ny", "x0_m", "y0_m", "dx_m", "dy_m")}
    ensemble = read_ensemble(args.ensemble)
    check_same_grid(grid, ensemble["x"], ensemble["y"])
    years = ensemble["years"]
    if years[0] != 2015:
        raise ValueError("the first keyframe must be 2015, the reference of the change")

    domain = present_day_ice(terrain["thickness"], terrain["mask"])
    frames = thickness_frames(terrain["thickness"], domain, ensemble["dthickness"], ensemble["start_thickness"])
    speed = speed_change_frames(ensemble["dspeed"], ensemble["n_speed"], domain, frames, args.min_speed_models)
    bed = terrain["bed"][domain]
    if np.any(~np.isfinite(bed)):
        raise ValueError("some present-day ice cells have no BedMachine bed")

    cell_area = abs(grid["dx_m"] * grid["dy_m"])
    sle_packed = sle_from_vaf(flotation_vaf_m3(bed, frames, density_ratio, cell_area), RHO_ICE)
    published = published_mean_sle(args.scalars, ensemble["models"], args.experiment)
    at_keys = np.array([published["mean"][years_index] for years_index in (np.array(years) - 2015)])
    packed_volume = (frames - frames[0][None, :]).sum(axis=1) * cell_area / 1e9
    ensemble_volume = ensemble["volume_change_mean_km3"]
    end_capture = float(packed_volume[-1] / ensemble_volume[-1]) if ensemble_volume[-1] else float("nan")

    info = EXPERIMENTS[args.experiment]
    extra_meta = {
        "title": f"ISMIP6 Antarctica 2300 {args.experiment}: mean of {len(ensemble['models'])} ice sheet models, {info['label']}",
        "product_version": "ISMIP6 Projections 2300 Antarctica (Seroussi et al. 2024), ensemble-mean change applied to BedMachine Antarctica v4",
        "release_status": "Public. Derived from CC BY 4.0 data; cite the sources listed under `sources`.",
        "experiment": {"id": args.experiment, **info},
        "models": ensemble["models"],
        "sources": SOURCES,
        "method": {
            "thickness": "BedMachine v4 thickness carrying the equal-weight mean, over the models, of each model's change in cell-mean thickness since its 2015 record: thickening is added as it is; thinning is scaled by BedMachine thickness over the models' mean 2015 thickness, so each cell keeps the fraction of its ice that the models keep of theirs (H = H_bm * (H_2015 + dH) / H_2015). Present-day ice cells only (no new ice)",
            "lost_ice": f"ice is removed where the models' mean thickness falls below {LOST_ICE_THICKNESS_M:g} m from at least that in 2015 (the film scaling would leave there is ice the models have all but lost), and returns only once their mean regrows past {REGROWN_ICE_THICKNESS_M:g} m so that cells near the threshold do not blink; remaining ice is at least {MIN_ICE_THICKNESS_M:g} m, the packing resolution",
            "thinning_scaled_because": "the models start from their own 2015 ice, which on the ice shelves is 26 % thinner to 18 % thicker than BedMachine cell by cell (10th to 90th percentile); where they thin a shelf by nearly its whole thickness, adding their mean thinning outright empties every cell in which it exceeds BedMachine thickness, riddling the shelf with holes, and leaves ice they have lost wherever BedMachine is the thicker",
            "speed_change": f"equal-weight mean of each model's change in depth-averaged speed since 2015 (ice-area-weighted within each 10 km cell), over the models with usable speeds in the cell at both times; set to 0 where fewer than {args.min_speed_models} models qualify or no ice remains",
            "speed_filter": ensemble["speed_filter"] or "none recorded",
            "regridding": "conservative area-weighted resampling of every model from its own grid (4-32 km) to the explorer's 10 km grid, done on NCI Gadi with regrid_ismip6_2300_run.py and combine_ismip6_2300_mean.py",
            "geometry": "the browser rebuilds surface and base from thickness and the BedMachine bed by flotation: base = max(bed, -H * rho_ice/rho_seawater), surface = base + H",
            "bed": "static BedMachine v4 bed; the models' bedrock adjustment is not shown",
            "time_convention": "record k of every model is the end of year 2015 + k; a 287-record run (LSCE) skips its initial state",
        },
        "physical_constants": {
            "rho_ice_kg_m3": RHO_ICE,
            "rho_seawater_kg_m3": RHO_SEAWATER,
            "density_ratio": density_ratio,
            "gt_per_mm_sle": GT_PER_MM_SLE,
        },
        "source_package": {"grid_from": args.terrain_meta.name, "ensemble_file": args.ensemble.name},
        "keyframes": {"interval_years": int(years[1] - years[0]), "interpolation": "linear in thickness and in speed change"},
        "validation": {
            "sea_level_of_packed_geometry_m": {
                "note": "volume above flotation of the packed frames on the BedMachine bed (grid area, freshwater 362.5 Gt/mm) against the mean of the models' published sle",
                "end_packed": round(float(sle_packed[-1]), 4),
                "end_published_mean": round(float(at_keys[-1]), 4),
                "max_abs_difference": round(float(np.max(np.abs(sle_packed - at_keys))), 4),
            },
            "volume_change_captured": {
                "note": "ice-volume change of the packed frames over the ensemble-mean change on the 10 km grid; departs from 1 where BedMachine and the models' 2015 thickness differ (thinning is scaled by their ratio) and where the change falls outside present-day ice",
                "end_ratio": round(end_capture, 4),
                "end_packed_km3": round(float(packed_volume[-1]), 1),
                "end_ensemble_km3": round(float(ensemble_volume[-1]), 1),
            },
        },
        "series": {
            "years": published["years"],
            "sea_level_contribution_m": [round(float(v), 5) for v in published["mean"]],
            "sea_level_contribution_min_m": [round(float(v), 5) for v in published["min"]],
            "sea_level_contribution_max_m": [round(float(v), 5) for v in published["max"]],
            "per_model_end_m": {k: round(v, 4) for k, v in published["per_model_end"].items()},
            "note": "Annual mean, min and max over the models of the published ISMIP6 sea-level-equivalent change since 2015 (control run not subtracted, as in Seroussi et al. 2024).",
        },
    }
    meta = write_package(
        args.output_dir, args.basename, grid=grid, domain=domain, bed=bed, thickness=frames,
        speed_change=speed, years=years, extra_meta=extra_meta,
    )
    size = (args.output_dir / f"{args.basename}.bin").stat().st_size
    print(f"Wrote {args.basename}: {meta['domain']['cell_count']} cells x {len(years)} keyframes, {size} bytes")
    print(json.dumps(meta["validation"], indent=2))


if __name__ == "__main__":
    main()
